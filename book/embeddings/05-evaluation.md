# 5. Evaluation

There is no single accuracy number for an embedding space. The embedding is not
the final product; the retrieval, ranking, or fraud model it powers is. Evaluate
the space by what it enables, and use multiple lenses because each catches a
different failure mode.

## Offline metric: recall@k against held-out future positives

**Input / output.** The model embeds a user query (or user vector) and issues
a nearest-neighbor lookup against the item index; Recall@k counts what fraction
of the user's held-out future positives appear among the $k$ returned items,
averaged over users, and outputs a scalar in $[0, 1]$.

$$\text{Recall@k} = \frac{1}{|U|}\sum_{u \in U} \frac{|\text{retrieved}_k(u) \cap \text{relevant}(u)|}{|\text{relevant}(u)|}$$

```python
def recall_at_k(retrieved, relevant, k):
    total = 0.0
    for ret, rel in zip(retrieved, relevant):   # one user per row
        rel = set(rel)                           # this user's held-out future positives
        if not rel: continue                     # skip users with nothing to recover
        hits = len(set(ret[:k]) & rel)           # relevant items among the top-k retrieved
        total += hits / len(rel)                 # fraction of this user's positives recovered
    return total / len(retrieved)                # average over users
# recall_at_k([[1,2,3],[4,5,6]], [[2,9],[4,5]], k=2) -> 0.75
```

Measure this at the $k$ you actually pass to downstream consumers, not at $k = 10$
if you pass 500 candidates to ranking. Use a **time-based split**: hold out future
interactions, not a random sample of existing ones. A random split leaks the future
and flatters the model.

Measure **tail recall separately from head recall**. A space that improves average
recall by pushing head items back into the top-k is a space that has gotten worse
for the long tail; average recall hides that. Report both.

## Alignment and uniformity diagnostics

Two properties define a well-formed embedding space, and a space can look fine on
cosine probes while quietly failing on one of them.

**Alignment** measures how close positive pairs are in the embedding space.
Input: a set of positive pairs $(x, x^+)$ drawn from the training signal (for
example, user-item co-engagements); the metric computes expected squared L2
distance (straight-line distance between the two vectors, then squared) between
each pair's embeddings. Output: a non-negative scalar; lower
is better, meaning positive pairs sit near each other:

$$\ell_{\text{align}} = \mathbb{E}_{(x,\, x^{+})} \bigl\lVert f(x) - f(x^{+}) \bigr\rVert^{2}$$

```python
import numpy as np
def alignment(x, xpos):
    x, xpos = np.asarray(x, float), np.asarray(xpos, float)  # each row is one embedding
    return float(np.sum((x - xpos)**2, axis=1).mean())       # mean squared L2 over positive pairs
# alignment([[1,0],[0,1]], [[0,1],[0,1]]) -> 1.0  (first pair differs, second is identical)
```

**Uniformity** measures how spread the embeddings are across the unit
hypersphere. Input: uniformly random pairs $(x, y)$ from the corpus; the metric
computes the log of their average Gaussian-kernel similarity. Output: a negative
scalar; more negative is better (the space is well spread and uses its full
capacity rather than collapsing to a tight cluster):

$$\ell_{\text{unif}} = \log \, \mathbb{E}_{x,\, y} \; e^{-2 \lVert f(x) - f(y) \rVert^{2}}$$

```python
import numpy as np
def uniformity(z):
    z = np.asarray(z, float)                                 # each row is one embedding
    sq = np.sum((z[:, None, :] - z[None, :, :])**2, axis=-1) # pairwise squared L2 distances
    iu = np.triu_indices(len(z), k=1)                        # each unordered pair (x, y) once
    return float(np.log(np.exp(-2 * sq[iu]).mean()))         # log mean Gaussian-kernel similarity
# uniformity([[1,0],[0,1],[-1,0]]) -> -4.3963  (more negative = more spread out)
```

These two losses trade off against each other; the ideal space has both small
alignment loss (positives are close) and small uniformity loss (the full space is
spread). A model with poor negatives or a weak loss can collapse to a tight cluster
where all similarities are high and ranking is meaningless. Only the uniformity
diagnostic catches that.

![Alignment vs uniformity across training methods](assets/fig-alignment-uniformity.png)

*Each point is a training configuration. The ideal region is the lower-left corner:
positives close (small alignment loss) and space spread (small uniformity loss).
Random-negative triplet loss lands far from the ideal corner; InfoNCE with hard
negatives lands closest. Schematic; relative positions are the point.*

### The anisotropy problem in text embeddings

Text encoders built on pretrained language models have a specific, well-documented
uniformity failure worth naming explicitly: **anisotropy**. The raw output vectors
of models like BERT and GPT-2 occupy a narrow cone rather than spreading over the
sphere, so two randomly chosen sentences already sit at high cosine similarity
(Ethayarajh, 2019 measured this geometry across BERT, ELMo, and GPT-2; Gao et al.
2019 traced the training cause and named it the representation degeneration
problem). In the vocabulary of this section, anisotropy is exactly a uniformity
failure, and it is invisible to a plain cosine probe: because every pair, related
or not, reports a high cosine, the absolute similarity number is meaningless and
only the relative ranking of scores carries any signal.

Two families of fix. Post-hoc, you can whiten or standardize the vectors to
re-center and de-correlate them so the cone opens up (BERT-flow, Li et al. 2020,
and related whitening transforms). More durably, contrastive fine-tuning such as
SimCSE directly optimizes uniformity and pulls the geometry off the cone at
training time. This is why dropping a raw pretrained encoder into an ANN index and
trusting cosine usually underperforms a smaller contrastively fine-tuned one:
the larger model may hold more knowledge, but its space is not shaped for
similarity search until something has forced it to be uniform.

## Downstream task lift

The best offline proxy for production value is whether the embedding improves a
downstream task trained on top of it:

- **Retrieval Recall@k** after loading vectors into a real ANN index (end-to-end,
  not just cosine on a probe set).
- **Ranking NDCG@k or MRR** when the embedding is used as a feature in a ranking
  model trained separately. NDCG@k rewards placing relevant items higher:
  $\text{NDCG@k} = \frac{1}{Z}\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$
  where $Z = \text{IDCG@k}$ is the ideal DCG and higher is better. MRR averages
  the reciprocal rank of the first relevant result:
  $\text{MRR} = \frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q}$, with
  $1/\text{rank}_q = 0$ when no relevant result is returned; higher is better.
- **Classification accuracy** on a held-out probe set if you have any labeled
  categories (item category, user segment).
- **Fraud PR-AUC** for systems like Wayfair's Melange, where the whole point is to
  improve a downstream fraud detector with scarce labels. PR-AUC (average
  precision) is the area under the precision-recall curve:
  $\text{AP} = \sum_{k}(R_k - R_{k-1})\cdot P_k$; it is preferred over ROC-AUC
  when fraud events are rare because it focuses entirely on the positive class
  and is not inflated by the large negative mass.

```python
import numpy as np
def dcg(rels):                                    # rels: relevance grades in predicted-rank order
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))  # i=0 -> rank 1 -> log2(2)=1
def ndcg_at_k(rels, k):
    ideal = dcg(sorted(rels, reverse=True)[:k])   # IDCG: grades sorted descending = best ordering
    return dcg(rels[:k]) / ideal if ideal else 0.0
# ndcg_at_k([3, 2, 3, 0, 1], k=5) -> 0.9575  (always in [0, 1]; 1.0 is a perfect ranking)
```

```python
import numpy as np
def mrr(ranks):
    # ranks: 1-based rank of the first relevant result per query, 0 (or None) if none returned
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks]))  # reciprocal rank, averaged
# mrr([1, 3, 0, 2]) -> 0.4583
```

```python
import numpy as np
def average_precision(labels):
    # labels: 0/1 relevance in predicted-rank order (rank 1 first)
    labels = np.asarray(labels, float)
    tp = np.cumsum(labels)                          # true positives seen up to each rank
    precision = tp / (np.arange(len(labels)) + 1)   # P_k: precision at each rank k
    n_pos = labels.sum()
    return float((precision * labels).sum() / n_pos) if n_pos else 0.0  # mean P_k at the hits
# average_precision([1, 0, 1, 0, 1]) -> 0.7556
```

The probe set should not overlap training. If it does, you are measuring
memorization, not generalization.

## Online eval

Offline metrics are necessary but not sufficient. Always gate a launch on an
online A/B experiment measuring:

- **Engagement rate** of the feed or search results the embedding powers (click,
  dwell, purchase).
- **Coverage and diversity**: does the new space surface long-tail items, or does
  it collapse to a head-item echo chamber? A recall win that shrinks catalog
  coverage often loses long-term.
- **New-entity retrievability**: what fraction of fresh items are surfaced within
  the freshness SLA (an hour in our requirements)?

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| Recall@k at the downstream k | evaluating the retrieval stage the embedding feeds | recall@10 when you pass 500 candidates, which optimizes the wrong operating point |
| Tail recall separately | checking for popularity collapse | average recall alone, which hides tail starvation |
| Alignment and uniformity | suspecting representation collapse or diagnosing why a space underperforms | cosine probe accuracy alone, which misses silent collapse |
| Downstream task lift (NDCG, MRR, PR-AUC) | the embedding will be reused across tasks | eval on one task, when the whole economic case is multi-task reuse |
| Online A/B engagement | the final launch decision | offline recall alone, which misses the ranking interaction and diversity effects |
| Time-based split | any offline retrieval eval | random split, which leaks the future |

**Tools.** End-to-end Recall@k is measured by loading vectors into a real ANN index such as FAISS (Meta) and querying it, rather than trusting a brute-force cosine probe. TorchMetrics provides RetrievalRecall, RetrievalNormalizedDCG, and RetrievalMRR for the downstream-lift metrics, and scikit-learn covers PR-AUC (average_precision_score) for scarce-label fraud probes. Alignment and uniformity are two short functions from the Wang-and-Isola formulation computed in PyTorch (Meta); the final online A/B leans on a stats library such as statsmodels.

**Worked example.** A streaming service evaluating a new item-embedding space measures Recall@k at the k it actually passes to ranking (not Recall@10 when it forwards 500 candidates), using a time-based split so held-out future interactions are not leaked by a random split. It reports tail recall separately from head recall to catch a space that lifts the average by pushing popular items back into the top-k. When a cosine probe looks healthy but ranking underperforms, it computes alignment and uniformity and finds the space has partly collapsed, a failure the probe missed. It then loads the vectors into FAISS for true end-to-end Recall@k and computes downstream NDCG and MRR with TorchMetrics, since the embedding is reused across tasks. The launch is gated on an online A/B measuring engagement plus catalog coverage, not offline recall alone.
