# 5. Evaluation

Link prediction is judged as ranking on a time-based split, and the online metric
is acceptance, not clicks.

## Offline metrics

- **AUC / average precision (AP)** over positive edges versus sampled negatives:
  can the model rank a true future edge above a non-edge. AP is more informative
  than AUC under the extreme non-edge imbalance.
- **Hits@k and MRR** per member: of the actual future connections, how many land in
  the top k suggestions, and how high. This matches the product (a short ranked
  list), so it is the metric to headline.
- **Always a time-based split.** Train on the graph as of T, evaluate on edges that
  formed after T. A random edge split leaks the answer.

$$\text{Hits@k} = \frac{1}{|U|}\sum_{u \in U} \mathbf{1}\!\left[\text{a true future edge of } u \in \text{top-}k(u)\right]$$

```python
import numpy as np
def hits_at_k(ranked_labels, k):
    # ranked_labels: per member, a 0/1 list of candidates sorted by score, best first
    hits = []
    for labels in ranked_labels:                # one member at a time
        topk = labels[:k]                       # the member's top-k suggestions
        hits.append(1.0 if any(topk) else 0.0)  # 1 if a true future edge landed in top-k
    return float(np.mean(hits))                 # average over members = Hits@k
# hits_at_k([[0,1,0],[0,0,1]], 2) -> 0.5  (member 1 hits at rank 2, member 2's positive is at rank 3)
```

MRR (mean reciprocal rank) rewards ranking the first true edge high, not just inside
top-k:

```python
def mrr(ranked_labels):
    # ranked_labels: per member, a 0/1 list of candidates sorted by score, best first
    rr = []
    for labels in ranked_labels:
        r = 0.0
        for rank, lab in enumerate(labels, start=1):  # ranks are 1-based
            if lab:
                r = 1.0 / rank                         # reciprocal rank of the first hit
                break
        rr.append(r)
    return sum(rr) / len(rr)                            # average over members
# mrr([[0,1,0],[0,0,1]]) -> (1/2 + 1/3)/2 = 0.41666666666666663
```

AUC is the chance a true edge outscores a sampled non-edge (ties count as half):

```python
def auc(pos_scores, neg_scores):
    # pos_scores: scores of true edges;  neg_scores: scores of sampled non-edges
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n: wins += 1.0
            elif p == n: wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))   # fraction of pos>neg pairs
# auc([0.9, 0.6], [0.5, 0.7, 0.2]) -> 5 of 6 pairs won = 0.8333333333333334
```

AP (average precision) averages the precision measured at each true edge in the
ranking, and is more informative than AUC under heavy non-edge imbalance:

```python
def average_precision(labels):
    # labels: 0/1 list for one member, sorted by score, best first
    hits = 0
    total = 0.0
    for i, lab in enumerate(labels, start=1):
        if lab:
            hits += 1
            total += hits / i        # precision at this true-edge position
    return total / hits if hits else 0.0
# average_precision([1,0,1,0]) -> (1/1 + 2/3)/2 = 0.8333333333333333
```

## Online metrics

- **Invitation acceptance rate**, not invitations sent. The product wins on
  accepted connections; a suggestion that gets invited and rejected is a cost (it
  annoys the recipient), so acceptance is the true objective.
- **Downstream network growth and engagement**, since a good connection drives
  long-term activity, and a bad one adds spammy invites.
- **Coverage and fairness.** Does the system only suggest hub nodes and dense
  communities, starving new or peripheral members? Track suggestion coverage across
  the degree distribution.

```python
def coverage(rec_lists, catalog):
    # rec_lists: the recommended-id lists shown to members
    # catalog: set of all member ids that could be suggested
    shown = set()
    for recs in rec_lists:
        shown.update(recs)
    return len(shown & catalog) / len(catalog)   # fraction of the catalog ever shown
# coverage([[1,2],[2,3]], {1,2,3,4}) -> 3 of 4 members shown = 0.75
```

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| Hits@k / MRR | judging the ranked suggestion list as the product sees it | AUC alone, which does not reflect a short top-k list |
| AP over AUC | the non-edge class massively outnumbers edges | AUC, which looks high even for a weak model under imbalance |
| Time-based split | any offline link-prediction eval | a random edge split, which leaks the future edge |
| Online acceptance rate | the launch decision | invitations sent, which rewards spammy over-suggesting |

The guardrail to state: an offline Hits@k gain must survive an online A/B on
**acceptance rate** and coverage, because a model that only re-suggests popular hubs
can win offline ranking while flooding people with unwanted invites.

**Tools.** Hits@k, MRR, AUC, and AP over positive-versus-sampled-negative edges are
computed with the ranking metrics in PyG (PyTorch Geometric), DGL, or TorchMetrics;
scikit-learn's average_precision_score and roc_auc_score cover the edge-scoring case
directly. Time-based splitting is a pandas or graph-library filter on edge timestamps
(train on the graph as of T, evaluate on edges formed after). Online acceptance rate,
coverage across the degree distribution, and downstream engagement come from the
production experimentation stack joined to invitation outcomes.

**Worked example.** A marketplace suggesting connections between members headlines
Hits@k and MRR, since the product surfaces a short ranked list and AUC alone would not
reflect that top-k shape. Because non-edges massively outnumber true future edges, it
prefers AP over AUC offline, which would otherwise look high even for a weak model
under that imbalance. Every offline number uses a time-based split (train on the graph
as of T, evaluate on edges that formed after) so a random edge split cannot leak the
future connection. The launch decision is made online on invitation acceptance rate
and coverage, not invitations sent, because a model that only re-suggests popular hubs
can win offline ranking while flooding people with unwanted invites.
