# 5. Evaluation

There is no single accuracy number for an embedding space. The embedding is not
the final product; the retrieval, ranking, or fraud model it powers is. Evaluate
the space by what it enables, and use multiple lenses because each catches a
different failure mode.

## Offline metric: recall@k against held-out future positives

Of the items a user will engage with in the future (held out), what fraction
appear in the top $k$ retrieved from the embedding index?

$$\text{Recall@k} = \frac{1}{|U|}\sum_{u \in U} \frac{|\text{retrieved}_k(u) \cap \text{relevant}(u)|}{|\text{relevant}(u)|}$$

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

**Alignment** measures how close positive pairs are in the embedding space. Lower
is better:

$$\ell_{\text{align}} = \mathbb{E}_{(x,\, x^{+})} \bigl\lVert f(x) - f(x^{+}) \bigr\rVert^{2}$$

**Uniformity** measures how spread the embeddings are across the unit
hypersphere. More spread (lower value of the log term) means the space uses its
capacity rather than collapsing:

$$\ell_{\text{unif}} = \log \, \mathbb{E}_{x,\, y} \; e^{-2 \lVert f(x) - f(y) \rVert^{2}}$$

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

## Downstream task lift

The best offline proxy for production value is whether the embedding improves a
downstream task trained on top of it:

- **Retrieval Recall@k** after loading vectors into a real ANN index (end-to-end,
  not just cosine on a probe set).
- **Ranking NDCG or MRR** when the embedding is used as a feature in a ranking
  model trained separately.
- **Classification accuracy** on a held-out probe set if you have any labeled
  categories (item category, user segment).
- **Fraud PR-AUC** for systems like Wayfair's Melange, where the whole point is to
  improve a downstream fraud detector with scarce labels.

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
