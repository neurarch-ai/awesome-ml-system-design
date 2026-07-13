# 5. Evaluation

Retrieval is judged differently from ranking, and using the wrong metric is a
classic mistake. Retrieval's job is recall, so measure recall.

## Offline metrics

- **Recall@k.** Of the items the user actually engaged with (held out), what
  fraction appear in the top k retrieved? This is the primary metric, because
  retrieval only has to get good items into the candidate set.
- **Do not headline precision here.** A retrieved set of a few hundred is mostly
  "not yet judged," so precision is low by construction and misleading. Precision
  is the ranking stage's metric.

$$\text{Recall@k} = \frac{1}{|U|}\sum_{u \in U} \frac{|\text{retrieved}_k(u) \cap \text{relevant}(u)|}{|\text{relevant}(u)|}$$

Evaluate at the k you actually pass downstream (say a few hundred), not at k=10,
or you will optimize the wrong operating point.

## Online metrics

Offline recall is necessary but not sufficient; a launch is decided online.

- **Engagement rate** of the final feed (click-through, dwell), since retrieval
  changes what ranking even sees.
- **Coverage and diversity.** Does retrieval surface the long tail, or collapse to
  popular items? A recall win that shrinks catalog coverage often loses long-term.
- **New-item retrievability.** The fraction of fresh items that get retrieved
  within minutes, tied directly to the freshness requirement.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| Recall@k (at the downstream k) | judging the retrieval stage in isolation | precision@k, which belongs to ranking |
| Coverage and diversity | you suspect popularity collapse | recall alone, which can improve while diversity craters |
| Online engagement A/B | the final launch decision | offline recall alone, which does not capture the ranking interaction |

The guardrail to state: an offline recall gain must survive an online A/B against
engagement **and** coverage before it ships, because retrieval that only
resurfaces popular items can win recall and lose the product.
