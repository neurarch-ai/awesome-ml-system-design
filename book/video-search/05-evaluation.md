# 5. Evaluation

## Different stages, different metrics

Evaluating the whole system with one number hides which stage broke.

| Stage | Metric | Why this one |
|---|---|---|
| Retrieval | Recall@k against the union of known-relevant videos, at the k the ranker receives | Retrieval's only job is not to lose the right answer; precision is the ranker's problem |
| Fusion | Recall@k lift over each retriever alone, and overlap between them | A fusion that adds no unique candidates is paying for nothing |
| Ranking | NDCG@k with graded relevance | Position-weighted, and it uses the grades rather than collapsing them |
| Reranking | NDCG@10 and the top-1 accuracy | Only the visible results matter here |
| Navigational subset | MRR | There is one right answer and its rank is the whole story |
| End to end (online) | Long-watch rate, reformulation rate, abandonment, time to first satisfying result | What the user actually experienced |

The metric that gets misused most is **recall@k for retrieval**: it is only
meaningful against a *known-relevant set* that was not produced by the system under
test. If the relevant set comes from the current production ranker's clicks, a new
retriever that surfaces genuinely better videos scores *worse*, because its finds are
labeled unknown rather than relevant. The fix is the human-rated anchor set plus
pooled judgments across systems.

## Slice or you will ship a regression

An aggregate NDCG hides exactly the thing this system is being built to fix.
Mandatory slices:

- **Query type**: navigational, informational, visual, moment-seeking.
- **Query length**: short keyword versus natural-language.
- **Language**, and specifically transcript-rich versus transcript-poor languages.
- **Video age**: fresh uploads versus the back catalog.
- **Head versus tail query frequency.**

The classic outcome of adding semantic retrieval is a tail win and a head
regression, invisible in the average, and fatal in production because the head is
most of the traffic.

## Online evaluation

Behavioral metrics with their known confounds handled:

- **Long-watch rate**, normalized by video duration so the metric does not simply
  prefer long videos.
- **Reformulation rate**: a user retyping the same intent is the cleanest negative
  signal search has.
- **Abandonment** and **time to first satisfying result**.
- **Guardrails**: latency (a semantic retriever adds a hop), result diversity, and
  the fraction of results from fresh uploads, which is where a new ranker quietly
  starves new creators.

For ranking changes specifically, **interleaving** is far more sensitive than a
conventional A/B because it compares two rankings within the same session, which is
why it is the standard first filter before a full experiment (see
[experimentation](../experimentation/06-interleaving-and-alternatives.md)).

## The offline-online gap

The gap has the same three causes as everywhere in this book, plus one specific to
search:

1. **Training-serving skew** in the query or field features.
2. **Position bias** in the labels: the offline metric rewards agreeing with the old
   ranker.
3. **The metric measures ordering, not the page**: NDCG can improve while the top
   three results are unchanged, and the top three are what users see.
4. **Search-specific: the candidate set changed.** A ranking metric computed over the
   old retriever's candidates is not comparable to one computed over a new
   retriever's candidates. Fix the candidate set when you evaluate ranking, and
   evaluate the retriever separately.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Recall@k on a pooled, human-anchored relevant set | Evaluating retrieval or fusion | Recall against the current system's clicks, which punishes new finds |
| NDCG@k with graded labels | Evaluating ranking | Binary relevance, which cannot express "good but not best" |
| MRR | Navigational queries with one right answer | NDCG, which dilutes a single-answer task |
| Interleaving | Comparing two rankers quickly | A full A/B for every candidate, when slots are scarce |
| Human-rated set | Adjudicating an offline-online disagreement | Arguing about which behavioral proxy is right |
| Duration-normalized watch | Any online quality metric | Raw watch time, which prefers long videos |
