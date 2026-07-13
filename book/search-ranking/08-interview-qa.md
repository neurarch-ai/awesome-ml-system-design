# 8. Interview Q&A

The questions an interviewer actually asks about search ranking, grouped by how
they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Why do you need two retrieval arms instead of one?**
A: Because BM25 and dense retrieval have complementary failure modes. BM25 is
fast and unbeatable on exact-term and rare-term queries (product codes, specific
names) but cannot match "laptop" to "notebook computer." A dense dual-encoder
closes that vocabulary gap via embedding proximity but drifts on exact strings and
rare proper nouns. Running both and taking the union is the only way to cover the
full query distribution without sacrificing either arm's strengths. Neither arm is
optional.

**Q: Why pairwise or listwise loss instead of just predicting a relevance score
per document?**
A: Because the metric is about *order* and is position-weighted. A pointwise model
optimizes absolute scores everywhere, including deep in the list where it does not
matter, and has no way to concentrate capacity on getting the top slots right.
Pairwise (RankNet) trains on "document A should rank above document B," which
matches the task. Listwise (LambdaMART) weights each pairwise gradient by the
NDCG change that swapping the pair would cause, directly aligning the loss with
the metric you report. That alignment is the senior point.

**Q: How do you evaluate a search ranking model offline?**
A: Use NDCG@k at the k you actually render (typically 10), computed against a
held-out, debiased relevance-label set with a time-based split. Use MRR as a
companion for navigational queries. Then gate the launch on an interleaving
experiment or A/B test against engagement and reformulation rate, because offline
NDCG can be optimistic. Never ship on offline NDCG alone.

**Q: Where do your training labels come from?**
A: Two sources, and both are necessary. Click-derived labels are abundant, cover
the full query distribution including the long tail, and reflect real users, but
they are biased by position and noisy (a click is not the same as relevance).
Human-judged labels are trustworthy and unbiased but expensive, slow, and cover
only the head. Fuse them: human judgments anchor calibration and validate the
click signal, click-derived labels provide volume, freshness, and tail coverage.

**Q: How does a brand-new product get retrieved and ranked?**
A: The lexical arm can retrieve it immediately via its title and category terms in
the inverted index. The dense arm needs its embedding, which is computed by a
streaming pipeline and upserted into the ANN index within minutes of listing.
Until the product has enough click history, its ranker features fall back to
content signals (title embedding, category, brand). Cold-start is a retrieval
freshness problem first, then a ranking-features problem.

## Tricky (the follow-ups that separate people)

**Q: Your offline NDCG went up but online engagement did not move. What happened?**
A: Three suspects. First, the label pipeline: if your click labels are not
properly position-debiased, NDCG is computed against biased signals and can lift
while true relevance did not. Check the IPW weights and the debiasing pipeline.
Second, point-in-time correctness: if features join post-query data naively, the
training set leaks future clicks into features and the offline score is inflated.
Third, the gain may be real but deep in the list, where users do not scroll. Check
NDCG@3 and NDCG@5 specifically.

**Q: Your query expansion lifted recall but users started reformulating more often.
What went wrong?**
A: Over-expansion drifted the query from the user's original intent. The expanded
terms brought in semantically related but not truly relevant documents, lowering
precision for the documents actually returned. The fix is conservative expansion
(restrict to high-confidence synonyms and stems), always keep the original query
as a fallback retrieval path, and measure reformulation rate as an explicit cost
alongside recall.

**Q: How is search ranking different from recommendation ranking?**
A: Three structural differences. First, there is an explicit query, so query
understanding is a first-class stage and its errors cascade into both retrieval
arms before ranking ever runs. Second, retrieval needs a lexical arm (for
exact-term and rare-term recall) in addition to the semantic arm; recommendation
retrieval only needs the semantic arm. Third, the dominant label problem is
position-biased clicks rather than multi-task engagement trade-offs. The funnel
shape (retrieve then rank) is shared; the query and the label challenge are what
change.

## Commonly answered wrong (the traps)

**Q: Should you replace BM25 with a dense retrieval model once you have a good
dual-encoder?**
A: No. Dense retrieval reliably outperforms BM25 on paraphrase and natural-language
queries but still loses on exact-term and rare-term queries (product codes, proper
nouns), because embeddings average semantics across many terms and can miss
low-frequency exact matches. Every production system that tried "dense only" found
a class of queries where BM25 was irreplaceable. Keep both arms. "There is no
silver bullet" (Spotify's language, confirmed by Instacart and LinkedIn).

**Q: Can you just train the ranker on raw click-through rates and use high CTR as
a proxy for relevance?**
A: No. Raw CTR is a function of position as much as of relevance; position 1 gets
clicks because it is on top. Training on raw CTR teaches the model to predict the
displayed rank, not the document's quality, and reinforces whatever order you
already shipped. The fix is IPW debiasing (weight each click by the inverse of its
position's propensity) or position-as-a-train-time-feature (fix it to a neutral
constant at serving). Without this correction, the system locks in a feedback loop
where popular positions stay popular.

**Q: Should retrieval optimize precision so the ranker has fewer irrelevant
documents to score?**
A: No. Retrieval is a recall stage: its job is to not miss documents the ranker
would want to see. If retrieval misses a relevant document, the ranker can never
recover it. Optimizing precision in retrieval wastes the division of labor; the
ranker handles precision on the few hundred survivors. The only exception is when
candidate count is the binding latency constraint, in which case you can prune
conservatively, but the default is to maximize recall at retrieval.

**Q: Is offline NDCG reliable enough to use as the ship gate?**
A: No. Offline NDCG is computed against labels that are themselves biased clicks
plus a thin layer of human judgments. It is the fastest feedback signal, not the
most reliable one. A model that learns to predict position better will lift NDCG
while degrading user experience. The ship gate is an interleaving experiment or
A/B test on engagement and reformulation rate. Treat offline NDCG as a pre-gate
that decides whether to spend live traffic on a model, not as the decision.
