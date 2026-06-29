# Question bank

This repo is organized by **component** (candidate retrieval, ranking, sequential
recommendation, and the production ML stack) because those are the reusable
building blocks. But interviews are posed as **end-to-end questions** ("design
YouTube recommendations"), and you are expected to assemble the right components
yourself.

This page is the bridge. It lists the ML system design questions that actually get
asked, what each is really testing, and which topics to draw on. Practice by
picking a question, drafting an answer with the
[framework](framework/answer-framework.md), then reading the linked topics to find
your gaps.

| Question | What it is really testing | Topics to draw on |
|---|---|---|
| Design video / YouTube recommendations | The retrieval-then-ranking funnel, personalization, freshness | [01 retrieval](topics/01-candidate-retrieval.md), [02 ranking](topics/02-ranking-model.md), [03 sequential](topics/03-sequential-recommendation.md) |
| Design a personalized news / home feed | Multi-stage ranking, recency, multi-objective scoring | [02 ranking](topics/02-ranking-model.md), [03 sequential](topics/03-sequential-recommendation.md), [01 retrieval](topics/01-candidate-retrieval.md) |
| Design e-commerce product recommendations | Candidate generation at catalog scale, cold start | [01 retrieval](topics/01-candidate-retrieval.md), [02 ranking](topics/02-ranking-model.md) |
| Design ads CTR prediction / ranking | Calibration, cross features, embedding tables, latency | [02 ranking](topics/02-ranking-model.md) (+ planned: Ads CTR) |
| Design search ranking | Two-stage retrieval + ranking, relevance signals, NDCG | [01 retrieval](topics/01-candidate-retrieval.md), [02 ranking](topics/02-ranking-model.md) |
| Design "people you may know" / friend recommendation | Graph-based candidates, retrieval, ranking | [01 retrieval](topics/01-candidate-retrieval.md), [02 ranking](topics/02-ranking-model.md) |
| Design real-time personalization (react to the last action) | Behavior-sequence modeling, real-time features, freshness | [03 sequential](topics/03-sequential-recommendation.md), [02 ranking](topics/02-ranking-model.md) |
| Design a content moderation / harmful-content system | Classification, imbalanced labels, precision/recall tradeoff, human loop | [framework](framework/answer-framework.md) (+ planned: a classification topic) |
| Design spam / bot detection | Classification under adversarial drift, features, feedback loops | [framework](framework/answer-framework.md) (+ planned) |
| Design fraud / anomaly detection | Rare-event detection, label delay, cost-sensitive thresholds | planned: Fraud and anomaly detection |
| Why is our model good offline but bad online? | Training-serving skew, leakage, position bias, calibration | [02 ranking](topics/02-ranking-model.md), [framework](framework/answer-framework.md) |

## How to use a question

1. **Scope it first.** Pin the objective, the scale (catalog size, QPS, latency
   budget), and how quality is measured before designing. See
   [the framework](framework/answer-framework.md).
2. **Reach for the two-stage funnel.** Most large-scale recommendation and ranking
   questions are retrieval (cheap recall over millions) then ranking (expensive
   accuracy over hundreds). Name that structure early.
3. **Name the dominant failure mode.** Training-serving skew, leakage, position
   bias, and cold start recur across almost every question here. Bring them up
   before the interviewer does.
4. **Always have an eval and an online story.** Offline AUC/NDCG plus an online
   A/B test, because the two routinely disagree.

Some questions point at **planned** topics not yet written. Want one prioritized?
Open an issue or a PR; see [topics/README.md](topics/README.md).
