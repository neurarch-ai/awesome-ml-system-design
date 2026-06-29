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
| Design ads CTR prediction / an ad ranking system | Calibration (it feeds the auction), cross features, embedding tables, delayed conversions | [10 ads CTR](topics/10-ads-ctr-prediction.md), [02 ranking](topics/02-ranking-model.md) |
| Design search ranking | Query understanding, two-stage retrieval + learning-to-rank, NDCG, position bias | [09 search ranking](topics/09-search-ranking.md), [01 retrieval](topics/01-candidate-retrieval.md), [02 ranking](topics/02-ranking-model.md) |
| Design "people you may know" / friend recommendation | Graph-based candidates and embeddings, retrieval, ranking | [07 embeddings](topics/07-embeddings-and-representation-learning.md), [01 retrieval](topics/01-candidate-retrieval.md), [02 ranking](topics/02-ranking-model.md) |
| Design real-time personalization (react to the last action) | Behavior-sequence modeling, real-time features, freshness | [03 sequential](topics/03-sequential-recommendation.md), [02 ranking](topics/02-ranking-model.md) |
| Design fraud / anomaly / abuse detection | Extreme class imbalance, cost-sensitive thresholds, label delay, adversaries | [08 fraud](topics/08-fraud-and-anomaly-detection.md), [11 monitoring](topics/11-ml-monitoring-and-drift.md) |
| Design spam / bot / content-moderation classification | Imbalanced classification under adversarial drift, precision/recall tradeoff, human loop | [08 fraud](topics/08-fraud-and-anomaly-detection.md), [framework](framework/answer-framework.md) |
| How do you learn embeddings for users / items / entities? | Contrastive learning, negative sampling, dimensionality, the ANN index | [07 embeddings](topics/07-embeddings-and-representation-learning.md), [01 retrieval](topics/01-candidate-retrieval.md) |
| Design how models get served and deployed safely | Model servers, batching, canary/shadow, rollback, autoscaling | [05 serving](topics/05-realtime-serving-and-deployment.md), [04 feature store](topics/04-feature-store-and-training-serving-skew.md) |
| Your model wins offline; how do you decide it ships? | A/B testing, guardrail metrics, sample sizing, interleaving, novelty effects | [06 experimentation](topics/06-online-experimentation-and-ab-testing.md), [02 ranking](topics/02-ranking-model.md) |
| Design a feature platform for our models | Online/offline parity, point-in-time correctness, reuse | [04 feature store](topics/04-feature-store-and-training-serving-skew.md), [02 ranking](topics/02-ranking-model.md) |
| Our model was great at launch and quietly got worse | Drift, label delay, monitoring, the retraining loop | [11 monitoring](topics/11-ml-monitoring-and-drift.md), [04 feature store](topics/04-feature-store-and-training-serving-skew.md) |
| Why is our model good offline but bad online? | Training-serving skew, leakage, position bias, calibration | [04 feature store](topics/04-feature-store-and-training-serving-skew.md), [06 experimentation](topics/06-online-experimentation-and-ab-testing.md), [02 ranking](topics/02-ranking-model.md) |

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

All eleven component topics are written. Missing a question you were asked, or
want a new angle covered? Open an issue or a PR; see
[topics/README.md](topics/README.md).
