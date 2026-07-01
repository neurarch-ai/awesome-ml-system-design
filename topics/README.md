# Topics

Each topic is a self-contained interview walkthrough. They follow the same
shape, mirroring the [answer framework](../framework/answer-framework.md):

1. The question, as an interviewer would pose it
2. Clarifying questions and scope
3. Requirements
4. High-level data flow (offline and online paths)
5. Deep dives on the components that carry the signal
6. Bottlenecks and scaling
7. Failure modes, safety, and eval
8. Likely follow-up questions
9. The relevant architecture, opened live

## Topics by ML use case

All eighteen topics are written. They are grouped to mirror how production ML
actually splits, the same use-case taxonomy the
[Evidently AI 800-case database](https://www.evidentlyai.com/ml-system-design)
uses, plus the cross-cutting platform layer. To navigate by interview question
instead, start from the [question bank](../questions.md).

**Recommender systems**
- [01 - Candidate retrieval (two-tower)](01-candidate-retrieval.md)
- [02 - Ranking model](02-ranking-model.md)
- [03 - Sequential and personalized recommendation](03-sequential-recommendation.md)
- [18 - Cold start and exploration](18-cold-start-and-exploration.md)

**Advertising**
- [10 - Ads CTR prediction](10-ads-ctr-prediction.md)

**Search and ranking**
- [09 - Search ranking](09-search-ranking.md)

**Fraud, abuse, and content moderation**
- [08 - Fraud and anomaly detection](08-fraud-and-anomaly-detection.md)
- [16 - Content moderation and trust and safety](16-content-moderation.md)

**Computer vision**
- [12 - Computer vision](12-computer-vision.md)

**Speech and audio**
- [17 - Speech and audio](17-speech-and-audio.md)

**Natural language processing**
- [13 - Natural language processing](13-natural-language-processing.md)

**Forecasting and predictive modeling**
- [14 - Demand forecasting and time series](14-demand-forecasting-and-time-series.md)
- [15 - Predictive modeling on tabular data](15-predictive-modeling-tabular.md)

**Platform, representation, and reliability**
- [07 - Embeddings and representation learning](07-embeddings-and-representation-learning.md)
- [04 - Feature store and training-serving skew](04-feature-store-and-training-serving-skew.md)
- [05 - Real-time ML serving and model deployment](05-realtime-serving-and-deployment.md)
- [06 - Online experimentation and A/B testing](06-online-experimentation-and-ab-testing.md)
- [11 - ML monitoring and drift](11-ml-monitoring-and-drift.md)

## Contributing a topic

Open a PR that follows the eight-section shape above. Rules:

- **Be honest about numbers.** If you cite a dimension, a latency, a funnel size,
  or a metric, it should be real or clearly labeled as an illustrative estimate.
- **No vendor pitch in the body.** The walkthrough teaches; the only product
  references are the architecture links at the end, which open the actual graph
  being discussed.
- **Link the architecture where it is real.** If your topic touches a specific
  model mechanism, link the validated graph so readers can trace it.
- **Prefer mechanism over name-dropping.** Explain what in-batch negatives do, do
  not just say a model uses them.
