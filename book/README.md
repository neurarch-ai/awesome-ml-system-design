# Machine Learning System Design: A Practical Guide

This folder holds the book-format edition of the classic-ML half of this project:
the same production ML system-design material as the topic walkthroughs, rewritten
as instructional book chapters. Where the top-level repo is organized for
interview practice, these chapters read as a teaching sequence you can work
through front to back.

Each chapter opens a set of **validated reference architectures** from the
Neurarch model zoo as figures. They are not screenshots: every figure is a
shape-checked graph you can open live and inspect layer by layer.

## What you will learn

- Build the recommendation stack end to end: two-tower retrieval, ranking, sequential personalization, cold start, and exploration.
- Design advertising and search systems: calibrated CTR prediction under an auction, query understanding, and learning-to-rank.
- Handle adversarial and imbalanced problems: fraud and anomaly detection, and content moderation across text, image, and audio.
- Ship perception systems: computer vision, speech and audio, and task-specific NLP.
- Predict the future and the tail: demand forecasting and time series, and predictive modeling on tabular data.
- Run the platform underneath it all: embeddings, feature stores, real-time serving, experimentation, and monitoring.
- Connect every modeling choice to the architecture underneath it, traced on a real graph rather than a box diagram.

## Chapters

| # | Chapter | Covers |
|---|---------|--------|
| 1 | [Candidate Retrieval with Two-Tower Models](chapter-01-candidate-retrieval.md) | The two-tower architecture, in-batch negatives, ANN serving, the candidate funnel, embedding freshness |
| 2 | [Ranking Models](chapter-02-ranking-models.md) | Feature engineering, wide-and-deep and DLRM, feature interactions, multi-task ranking, calibration |
| 3 | [Sequential and Personalized Recommendation](chapter-03-sequential-recommendation.md) | Behavior-sequence modeling, attention over interactions, session-based recsys, cold start, real-time features |
| 4 | [Cold Start and Exploration](chapter-04-cold-start-and-exploration.md) | New-user and new-item cold start, content towers, explore-exploit, contextual bandits, off-policy evaluation |
| 5 | [Ads Click-Through-Rate Prediction](chapter-05-ads-ctr-prediction.md) | Calibration (it feeds the auction), cross features, bidding signals, delayed conversions |
| 6 | [Search Ranking](chapter-06-search-ranking.md) | Query understanding, learning-to-rank, relevance labels, position bias |
| 7 | [Fraud and Anomaly Detection](chapter-07-fraud-and-anomaly-detection.md) | Class imbalance, label delay, cost-sensitive thresholds, graph features, adversaries |
| 8 | [Content Moderation and Trust and Safety](chapter-08-content-moderation.md) | Per-policy harm taxonomy, recall at a precision floor, multi-modal, human-in-the-loop, adversarial evasion |
| 9 | [Computer Vision](chapter-09-computer-vision.md) | Task taxonomy, transfer learning, labeling cost, GPU serving cost, moderation recall, visual search |
| 10 | [Speech and Audio](chapter-10-speech-and-audio.md) | Streaming vs batch ASR, RNN-T and Conformer, WER pitfalls, wake word, diarization, TTS, on-device |
| 11 | [Natural Language Processing](chapter-11-natural-language-processing.md) | Text classification, NER and extraction, intent, entity resolution, translation, encoder vs LLM |
| 12 | [Demand Forecasting and Time Series](chapter-12-demand-forecasting.md) | Probabilistic and hierarchical forecasts, classical vs deep, backtesting, forecast-then-optimize |
| 13 | [Predictive Modeling on Tabular Data](chapter-13-predictive-modeling-tabular.md) | Why trees still win, calibration, uplift and causal decisions, delayed labels, survival, LTV, fairness |
| 14 | [Embeddings and Representation Learning](chapter-14-embeddings.md) | Contrastive learning, negative sampling, dimensionality, index choice |
| 15 | [Feature Stores and Training-Serving Skew](chapter-15-feature-stores.md) | Online and offline parity, point-in-time correctness, backfills, freshness |
| 16 | [Real-Time Serving and Deployment](chapter-16-realtime-serving.md) | Model servers, batching, shadow and canary, rollback, autoscaling |
| 17 | [Online Experimentation and A/B Testing](chapter-17-experimentation-ab-testing.md) | Metrics, guardrails, interleaving, novelty effects, sample sizing |
| 18 | [ML Monitoring and Drift](chapter-18-monitoring-and-drift.md) | Feature drift, label drift, performance decay, alerting |

A companion book covers the LLM half (RAG, serving, agents, evaluation, safety)
in the
[LLM System Design Interview](https://github.com/neurarch-ai/awesome-llm-system-design)
repository.

## Technical requirements

You need only a modern web browser to open the validated reference graphs used as
figures throughout the book. Each figure links to the
[Neurarch model zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo), where
the architecture opens live in the editor at real dimensions. The chapters name
the supporting tooling (an ANN index, a feature store, a serving stack, an
experimentation platform) but do not require you to install anything to read them.

## How to use this book

Each chapter ends with a **Questions** section for self-review and a **Further
reading** list of first-party production engineering writeups. Read a chapter,
open its figures, attempt the questions, then follow the further reading to see
how real teams shipped the same system. Built by [Neurarch](https://www.neurarch.com).
