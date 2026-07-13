# Machine Learning System Design: A Practical Guide

This folder holds the book-format edition of the classic-ML half of this project:
the same production ML system-design material as the topic walkthroughs, rewritten
as teach-first book chapters. Where the top-level repo is organized for interview
practice, these chapters read as a teaching sequence you can work through front to
back.

Each chapter is a folder of one file per section. It opens with a
Candidate/Interviewer dialogue to scope the problem, then walks the frame-data-
model-evaluate-serve arc, borrows how real teams shipped it in production (with
first-party links), and closes with an interview Q&A and a self-test. Figures are
worked matplotlib plots and mermaid diagrams, plus **validated reference
architectures** from the Neurarch model zoo that open live at real dimensions, not
screenshots.

## What you will learn

- Build the recommendation stack end to end: two-tower retrieval, ranking, sequential personalization, cold start, exploration, and graph-based link prediction.
- Design advertising and search systems: calibrated CTR prediction under an auction, query understanding, and learning-to-rank.
- Handle adversarial and imbalanced problems: fraud and anomaly detection, and content moderation across text, image, and audio.
- Ship perception systems: computer vision, speech and audio, and task-specific NLP.
- Predict the future and the tail: demand forecasting and time series, and predictive modeling on tabular data.
- Run the platform underneath it all: embeddings, feature stores, real-time serving, experimentation, and monitoring.
- Connect every modeling choice to the architecture underneath it, traced on a real graph rather than a box diagram.

## Chapters

| # | Chapter | Covers |
|---|---------|--------|
| 1 | [Candidate Retrieval with Two-Tower Models](candidate-retrieval/) | The two-tower architecture, in-batch negatives, ANN serving, the candidate funnel, embedding freshness |
| 2 | [Ranking Models](ranking/) | Feature engineering, wide-and-deep and DLRM, feature interactions, multi-task ranking, calibration |
| 3 | [Sequential and Personalized Recommendation](sequential-recommendation/) | Behavior-sequence modeling, attention over interactions, session-based recsys, cold start, real-time features |
| 4 | [Cold Start and Exploration](cold-start/) | New-user and new-item cold start, content towers, explore-exploit, contextual bandits, off-policy evaluation |
| 5 | [Ads Click-Through-Rate Prediction](ads-ctr/) | Calibration (it feeds the auction), cross features, bidding signals, delayed conversions |
| 6 | [Search Ranking](search-ranking/) | Query understanding, learning-to-rank, relevance labels, position bias |
| 7 | [Fraud and Anomaly Detection](fraud-detection/) | Class imbalance, label delay, cost-sensitive thresholds, graph features, adversaries |
| 8 | [Content Moderation and Trust and Safety](content-moderation/) | Per-policy harm taxonomy, recall at a precision floor, multi-modal, human-in-the-loop, adversarial evasion |
| 9 | [Computer Vision](computer-vision/) | Task taxonomy, transfer learning, labeling cost, GPU serving cost, moderation recall, visual search |
| 10 | [Speech and Audio](speech/) | Streaming vs batch ASR, RNN-T and Conformer, WER pitfalls, wake word, diarization, TTS, on-device |
| 11 | [Natural Language Processing](nlp/) | Text classification, NER and extraction, intent, entity resolution, translation, encoder vs LLM |
| 12 | [Demand Forecasting and Time Series](forecasting/) | Probabilistic and hierarchical forecasts, classical vs deep, backtesting, forecast-then-optimize |
| 13 | [Predictive Modeling on Tabular Data](tabular/) | Why trees still win, calibration, uplift and causal decisions, delayed labels, survival, LTV, fairness |
| 14 | [Embeddings and Representation Learning](embeddings/) | Contrastive learning, negative sampling, dimensionality, index choice |
| 15 | [Feature Stores and Training-Serving Skew](feature-store/) | Online and offline parity, point-in-time correctness, backfills, freshness |
| 16 | [Real-Time Serving and Deployment](realtime-serving/) | Model servers, batching, shadow and canary, rollback, autoscaling |
| 17 | [Online Experimentation and A/B Testing](experimentation/) | Metrics, guardrails, interleaving, novelty effects, sample sizing |
| 18 | [ML Monitoring and Drift](monitoring/) | Feature drift, label drift, performance decay, alerting |
| 19 | [Graph Recommendation and Link Prediction](graph-recommendation/) | People You May Know, GNNs (GraphSAGE, PinSage), link prediction, negative sampling on graphs, serving embeddings |

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

Open a chapter folder and read its sections in order (start at the folder's
README). Each chapter ends with an interview Q&A and a self-test, and links a set
of first-party production engineering writeups. Read a chapter, open its figures,
attempt the questions, then follow the further reading to see how real teams
shipped the same system. Built by [Neurarch](https://www.neurarch.com).
