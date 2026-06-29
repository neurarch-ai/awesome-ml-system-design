# ML System Design Interview

A practical guide to the system design questions you actually get asked when
interviewing for classic machine learning roles: applied scientist, ML engineer,
recommendations and ranking, search, and the broad "ML platform" bucket that
keeps production models alive.

This is the non-LLM stack: recommenders, ranking, retrieval, feature pipelines,
embeddings, and the offline-to-online machinery that turns a model into a
product. The questions here predate the LLM hype and outlast it, because almost
every company that serves predictions at scale runs some version of the same
pattern: retrieve a candidate set cheaply, score it carefully, log everything,
and close the loop with training data the system generates about itself.

This repo covers that pattern, organized as interview-ready walkthroughs.

Every model architecture in here is a **validated reference graph**, not a
hand-drawn box diagram. Each one is checked end to end: tensor shapes, embedding
table dimensions, feature-interaction wiring, parameter counts within roughly 10
percent of the published design. You can open any of them, edit the structure,
and watch the numbers change. More on why that matters [below](#about-the-diagrams).

---

## How to use this repo

1. **Read the [answer framework](framework/answer-framework.md) first.** It is the
   spine every walkthrough hangs off. Interviewers grade structure as much as
   content; a strong framework keeps you from rambling.
2. **Work through the topics in order.** They build on each other. Retrieval
   introduces the two-tower and the candidate funnel; ranking scores what
   retrieval returns; sequential recommendation personalizes the whole thing
   with the user's behavior history.
3. **For each topic, try to draft your own answer before reading ours.** Then
   compare. The gaps are your study list.
4. **Open the architectures live.** Where a topic touches a real model (two-tower
   retrieval, DLRM-style feature interactions, a behavior sequence transformer),
   there is a link that loads the actual graph so you can trace the data flow
   yourself instead of trusting a static picture.

> **Interviews ask end-to-end questions, not "explain ranking."** The
> [**question bank**](questions.md) maps the questions you actually get ("design
> YouTube recommendations", "why is my model good offline but bad online") to the
> component topics below. Start there if you are practicing for a specific loop.

---

## Topics

Two ways in. **By use case** (how interviews are actually posed): start with the
[**question bank**](questions.md), which maps prompts like "design YouTube
recommendations" to the topics below. **By pipeline stage** (to study a system end
to end): the topics are grouped below by where they sit in a production ML system.

### Retrieval and representation
*Get a small set of good candidates out of millions, and learn the vectors that make that possible.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 01 | [Candidate retrieval (two-tower)](topics/01-candidate-retrieval.md) | The two-tower architecture, in-batch negatives, ANN serving, the candidate funnel, embedding freshness |
| 07 | [Embeddings and representation learning](topics/07-embeddings-and-representation-learning.md) | Contrastive learning, negative sampling, dimensionality, index choice |

### Ranking
*Score and order the candidates retrieval hands you.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 02 | [Ranking model](topics/02-ranking-model.md) | Feature engineering, wide-and-deep and DLRM, feature interactions, multi-task ranking, calibration, the scoring latency budget |
| 03 | [Sequential and personalized recommendation](topics/03-sequential-recommendation.md) | Behavior sequence modeling, attention over interactions, session-based recsys, cold start, real-time features |

### Data and serving infrastructure
*Make the model runnable in production, with consistent features and safe deploys.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 04 | [Feature store and training-serving skew](topics/04-feature-store-and-training-serving-skew.md) | Online/offline parity, point-in-time correctness, backfills, freshness |
| 05 | [Real-time ML serving and model deployment](topics/05-realtime-serving-and-deployment.md) | Model servers, batching, shadow and canary, rollback, autoscaling |

### Measurement and reliability
*Know it works, and keep it working as the world moves.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 06 | [Online experimentation and A/B testing](topics/06-online-experimentation-and-ab-testing.md) | Metrics, guardrails, interleaving, novelty effects, sample sizing |
| 11 | [ML monitoring and drift](topics/11-ml-monitoring-and-drift.md) | Feature drift, label drift, performance decay, alerting |

### End-to-end systems
*Compose the stages above into a full system for one domain.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 08 | [Fraud and anomaly detection](topics/08-fraud-and-anomaly-detection.md) | Class imbalance, label delay, cost-sensitive thresholds, adversaries |
| 09 | [Search ranking](topics/09-search-ranking.md) | Query understanding, learning-to-rank, relevance labels, position bias |
| 10 | [Ads CTR prediction](topics/10-ads-ctr-prediction.md) | Calibration, auctions, bidding signals, delayed conversions |

All eleven topics are written and ready.

See [topics/README.md](topics/README.md) for the full roadmap and how to
contribute a topic.

Every ready topic ends with a **Seen in production** section: real first-party
engineering writeups (Pinterest, Meta, Google, Airbnb, Instacart, Alibaba,
Netflix, and more) tagged by what they illustrate, so the framework is grounded
in shipped systems rather than whiteboard theory. For the broadest index of
production ML systems, see the
[Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design)
(800 case studies from 150+ companies).

> Building with large language models instead? The companion
> [**LLM System Design Interview**](https://github.com/neurarch-ai/awesome-llm-system-design)
> repo covers the LLM-era stack: RAG, the KV cache, agent orchestration,
> inference serving, evals, and guardrails. This repo is the classic ML half of
> the same project.

---

## About the diagrams

Every architecture diagram links back to [Neurarch](https://www.neurarch.com),
where it lives as a structured graph rather than an image.

This is a deliberate choice, and it matters for interview prep specifically. A
picture of an architecture cannot be wrong in a way you can catch. A validated
graph can. Recommender diagrams are a quiet disaster zone for this: people draw a
"two-tower" with the user and item towers sharing weights (they usually do not),
or a DLRM with the feature interactions wired into the wrong layer, and nobody
notices because a `.png` has no ground truth to check against.

When you are about to explain in-batch negatives, or where exactly the dot
product happens in a two-tower, or how a DLRM crosses sparse and dense features,
you want the structure in your head to be the real one. So the diagrams here are
not screenshots. Each is a link that opens the actual graph, at real dimensions,
that you can inspect and modify:

- **Click to open it live** and the architecture loads onto a canvas where you
  can trace every tensor shape, follow the embedding tables into the interaction
  layer, and change a hyperparameter to see what breaks.
- **The source graphs are open** in the
  [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo) (MIT). Each
  entry is a real graph at real dimensions plus verified hyperparameters.

If you have never traced a recommender's shapes by hand, doing it once before an
interview is worth more than reading ten blog posts. The links are there so you
can.

---

## License

MIT. See [LICENSE](LICENSE). Contributions welcome; see
[topics/README.md](topics/README.md).

Built and maintained by the team behind [Neurarch](https://www.neurarch.com).
