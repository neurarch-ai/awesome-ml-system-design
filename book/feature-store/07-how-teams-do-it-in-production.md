# 7. How teams do it in production

Every large-scale ML platform converges on the same dual-store skeleton: a shared
feature definition drives a batch pipeline into a timestamped offline store and a
streaming pipeline into a low-latency online store; training reads from the offline
store via an as-of join; serving reads from the online store via a batch KV lookup.
What actually differs between teams is three decisions: **how the shared definition
is enforced** (a DSL, a unified API, or just a disciplinary convention), **which
online store technology is used**, and **who operates the infrastructure** (in-house,
open-source, or managed SaaS). The architecture everyone shares; the leverage is
in how those three choices are made.

## Where the real designs diverge

| System | Shared definition enforcement | Online store | Point-in-time join | When it wins | Watch out |
|---|---|---|---|---|---|
| Uber Michelangelo | Scala DSL in model config; runs identically at train and serve time | Cassandra (P95 \lt 10ms) | Log streaming features at compute time; replay logs for training | Thousands of shared features, tight latency, large Spark shop | Heavy in-house platform; DSL versioning becomes its own problem |
| LinkedIn Feathr | Unified Spark transformation API across batch, streaming, and online | Redis or Cosmos DB | As-of join APIs built into the framework | Teams already on Spark; one API prevents drift across runtimes | Coupled to Spark; not ideal if compute is not Spark-based |
| Feast | Python SDK with feature definitions in code; bring your own compute | Pluggable: Redis, DynamoDB, Bigtable, Postgres, Qdrant, 20-plus others | `get_historical_features` as-of join API built in; simple materialization drops the guarantee | No single technology mandate; open-source; storage-agnostic | Framework, not a full pipeline; team must supply compute and orchestration |
| Tecton | Managed platform: feature definitions in Python, transforms run in managed infra | Managed DynamoDB or Redis-backed | Managed point-in-time dataset generation | Teams that want real-time features without operating streaming infra | Commercial, vendor dependency; less control over compute |
| Google (Rules of ML) | Discipline: reuse code between training and serving; log exact features at serving time | Not a store; log-replay is the source of truth | Temporal testing: measure on data gathered after training data ends | Any team; cheapest starting point; nearly zero infra | Guidance only; without enforcement the discipline quietly erodes at scale |

The dividing line is simple: **build vs. buy vs. discipline.** In-house platforms
(Uber) give maximum control and are justified when feature reuse across thousands
of models makes the investment pay off. Open-source frameworks (Feast, Feathr)
give flexibility without vendor lock-in but require operating compute. Managed SaaS
(Tecton) lowers operational burden at the cost of control and price. Discipline
alone (Google) works at small scale or as a starting point, but erodes without
tooling to enforce it.

## The systems (first-party write-ups)

- **Uber Michelangelo:** [Meet Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/) -- introduces the Scala DSL that runs identically at train and predict time, the Cassandra online store, and the Kafka/Samza streaming pipeline that logs aggregates back to HDFS for training reconstruction.
- **LinkedIn Feathr:** [Open Sourcing Feathr: LinkedIn's Feature Store for Productive Machine Learning](https://github.com/feathr-ai/feathr) -- the unified Spark transformation API across batch, streaming, and online paths; point-in-time sliding-window joins; Azure Purview for lineage.
- **Feast:** [Feast: Bridging ML Models and Data](https://feast.dev/) and the [Feast documentation](https://docs.feast.dev/) -- pluggable offline and online store backends; three materialization modes; `get_historical_features` as-of join; native drift and serving-log monitoring.
- **Tecton:** [Tecton: The Enterprise Feature Store](https://www.tecton.ai/blog/what-is-a-feature-store/) -- managed feature platform with real-time streaming transforms, managed point-in-time dataset generation, and no operational burden on the user.
- **Google Rules of ML:** [Rules of Machine Learning (Martin Zinkevich)](https://developers.google.com/machine-learning/guides/rules-of-ml) -- the discipline that every feature store encodes: train the way you serve; log features at serving time; test on data gathered after the training window.

For the full comparison (divergence diagram, choices table, math, quadrant plot),
see the dense reference in [topics/04-feature-store-and-training-serving-skew.md](../../topics/04-feature-store-and-training-serving-skew.md).
