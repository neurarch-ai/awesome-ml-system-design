# 4. Architecture: offline store, online store, read and write paths

## The dual-store architecture

One feature definition drives two stores. That is the whole design in one
sentence. The offline store holds full timestamped history and is optimized for
bulk scans; the online store holds the latest (or recent-window) value per entity
and is optimized for single-entity point reads at millisecond latency. Both are
populated by the same shared definition, so code skew is structurally impossible.

```mermaid
flowchart TB
  SRC["raw events\n+ upstream tables"]
  DEF["shared feature definition\n(one DSL / SDK)"]
  BATCH["batch pipeline\n(Spark / warehouse job)"]
  STREAM["streaming pipeline\n(Kafka + Flink / Samza)"]
  OFF["offline store\ntimestamped history\n(BigQuery / Snowflake / Hive / S3)"]
  ON["online store\nlatest value per entity\n(Redis / Cassandra / DynamoDB / Bigtable)"]
  PTJOIN["point-in-time\nas-of join"]
  TRAINDS["training dataset"]
  SVCREQ["feature lookup\n(ranking request)"]

  SRC --> BATCH
  SRC --> STREAM
  DEF --> BATCH
  DEF --> STREAM
  BATCH -->|"timestamped rows"| OFF
  BATCH -->|"materialize latest"| ON
  STREAM -->|"push fresh value"| ON
  OFF --> PTJOIN
  PTJOIN --> TRAINDS
  ON --> SVCREQ
```

## The write path

**Batch pipeline.** A scheduled job (daily, hourly, or finer) reads from the data
warehouse or object store, applies the shared feature definition, and writes two
things: (1) timestamped rows to the offline store for full history, and (2) the
latest value per entity to the online store via a materialization step. Spark is
the standard compute engine; the offline store is typically a columnar warehouse
(BigQuery, Snowflake) or partitioned Parquet on object storage.

**Streaming pipeline.** An event bus (Kafka, Kinesis) feeds a stream processor
(Flink, Samza, Spark Streaming) that applies the same shared definition to compute
the real-time aggregate, then pushes the result to the online store within seconds.
The hardest constraint here is that the streaming aggregate must be numerically
identical to what the batch backfill would compute for the same window over the
same data. This is the seam where data skew most often enters. Uber's DSL runs the
same code in both contexts. Feathr compiles the same transformation API to both
Spark batch and Spark Streaming.

## The read path

**Training read.** The model-training job reads from the offline store and executes
an as-of join (section 3) against the label table. The result is a training dataset
where each row carries the feature value as it existed just before the labeled event.
This is a bulk scan over potentially billions of rows; columnar formats and
predicate pushdown matter here.

**Serving read.** At request time, the ranking service issues a batch key lookup to
the online store: one entity key per feature, typically batched as "fetch all
features for user U and items I1, I2, I3 in a single round-trip." The result is
returned in a few milliseconds and injected into the model's feature vector. This
is a pure point-read workload; columnar formats are wrong here. Redis or Cassandra
are right.

## What lives in each store

| | Offline store | Online store |
|---|---|---|
| Data model | timestamped rows: (entity\_id, feature\_time, value) | latest row per entity: (entity\_id, value) |
| Access pattern | bulk scan with time filter (as-of join) | single-entity point read |
| Retention | full history (months to years) | latest value or recent window |
| Technology | BigQuery, Snowflake, Hive, Delta Lake, Parquet | Redis, Cassandra, DynamoDB, Bigtable |
| Latency | seconds to minutes | single-digit milliseconds |
| Cost driver | storage + scan compute | memory + IOPS |

## When to use which store design

| Reach for | When | Instead of |
|---|---|---|
| Redis as online store | sub-2ms p99 required, entity count fits in memory budget | Cassandra, whose disk-backed reads add latency under memory pressure |
| Cassandra as online store | entity count is large (tens of billions), disk-backed is acceptable at P95 \lt 10ms | Redis, whose memory cost grows linearly with entity count |
| DynamoDB as online store | team runs on AWS, no on-call appetite for operating Redis/Cassandra, cost over latency acceptable | self-managed stores, when managed is the constraint |
| Bigtable as online store | team runs on GCP, sub-10ms acceptable, very wide rows (many features per entity) | Redis, for very large wide-row feature sets that blow memory budget |
| Single shared DSL / unified API | multiple models, multiple teams, skew is already a problem | separate SQL and service code, which guarantees code skew at scale |
| Pluggable backends (Feast) | no single technology mandate, different models need different stores | a fixed-backend platform, when the team cannot commit to one online store |

## The feature registry

The registry is the catalog layer above the stores. It records: feature name,
owner, description, schema, freshness SLA, which models consume it, and lineage
(what upstream tables and transformations produced it). Without a registry, the
feature store becomes a dumping ground: 10,000 features at Uber with no metadata
is the canonical cautionary tale. A registry does not need to be a separate
service; it can be a versioned YAML or JSON schema checked into the same repository
as the feature definitions. Azure Purview (Feathr), Amundsen, and DataHub are the
common external choices.
