# 10. Putting it together: the complete build

Sections 1 through 6 taught each piece with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single platform with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it shows
how the same decisions flip when the constraints change. It closes with the
smallest runnable point-in-time join, one file, no installs.

## The default stack: start here, deviate with reason

Every piece of this chapter has three to six credible options, and a first-time
builder can burn a week comparing online stores before serving a single feature.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Tools change yearly,
but the interface of each piece (define, materialize, store history, join, serve,
monitor) does not, so pick per piece by interface and treat any specific
technology as replaceable.

| Piece | Default | Deviate when | Why (section) |
|---|---|---|---|
| Shared definition | One definition compiled to both paths (Feast-class SDK, versioned in a repo) | One model, one team: Google-style discipline plus serve-time logging | [2](02-the-core-problem.md), [4](04-architecture.md) |
| Offline store | Columnar warehouse or Parquet with (entity, event\_time, value) rows, full history | History cost dominates: tier old partitions to cheap object storage | [4](04-architecture.md) |
| Online store | Redis, batch key lookup, one round-trip per request | Entity-feature pairs outgrow cluster memory: Cassandra; no on-call appetite: DynamoDB / Bigtable | [4](04-architecture.md), [6](06-serving-and-scaling.md) |
| Point-in-time join | As-of join on event time with a ttl staleness bound matching online expiry | Serve-time feature logs exist: train on the logs (log-replay) instead | [3](03-point-in-time-correctness.md) |
| Streaming vs batch | Every feature starts on daily batch; promote per feature to hourly or streaming | A one-minute lag would meaningfully change the prediction: streaming tier | [5](05-freshness-and-backfills.md) |
| Backfills | Full recompute with proper event timestamps, then a parity check | Serve-time logs cover the window: log-replay; minor variant of an existing feature: incremental | [5](05-freshness-and-backfills.md) |
| Monitoring | Per-feature parity, PSI against live traffic, staleness vs declared SLA, on a schedule | Never. Wire it before the first model ships | [2](02-the-core-problem.md), [5](05-freshness-and-backfills.md) |

The last row is the one beginners skip and regret: skew is silent by
construction, no error is thrown and offline metrics look fine, so parity and
staleness alerts are the only warning that arrives before an A/B result does.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): 50
million active users, 10 million items, writes peaking at 500k events per
second, feature fetch in single-digit milliseconds on the ranking critical
path, a dozen models across five teams, both daily batch aggregates and
session signals fresh within seconds, point-in-time correctness required after
a past leakage incident, and a migration from 200-plus existing pipelines.
Here is the whole platform with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Shared definition | Feast-class SDK with pluggable backends, definitions in one repo | 200-plus pipelines migrate incrementally; a big-bang DSL rewrite would stall the migration |
| Offline store | Columnar warehouse, timestamped rows, full history retained | Point-in-time correctness is impossible without history; bulk as-of scans are the access pattern |
| Online store | Redis cluster, batch key lookup | Single-digit-millisecond budget rules out disk-backed tails; the working set fits memory (below) |
| Point-in-time join | As-of join on event time, ttl matching the online store's expiry | Labels arrive hours late; the ttl bound keeps offline joins consistent with online expiry |
| Streaming path | Kafka plus Flink, compiled from the same definition as batch | Session signals need seconds-level freshness at 500k events per second peak |
| Batch path | Hourly and daily Spark jobs; materialization rate-limited by a token bucket | Slow features stay on the cheap tier; an unthrottled burst would spike live read latency |
| Freshness assignment | Least-strict tier per feature, declared SLA in the registry | Streaming everything costs roughly 7x daily batch with no accuracy gain on slow features |
| Backfills | Serve-time feature logging from day one; log-replay preferred, recompute plus parity check otherwise | Recomputation is where backfill skew enters; logs make parity hold by construction |
| Monitoring | Parity above 0.999 per feature, PSI against live traffic, staleness alerts per SLA | The earliest skew warning that does not wait for an A/B result |
| Registry | Owner, description, freshness SLA, consumers, lineage per feature | A dozen models across five teams; without governance the store only grows |

**Online store sizing.** 50 million users at 100 features and 4 bytes per value
is 50M x 100 x 4 = 20 GB ([section 6](06-serving-and-scaling.md)); 10 million
items at the same width add 4 GB. Call the raw working set ~24 GB, comfortable
in a Redis cluster even with replication. The same math at 500 million users or
1,000 features per entity crosses into hundreds of gigabytes, which is the point
where Cassandra's disk-backed storage becomes cheaper than RAM; the store choice
was made from this multiplication, not from a benchmark table. Illustrative.

**Serving latency.** The ranking request has a 50ms overall budget with 5-10ms
allocated to feature fetch ([section 6](06-serving-and-scaling.md)). One batched
lookup fetches the user's features and a few hundred items' features in a
single round-trip; Redis returns it at sub-2ms p99, leaving the rest of the
budget for model inference. Issuing one request per entity instead would
multiply latency by the number of items and blow the budget on its own.

**Write path and freshness lag.** Peak traffic is 500k events per second into
Kafka; Flink computes the session aggregates and pushes to Redis within
seconds, and the staleness metric from [section 5](05-freshness-and-backfills.md)
is alerted per feature against its declared SLA. The batch side is where the
write path bites: materializing 60 million entities in one unthrottled burst
competes with live reads on the same nodes, so the job runs behind a token
bucket at, say, 50k writes per second, stretching a full materialization to
60M / 50k = 1,200 seconds, about 20 minutes, in exchange for a flat read p99.
Illustrative.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: read-latency spikes during the first
full-catalog materialization (online store p99 against the write-burst
schedule; the fix is the rate limit above), streaming lag at peak (per-feature
staleness against the seconds-level SLA while Flink parallelism is still being
tuned), and parity drops on migrated features (one of the 200 legacy pipelines
recomputed a null or a time zone differently, and parity below 0.999 on that
feature is the only place the difference shows before model quality does).

## The same techniques under different constraints

The review question that matters in practice is not "which online store is
best" but "which online store is best under my constraints." Here is the same
platform built three times. Only the middle column is the build above; the
other two keep the identical piece interfaces and swap nearly every
implementation choice.

| | Single-model startup | Marketplace platform (this chapter) | Nightly batch scorer |
|---|---|---|---|
| Scale / traffic | ~100k users, one model, two engineers | 50M users, 10M items, 500k events/s peak, a dozen models | 5M customers, one churn model, scored once nightly |
| Latency budget | Tens of ms are fine at low QPS | Feature fetch in single-digit ms inside a 50ms request | None; the scoring job runs for an hour and nobody waits |
| Shared definition | Discipline: reuse the code, log features at serving time (Google style) | SDK-enforced definitions with a registry across five teams | One SQL pipeline in the warehouse; nothing to share |
| Online store | Postgres with a small cache layer, or none at all | Redis, batch key lookup, rate-limited materialization | None: predictions are written back to the warehouse |
| Point-in-time join | Train directly on serve-time logs; no reconstruction needed | As-of join with ttl over full timestamped history | As-of join in SQL over warehouse history; same invariant, zero new infra |
| Freshness | Daily batch for everything | Tiered: daily, hourly, streaming, request-time | Daily is the product, not a compromise |
| Backfills | Recompute the whole small history; it fits in one job | Log-replay preferred; recompute validated with parity | Full recompute is cheap; parity check still runs |
| What would be over-engineering | Streaming infra, a registry service, a second store | A big-bang DSL rewrite of the 200 legacy pipelines | Any online store at all |

Two lessons fall out. First, the startup column is mostly deletions: with one
model and serve-time logging, training data is the logs, parity holds by
construction, and the entire dual-store apparatus can wait until a second team
shows up, which is exactly the Google row of [section 7](07-how-teams-do-it-in-production.md).
Second, the batch column shows that the online store is not what makes a
feature store: it exists only to meet a serving latency budget, and when no
request path exists the offline store plus the as-of join of
[section 3](03-point-in-time-correctness.md) is the whole platform. Point-in-time
correctness is the invariant all three columns keep; everything else is
negotiable.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Entity count x feature count | Online store technology | Working set under a few hundred GB: Redis. Beyond: Cassandra; managed KV when ops appetite is the real limit |
| Feature-fetch latency budget | Store choice and request shape | Single-digit ms: in-memory store, one batched get per request. 10ms-plus acceptable: managed KV is fine |
| Label arrival delay | Join anchor and history retention | Labels arrive late: as-of join on event time over timestamped history, with a ttl bound; never join-on-latest |
| How fast the feature changes | Freshness tier | Least-strict tier accuracy tolerates; streaming costs roughly 7x daily batch, so promote per feature, never by default |
| Peak write rate | Streaming parallelism, materialization throttle | Size Flink for peak; rate-limit batch materialization so the burst never competes with live reads |
| Teams times models | Definition enforcement | One model: discipline plus serve-time logs. Several teams: an SDK or DSL that compiles to both paths, plus a registry |
| Serve-time logs exist | Training-data source, backfill path | Logs: train and backfill by log-replay, parity by construction. No logs: recompute and validate with parity |
| Ops appetite | Build vs buy vs discipline | No cluster on-call: DynamoDB / Bigtable or a managed platform; the architecture stays the same either way |
| Quality floor | Monitoring set | Parity above 0.999 per feature, PSI under 0.1 vs live traffic, staleness alerts per declared SLA |

## The smallest runnable point-in-time join

The chapter's core mechanism fits in one file with zero installs. Every
production component is swapped for the smallest thing with the same interface:
the event log becomes a seeded loop, the offline store's timestamped history
becomes the value captured at event time, the online "latest value" becomes a
dict, and the model becomes a single learned threshold. The naive join trains
on the latest value, the point-in-time join trains on the value as of the
event, and simulated serving shows which one survives contact with live
traffic. The shape is the lesson; every section of this chapter upgrades one
piece of this file.

```python
"""Point-in-time join vs join-on-latest, runnable with no installs."""
import random

random.seed(7)

# --- event log --------------------------------------------------------------
# Feature: lifetime purchase count per user. Each user gets one labeled event
# ("will they purchase now?"). The true rule uses the count AS IT EXISTED at
# the event: active users (count >= 2) convert with p=0.65, others p=0.35.
# A converting user then keeps purchasing, so the END-OF-LOG total encodes
# the label it is supposed to predict.

rows = []                                     # (user, count_at_event, label)
latest = {}                                   # end-of-log count: the naive join
for user in range(400):
    rate = random.uniform(0.0, 0.05)          # background purchases per tick
    t_event = random.randint(20, 80)
    count_at = sum(random.random() < rate for _ in range(t_event))
    p = 0.65 if count_at >= 2 else 0.35       # rule uses value AT event time
    label = 1 if random.random() < p else 0
    after = sum(random.random() < rate for _ in range(t_event, 100))
    latest[user] = count_at + after + (5 if label else 0)   # converts keep buying
    rows.append((user, count_at, label))

# --- two joins, one label table ---------------------------------------------
train, serve = rows[:200], rows[200:]         # serve = users scored later

pit_rows = [(count_at, y) for _, count_at, y in train]      # as-of join
naive_rows = [(latest[u], y) for u, _, y in train]          # join-on-latest

# --- trivial threshold model ------------------------------------------------

def fit(pairs):
    """Best count threshold by train accuracy; production: any model."""
    return max(range(0, 15),
               key=lambda k: sum((x >= k) == bool(y) for x, y in pairs))

def accuracy(pairs, k):
    return sum((x >= k) == bool(y) for x, y in pairs) / len(pairs)

k_pit, k_naive = fit(pit_rows), fit(naive_rows)

# --- simulated serving ------------------------------------------------------
# Live traffic can only see the count as of the request; the future has not
# happened yet. Both models score the same point-in-time feature.
serve_rows = [(count_at, y) for _, count_at, y in serve]

print(f"naive join  train accuracy: {accuracy(naive_rows, k_naive):.3f}  (threshold {k_naive})")
print(f"pit join    train accuracy: {accuracy(pit_rows, k_pit):.3f}  (threshold {k_pit})")
print(f"naive join  serving accuracy: {accuracy(serve_rows, k_naive):.3f}")
print(f"pit join    serving accuracy: {accuracy(serve_rows, k_pit):.3f}")
```

Run it and four numbers tell the whole chapter's story. The naive join trains
at 0.945 accuracy and the point-in-time join at 0.645, so on offline metrics
alone the naive model wins by thirty points and ships. At simulated serving the
naive model scores 0.555, barely above guessing the majority class, while the
point-in-time model holds at 0.640, within half a point of its training
accuracy. The naive model was never predicting purchases; it was reading them
back out of a feature that the label itself had incremented, and its learned
threshold of 5 is calibrated to post-outcome counts that no live request will
ever contain. This is the "great offline, bad online" pattern of
[section 2](02-the-core-problem.md) reproduced in fifty lines. In production
terms: `latest` is the online store's latest-value-per-entity, `count_at` is
the timestamped history the offline store keeps so the as-of join of
[section 3](03-point-in-time-correctness.md) has something to look up, the
seeded loop is the event bus feeding the write path of
[section 4](04-architecture.md), and the serving block is the rule that a model
must be evaluated on the features the request path can actually produce. Swap
the dict for Redis, the captured history for warehouse rows, and the threshold
for a real model, and you have rebuilt this chapter.
