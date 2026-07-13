# 6. Reliability

## Timeouts and circuit breakers

A model server that is slow is worse than a model server that is down. A slow
downstream blocks threads, exhausts connection pools, and turns one overloaded
replica into a latency cascade across the whole fleet. Every call into the
model server from the application layer needs a **hard timeout**: if no
response within, say, 45 ms on a 50 ms p99 budget, return a fallback rather
than waiting.

A **circuit breaker** goes one step further: if the error rate or timeout rate
from a replica exceeds a threshold, stop sending it traffic for a cool-down
period. This breaks the feedback loop where a struggling replica receives more
retries, making it struggle more.

The practical choice of timeout value: set it just inside the p99 budget so the
slow-tail requests trip the timeout rather than breaching the SLA by a large
margin.

## Fallbacks and graceful degradation

Every online serving path should have a defined fallback for when the model is
unavailable or too slow. Options in roughly increasing freshness cost:

- **Pre-computed static scores.** Booking.com precomputes static fallback
  scores for every property so the ranking page can still answer when the live
  model is down. The output is stale but present.
- **A cheaper model.** Serve a smaller, faster model version when the full
  ranker is overloaded. Quality degrades gracefully rather than the service
  failing.
- **Popularity-based default.** Return popular items without personalization.
  Users get a valid response; the team loses the incremental lift of the model
  for that period.
- **Cached prediction.** If the same entity was scored recently and the cache
  is warm, return the cached result. Works well for slowly-changing inputs.

The key principle: **define the fallback chain before the incident, not during
it.** At incident time the only question should be "is the fallback already
triggering?" not "what should our fallback be?"

## What "high availability" actually means for serving

High availability for a model serving fleet means:

1. No single host failure takes the service down (replicas absorb the
   load when one drops; the load balancer's health check removes the
   failed replica quickly).
2. No single model version deploy takes the service down (gradual rollout
   with the old version staying warm; serve-while-loading prevents cold
   replicas from receiving traffic before they are ready).
3. No upstream dependency failure takes the service down (feature store
   unavailability triggers the fallback chain, not a 5xx cascade to the
   caller).

## p99 as a first-class alert

![Latency percentiles: the tail hidden by the mean](assets/fig-latency-percentiles.png)

*The mean hides the fat tail. Here 7% of requests take 10x longer than the
median, but the mean looks reasonable. The p99 is what breaches the SLA.
Alerting on the mean lets the problem grow silently. Illustrative simulation.*

Set your on-call alert thresholds on p99 (and p999 for fan-out services), not
on average latency or error rate alone. A latency spike that only shows in the
tail is an SLA breach for every user whose request lands there, even if the
average looks fine.

## Bottlenecks and how to attack them

| Bottleneck | First sign | Fix | Tradeoff |
|---|---|---|---|
| Tail latency over budget | p99 above SLA while p50 is fine | Tune batch window, add replicas, cap model size | Throughput vs tail latency |
| GPU lanes idle under load | Hardware underutilized, QPS below capacity | Increase dynamic batch size or window | Adds batch wait latency |
| Cold start on new replicas | Replicas take traffic before warming, p99 spikes during scale-out | Readiness probes, pre-warm synthetic requests, keep headroom | Idle capacity cost |
| Autoscaling on wrong signal | Fleet scales late or thrashes on CPU spikes | Scale on queue depth, GPU utilization, or batch latency | Tuning and observability cost |
| Embedding table too large for one replica | OOM errors, slow model loads | Shard tables across replicas, quantize (int8, 4-bit), host-memory tiering | Lookup latency, slight quality loss |
| Deploy blast radius | A bad version hits 100% of traffic | Shadow then canary then gradual ramp | Slower, higher-cost rollouts |
| Slow rollback | Incident drags on while old version is rebuilt | Registry pointer rollback, keep prior version warm | Cost of idle warm replica |
| Feature fetch on the critical path | Latency before inference dominates | Cache hot keys, batch feature reads, co-locate the online store | Freshness vs latency |
| Training-serving skew | Model serves fine but degrades over time | Log served features and compare to training distribution | Logging cost and storage |
