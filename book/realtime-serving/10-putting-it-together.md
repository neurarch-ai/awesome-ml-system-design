# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable replica pool, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing serving frameworks before answering a single
request. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Tools
change yearly, but the interface of each stage (serve, batch, deploy, scale,
degrade) does not, so pick per stage by interface and treat any specific
framework as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Deployment type | Online real-time only if a caller blocks on the result | Result can wait seconds to minutes: async queue; scheduled scoring: offline batch, up to an order of magnitude cheaper | [2](02-the-serving-problem.md) |
| Serving runtime | Dedicated model server (TF Serving / Triton / TorchServe class) behind the app, loaded from a versioned registry | Tiny CPU model, single consumer, one small team: in-process monolith, no network hop | [2](02-the-serving-problem.md) |
| Batching | Dynamic batching, window sized backwards from the p99 budget | Low QPS or a p50 SLA: request-at-a-time; candidates already arrive together: micro-batch within the request | [3](03-batching-and-throughput.md) |
| Deployment strategy | Shadow briefly, then 5% canary, then step ramp with automated rollback | Rare, high-stakes deploy (architecture change): blue-green with two full fleets | [4](04-deployment-strategies.md) |
| Rollback | Automated trigger on p99 or error-rate breach; registry pointer move | Never. Define the trigger before the deploy, not during the incident | [4](04-deployment-strategies.md) |
| Autoscaling signal | Request queue depth or GPU utilization | CPU-only fleet where CPU genuinely is the bottleneck: CPU utilization is then honest | [5](05-autoscaling-and-cost.md) |
| Headroom | Provision (1 + h) times peak, h at least cold-start time over scaling interval | Cold start near zero (small CPU model, fast load): h shrinks toward plain peak sizing | [5](05-autoscaling-and-cost.md) |
| Reliability | Hard timeout inside the caller's budget, circuit breaker, fallback chain defined before launch | Never on the fallback chain. The timeout value moves with the budget; the pattern does not | [6](06-reliability.md) |

The last two rows are the ones teams skip and regret: an autoscaler on the
wrong signal and a fallback chain invented mid-incident are each responsible
for more serving outages than any model bug. One afternoon wiring queue-depth
scaling and a static-score fallback pays for itself the first bad night.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a large
ranking model with embedding tables and a few dense layers, online and
synchronous, p99 under 50 ms end-to-end, 50 000 QPS at peak with launch spikes
to 3x, daily model updates with gradual rollout and automatic rollback. Here is
the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Deployment type | Online real-time | The caller blocks on the response; batch and async are ruled out by the requirement, not by preference |
| Architecture | Separate model server fleet behind the app, gRPC | Embedding tables of tens of gigabytes and GPU scaling force the split; the app scales on different resources |
| Registry | Versioned immutable artifacts; deploy and rollback are pointer changes | Daily deploys make rollback speed load-bearing; a rebuild under pressure is not a rollback |
| Hardware | GPU replicas with dynamic batching | Sub-linear GPU cost curve at sustained high QPS; the table sizes rule out many small CPU replicas |
| Batching | Dynamic window W = 5 ms, max batch B = 32 | Sized backwards from the budget slice left after network and feature fetch, not for peak idle throughput |
| Deployment strategy | Short shadow, 5% canary, ramp 5-25-50-100 with health gates | Shadow proves no breakage at zero user risk; canary proves user impact; the ramp catches scale-only failures |
| Blue-green | Rejected for the daily path | Two full GPU fleets for a routine checkpoint buys nothing the canary-plus-registry setup lacks |
| Autoscaling | Queue depth as the signal, readiness probes gating warm-up | CPU stays quiet while the GPU saturates; a cold replica taking traffic blows the budget for every request it touches |
| Fallback chain | Cached prediction, then popularity-based default | Defined before launch; the caller always gets an answer inside the timeout |
| Alerting | p99 and p999 first-class, per version | The mean hides the tail; a per-version histogram is what distinguishes a slow model from a slow host |

**The latency budget.** The 50 ms p99 target is spent per
[section 2](02-the-serving-problem.md): 5 ms network round-trip, 10 ms feature
store fetch, 5 ms batch wait window, 25 ms model forward pass at batch 32,
totalling 45 ms with 5 ms of headroom. The application-side timeout sits at
45 ms so tail requests trip the fallback instead of breaching the SLA by a
wide margin ([section 6](06-reliability.md)).

**Replica math.** One replica running batches of 32 through a 5 ms window and a
25 ms forward pass delivers about 32 / 0.030, roughly 1 070 QPS
([section 3](03-batching-and-throughput.md)). The saturation floor at 50 000 QPS
is 47 replicas; at the chapter's target utilization near 0.83 the steady peak
fleet is about 57. Little's law puts roughly 1 000 requests in flight across
the fleet at 50 000 QPS with a 20 ms mean time in system, which is the queue
the monitoring watches.

**Headroom and spikes.** Cold start on these replicas is about 60 seconds
(weights, warm-up pass, cache priming) against a 30-second autoscaler interval,
so h = 2 and the provisioned ceiling is (1 + 2) x 47, about 141 replicas
([section 5](05-autoscaling-and-cost.md)). That is the same number the 3x
launch spike demands as its bare floor (150 000 / 1 070), so cold-start
headroom and spike coverage are one budget here, not two. The diurnal curve
lets autoscaling shrink the fleet well below 57 overnight.

**Cost.** Illustrative: at \$2 per GPU-hour the steady peak fleet runs about
\$114 per hour and the launch-window ceiling about \$282 per hour, with the
diurnal trough well under half of peak. A full shadow mirror doubles inference
spend for as long as it runs, which is why the shadow phase is sized in hours,
not weeks ([section 5](05-autoscaling-and-cost.md)). The CPU alternative fails
before the price comparison starts: at roughly 100 QPS per CPU replica the
floor is 500 replicas, each needing the tens-of-gigabytes embedding tables in
memory, which is the situation that pushed Pinterest to GPU batch economics in
the first place.

**What breaks in month one.** Three failure signals dominate early operations,
so wire them before launch: p99 spikes time-aligned with scale-out events (cold
replicas taking traffic before the warm-up pass returns; the readiness probe is
lying or missing), queue depth climbing while host CPU sits flat (the
bottleneck is the accelerator, and anyone who "fixes" it by scaling on CPU will
watch the queue grow while the autoscaler waits), and a rising fallback-serve
rate (the feature store having a slow day pushes traffic down the fallback
chain, and the model quietly serves stale or unpersonalized answers while every
serving dashboard stays green).

## The same techniques under different constraints

The review question that matters in practice is not "which serving framework is
best" but "which serving design is best under my constraints." Here is the same
stack built three times. Only the middle column is the build above; the other
two keep the identical stage interfaces and swap nearly every choice.

| | Checkout fraud check | Ranking fleet (this chapter) | Nightly recommendation refresh |
|---|---|---|---|
| Model / traffic | Small dense net or boosted trees; ~200 QPS | Large ranker, tens-of-GB embeddings; 50 000 QPS, 3x spikes | Same ranker class; whole catalog scored once nightly, no per-request path |
| Latency contract | p99 tight but the model is milliseconds on CPU | p99 < 50 ms end-to-end | None; finish before morning, cost is the metric |
| Architecture | In-process or one small CPU fleet; the network hop may cost more than the model | GPU model server fleet behind the app, registry-backed | Batch job on spot capacity writing scores to a lookup store |
| Batching | None; request-at-a-time, the wait window would add latency for near-zero fill | Dynamic, W and B sized backwards from the budget | The largest batches the hardware takes; no window, nobody is waiting |
| Deploy strategy | Blue-green is nearly free at three replicas; instant cutover, instant rollback | Shadow, canary, step ramp, automated rollback | Rerun yesterday's job with yesterday's model; "rollback" is re-pointing the lookup table |
| Autoscaling | Barely needed; provision a small fixed fleet | Queue depth, h = 2 headroom, diurnal shrink | None; the job scales out and terminates |
| Fallback | Rules-based approve/decline when the model times out | Cached prediction, then popularity default | Serve yesterday's scores; staleness is the built-in fallback |
| What would be over-engineering | GPU, dynamic batching, canary tooling, autoscaling | Blue-green on the daily path, prediction cache on personalized inputs | Any real-time serving infrastructure at all |

Two lessons fall out. First, the fraud column is mostly deletions: at 200 QPS
on a millisecond CPU model, batching, GPUs, and canary automation are dead
weight, and blue-green, prohibitive for the GPU fleet, becomes the cheapest
safe deploy available. Second, the nightly column shows the
[section 2](02-the-serving-problem.md) deployment-type table acting as the
biggest cost lever in the chapter: moving the same model off the synchronous
path deletes the latency budget, the batch window, the autoscaler, and the
fallback chain in one decision, which is why "does anyone block on this?" is
the first clarifying question and not a detail.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Who waits on the result | Deployment type | Caller blocks: online real-time. Minutes are fine: async queue. Scheduled: batch, up to an order of magnitude cheaper |
| p99 budget | Batch window, timeout | Spend the budget backwards: network and feature fetch first, W and the forward pass get the remainder; timeout just inside the caller's budget |
| Model size and QPS | CPU vs GPU | Small model or low QPS: CPU's linear curve is cheaper. Large model at sustained load: GPU's sub-linear batch curve wins |
| Deploy cadence | Deployment strategy | Daily: shadow, canary, ramp, automated rollback. Rare and high-stakes: blue-green earns its two fleets |
| Cold-start time | Headroom h | h at least cold-start over scaling interval; 60 s starts on a 30 s poll means h = 2 |
| Traffic spikiness | Scaling signal, provisioned ceiling | Scale on queue depth or GPU utilization, never CPU; spikes faster than cold start must be pre-provisioned, not reacted to |
| Freshness tolerance | Online vs precomputed | Hours-old is fine and no request-time context: batch precompute into a lookup store, off the critical path |
| Input repetition | Prediction caching | Low-cardinality repeating inputs: cache pays. Personalized high-cardinality inputs: near-zero hit rate, pure overhead |
| Fan-out width | Percentile target, hedging | Tails compound across hops and fan-out takes the max of K tails: budget per hop, alert on p999, hedge the stragglers |

## The smallest runnable replica pool

The claim this chapter leans on hardest is the one people trust least until
they see it: a fleet can look half idle on average and still be breaching its
p99 SLA, because queueing delay is nonlinear in utilization. So here is the
mechanism in one file with zero installs. Every production component is swapped
for the smallest thing with the same interface: traffic becomes seeded Poisson
arrivals, the model server fleet becomes a heap of replica-free timestamps, the
forward pass becomes a fixed 10 ms service time, and the load balancer becomes
"pop the earliest-idle replica." Latency is queue wait plus service, exactly
the in-server portion of the [section 2](02-the-serving-problem.md) budget.

```python
"""A replica pool as a queue: why autoscaling watches p99, not mean CPU."""
import heapq, random

def simulate(n_requests, arrival_rate, service_ms, replicas, seed=7):
    """Poisson arrivals into a FIFO queue served by identical replicas.
    Each forward pass takes a fixed service_ms, so every millisecond of
    latency beyond service_ms is time spent waiting in the queue."""
    rng = random.Random(seed)
    free_at = [0.0] * replicas                    # when each replica next goes idle
    heapq.heapify(free_at)
    t, latencies = 0.0, []
    for _ in range(n_requests):
        t += rng.expovariate(arrival_rate)        # next arrival (ms apart)
        free = heapq.heappop(free_at)             # earliest-idle replica
        start = max(t, free)                      # wait if every replica is busy
        heapq.heappush(free_at, start + service_ms)
        latencies.append(start - t + service_ms)  # queue wait + forward pass
    latencies.sort()
    mean = sum(latencies) / len(latencies)
    return mean, latencies[int(0.99 * len(latencies))]

SERVICE_MS, REPLICAS, N = 10.0, 8, 200_000
print(f"{REPLICAS} replicas, fixed {SERVICE_MS:.0f} ms forward pass, {N} requests per row")
print(f"{'util':>5} {'mean ms':>8} {'p99 ms':>7}")
for rho in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
    lam = rho * REPLICAS / SERVICE_MS             # arrivals/ms giving this utilization
    mean, p99 = simulate(N, lam, SERVICE_MS, REPLICAS)
    print(f"{rho:>5.2f} {mean:>8.1f} {p99:>7.1f}")

lam = 0.90 * REPLICAS / SERVICE_MS                # hold offered load at the knee
mean, p99 = simulate(N, lam, SERVICE_MS, REPLICAS + 1)
print(f"same load, {REPLICAS + 1} replicas "
      f"(util {0.90 * REPLICAS / (REPLICAS + 1):.2f}): "
      f"mean {mean:.1f} ms, p99 {p99:.1f} ms")
```

Run it and the sweep prints the hockey stick in about forty lines. Between 50
and 90 percent utilization the mean creeps from 10.1 to 14.7 ms, numbers a
dashboard averaging latency would call healthy, while the p99 nearly triples
from 13.0 to 37.4 ms; at 95 percent the mean is still only 21 ms but the p99
hits 64 ms, past the knee where waits compound. The last line is the fix: the
same offered load with one more replica drops utilization from 0.90 to 0.80
and the p99 falls from 37.4 back to 20.8 ms. That single line is
[section 5](05-autoscaling-and-cost.md)'s argument in miniature: the quantity
that degrades first is tail latency driven by queue depth, so the autoscaler
must watch queue depth or a latency percentile, because any mean-shaped signal
stays calm until the SLA is already gone. What the toy leaves out is exactly
what the rest of the chapter adds back: dynamic batching
([section 3](03-batching-and-throughput.md)) changes the service time as a
function of load, cold start ([section 5](05-autoscaling-and-cost.md)) delays
the moment that ninth replica actually helps, and the timeout-and-fallback
chain ([section 6](06-reliability.md)) is what answers the requests stuck in
the queue past 45 ms while the new replica warms.
