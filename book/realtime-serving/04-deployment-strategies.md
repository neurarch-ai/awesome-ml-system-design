# 4. Deployment strategies

## The cardinal rule

Never flip 100 percent of traffic to a new version at once. A model version that
passed offline evaluation can still crash on real inputs, regress tail latency,
or hurt business metrics in ways offline eval cannot see. The safe-deploy toolkit
exists to retire each risk category in the cheapest order: prove no breakage
first, then prove it helps, then widen.

The model registry is what makes this possible. Every deployable model is an
immutable, versioned artifact with its training provenance, offline metrics, and
a stage (staging, production, archived). The rollout controller chooses how much
traffic each version sees. A deploy is a pointer change and a traffic ramp; a
rollback is a pointer change back.

```mermaid
flowchart LR
  REG["model registry<br/>(versioned artifact)"]
  SHADOW["shadow replica<br/>(throwaway output)"]
  CANARY["canary slice<br/>(5% real traffic)"]
  RAMP["gradual ramp<br/>(25, 50, 100%)"]
  ROLLBACK["rollback<br/>(pointer back)"]
  MON["monitoring<br/>(health + metrics)"]

  REG -->|"mirror"| SHADOW
  SHADOW -->|"no breakage"| CANARY
  CANARY -->|"metrics hold"| RAMP
  RAMP -->|"regression"| ROLLBACK
  MON -->|"regression"| ROLLBACK
  ROLLBACK --> REG
```

## Shadow (dark launch, mirroring)

Send a copy of live traffic to the candidate version, run inference, and **throw
the output away**. No user sees it. Compare the new version's latency and
prediction distribution against production offline.

What shadow proves: the new version does not crash on real inputs, does not
regress p99 latency, and does not produce wildly different prediction values
relative to the current version.

What shadow cannot prove: whether the new version actually helps users. Its
predictions never reach anyone, so feedback loops, personalization effects, and
engagement changes are invisible until real traffic is routed.

The cost: shadow doubles inference spend while it runs. At 50 000 QPS on GPU
hardware, this is not cheap. Run shadow for the minimum time to build confidence,
not indefinitely.

## Canary

Route a small real slice (typically 5 percent) to the new version. Watch
per-version serving health (p99, error rate, availability) and online business
metrics (engagement, conversion, coverage) on both the canary and the holdout.
Widen only if the canary holds.

What canary proves: real user impact under a bounded blast radius. If the new
model regresses engagement, 5 percent of traffic is affected, not everyone.

What canary cannot prove: behavior at full scale. Some failure modes only show up
at high concurrency or during traffic spikes. Ramp in steps rather than jumping
from 5 to 100 percent.

Netflix Kayenta automates this entirely: it compares canary and baseline metrics
with statistical rigor and gates the ramp without a human in the loop. That is
the production standard for teams that ship daily.

### Compare and contrast: canary vs shadow

Both send live production traffic through the new model, both run it on real
hardware next to the incumbent, and both exist to catch a bad version before
full rollout, which is why people treat them as interchangeable. The pivot is a
single wiring decision: whether the new model's output is returned to the user
or discarded.

| Aspect | Shadow | Canary |
|---|---|---|
| Input traffic | Real production requests (mirrored copy) | Real production requests (routed slice) |
| Runs on production infra | Yes | Yes |
| Output reaches the user | Never; it is thrown away | Yes, for the slice |
| User risk | Zero | Bounded by the slice size, never zero |
| Can measure | Crashes, latency, prediction-distribution shifts | All of that plus actual user impact (engagement, conversion) |
| Cannot measure | Any user-behavior effect, by construction | Full-scale-only failure modes (needs the ramp) |
| Marginal cost | Doubles inference spend on mirrored traffic | Roughly free; the slice replaces incumbent traffic |

The difference changes the design because the risk being managed is different:
if the question is "will this break," wire a shadow and accept the compute
bill; if the question is "does this help," no amount of shadow time can answer
it, so you must let some users see the output and pay in bounded user risk
instead.

## Gradual rollout (step ramp)

Widen the canary in steps: 5, 25, 50, 100 percent, with health gates between
each. A problem that only manifests at high concurrency (a memory leak, a
thundering-herd in the embedding cache) surfaces on 25 or 50 percent of traffic
rather than on everyone.

![Canary rollout: shadow then step ramp](assets/fig-canary-rollout.png)

*Shadow phase proves no breakage. Canary at 5% exposes the model to real users
on a small blast radius. The ramp widens in steps as health gates clear. Traffic
only reaches 100% after each gate holds. Illustrative numbers.*

## Blue-green

Stand up the candidate version (green) as a complete parallel fleet alongside
the current production fleet (blue). Once green is verified, cut all traffic over
in a single fast switch. Keep blue warm so rollback is an instant traffic switch
back to it, not a rebuild.

Blue-green gives the fastest possible cutover and the fastest possible rollback,
at the cost of running two full fleets simultaneously. At 50 000 QPS on GPU
hardware, the capacity cost may be prohibitive except for infrequent, high-stakes
deploys (a major architecture change, not a daily checkpoint).

## Rollback

Rollback must be faster and more boring than the original deploy. Because the
registry holds the previous version and the rollout controller owns the traffic
split, a rollback is "point production at the last known-good version." The
trigger should be automatic off a health or metric regression, not manual off a
human noticing an incident.

**Define the rollback trigger before you deploy, not during an incident.** Is it
p99 latency exceeding the budget? Error rate above a threshold? An engagement
drop outside the canary's confidence interval? Wire the trigger into the rollout
controller so that reverting takes seconds. Concretely, the trigger is a boolean
the controller polls on every health scrape:

```python
def should_rollback(p99_ms, error_rate, p99_budget_ms, err_budget):
    # any single breach flips the pointer back to the last known-good version
    return p99_ms > p99_budget_ms or error_rate > err_budget
# should_rollback(58, 0.002, 50, 0.01) -> True  (p99 over the 50 ms budget, revert)
```

## Model freshness: cadence, warm start, and incremental updates

Deployment strategy answers "how does a new version reach users." Freshness answers
the question that produces new versions in the first place: **how often, and by what
mechanism.** Interviewers ask it as "how often do you retrain?" and the wrong answer
is a number without a derivation.

**Derive the cadence from a staleness curve.** Train on data up to time $T$, then
evaluate that fixed model on held-out data from $T+1$, $T+7$, $T+30$, and plot the
metric decay. The curve tells you three things at once: how fast quality decays
(the slope), when it crosses the smallest lift worth shipping (the cadence), and
whether the decay is drift or seasonality (a curve that recovers is seasonal). This
is a day of work and it converts a religious argument into a measurement.

Decay speed is driven by what is actually moving:

| Driver | Typical decay | Implication |
|---|---|---|
| Adversarial adaptation (fraud, abuse, spam) | Hours to days | Fastest cadence, plus rules that ship faster than models |
| Catalog and content churn (feed, ads, marketplace) | Days | Daily retraining, and item cold-start handling matters more than cadence |
| User preference drift | Weeks | Weekly is usually enough for the model; features carry the freshness |
| Stable physical or seasonal processes (demand, capacity) | Weeks to months | Retrain on a calendar, and watch for regime changes rather than slow drift |

**Warm start versus cold rebuild.** Warm starting (initialize from the previous
checkpoint, continue on new data) is cheaper and converges faster, and it silently
accumulates state: a bug or a poisoned batch from three weeks ago is still in the
weights, and the model is no longer reproducible from a data snapshot alone. The
standard compromise is a warm-started daily model with a **periodic cold rebuild**
as the anchor, plus the rule that a cold rebuild must reproduce the warm-started
model's quality within tolerance before you trust the warm chain.

**Incremental and online updates.** Some systems update continuously: embedding
tables and CTR models trained on a stream, minutes behind live traffic. What you buy
is freshness on the fastest-moving parts. What you pay:

- **Feedback loops.** The model trains on data generated by its own ranking, so
  positions and exposure are confounded. This is the position-bias and
  logging-policy problem from the [ranking](../ranking/) chapter, and it gets worse
  the tighter the loop.
- **Delayed labels.** A conversion arrives hours later, so an online learner sees a
  biased snapshot of the recent past unless the delay is modelled.
- **No clean rollback.** You can revert a model artifact; you cannot revert learned
  state that has already absorbed a bad hour of data without a checkpoint to fall
  back to. Snapshot the online model on a schedule and treat those snapshots as the
  rollback targets.
- **Harder experimentation.** Comparing two continuously updating models means
  comparing two moving targets, so the experiment design has to hold the update
  policy constant.

**What freshness requires you to version.** Freshness is a reproducibility problem
before it is a quality problem: the data snapshot (or the exact time window),
feature definitions, the training code, and the artifact. If the snapshot is not
versioned, "retrain and compare" is not a repeatable operation and neither is a
rollback.

**Monitor staleness explicitly.** The metric is not "days since last training." It
is the gap between the live metric and the metric the current model achieved on
fresh held-out data at deploy time. Alert on the gap, because that is the signal
that the cadence you derived has stopped being right.

| Reach for | When | Instead of |
|---|---|---|
| Scheduled full retrain | Stable process, decay measured in weeks | A cadence chosen by habit rather than by the staleness curve |
| Warm-started frequent retrain plus periodic cold rebuild | Daily cadence, large models, cost matters | Pure warm start, which accumulates unreproducible state |
| Incremental or online updates | The fastest-moving parts (item embeddings, CTR) and freshness is worth the operational cost | A daily batch job, when the decay is measured in hours |
| Feature freshness instead of model freshness | The model is stable but the inputs move (counts, recency, prices) | Retraining more often to compensate for a stale feature pipeline |
| Rules or overrides alongside the model | Adversarial domains where the response must be faster than any training loop | Waiting for the next retrain during an active attack |

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Shadow | Proving the new version does not crash or regress latency, with zero user risk | Canary, when you need to prove zero-impact before any user sees the model |
| Canary plus step ramp (Kayenta) | Measuring real user impact on a bounded blast radius; gating on online metrics | Shadow alone, which cannot measure user impact |
| Gradual rollout in steps | A failure mode that only appears at scale (memory leak, cache stampede) | A direct 5-to-100 jump that misses scale-dependent bugs |
| Blue-green | A high-stakes deploy where instant cutover and instant rollback justify two full fleets | Routine daily checkpoint updates, where the capacity cost is not warranted |
| Automated rollback trigger | Any production system shipping at daily cadence | Manual rollback, which adds minutes of incident time per bad deploy |
| Serve-while-loading (Grab Catwalk) | A gapless hot-swap where the new version must warm before the old stops taking traffic | A swap that routes traffic before the new replica is ready |

**Provenance.** Statistical automated canary analysis (score the canary against the baseline on a metric panel, gate the ramp on the result) was popularized by Kayenta in Netflix's Spinnaker, named in the table above; the serve-while-loading gapless hot-swap is Grab's Catwalk pattern. Both origins are as stated in this file, not attributed beyond it.

**Tools.** On Kubernetes, Argo Rollouts and Flagger drive canary, blue-green, and step-ramp traffic shifting with automated health gates; Spinnaker with Kayenta (Netflix) runs statistical automated canary analysis. Traffic splitting for shadow and percentage canaries rides on a service mesh such as Istio or Linkerd. Model-serving layers KServe and Seldon Core expose shadow (mirror) and canary primitives directly, and a registry (MLflow Model Registry) holds the versioned, stage-tagged artifacts a rollback points back to. Prometheus plus Grafana or Alertmanager supply the health and metric signals that fire the automated rollback trigger.

**Worked example.** A marketplace ships a new listing-ranker checkpoint on a daily cadence. It first mirrors live traffic to the candidate with a KServe shadow to prove no crash and no p99 regression at zero user risk, since shadow alone cannot show whether the model helps. It then routes a 5 percent canary via Flagger with Kayenta-style analysis gating on conversion and latency, widening in 5, 25, 50, 100 steps so a cache stampede that only appears at scale surfaces before full exposure. Blue-green with two full fleets is held back for the rare high-stakes migration (a serving-stack rewrite), not the routine checkpoint, because two fleets on GPU are costly. Throughout, an automated rollback trigger wired into the rollout controller reverts on a health or metric breach in seconds rather than waiting for a human to notice.
