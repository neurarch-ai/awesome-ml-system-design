# 10. Putting it together: the complete build

Sections 1 through 6 taught each layer with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single monitoring
system with every decision made. This capstone does three things: it gives you
an opinionated default stack so option paralysis never blocks a first build, it
walks the chapter's scenario end to end with every choice committed and sized,
and it shows how the same decisions flip when the constraints change. It closes
with the smallest runnable drift monitor, one file, no installs.

## The default stack: start here, deviate with reason

Every layer in this chapter has several credible detectors and policies, and a
first-time builder can burn a week comparing drift libraries before logging a
single prediction. Skip that. The stack below is a sane default for a first
production monitor; each row names when to deviate and which section explains
why. Tools change yearly, but the interface of each layer (log, check health,
test drift, join labels, alert, respond) does not, so pick per layer by
interface and treat any specific library as replaceable.

| Layer | Default | Deviate when | Why (section) |
|---|---|---|---|
| Prediction log | Async writer off the request path; score, served features, timestamp, model version | Feature set is very wide and traffic huge: stratified sampling | [6](06-serving-and-scaling.md) |
| Data-quality checks | Null rate, schema, freshness, ranges on every feature, before any drift test | Never. This is always the first check | [2](02-what-to-monitor.md) |
| Feature drift | PSI per feature, fixed decile bins from the training reference, daily windows, 0.10/0.25 bands | Categorical features: chi-square; you need a p-value on a continuous feature: KS, gated on effect size | [3](03-detecting-drift.md) |
| Prediction drift | Score-distribution PSI plus mean and entropy of the score stream | Serving layer is a black box and only outcomes are visible: skip to performance | [2](02-what-to-monitor.md), [4](04-monitoring-without-labels.md) |
| Delayed-label performance | AUC and calibration by segment once labels join, respecting the feedback window | Labels take weeks: lead with the drift proxies and treat labels as confirmation | [4](04-monitoring-without-labels.md) |
| Alerting policy | Sustained breach (three consecutive windows), thresholds calibrated per feature from history, tiered severity | Model is P0 (fraud, pricing): page on the first confirmed breach | [5](05-alerting-and-response.md) |
| Response runbook | Triage bug-vs-drift first, triggered retrain through the eval gate, one-step rollback | Drift is slow and steady: scheduled retraining as the baseline instead | [5](05-alerting-and-response.md) |

The second row is the one beginners skip and regret: Google found 60 of 96
production ML failures were data and pipeline bugs, not model-quality problems.
A null-returning feature reads exactly like distribution drift, and retraining
on the corrupted window bakes the bug into the next model. Health checks run
first for this reason.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
recommendation ranker scoring a few hundred candidates per request, click
labels in seconds, long-term engagement in weeks, a history of six months of
silent decay, an alert-fatigued team, and on-demand retraining that costs a
day. Here is the whole system with every choice committed and the reason it
won.

| Decision | Choice | Why it won |
|---|---|---|
| Logging | Async log of score, served features, timestamp, model version; stratified sample | Served features are the only way to tell a pipeline bug from drift; async keeps it off the request path |
| Sampling | 10% of bulk traffic, 100% of new users | New users are the cohort most likely to regress unseen; a global sample underrepresents them |
| Data health | Null rate, schema, freshness, ranges on every feature, hourly | The most common real failure; a frozen or null feature must be named before any drift test runs |
| Feature drift | PSI, fixed decile bins from the training reference, daily windows | Symmetric, window-agnostic, thresholds are effect sizes not p-values; the industry default at scale |
| Prediction drift | Score-distribution PSI plus mean and entropy, hourly | Label-free and observable immediately; a mean-score slide from 0.62 to 0.48 precedes the measured AUC drop |
| Fast performance | Click AUC and calibration, hourly windows that respect the 30-minute feedback window | Clicks arrive in seconds; closing the window first avoids the premature-labeling bias |
| Slow performance | Weekly engagement and retention metrics by segment | The six-month silent decay lived here; segments stop a global 0.81 AUC from hiding a 0.67 cohort |
| Thresholds | PSI 0.10 warn / 0.25 page as the starting rule, then calibrated per feature from history | The team is alert-fatigued; a feature that swings 0.15 on weekends must not page every Monday |
| Alert policy | Sustained breach over three consecutive windows, tiered P2 baseline escalating on confirmed decay | A single noisy PSI spike is noise; three windows is signal; a feed ranker does not wake anyone at 3am |
| Response | Triage bug-vs-drift, then triggered retrain through the eval gate; one-step rollback for bad promotes | Retraining costs a day, so triggers must be deliberate and never fire on a pipeline bug |

**Metric cardinality.** The ranker has ~120 features (Illustrative). PSI per
feature globally plus six tracked segments ([section 2](02-what-to-monitor.md))
is 120 x 7 = 840 feature monitors, plus prediction drift and per-segment
performance, call it ~900 series. That is the multiple-testing regime from
[section 3](03-detecting-drift.md): at any moment something looks drifted, so
page only on prediction-drift and performance breaches, and let single-feature
PSI rank the diagnosis instead of driving the page. Uber D3 runs 100,000+
column-level monitors at about \$0.01 per dataset; the cost of a monitor is
storage and compute, but the cost of an undisciplined monitor is the on-call's
trust.

**Log volume.** At 20M impressions per day (Illustrative) and roughly 1 KB per
record for 120 served features, full logging is ~20 GB/day. The 10% stratified
sample cuts that to ~2 GB/day plus the full-rate new-user slice, and drift
statistics on a 2M-row daily window are indistinguishable from the full
population ([section 6](06-serving-and-scaling.md)). Sampling is the lever;
dropping features is not, because a low-weight feature can become important
after the world moves.

**Label-delay windows.** The two-speed design from
[section 4](04-monitoring-without-labels.md) falls straight out of the label
schedule. Fast path: data health, feature PSI, prediction drift, and click AUC
run hourly to daily, with the click metric waiting out its 30-minute feedback
window so early impressions are not scored as negatives. Slow path: engagement
and retention join back over weeks and run as weekly segment reports. The fast
path is the early warning; the slow path is the confirmation. An incident
flagged only by the fast path is often benign drift in a low-weight feature; an
incident confirmed on both is the six-month decay caught in week one.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: the multiple-testing alarm storm (with ~900
series something always breaches; if the on-call gets more than a few pages in
week one, the thresholds were not calibrated from history), the pipeline bug
read as drift (a PSI spike that arrives together with a null-rate or freshness
breach on the same feature is a broken upstream join, and the runbook must
route to fix-and-backfill, never to retrain), and the Monday-morning seasonal
page (a feature that naturally swings PSI 0.15 over weekends will page every
Monday until its threshold is raised or replaced with a learned band, which is
exactly why Uber D3 fits Prophet bounds per monitor).

## The same techniques under different constraints

The review question that matters in practice is not "which drift test is best"
but "which monitoring design is right under my label delay and my stakes." Here
is the same stack built three times. Only the middle column is the build above;
the other two keep the identical layer interfaces and swap nearly every policy
choice.

| | Fraud classifier | Feed ranker (this chapter) | Internal churn model |
|---|---|---|---|
| Label delay | Weeks (dispute resolution) | Seconds (clicks), weeks (retention) | Months (churn is defined over a quarter) |
| Dominant drift risk | Concept drift; adversaries adapt, marginals can stay flat | Slow covariate shift; silent decay | Slow covariate shift plus base-rate (label) drift |
| Primary signal | Feature-to-label relationship monitoring plus prediction drift; input marginals alone are structurally blind here | Feature + prediction PSI as leading indicators, click AUC as fast confirmation | Monthly PSI on inputs and scores; performance is a quarterly report |
| Severity tier | P0: page immediately on any confirmed breach, rollback ready | P2 baseline, escalate on confirmed segment decay | P3: dashboard and logging only |
| Windows | Hourly fast path; every day of delay is money lost | Hourly fast path, daily drift, weekly slow path | Monthly windows; anything shorter is noise on a small population |
| Retraining | Triggered aggressively; fresh labels encode the new adversarial mapping, old data reinforces the wrong one | Scheduled baseline plus deliberate triggered retrain through the gate | Scheduled quarterly; a triggered pipeline is over-engineering |
| What would be over-engineering | Nothing on this list; the stakes justify the platform | Auto-rollback infrastructure on every alert | Prophet bands, shadow traffic, triggered retrains |

Two lessons fall out. First, the fraud column is the one place where
input-marginal monitoring is structurally insufficient: concept drift moves
$P(y \mid X)$ while every feature histogram stays flat ([section
2](02-what-to-monitor.md)), so the spend goes into relationship monitoring and
fast escalation rather than more input tests. Second, the churn column shows
that monitoring maturity should match decision cadence: when nobody can act
faster than quarterly, hourly drift checks produce cost and fatigue, not
safety.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Label delay | Fast-path vs slow-path weighting | Seconds: performance is the primary monitor. Weeks or months: PSI and prediction drift lead, labels confirm |
| Alert-fatigued team | Threshold calibration and breach policy | Sustained breaches only, thresholds from each feature's history; the 0.10/0.25 rule is a starting point, never a constant |
| Cost of a miss (stakes) | Severity tier and loop closure | Fraud or pricing: P0 with auto-rollback. Feed ranking: P2 dashboard first. Internal scoring: log and review |
| Feature count | Page condition | Hundreds of features: page on prediction drift and performance, use per-feature PSI to diagnose, correct for monitor count |
| Traffic volume | Sampling strategy | Drift statistics survive a 1-10% sample; stratify to protect small high-value cohorts before thinning the bulk |
| Segment structure | Metric slicing | Any cohort that can regress independently (new users, new device) gets its own sliced series; aggregates hide cliffs |
| Retrain cost | Trigger discipline | Cheap retrains: schedule them. A day or more: triggered retrains through the eval gate, never straight to production |
| Seasonality | Reference and threshold form | Weekly rhythms: learned bands (Prophet-style) or same-period references; a static reference flags every weekend |
| Adversarial domain | What you monitor | Marginal drift is not enough; monitor the feature-to-label relationship or the drift dashboard stays green while quality dies |

## The smallest runnable drift monitor

The review of every monitoring tutorial is the same: the reader assembles a
metrics stack and still cannot see why the fast path exists. So here is the
chapter's core mechanism in one file with zero installs: a training-time
reference, a stream of daily serving windows where one feature drifts from an
injected day, PSI read against the 0.10/0.25 bands, and a label-based
engagement metric that arrives 14 days late. Every production component is
swapped for the smallest thing with the same interface: the feature store
becomes two lists of floats, the drift job becomes a PSI function over fixed
decile bins, and the label join becomes a dictionary lookup with a delay.

```python
"""PSI as an early warning vs a delayed label metric. Stdlib only; seeded."""
import math, random

random.seed(11)
BINS = 10
WARN, ALERT = 0.10, 0.25          # the PSI field rule from section 3
DRIFT_START, LABEL_DELAY = 8, 14  # drift injected day 8; labels join 14 days late

def quantile_edges(sample, bins=BINS):
    """Fixed bin edges from the reference window, as section 3 prescribes."""
    s = sorted(sample)
    return [s[int(len(s) * i / bins)] for i in range(1, bins)]

def fractions(sample, edges):
    counts = [0] * (len(edges) + 1)
    for x in sample:
        i = 0
        while i < len(edges) and x > edges[i]:
            i += 1
        counts[i] += 1
    n = len(sample)
    return [max(c / n, 1e-6) for c in counts]  # floor empty bins, no log(0)

def psi(ref_frac, cur_frac):
    return sum((p - q) * math.log(p / q) for p, q in zip(ref_frac, cur_frac))

def day_mean(day):
    """The drifting feature's true mean: flat, then +0.06 sd per day."""
    return 0.0 if day < DRIFT_START else 0.06 * (day - DRIFT_START + 1)

# Training reference: both features healthy, N(0, 1).
ref = [random.gauss(0, 1) for _ in range(50_000)]
edges = quantile_edges(ref)
ref_frac = fractions(ref, edges)

# True daily quality (a 14-day engagement rate), known only LABEL_DELAY later.
# Quality erodes as the model scores inputs outside its training support.
true_rate = {d: 0.300 - 0.05 * day_mean(d) ** 2 for d in range(1, 31)}

first_warn = None
print(f"{'day':>3} {'PSI stable':>10} {'PSI drift':>10} {'flag':>6}"
      f"   labels that just arrived")
for day in range(1, 31):
    stable = [random.gauss(0, 1) for _ in range(5_000)]
    drifted = [random.gauss(day_mean(day), 1) for _ in range(5_000)]
    ps = psi(ref_frac, fractions(stable, edges))
    pd = psi(ref_frac, fractions(drifted, edges))
    flag = "ALERT" if pd >= ALERT else "warn" if pd >= WARN else "-"
    if first_warn is None and pd >= WARN:
        first_warn = day
    lday = day - LABEL_DELAY
    if lday >= 1:
        r = true_rate[lday]
        label = f"day {lday:>2} engagement = {r:.3f}" + \
                ("  <- confirmed drop" if r < 0.294 else "")
    else:
        label = f"none yet (day 1 labels land on day {1 + LABEL_DELAY})"
    print(f"{day:>3} {ps:>10.4f} {pd:>10.4f} {flag:>6}   {label}")

print(f"\nPSI first warned on day {first_warn}; labels confirming the damage "
      f"for that day arrive on day {first_warn + LABEL_DELAY}.")
```

Run it and the printed table is the two-speed design of
[section 4](04-monitoring-without-labels.md) in thirty rows. The stable
feature's PSI stays near 0.002 for all thirty days. The drifting feature
crosses the 0.10 warn band on day 13, five days after the injected shift, and
the 0.25 page band on day 16, while the labels column still reads "none yet"
and then a healthy 0.300. The first confirmed engagement drop does not print
until day 27, fourteen days after the drift was already flagged, and the
closing line states the gap: PSI first warned on day 13; labels confirming the
damage for that day arrive on day 27. Swap the two lists for a feature store,
the loop for a scheduled drift job with per-feature calibrated thresholds and a
sustained-breach rule ([section 5](05-alerting-and-response.md)), and the
dictionary lookup for a real label join, and you have rebuilt this chapter's
fast path and slow path, and the two-week head start is the argument for
building the fast path at all.
