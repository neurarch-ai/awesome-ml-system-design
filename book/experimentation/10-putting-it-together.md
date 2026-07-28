# 10. Putting It Together: The Complete Build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single experiment
with every decision made. This capstone does three things: it gives you an
opinionated default stack so option paralysis never blocks a first design, it
walks the chapter's scenario end to end with every choice committed and sized,
and it shows how the same decisions flip when the constraints change. It closes
with the smallest runnable experiment, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has two to five credible options, and a first-time
designer can burn a week debating sequential methods before assigning a single
user. Skip that. The stack below is a sane default for a first trustworthy
experiment; each row names when to deviate and which section explains why.
Platforms and libraries change yearly, but the interface of each stage
(randomize, declare metrics, size, validate, analyze, decide) does not.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Unit of randomization | Per user, deterministic hash(user_id, experiment_id), assignment held for the whole run | Arms share inventory, supply, or a social graph: switchback or cluster randomization | [2](02-the-experiment-design.md), [6](06-interleaving-and-alternatives.md) |
| Metric hierarchy | One pre-registered OEC decides; guardrails must not regress; counter-metrics watch for gaming | Never on the count of deciders. Multiple variants or subgroups: add Bonferroni or FDR correction | [2](02-the-experiment-design.md), [5](05-pitfalls.md) |
| Sizing and power | Solve n per arm from sigma and MDE at alpha 0.05, power 0.80, before launch; round duration up to whole weeks | All G guardrails must pass jointly: tighten to beta star = beta / (G + 1) per metric | [3](03-sizing-and-power.md) |
| Validity checks | SRM chi-squared, pre-exposure balance, and flicker exclusion gate every readout | Never. A failed check voids the result before any metric is read | [2](02-the-experiment-design.md), [5](05-pitfalls.md) |
| Analysis method | t-test on per-user aggregates; CUPED with a pre-period covariate when one exists | Metric is a ratio of sums: delta-method variance. Early looks needed: mSPRT, not daily peeks at 0.05 | [4](04-analysis.md) |
| Guardrails | Non-inferiority with explicit margins declared before launch | Never read "not significant" as "safe"; an underpowered guardrail passes silently | [2](02-the-experiment-design.md), [4](04-analysis.md) |
| Ramp policy | Hold the planned window, require the daily effect curve to stabilize, then ramp to 100% with a long-term holdback | Guardrail breach at any point: kill and roll back, no matter how the primary looks | [4](04-analysis.md), [5](05-pitfalls.md) |

The pre-registration embedded in the first three rows is the part beginners skip
and regret: without the OEC, MDE, margins, and duration written down before
launch, every post-hoc reading is a rationalization, and you cannot tell a real
win from a lucky slice. One page of pre-registration pays for itself the first
time a secondary metric "wins" and the primary does not.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a new
ranking model that beats the current one on every offline metric, a session
engagement rate OEC, four guardrails (p99 latency, error rate, revenue per
session, complaint rate), a 1% relative lift as the smallest effect worth
shipping, up to 50% of users available for treatment, and a decision wanted in
two to three weeks. Here is the whole experiment with every choice committed
and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Hypothesis | "The new ranker improves session engagement rate because better calibration surfaces relevant content earlier" | Direction and mechanism written before data exist; this is what separates a hypothesis from a rationalization |
| Primary metric (OEC) | Session engagement rate, one metric decides | Sensitive to a ranking change, measurable in-window, correlated with retention; letting metrics vote always finds a winner |
| Guardrails | p99 latency (margin +10 ms), error rate, revenue per session (margin -0.5%), complaint rate, all non-inferiority | "Not significant" rewards under-powering; the CI must exclude the margin to pass |
| Counter-metrics | Dwell time per click, return visit rate | Engagement can be won by clickbait; these catch the metric being gamed |
| Randomization unit | Per user, hash(user_id, experiment_id) mod 100, 50/50 split | The effect carries across a user's requests; per-request diversion contaminates users with both arms |
| Sizing | ~24,800 users per arm before CUPED, ~12,600 after (math below) | Powered for the primary and jointly for four guardrails, not just the primary |
| Analysis | t-test on per-user aggregates, CUPED with prior-week covariate, delta method for the ratio OEC | Diversion is per user, so variance clusters per user; the OEC is a ratio of sums |
| Duration | 14 days, committed before launch, one look at the end | Whole-week multiples absorb seasonality; the daily effect curve must stabilize past any novelty spike |
| Validity gates | SRM chi-squared and pre-exposure balance on the readout; flicker users excluded | A broken split or contaminated user invalidates everything downstream |
| Ship rule | Lift above the 1% MDE, all four guardrail CIs exclude their margins, no key segment reverses | Significant-but-below-MDE is not a ship; a segment reversal is Simpson's paradox, not a win |

**Sample size.** [Section 3](03-sizing-and-power.md)'s worked figure is the
anchor: at per-user sigma 0.15 and an absolute MDE of 0.01, the formula gives
3,528 users per arm. Our bar is a 1% relative lift on an illustrative baseline
engagement rate of 0.50, so the absolute MDE is 0.005, half the anchor's, and
because n scales as 1 / MDE squared, halving the MDE quadruples the requirement
to 14,112 per arm. That is the number for the primary alone.

**Guardrail powering.** The ship rule requires all four guardrails to pass
alongside the primary, so the Spotify joint correction applies: beta star =
0.20 / (4 + 1) = 0.04, power 0.96 per metric, which replaces z beta = 0.84 with
roughly 1.75. The z-sum factor grows from (1.96 + 0.84)^2 = 7.84 to
(1.96 + 1.75)^2 = 13.76, about 1.76x, lifting the requirement to roughly 24,800
users per arm. Then CUPED with the prior week of the same metric (correlation
around 0.7, removing about half the variance) cuts it to roughly 12,600 per
arm. CUPED did not shorten the calendar here; it bought the joint guardrail
power that honest sizing demanded.

**Duration.** Illustrative traffic: 4,000 new unique users enter the experiment
per day, 2,000 per arm at the 50/50 split. Without CUPED, 24,800 per arm takes
about 12.4 days; with CUPED, about 6.3. Both round up to whole weeks, and the
novelty-effect floor from [section 4](04-analysis.md) demands one to two full
weeks regardless, so the committed window is 14 days. That commitment is the
whole point: the dashboard will cross 0.05 somewhere in the first week by
chance alone, and the plan is what makes ignoring it a policy rather than a
daily act of willpower.

**What breaks in month one.** Three failure signals dominate early operations,
so wire them before launch: SRM drift (a chi-squared alarm on the observed
split; the first cache or bot-filter change that touches one arm asymmetrically
will fire it, and the correct response is to void the result, not to explain it
away), a decaying daily effect curve (a lift that shrinks day over day is a
novelty spike being mistaken for a win; the curve must flatten before the
readout counts), and guardrail intervals too wide to exclude their margins (a
"passing" guardrail whose CI still contains the margin means the test was
underpowered for it, and the ship decision must wait, not proceed on silence).

## The same techniques under different constraints

The question that matters in practice is not "which test is best" but "which
test is best under my constraints." Here is the same experiment designed three
times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every choice. Traffic figures in the
outer columns are illustrative.

| | Low-traffic B2B SaaS | Feed ranker (this chapter) | Marketplace pricing change |
|---|---|---|---|
| Traffic / decision cadence | 2,000 weekly active accounts; quarterly release train | 4,000 users/day entering; decision in 2-3 weeks | City-level supply and demand; treatment shifts driver availability |
| Unit of randomization | Account (seats within an account interact), stratified by account size | User, deterministic hash, 50/50 | Time windows per city (switchback): a user split is biased by shared supply |
| Sizing reality | A 1% MDE needs more accounts than exist; honest MDE is 10%+ or the metric is a within-account rate | 12,600 per arm with CUPED; 14-day window | Effective n = number of windows, not users; longer runs and window-size calibration |
| Variance reduction | CUPED on prior-quarter usage is mandatory, not optional; it is the only way to reach power | CUPED with prior-week covariate, correlation ~0.7 | Window-level CUPED against same-window-last-week; burn-in gaps between windows |
| Analysis | Fixed horizon, one look; sequential methods spend power the traffic cannot afford | t-test on per-user aggregates, delta method for the ratio OEC | Window-level aggregates; carryover checked by comparing early vs late windows |
| Guardrails | Churn and support-ticket rate, read with non-inferiority over a longer horizon | Latency, errors, revenue, complaints, jointly powered | Marketplace health both sides: driver utilization, rider wait time |
| What would be over-engineering | Daily monitoring, mSPRT, interleaving infrastructure | Cluster randomization (no interference in a standard feed) | Per-user CUPED, per-user analysis of any kind |

Two lessons fall out. First, the B2B column is mostly honesty about power: with
2,000 accounts you cannot detect a 1% effect, and the design choices that
matter are raising the MDE to what the traffic can support, randomizing at the
account level where the effect operates, and treating CUPED as load-bearing
rather than optional. Pretending otherwise ships noise on a quarterly cadence.
Second, the marketplace column shows the unit of randomization absorbing the
interference problem: once arms compete for shared supply, [section
6](06-interleaving-and-alternatives.md)'s switchback redraws the arm boundary
around the resource, and every downstream choice (sizing, variance reduction,
analysis) follows the new unit, windows instead of users.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you debate any method.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Daily traffic vs required n | MDE, or the method itself | n scales as 1 / MDE^2: halving MDE quadruples traffic. If n exceeds the traffic, raise the MDE honestly or change method |
| Where the effect carries | Unit of randomization | Divert at the level the effect operates: user for a ranker, account for seats, time window for shared supply |
| Interference (marketplace, social graph) | Randomization design | Cluster or switchback; budget for the power hit, roughly a 1 + (m - 1) rho design effect on clustered units |
| Need for early looks | Testing method | Continuous monitoring needs mSPRT or always-valid p-values; a fixed-horizon threshold checked daily inflates false positives several-fold |
| Pre-period covariate exists | Variance reduction | CUPED at correlation 0.6-0.8 removes 35-65% of variance, the same power with roughly half the users |
| Number of guardrails G | Per-metric power | All must pass jointly: power each at beta star = beta / (G + 1), then re-solve n |
| Metric is a ratio of sums | Variance estimator | Delta method (or bootstrap); naive per-row variance is wrong twice, clustering and denominator |
| Many candidate rankers | Screening method | Interleaving prunes with ~100x less traffic; the A/B on the survivors still makes the ship call |
| Weekly seasonality | Duration | Whole-week multiples, one to two weeks minimum, and the daily effect curve must stabilize |

## The smallest runnable experiment

Every pitfall in [section 5](05-pitfalls.md) is a claim about what happens on
null data, and null data is free to generate. So here is the chapter's central
discipline demonstrated in one file with zero installs: thousands of A/A
experiments (both arms draw from the same distribution, so every significant
result is false by construction), each analyzed twice on identical data. The
fixed-horizon reader looks once at the planned end; the peeker checks the same
z-statistic every day and stops at the first crossing of 1.96. The per-user
metric uses the chapter's own figures, mean 0.50 and sigma 0.15.

```python
"""A/A peeking simulation: the same null data read two ways, stdlib only."""
import math, random

DAYS, USERS_PER_DAY, SIMS, Z_CRIT = 20, 200, 4000, 1.96
random.seed(11)

def z_stat(n_a, sum_a, sq_a, n_b, sum_b, sq_b):
    """Two-sample z on running sums; production: scipy.stats.ttest_ind."""
    mean_a, mean_b = sum_a / n_a, sum_b / n_b
    var_a = (sq_a - n_a * mean_a * mean_a) / (n_a - 1)
    var_b = (sq_b - n_b * mean_b * mean_b) / (n_b - 1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    return (mean_b - mean_a) / se if se else 0.0

fixed_hits, peek_hits, stop_days = 0, 0, []
for _ in range(SIMS):
    na = sa = qa = nb = sb = qb = 0
    stopped = False
    for day in range(1, DAYS + 1):
        for _ in range(USERS_PER_DAY):          # one row per user per arm
            x = random.gauss(0.50, 0.15)        # control user's metric
            y = random.gauss(0.50, 0.15)        # treatment user: SAME distribution
            na, sa, qa = na + 1, sa + x, qa + x * x
            nb, sb, qb = nb + 1, sb + y, qb + y * y
        z = abs(z_stat(na, sa, qa, nb, sb, qb))
        if not stopped and z > Z_CRIT:          # the peeker stops at first crossing
            stopped = True
            stop_days.append(day)
        if day == DAYS and z > Z_CRIT:          # the fixed-horizon reader looks once
            fixed_hits += 1
    peek_hits += stopped

print(f"experiments simulated : {SIMS} A/A tests (no true effect exists)")
print(f"looks per experiment  : 1 (fixed horizon) vs {DAYS} (peek daily, stop when significant)")
print(f"fixed-horizon FPR     : {fixed_hits / SIMS:.1%}  (nominal alpha = 5.0%)")
print(f"peeking FPR           : {peek_hits / SIMS:.1%}  ({peek_hits / fixed_hits:.1f}x the fixed-horizon rate)")
print(f"median false-stop day : day {sorted(stop_days)[len(stop_days) // 2]} of {DAYS}")
```

Run it and the output makes the pitfall quantitative: the fixed-horizon reader
comes back at 5.2%, right at the nominal alpha, while the peeker, reading the
identical data, false-positives on 24.3% of experiments, 4.6 times the stated
rate, with a median false stop on day 4 of 20. That is the "ranking test wins
in the first week" trap measured: nearly one experiment in four hands the
peeker a shippable-looking result from pure noise, and it usually arrives early
enough to feel like a fast win. Every toy piece has a production counterpart:
`random.gauss` stands in for the per-user aggregates of the primary metric, the
running sums are the exposure-outcome join from [section
2](02-the-experiment-design.md), `z_stat` is the t-test at scale, the daily
loop is the live dashboard, and the `stopped` flag is a team without a
pre-registered stopping rule. What the file deliberately omits is the repair
kit: an SRM check before any reading, CUPED to shrink `sigma`, and the
sequential thresholds from [section 4](04-analysis.md) that widen Z_CRIT per
look so the peeker's 24.3% is pulled back down to 5%. Add those and you have
rebuilt this chapter.
