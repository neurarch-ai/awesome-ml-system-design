# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's credit-risk scenario end to end with every choice committed and sized,
and it shows how the same decisions flip when the constraints change. It closes
with the smallest runnable gradient booster, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing encoders before scoring a single row. Skip
that. The stack below is a sane default for a first production build; each row
names when to deviate and which section explains why. Libraries change yearly,
but the interface of each stage (frame, join, encode, train, validate,
calibrate, explain, serve) does not, so pick per stage by interface and treat
any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Task framing | Fixed-window binary classification, calibrated | Timing matters with censored rows: survival. Question is WHETHER to intervene: uplift | [2](02-frame-as-ml-task.md) |
| Feature pipeline | Point-in-time joins from a feature store; static + rolling aggregates + ratios; log features at serve time | Never. The time-travel join is load-bearing | [3](03-data-preparation.md) |
| Encoding | One-hot for low cardinality; cross-fitted target encoding for hundreds of values | Millions of IDs: hashing or learned embeddings; CatBoost's ordered encoding if staying tree-native | [3](03-data-preparation.md), [4](04-model-development.md) |
| Model family | Gradient-boosted trees (LightGBM / XGBoost / CatBoost) | Millions of ID values or fusion with text/images: neural with embeddings | [4](04-model-development.md) |
| Validation | Time-based split with an out-of-time holdout; early stopping on a validation fold | Never use a random split; shared behavioral windows leak the future | [5](05-evaluation.md) |
| Calibration | Platt on a small holdout; isotonic once the holdout has enough events per bin | Score is consumed as a pure ranking (tiering, prioritization): calibration is optional | [4](04-model-development.md), [5](05-evaluation.md) |
| Interpretability | Tree SHAP reason codes; monotone constraints where a regulator or auditor can ask why | Unregulated ranking-only use: SHAP stays useful for debugging, constraints become optional | [6](06-serving-and-scaling.md) |
| Serving | Batch scoring into a score store; seconds-scale realtime only where the decision follows a fresh action | Never sub-millisecond first; tabular bottlenecks are labels and calibration, not QPS | [6](06-serving-and-scaling.md) |

The framing row is the one beginners skip and regret: an intervention question
answered with a propensity model spends the budget on sure things and lost
causes, and no amount of downstream tuning recovers it. Pin the decision first,
then pick the row; [section 2](02-frame-as-ml-task.md) is one page and it is the
highest-leverage page in the chapter.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
regulated credit issuer scoring card applicants. The output is a calibrated
probability of default (90+ days past due within 12 months) that feeds an
approve/decline rule, a limit formula, and a risk-based price. Labels mature
over 12 months and exist only for approved applicants; every decline owes
plain-language adverse-action reasons. Here is the whole system with every
choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Framing | Fixed-window binary classification, calibrated | The 12-month horizon is fixed by the label definition; the decision reads one number per applicant |
| Label | 90+ DPD within 12 months, matured vintages only; 30-day early delinquency tracked as a proxy | Counting immature accounts as good biases risk downward on exactly the newest applicants |
| Selection bias | Reject inference from bureau data plus a 1% randomized approval slice below the cutoff | Imputed outcomes reduce the bias; only randomization produces unbiased ground truth for the borderline region |
| Features | ~300 point-in-time features: static attributes, rolling aggregates, ratios (utilization, payment ratio); served values logged | The time-travel join kills leakage; logging kills train/serve skew |
| Encoding | One-hot for product and channel; cross-fitted target encoding for moderate-cardinality categoricals | No million-ID tables here, so embeddings buy nothing; cross-fitting closes the target-encoding leak |
| Model | Monotone-constrained gradient-boosted trees | Heterogeneous columns with missing values; constraints make each decline defensible to a regulator |
| Validation | Time-based split; most recent matured quarter held out as out-of-time test | A random split lets shared behavioral windows leak the future and inflates every metric |
| Calibration | Isotonic on a held-out matured quarter, monitored with reliability curves sliced by segment and vintage | The probability multiplies into a limit and a price; the holdout has enough events per bin for isotonic |
| Decision layer | Expected-value threshold plus limit formula, in a separate artifact from the model | The cost matrix changes with interest rates; the policy must update without retraining the model |
| Explainability | Tree SHAP, top contributions rendered as adverse-action reasons | Exact and fast for trees; satisfies the per-decline reason requirement |
| Serving | Score at application time, seconds-scale budget; feature store supplies as-of-now features | Decisions run in seconds, not microseconds; the hard part was never QPS |
| Monitoring | The ladder: PSI and score drift now, proxy-label drift in weeks, sliced calibration as labels mature | A 12-month label cannot be the alarm; the label-free rungs have to fire first |

**Training data sizing.** Take 1.5 million applications per year with a 60%
approval rate (Illustrative). Only approvals carry labels, so one year yields
900k labeled rows, and the 12-month maturation means the most recent year is
unusable. Training on the three matured years before that gives roughly 2.7M
rows. At ~300 features in float32 that is 2.7M x 300 x 4 bytes, about 3.2 GB:
one machine, and [section 4](04-model-development.md)'s observation that a
boosted model on millions of rows trains in minutes holds with room to spare.
Nothing in this build needs a GPU or a cluster; the scarce resources are
matured labels and clean features.

**Validation-fold design.** The three matured years split by time, never at
random: the oldest ten quarters train the model with early stopping on the next
quarter, the following quarter fits the isotonic calibrator, and the most
recent matured quarter is the untouched out-of-time test. At a 2.5% default
rate (Illustrative, inside the chapter's 1-to-5% band) the calibration quarter
holds about 225k rows and ~5,600 default events, enough events per bin for
isotonic to beat Platt without encoding noise, which is exactly the rule
[section 8](08-interview-qa.md) gives for choosing between them.

**The cost that matters.** Compute is a rounding error: a monthly full retrain
on 3.2 GB and a weekly calibrator re-fit run on one box for tens of dollars.
The real training cost is the randomized approval slice. Declining 40% of 1.5M
applications leaves 600k declines a year; approving 1% of them at random is
6,000 deliberately risky accounts. If that population defaults at 8% against
the portfolio's 2.5%, and a default costs about \$2,000 of exposure times loss
given default, the slice burns roughly 6,000 x 5.5% x \$2,000, about \$650k per
year (Illustrative). That is the price of unbiased labels for the borderline
region, it is a business decision, and it dwarfs every infrastructure line in
the budget.

**What breaks in month one.** With a 12-month label, nothing that matters shows
up in the label for a year, so wire the label-free signals before launch: PSI
above 0.2 on any top-SHAP feature (a marketing change shifts the applicant mix
long before a default can mature), approval rate moving at a fixed threshold
(score drift that means the population changed, not that the model improved),
and offline-online score parity on sampled requests (the first sign of
train/serve skew is the same applicant scoring differently in the batch
recompute than at serve time, and it is invisible unless you log served
features and compare).

## The same techniques under different constraints

The review question that matters in practice is not "which model is best" but
"which model is best under my constraints." Here is the same pipeline built
three times. Only the credit column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Regulated credit risk (this chapter) | Subscription churn save-offers | Advertiser churn tiering |
|---|---|---|---|
| Decision it feeds | Approve/decline, limit, price | Which customers get a discount | Which accounts a fixed team of account managers calls |
| Label | 90+ DPD in 12 months; matures in a year; approvals only | Cancel within 4 weeks; resolves in a month | Spend stops within 14 days; resolves in two weeks |
| Selection bias | Structural: rejects vanish from the label; reject inference + randomized slice | Only past offer recipients reveal treatment effects; requires an RCT slice | Mild: all advertisers observed |
| Framing and model | Calibrated binary GBDT, monotone-constrained | Uplift/CATE (two-model or causal forest) over a GBDT feature base; survival if timing drives contact | Plain GBDT snapshot, precision-recall tiering |
| Calibration | Load-bearing: the number multiplies into money | Load-bearing on the uplift estimate: budget fills a knapsack by uplift-per-dollar | Skipped: only the ranking is consumed |
| Explainability | Regulatory: SHAP adverse-action reasons per decline | Internal only: SHAP for debugging | Operational: SHAP tells the account manager what to say |
| Validation cadence | Monthly retrain, weekly recalibration, quarterly matured re-eval | Retrain on each resolved 4-week cohort | Daily scoring, frequent retrain; labels are cheap and fast |
| Serving | Request-time scoring, seconds budget | Weekly batch into the campaign tool | Daily batch into the CRM |
| What would be over-engineering | Neural embeddings, realtime feature streaming | Regulation-grade monotone constraints, reason-code plumbing | Any calibration layer, uplift machinery, realtime anything |

Two lessons fall out. First, moving right strips cost: the tiering column
drops calibration, constraints, reject inference, and the exploration budget
entirely, because a fast clean label and a ranking-only decision delete the
four most expensive parts of the credit build. A two-week label is worth more
than any modeling trick in the left column. Second, the middle column shows the
framing, not the infrastructure, doing the work: the feature store, the GBDT,
and the time-based split all survive, but the question changes from "who will
churn" to "whom does the discount change," and [section 2](02-frame-as-ml-task.md)'s
warning applies with money attached: a propensity model here would spend the
retention budget on sure things and lost causes.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| The absolute number sets money | Calibration layer and its monitoring | Platt for small holdouts, isotonic with enough events per bin; reliability curves sliced by segment and vintage, always |
| Label maturation lag | Training-data strategy | Matured vintages, a faster proxy, or survival censoring; never count immature accounts as good |
| Outcomes observed only for chosen entities | Selection-bias correction | Reject inference reduces bias; only a randomized slice produces ground truth, and its cost is a business decision |
| Regulator can ask "why this decision" | Model family and explanation path | Monotone-constrained GBDT plus tree SHAP reason codes; decide this before training, not after |
| Positives under ~5% | Metric and imbalance handling | AUC-PR over AUC-ROC; class weights over SMOTE; recalibrate after any resampling |
| Millions of ID values | Encoding | Hashing or learned embeddings; cross-fitted target encoding for hundreds of values; ordered encoding if staying in CatBoost |
| The question is WHETHER to intervene | Framing and data | Uplift/CATE, which needs randomized treatment data; the optimizer that allocates budget is a separate box |
| WHEN the event lands matters, rows censored | Framing | Survival analysis; a fixed-window binary discards the censored rows and the timing |
| Score freshness needed | Batch vs realtime | Batch into a score store unless the decision follows a fresh user action; the latency budget is seconds |
| Drift with slow labels | Monitoring ladder | PSI and score drift immediately, proxy labels in weeks, sliced calibration in months; recalibrate before you retrain |

## The smallest runnable gradient booster

The chapter's default model family is gradient boosting, and the review of
every library tutorial is the same: the reader tunes twelve hyperparameters and
never sees the mechanism. So here is the mechanism in one file with zero
installs. Every production component is swapped for the smallest thing with the
same interface: the depth-6 trees become depth-1 stumps, the library's split
finder becomes an exhaustive scan, the credit dataset becomes a seeded
three-feature applicant with a non-smooth interaction target (income below 0.3
AND utilization above 0.6, exactly the shape [section 4](04-model-development.md)
says trees win on), and the validation fold becomes a held-out sample printed
every fifteen rounds. The training set is small on purpose: overfitting is the
lesson.

```python
"""Gradient boosting with depth-1 stumps, runnable with no installs."""
import random

random.seed(7)

def make_row():
    """One applicant: income, utilization, tenure, all scaled to [0, 1]."""
    return [random.random(), random.random(), random.random()]

def true_risk(x):
    """Non-smooth interaction target: exactly the shape trees win on."""
    income, util, tenure = x
    risk = 0.0
    if income < 0.3 and util > 0.6: risk += 1.0     # the interaction
    if util > 0.85:                 risk += 0.5     # a lone threshold
    return risk + 0.25 * (random.random() - 0.5)    # label noise

TRAIN = [make_row() for _ in range(60)]             # small on purpose: overfit is the lesson
HELD  = [make_row() for _ in range(400)]
Y_TR  = [true_risk(x) for x in TRAIN]
Y_HD  = [true_risk(x) for x in HELD]

def fit_stump(X, resid):
    """Best single split over all features: two leaf means, min squared error."""
    best = None
    for f in range(len(X[0])):
        for thr in sorted({x[f] for x in X}):
            left  = [r for x, r in zip(X, resid) if x[f] <  thr]
            right = [r for x, r in zip(X, resid) if x[f] >= thr]
            if not left or not right: continue
            ml, mr = sum(left) / len(left), sum(right) / len(right)
            sse = sum((r - ml) ** 2 for r in left) + sum((r - mr) ** 2 for r in right)
            if best is None or sse < best[0]: best = (sse, f, thr, ml, mr)
    _, f, thr, ml, mr = best
    return lambda x: ml if x[f] < thr else mr

def rmse(pred, y):
    return (sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)) ** 0.5

# --- the boosting loop: each stump fits what the ensemble still gets wrong ----
LR, ROUNDS = 0.3, 120
base = sum(Y_TR) / len(Y_TR)                        # round 0: the mean predictor
f_tr = [base] * len(TRAIN)
f_hd = [base] * len(HELD)
best_round, best_err, stumps = 0, rmse(f_hd, Y_HD), []
print(f"round {0:>3}  train {rmse(f_tr, Y_TR):.3f}  held-out {rmse(f_hd, Y_HD):.3f}   (mean predictor)")
for t in range(1, ROUNDS + 1):
    stump = fit_stump(TRAIN, [y - p for y, p in zip(Y_TR, f_tr)])   # fit the residual
    stumps.append(stump)
    f_tr = [p + LR * stump(x) for p, x in zip(f_tr, TRAIN)]
    f_hd = [p + LR * stump(x) for p, x in zip(f_hd, HELD)]
    err = rmse(f_hd, Y_HD)
    if err < best_err: best_round, best_err = t, err
    if t % 15 == 0 or t == 1:
        note = "  (single stump)" if t == 1 else ""
        print(f"round {t:>3}  train {rmse(f_tr, Y_TR):.3f}  held-out {err:.3f}{note}")
print(f"\nbest held-out {best_err:.3f} at round {best_round}: early stopping would halt there;")
print(f"round {ROUNDS} kept training error falling while held-out error rose.")
```

Run it and the whole of [section 4](04-model-development.md)'s training-curve
diagnostics play out in the printout. The mean predictor scores held-out RMSE
0.403 and a single stump 0.363; boosting drives held-out error down to its
minimum of 0.293 at round 24, and from there training error keeps falling every
round (0.139 at round 30 to 0.080 at round 120) while held-out error climbs
back to 0.306: the ensemble has stopped learning the interaction and started
memorizing the 60 rows and their noise. That divergence is the overfitting row
of the pitfalls table made visible, and the `best_round` tracker is early
stopping in two lines. The mapping to production is direct: `fit_stump` stands
in for a library's histogram-based split finder, `LR` is the shrinkage that
makes hundreds of weak trees sum to a strong model, the residual in the loop is
the gradient of squared loss (swap in the log-loss gradient and this becomes a
classifier), and `HELD` is the validation fold your early-stopping callback
watches. What the toy deliberately omits is everything the rest of this chapter
adds: the point-in-time join that keeps `TRAIN` honest, the calibration layer
that turns the sum of stumps into a probability you can price with, and the
monotone constraints that make each score defensible. Swap those in and you
have rebuilt this chapter.
