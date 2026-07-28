# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable fraud scorer, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing gradient-boosting libraries before scoring a
single transaction. Skip that. The stack below is a sane default for a first
production build; each row names when to deviate and which section explains why.
Model families change yearly, but the interface of each stage (screen, assemble
features, score, calibrate, threshold, mature labels, close the loop, evaluate)
does not, so pick per stage by interface and treat any library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Rules layer | A thin deterministic screen ahead of the model: known-compromised BIN ranges, hard velocity stops, sanctioned geographies. Everything else goes to the model | The rule list grows past a few dozen and starts encoding soft judgement: that is a model, so move it into the model | [6](06-serving-and-scaling.md), [7](07-how-teams-do-it-in-production.md) |
| Velocity and aggregate features | One streaming aggregation (Flink or equivalent) writing counters to a low-latency store, and the same computation path used for training and serving | Never split the path. If streaming infrastructure does not exist yet, ship identity and raw transaction fields only and accept the recall ceiling | [3](03-data-preparation.md), [6](06-serving-and-scaling.md) |
| Model family | Gradient-boosted trees (XGBoost or LightGBM) over the tabular feature set | High-cardinality sparse ids (device, merchant) carry real signal: wide-and-deep. Fraud is coordinated across shared entities: RGCN scores computed offline and injected as features | [4](04-model-development.md) |
| Imbalance handling | Class weights in the loss; focal loss if the weight ratio needs constant hand-tuning | Labeled fraud is critically scarce and recall stays flat under weighting: SMOTE, accepting the boundary risk. Never rebalance the eval set | [3](03-data-preparation.md), [4](04-model-development.md) |
| Calibration | Isotonic or Platt fit on a held-out slice drawn at the true base rate, refit after every retrain | Never skip it. Any resampling, focal loss, or joint wide-and-deep logit leaves the raw score unusable by the threshold formula | [4](04-model-development.md) |
| Threshold policy | Two thresholds from the cost matrix: block above $\tau^{\star} = c_{\text{FP}} / (c_{\text{FP}} + c_{\text{FN}})$, review down to a floor set by analyst capacity, allow below | A friction step (step-up auth, micro-auth) exists as a third action: switch to the Airbnb three-action loss and re-derive both bands | [2](02-frame-as-ml-task.md), [4](04-model-development.md) |
| Label maturation | A 60-to-90-day window; rows inside it are excluded, never labeled negative. Analyst verdicts serve as leading indicators until chargebacks settle | Never on the "never negative" half. Shorten the window only if your own chargeback arrival curve says it is safe | [3](03-data-preparation.md) |
| Novelty path | Isolation Forest or an autoencoder scoring in parallel, routed to human review, never to an auto-block | No analyst capacity at all: ship supervised only and write down that you are blind to novel attacks | [2](02-frame-as-ml-task.md), [4](04-model-development.md) |
| Feedback loop | A randomized allow-through hold-out on a few basis points of would-be blocks, tracked to settled labels | Loss tolerance or regulation forbids letting known-risky traffic through: lean on review verdicts and accept a biased block-precision estimate | [6](06-serving-and-scaling.md), [8](08-interview-qa.md) |
| Evaluation | PR-AUC as the model gate, expected cost at the shipping threshold as the ship gate, time-based and entity-disjoint splits | Never accuracy. Never ROC-AUC alone | [5](05-evaluation.md) |

Two rows carry more weight than the rest. The calibration row is the hinge: the
threshold formula in the row below it is arithmetic on a true probability, so an
uncalibrated score does not produce a slightly-wrong operating point, it
produces an unrelated one. And the label-maturation row is the one beginners
skip and regret, because mislabeling unmatured fraud as legitimate does not
throw an error, it quietly teaches the model that recent fraud is fine.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md):
card-not-present payment fraud, roughly 0.2 percent of transactions, scored
inline with authorization at a p99 in the low tens of milliseconds, three
actions (allow, block, review), chargeback labels arriving 30 to 120 days late,
historical labeled fraud available, and a requirement to catch novel attacks.
Here is the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Screening | Deterministic rules for compromised BINs and hard velocity stops, ahead of the model | Some blocks are policy, not prediction, and a rule needs no maturation window to take effect |
| Features | Velocity counters and graph scalars from one streaming path, plus raw authorization fields and identity signals; served feature vectors logged | Velocity is the most predictive group and the biggest skew risk; logging the served vector is what makes the skew diagnosable |
| Model | LightGBM over the tabular set, with offline RGCN entity scores and a bounded-BFS `hops_to_fraud` scalar as input features | Trees match or beat a DNN on engineered tabular fraud features, retrain in minutes, and give the attribution an audit will ask for |
| Imbalance | Class weights in the loss; no resampling of the training set, no resampling of eval | Weighting keeps every legitimate example and avoids the prior shift that resampling would force us to undo |
| Calibration | Isotonic regression fit on a held-out slice at the true 0.2 percent base rate, refit on every retrain | The block threshold is a probability, so a monotone score is not enough; the number has to mean what it says |
| Thresholds | Block above 0.056, review down to the score that fills analyst capacity, allow below | 0.056 is $c_{\text{FP}} / (c_{\text{FP}} + c_{\text{FN}})$ on the measured cost matrix; the review floor is a staffing constraint, not a modeling one |
| Novelty | Isolation Forest in parallel, routed to review with its own quantile threshold | An anomaly score has no probability semantics, so it cannot enter the cost formula and must not auto-block |
| Label pipeline | 90-day maturation window; analyst verdicts as leading indicators; randomized allow-through on 1 percent of would-be blocks | Chargebacks are the ground truth and they are slow; without the hold-out, block precision drifts free of reality |
| Retraining | Daily retrain on matured labels, recalibrate every time, drift alarms on input and score distributions | The adversary sets the cadence, and score-distribution drift fires weeks before any label-based metric moves |
| Evaluation | PR-AUC offline on a time-based entity-disjoint split; expected cost at the shipping threshold; online blocked-fraud dollars against settled-label false-positive rate | The model gate and the ship gate answer different questions and neither substitutes for the other |

**Transaction volume and fraud exposure.** Illustrative: 25 million
transactions per month, roughly 10 per second on average and 100 per second at
seasonal peak. At the scenario's 0.2 percent base rate that is 50,000 fraud
attempts per month. Average legitimate ticket is 70 USD; fraud attempts skew
larger at about 153 USD, because a fraudster optimizing a stolen card buys up.
Monthly fraud exposure is therefore roughly 50,000 x 153, about 7.7 million USD,
plus card-network fees. Doing nothing scores 99.8 percent accuracy on this
stream and loses all of it.

**Cost matrix and the threshold.** A false positive costs the merchant margin
on a lost sale plus a support contact: about 10 percent of a 70 USD ticket plus
3 USD, near 10 USD. A false negative costs the chargeback amount plus a 15 USD
network fee, near 168 USD on the fraud-ticket average. The closed form from
[section 4](04-model-development.md) gives
$\tau^{\star} = 10 / (10 + 168) \approx 0.056$. That is the block threshold: the
system blocks any transaction whose calibrated fraud probability exceeds 5.6
percent, roughly one ninth of the default 0.5. The runnable at the end of this
section confirms the number empirically by sweeping the curve, and shows what it
costs to ignore it.

**Review queue capacity.** The lower threshold is not derived from the cost
matrix at all; it is derived from staffing. Illustrative: 20 analysts working 8
hours at 60 cases per hour clear about 9,600 cases per day. At 833,000
transactions per day, that is 1.15 percent of traffic, so the review floor is
set to whatever score puts a little under 1 percent of transactions in the band
and re-derived whenever headcount or case-handling time changes. This is why the
review band drifts while the block threshold does not: one tracks money, the
other tracks people.

**Decision latency.** Illustrative decomposition of a 25 ms p99 budget: network
and deserialization on the authorization hop, about 3 ms; a batched read of
velocity counters from the low-latency store, 4 ms; the graph feature fetch,
which is a precomputed RGCN entity score plus a depth-capped BFS for
`hops_to_fraud`, 8 ms and the most variable term; feature assembly of the raw
fields, 1 ms; a 500-tree LightGBM forward pass, under 1 ms; the isotonic
calibration map and the two threshold comparisons, microseconds. The total lands
near 17 ms. The model is not the cost. The graph traversal sets the p99, which is
why [section 6](06-serving-and-scaling.md) caps BFS depth and prunes high-degree
nodes rather than tuning the tree count.

**Label maturation and retrain cadence.** With a 90-day window, the newest row
the daily retrain may use is 90 days old, so the supervised model is always
fighting the adversary of a quarter ago. Two things close the gap and neither is
optional: analyst verdicts from the review queue land in minutes and give a
running precision estimate on the reviewed band, and score-distribution drift
alarms fire when attempts start bunching just below the block threshold, which
is what probing looks like before any chargeback exists to label it.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: training-serving skew on velocity counters
(offline PR-AUC holds while live precision sags; the tell is the logged serving
feature vector diverging from the training-time recomputation for the same card
and timestamp, and it is a one-day diff if you logged the vector and a
month-long argument if you did not), review-queue overflow (the band was set at
launch traffic and seasonal peak triples it; the queue ages, analysts start
rubber-stamping, and their verdicts silently degrade as training labels, so
monitor queue age and per-analyst agreement, not just volume), and the block-side
blind spot closing (if the randomized allow-through hold-out is switched off for
a "temporary" loss-reduction push, the block region stops producing ground truth
within one retrain cycle and the estimated block precision starts describing a
population the model no longer sees).

## The same techniques under different constraints

The review question that matters in practice is not "which model family is best"
but "which is best under my constraints." Here is the same system built three
times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Seed-stage marketplace, no fraud history | Card-not-present PSP (this chapter) | Bank AML alert triage |
|---|---|---|---|
| Scale and traffic | ~3,000 transactions per day; one engineer owns the whole loop | 25M transactions per month, inline with authorization | Millions of accounts scored on a nightly batch, no inline decision at all |
| Latency budget | Inline but generous; nothing is tuned for it | p99 in the low tens of ms, imposed by the auth flow | None. The constraint is the investigator backlog, not the clock |
| Labels | Almost none. A handful of confirmed chargebacks | Historical labeled fraud plus a 90-day maturation window | Investigator dispositions and filed reports; slow, expensive, and biased toward what past rules surfaced |
| Model | Rules plus Isolation Forest. A supervised model has nothing to learn from yet | LightGBM with graph features, plus a parallel anomaly path | Random forest over several hundred features; explainability is a regulatory requirement, not a preference |
| Imbalance | Irrelevant at this volume; every flagged case is looked at by a human | Class weights, no resampling, isotonic recalibration | Class weights; the far bigger problem is that "no alert" is not evidence of no crime |
| Threshold policy | Whatever the founder can review over coffee. Capacity is one person | Block at 0.056 from the cost matrix, review floor from analyst capacity | No block action exists. Score into severity buckets sized to investigator throughput |
| Novelty path | This is the whole system | Isolation Forest into review, never auto-block | Central: novel typologies are the point, and unexplained alerts still have to be defensible |
| Feedback loop | Manual. Every decision is labeled because a human made it | Randomized allow-through hold-out plus review verdicts | Dispositions only; there is no counterfactual arm to run |
| Eval | Count of caught cases, honestly. PR-AUC on 40 positives is noise | PR-AUC, cost at threshold, online blocked-fraud dollars vs FP rate | Precision at the investigator capacity point, plus regulator-facing recall on known typologies |
| What would be over-engineering | A feature store, a GNN, a calibration job, a retrain pipeline | A single global PR-AUC with no cost curve behind it | Sub-second serving, streaming velocity counters, an inline graph traversal |

Two lessons fall out. First, the seed-stage column is mostly deletions: with 40
confirmed fraud cases there is no supervised model to fit, no PR-AUC worth
reporting, and no threshold worth deriving, so the honest build is rules plus an
anomaly score plus a human, and its job is to manufacture the labels that make
the middle column possible a year later. Second, the AML column shows the
threshold thread surviving the loss of its own formula: with no chargeback and
no dollar-denominated false-negative cost, $\tau^{\star}$ has nothing to compute
from, so the operating point falls back to the capacity constraint that was
already setting the review band in the middle column. The cost matrix was never
the only thing choosing thresholds; it was the part that could be priced.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any model families.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Base rate | Metric choice and eval-set construction | Under about 5 percent positives: PR-AUC leads, ROC-AUC is a sanity check, accuracy is never reported. Never rebalance the eval set |
| Cost ratio $c_{\text{FN}} / c_{\text{FP}}$ | The block threshold, directly | $\tau^{\star} = c_{\text{FP}} / (c_{\text{FP}} + c_{\text{FN}})$. At 10:1 it is 0.09, at 20:1 it is 0.048. It is never 0.5, and it moves when average ticket size moves |
| Label delay | Maturation window and retrain cadence | Exclude the window, never label it negative. Retrain on matured labels, and use analyst verdicts as the fast signal in between |
| Human review capacity | The lower threshold | The review floor is a staffing number. Re-derive it whenever headcount, handling time, or traffic changes, and monitor queue age before volume |
| Latency budget | Feature fetch strategy, not model size | The tree ensemble is under 1 ms. Cap graph traversal depth, batch the counter reads, and precompute anything that does not depend on this transaction |
| Adversary cadence | Retrain frequency and drift alarms | Alarm on score-distribution shift, not only feature drift: probing bunches attempts just under the threshold weeks before any chargeback settles |
| Fraud coordination structure | Whether graph enters at all | Isolated per-account fraud: skip the graph. Shared devices, cards, addresses: RGCN offline as features, or bounded BFS inline when the budget allows |
| Regulatory explainability | Model family | Audit-facing decisions pin you to trees or linear models with per-decision attributions; a deep model buys accuracy you may not be allowed to use |
| Blocked-side observability | Whether a hold-out exists | Without a randomized allow-through arm, block precision is an estimate of a population you stopped observing. A few basis points buys back ground truth |
| Volume of labeled fraud | Imbalance strategy | Plenty: class weights alone. Thin: focal loss. Critically thin and recall is flat: SMOTE, knowing it invents points near the boundary |

## The smallest runnable fraud scorer

The chapter's central claim is that a fraud system is a threshold decision
wearing a classifier costume, and the fastest way to believe it is to watch the
best-ranked model lose money at the wrong operating point. The file below is
the whole argument with zero installs: a seeded transaction stream at a 0.2
percent base rate, the do-nothing classifier scoring its 99.8 percent, and then
a threshold sweep priced with asymmetric unit costs. Every production component
is swapped for the smallest thing with the same interface: the latent risk
signal stands in for the feature vector, the posterior average stands in for the
isotonic calibrator, and the sweep stands in for the offline cost curve of
[section 5](05-evaluation.md). The shape is the lesson; every section of this
chapter upgrades one function of this file.

```python
"""Fraud at a 0.2 percent base rate: the accuracy trap, then the cost-optimal threshold."""
import math, random

random.seed(11)
N = 200_000
FEE, MARGIN, SUPPORT = 15.0, 0.10, 3.0   # network fee; margin lost on a decline; support cost

def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))

def unit_costs(amount):
    """The asymmetry the chapter turns on; production: the cost matrix, in USD."""
    return MARGIN * amount + SUPPORT, amount + FEE      # (false decline, missed fraud)

# --- the stream: latent risk z drives the label, the model sees a noisy copy ---
B, K, NOISE = -12.5, 4.0, 0.2

def stream(n):
    out = []
    for _ in range(n):
        z = random.gauss(0.0, 1.0)
        y = 1 if random.random() < sigmoid(B + K * z) else 0
        amount = round(math.exp(random.gauss(3.9, 0.9)) * (2.2 if y else 1.0), 2)
        z_hat = z + random.gauss(0.0, NOISE)            # production: an imperfect classifier
        mean = B + K * z_hat / (1 + NOISE ** 2)         # shrink toward the prior, then average
        var = (K * NOISE) ** 2 / (1 + NOISE ** 2)       # over it: a CALIBRATED probability,
        score = sigmoid(mean / math.sqrt(1 + math.pi * var / 8))   # which the formula requires
        out.append((score, y, amount))
    return out

txns = stream(N)
frauds = sum(y for _, y, _ in txns)
fraud_usd = sum(a for _, y, a in txns if y)
print(f"{N} transactions, {frauds} fraudulent ({frauds / N:.3%}), "
      f"USD {fraud_usd:,.0f} of fraud in the stream")
print(f"do-nothing classifier: accuracy {1 - frauds / N:.2%}, recall 0.0%, "
      f"cost USD {fraud_usd + FEE * frauds:,.0f}\n")

def evaluate(tau):
    tp = fp = fn = 0
    caught = declined = cost = 0.0
    for s, y, a in txns:
        c_fp, c_fn = unit_costs(a)
        if s >= tau:
            if y: tp, caught = tp + 1, caught + a
            else: fp, declined, cost = fp + 1, declined + a, cost + c_fp
        elif y:
            fn, cost = fn + 1, cost + c_fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, caught, declined, cost

print(f"{'tau':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'fraud caught':>13}"
      f" {'good declined':>14} {'total cost':>11}")
rows = []
for tau in (0.60, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01):
    prec, rec, f1, caught, declined, cost = evaluate(tau)
    rows.append((tau, f1, cost))
    print(f"{tau:>6.2f} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} {caught:>13,.0f}"
          f" {declined:>14,.0f} {cost:>11,.0f}")

f1_tau, f1_val, f1_cost = max(rows, key=lambda r: r[1])
best_tau, best_f1, best_cost = min(rows, key=lambda r: r[2])
c_fp = sum(unit_costs(a)[0] for _, y, a in txns if not y) / (N - frauds)  # over legit traffic
c_fn = sum(unit_costs(a)[1] for _, y, a in txns if y) / frauds            # over real fraud
print(f"\nmax-F1 threshold       tau={f1_tau:.2f}  F1={f1_val:.3f}  cost USD {f1_cost:,.0f}")
print(f"cost-optimal threshold tau={best_tau:.2f}  F1={best_f1:.3f}  cost USD {best_cost:,.0f}")
print(f"shipping the F1 optimum costs USD {f1_cost - best_cost:,.0f} more over {N} transactions")
print(f"closed form, mean costs: c_FP/(c_FP + c_FN) = {c_fp:.0f}/({c_fp:.0f} + {c_fn:.0f}) "
      f"= {c_fp / (c_fp + c_fn):.3f}")
```

Run it and the output tells the chapter's story in three acts. The stream lands
453 frauds in 200,000 transactions, 0.227 percent, and the do-nothing classifier
posts 99.77 percent accuracy while losing every one of the 69,234 USD on the
table: the accuracy trap of [section 3](03-data-preparation.md), stated in the
only units that matter. The sweep then separates the two questions everyone
conflates. F1 peaks at tau = 0.20, where precision 0.402 and recall 0.468 are
nicely balanced and the system still burns 45,273 USD. Total cost bottoms out
four times lower, at tau = 0.05, where precision has collapsed to 0.183 and
three of every four flagged transactions are good customers, and that ugly
operating point is the correct one: it costs 38,072 USD, or 7,201 USD less per
200,000 transactions, because it recovers 48,685 USD of fraud instead of 30,870.
A metric that treats a false decline and a chargeback as the same unit will
always land in the wrong place. The last line closes the loop back to
[section 4](04-model-development.md): the empirical minimum sits where the closed
form says it should, at 0.058 on mean costs that print as 10 and 168 USD, the
same 0.056 the build above derived. That agreement holds only because the score
was calibrated before the sweep ran. Swap the latent signal for a real vector, the
posterior average for an isotonic fit, the sweep for the offline cost curve, and
the printed threshold for the two bands feeding allow, block, and review, and
you have rebuilt this chapter.
