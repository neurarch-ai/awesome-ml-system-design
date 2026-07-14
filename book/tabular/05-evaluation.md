# 5. Evaluation

Evaluating a tabular model is not the same as evaluating a recommender. The
deliverable is a calibrated probability, not a ranked list. That changes which
metrics carry signal.

## The three layers of evaluation

Report all three, in order of business proximity. An impressive AUC with poor
calibration is a model that ranks correctly but prices incorrectly. An impressive
calibration with poor business value means you optimized the wrong objective.

### Layer 1: ranking metrics

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve).** Measures
whether the model separates positives from negatives across all thresholds.

- **Input / output.** Takes model scores paired with binary labels; returns a
  scalar in $[0, 1]$: 0.5 is random, 1.0 is perfect separation.
- **How it is computed.** AUC equals the probability that a randomly drawn
  positive example is scored higher than a randomly drawn negative:

$$\text{AUC} = \Pr(\hat{p}(x^+) \gt \hat{p}(x^-))$$

For a credit model, AUC tells you whether higher scores predict higher default
rates, not whether the absolute probability is right.

**AUC-PR (Precision-Recall AUC).** Preferable when positives are rare (default
rates of 1 to 5%) and false negatives carry high cost.

- **Input / output.** Takes model scores and binary labels; returns a scalar in
  $[0, 1]$. More sensitive to improvements at the operating point than AUC-ROC on
  heavily imbalanced datasets.
- **How it is computed.** Plot precision $p_k$ against recall $r_k$ as the
  decision threshold sweeps from 1 to 0 through each unique predicted score, then
  integrate (commonly approximated as Average Precision):

$$\text{PR-AUC} \approx \sum_{k} p_k \cdot (r_k - r_{k-1})$$

where $p_k$ and $r_k$ are precision and recall at the $k$-th threshold step in
decreasing score order.

**C-index (concordance index, for survival models).** Measures whether predicted
hazards rank correctly relative to observed event times; the survival analogue of
AUC-ROC.

- **Input / output.** Takes predicted survival probabilities and event times with
  censoring indicators; returns a scalar in $[0.5, 1.0]$: 0.5 is random, 1.0 is
  perfect concordance.
- **How it is computed.** For every comparable pair $(i, j)$ where $T_i \lt T_j$
  and observation $i$ is not censored, check whether the predicted survival for
  $i$ is lower than for $j$ (lower survival = higher risk). Using survival
  function $\hat{S}$ (smaller value means higher predicted risk):

$$\text{C-index} = \Pr\!\left(\hat{S}(t|x_i) \lt \hat{S}(t|x_j) \;\middle|\; T_i \lt T_j,\, T_i \text{ not censored}\right)$$

Block Square's survival forest achieved C-index 0.83.

### Layer 2: calibration metrics

Calibration is the most important layer when the absolute probability sets money.
A model calibrated on average can be badly miscalibrated on the segments the
decision cares about.

**Reliability curves (calibration plots).** Bin predicted probabilities into
deciles; for each bin, plot mean prediction against observed positive rate. A
perfectly calibrated model falls on the diagonal. Deviations show overconfidence
(predictions too extreme) or underconfidence (predictions compressed toward 0.5).

**Expected Calibration Error (ECE).** A scalar summary of the reliability curve.

- **Input / output.** Takes predicted probabilities bucketed into $M$ bins,
  paired with binary labels; returns a scalar in $[0, 1]$ where 0 is perfect
  calibration.
- **How it is computed.**

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where $\text{acc}(B_m)$ is the observed positive rate in bin $B_m$ and
$\text{conf}(B_m)$ is the mean predicted probability. In code it is a single pass
over the bins, weighting each bin's confidence-accuracy gap by its share of the
data:

```python
import numpy as np
def ece(probs, labels, n_bins=10):
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)      # predictions falling in this bin
        if m.sum() == 0: continue
        conf = probs[m].mean()                # mean predicted prob in the bin
        acc = labels[m].mean()                # observed positive rate in the bin
        e += m.mean() * abs(acc - conf)       # weight the gap by bin fraction
    return e
# perfectly calibrated probabilities give ece ~ 0; systematic overconfidence inflates it
```

![Reliability (calibration) curve: predicted probability vs observed positive rate](assets/fig-calibration-curve.png)

*A well-calibrated model (blue, left panel) tracks the diagonal: a predicted probability of 0.7 means roughly 70% of those examples are true positives. A miscalibrated model (red, right panel) bows away from the diagonal; the shaded gap between the curve and the diagonal is exactly what ECE integrates. Illustrative.*

**Brier Score.** Mean squared error between predicted probability and binary
label; rewards both discrimination and calibration simultaneously.

- **Input / output.** Takes predicted probabilities $\hat{p}_i$ and binary labels
  $y_i \in \{0, 1\}$; returns a scalar in $[0, 1]$ where 0 is perfect.
- **How it is computed.**

$$\text{Brier} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$$

Block Square reports both C-index 0.83 (ranking) and Integrated Brier Score
(calibration) for exactly this reason: C-index alone hides calibration failures.

**Sliced calibration.** Always report calibration by segment (product type, vintage,
region, protected group). A global ECE of 0.02 can mask a 0.10 ECE on a new
product or a demographic slice that drives most of the risk exposure.

### Layer 3: business-value metrics

Calibration is necessary but not sufficient. A well-calibrated model with the wrong
objective still fails the business.

**Expected value under the cost matrix.** For approve/decline decisions, compute
the expected profit and expected loss at the operating threshold. This is the number
the business decision maker cares about.

**Uplift metrics.** For intervention models, use the **area under the uplift curve** (AUUC) or the **Qini coefficient**.

- **Input / output.** Both metrics take a model that scores each individual's
  predicted incremental treatment effect (uplift) and a held-out randomized
  experiment; they return a scalar summarizing how well the model concentrates
  incremental conversions in the top-targeted fraction.
- **How they are computed.** Sort the population by descending predicted uplift.
  Plot cumulative incremental conversions (treatment minus control outcomes) against
  the fraction of the population targeted. AUUC is the raw area under that curve;
  the Qini coefficient subtracts the area of the random-targeting diagonal:

$$\text{AUUC} = \int_0^1 U(\phi)\,d\phi, \qquad \text{Qini} = \text{AUUC} - \text{area under diagonal}$$

where $U(\phi)$ is cumulative incremental conversions when targeting the top $\phi$
fraction. These are related but distinct: AUUC is the raw area; Qini is the area
above the random baseline. Both measure how much better than random the model
concentrates incremental conversions. A churn model evaluated on either will look
worse than an uplift model because it wastes budget on sure things.

![Uplift and Qini curve](assets/fig-uplift-qini.png)

*Left: Qini curve comparing a propensity model against an uplift model and a random
baseline. The uplift model concentrates incremental conversions in the top fraction
of the population (steeper early slope), making it more budget-efficient. Right:
the four persuadability segments: persuadables (high uplift), sure things (convert
regardless), lost causes (never convert), and do-not-disturbs (backfire under
treatment). Budget should flow only to persuadables.*

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| AUC-ROC | comparing models when the threshold is not yet set; overall discrimination quality | raw accuracy, which is misleading on imbalanced datasets |
| AUC-PR / Average Precision | rare positive class (default rate below 5%) and false negatives are costly | AUC-ROC, which is insensitive to minority class recall on heavily imbalanced data |
| C-index | survival models; you want to know if predicted hazards rank correctly | AUC-ROC applied to the event indicator, which ignores censoring |
| Reliability curve + ECE | any model whose probability feeds a threshold, limit, or price | AUC alone, which says nothing about calibration |
| Brier Score | combined discrimination and calibration in one number | two separate metrics when a scalar summary is enough |
| AUUC / Qini coefficient | uplift and intervention models where budget efficiency matters (AUUC = raw area; Qini = area above random baseline) | accuracy or AUC on the treatment-response label, which does not measure incremental value |
| Sliced metrics (by segment, vintage, protected group) | always, especially for regulated decisions | global aggregates that mask slice-level failures |

## The evaluation discipline

**Use a time-based split, not a random split.** Hold out future events and evaluate
whether today's model predicts them. A random split allows the model to "memorize"
future information through shared behavioral windows and inflates all metrics.

**Report business value under the actual cost matrix.** AUC is not what a credit
risk officer cares about. They care about approval rates, expected losses, and
expected margin at the operating threshold. Report those.

**The online gate is the real launch gate.** Offline metrics are necessary for
screening. The final launch decision on a money-setting model is always a
champion-challenger (A/B) test against the business outcome: actual defaults,
actual retention rates, actual incremental revenue. Offline evaluation optimizes
a proxy; the online test evaluates the real thing.
