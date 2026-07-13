# 5. Evaluation

## Why PR-AUC beats ROC-AUC under extreme imbalance

ROC-AUC is the standard metric for binary classifiers, but it is silently
misleading when positives are rare. The reason is structural: ROC-AUC is driven
by the true-positive rate and the false-positive rate. At a 0.2 percent base
rate, there are roughly 500 negative examples for every positive. Even a model
with very poor precision can achieve a high true-positive rate without moving
the false-positive rate much, because the vast true-negative mass absorbs false
positives invisibly. The "always-legit" baseline has an ROC-AUC of exactly 0.5
(random), which correctly identifies it as useless. But a model that is only
slightly better than random on the rare positive class can still show an ROC-AUC
of 0.85+ by being good at the easy negatives.

PR-AUC (average precision) does not have this property. It is computed purely
from the precision-recall curve, which focuses entirely on the positive class.
The baseline curve for a classifier that outputs random scores sits at
approximately the base rate (0.002 at a 0.2 percent prevalence). Any model
that does better than this shows its improvement clearly.

The figure below makes this concrete.

![PR vs ROC curves](assets/fig-pr-vs-roc.png)

*Left: ROC curves for a decent model (A) and an always-legit baseline (B). The
baseline sits at the diagonal (ROC-AUC=0.50), which correctly reads as random.
Right: PR curves for the same two classifiers. The baseline PR-AUC is
approximately the base rate (near zero), and the model's improvement is visible
and honest. Illustrative with synthetic distributions.*

## PR-AUC formula (average precision)

$$\text{AP} = \sum_{k}(R_k - R_{k-1}) \cdot P_k$$

where $P_k = \text{TP}_k / (\text{TP}_k + \text{FP}_k)$ is the precision and
$R_k = \text{TP}_k / (\text{TP}_k + \text{FN}_k)$ is the recall at threshold $k$.
Average precision is the recall-weighted mean of precision values, summarizing
the entire curve into one number. It is the primary offline metric for fraud
scoring.

## Precision at fixed recall

For a given operating constraint, report precision at a fixed recall target.
For example: "what is the precision when recall equals 0.80?" This is the
right question for a cost-sensitive system where the business has decided it
must catch at least 80 percent of fraud and wants to know how many legitimate
transactions that requires blocking. PR-AUC summarizes the full curve, but
the precision-at-recall-target is what feeds the cost model and justifies
the threshold choice to stakeholders.

## Cost-weighted operating point

The correct offline evaluation is not just the PR curve shape; it is the
expected cost at each threshold. For each candidate threshold $\tau$ on the
validation set, compute:

$$L(\tau) = c_{\text{FP}} \cdot \text{FP}(\tau) + c_{\text{FN}} \cdot \text{FN}(\tau)$$

Plot this curve (normalized) and mark the minimum. The threshold that minimizes
expected cost on the validation set is the threshold that ships. This is a
decision-theory evaluation, not a leaderboard metric.

Report both:
- **PR-AUC** as the model-quality gate (model A is better than model B across
  the full operating range).
- **Cost at the chosen operating point** as the shipping gate (model A saves X
  dollars per thousand transactions versus the current system at the cost ratio
  in production).

## Time-based split is mandatory

The validation set must be future data relative to the training set. A random
split allows the model to train on chargebacks that postdate the transactions
in the validation set, which is a form of label leakage. Always split by time
and respect the maturation window: do not include transactions in the validation
set that do not yet have settled labels.

## Online metrics

Offline PR-AUC and cost-at-threshold are the pre-ship gates. The real ship
decision is made on live metrics measured against settled labels:

- **Blocked fraud dollars.** The dollar value of fraud stopped. The goal.
- **False-positive rate on settled labels.** Of the transactions blocked or
  routed to review, what fraction turned out to be legitimate (measured once
  chargebacks settle). The constraint.
- **Review queue precision and throughput.** Of cases sent to analysts, what
  fraction are confirmed fraud? If it falls, the model is sending too many
  easy legitimates to the queue.
- **Human review audit.** Periodically sample analyst decisions and audit for
  consistency, since analyst verdicts become training labels.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| PR-AUC (average precision) | primary offline model quality gate at any base rate under 5% | ROC-AUC, which a near-random model can game via true-negative mass |
| Precision at fixed recall target | communicating the operating point to stakeholders ("we catch 80% of fraud, how often do we bother a good user?") | PR-AUC alone, which does not answer a specific operating question |
| Cost at the chosen threshold | choosing the threshold that ships, given the cost matrix | a default 0.5 threshold, which ignores the asymmetric cost structure |
| Accuracy | never, for fraud | PR-AUC; accuracy rewards predicting the majority class |
| ROC-AUC | only as a secondary sanity check alongside PR-AUC | as the primary metric; it flatters models on rare positives |
| Online blocked-fraud-dollars vs FP-rate | the final ship decision against settled labels | offline metrics alone, which do not reflect the live cost structure |
| Time-based val split | always | random split, which leaks future label information |
