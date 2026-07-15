# 8. Interview Q&A

The questions an interviewer actually asks about fraud and anomaly detection,
grouped by how they are used. The commonly-missed ones are where interviews are
won or lost.

## Commonly asked

**Q: Why not just use accuracy?**

A: At a 0.2 percent base rate, a model that predicts "never fraud" scores 99.8
percent accuracy and catches zero fraud. Accuracy rewards predicting the majority
class, which is the opposite of the job. Use precision, recall, and PR-AUC.
State this at the beginning of any fraud design discussion; it signals you
understand the imbalance problem.

**Q: How do you set the classification threshold?**

A: From the cost matrix, not a default. Build the cost matrix: a false positive
blocks a real customer (lost sale, support, churn risk); a false negative is a
realized loss (chargeback plus fees). For a calibrated probability, the cost-
optimal threshold is $\tau^{\star} = c_{\text{FP}} / (c_{\text{FP}} +
c_{\text{FN}})$. When fraud loss is ten times the friction cost, the threshold
is $1/11 \approx 0.09$, not 0.5. State that this is a business decision, not a
modeling default, and that it is revisited whenever costs or the base rate shift.

**Q: Labels arrive weeks late. How do you handle that?**

A: Three things. First, respect a maturation window: only train on examples
whose labels have had enough time to settle (typically 60 to 90 days); never
treat unmatured recent data as "not fraud." Second, use fast signals as leading
indicators: analyst verdicts from the review queue close in minutes and give a
partial precision-recall estimate before chargebacks arrive. Third, reconcile
the leading indicators against settled labels once available, and accept that
the live eval metric always lags reality by at least the maturation window.

**Q: Supervised or anomaly detection?**

A: Supervised for known fraud at high precision (it learns exactly what labeled
fraud looks like). Unsupervised anomaly detection (Isolation Forest, autoencoder,
GraphBEAN) for novel attacks with no labels. The mature answer runs both: the
supervised model handles the known; the anomaly path handles the unknown; the
human review queue converts anomaly hits into labels for the next supervised
retrain.

**Q: How do you catch fraud rings?**

A: Graph signals. Fraud rings share devices, cards, IPs, and addresses across
many accounts. Individual transactions look clean; the network screams. Feed
graph features into the tabular model (component size, hops-to-fraud, shared-
device count), or run a GNN (RGCN for relation-specific message passing) over
the entity graph. PayPal runs a live graph DB for sub-second ring detection;
Uber and Grab train RGCN offline and inject scores as features. State that ring
fraud is coordinated, not per-event.

## Tricky (the follow-ups that separate people)

**Q: The fraudsters keep adapting. How do you stay ahead?**

A: Treat drift as the default, not an edge case: an adversary changes behavior
the moment a tactic stops working. Short retrain cadence (daily to weekly),
drift alarms on both input feature distributions and score distributions (a
sudden shift in the mix of declined transactions is an attack signature), and
an anomaly path that catches new attacks before labels exist. Cross-link to
monitoring; fraud is the canonical case where drift detection is a safety system.

**Q: Your PR-AUC improved offline but live precision fell. What happened?**

A: Several possibilities. Most likely: (1) training-serving skew in velocity
features (batch and streaming compute different numbers; the model is served
garbage); (2) label delay in the training set (recent unmatured fraud was
treated as negative, teaching the model that fresh fraud is fine); or (3) the
adversary shifted between your training window and production. Diagnose by
logging served feature values and comparing against training distributions.

**Q: How do you handle the block-side blind spot?**

A: Blocked transactions never generate a chargeback, so the model never learns
it was wrong to block a good customer. Mitigate with a small randomized allow-
through hold-out: occasionally allow a fraction of would-be-blocked transactions
(say 1 to 2 percent), accept the small fraud cost, and observe what happens.
Their settled labels give you block-precision estimates. Also lean on analyst
review verdicts for borderline cases, since those cases span both sides.

**Q: Is calibration important for fraud?**

A: Critical. The cost-optimal threshold formula ($c_{\text{FP}} /
(c_{\text{FP}} + c_{\text{FN}})$) only holds if the model output is a true
probability. A wide-and-deep joint logit, a tree ensemble trained on a
resampled set, or focal-loss outputs are all miscalibrated out of the box.
Recalibrate after every retrain with isotonic regression or Platt scaling.
Deploying an uncalibrated model with a cost-derived threshold places the
operating point at the wrong precision-recall level. Mechanism for why focal
loss miscalibrates: it multiplies each example's loss by $(1 - p_t)^\gamma$, so
confident correct predictions contribute almost nothing to the gradient. That is
what forces the model to focus on hard fraud, but it also means the model is no
longer optimizing proper-scoring-rule log loss, so its raw outputs stop being
frequency-accurate probabilities and must be mapped back with a post-hoc calibrator.

**Q: You undersampled the majority class to train faster. The ranking looks fine but
the cost-optimal threshold is now far too aggressive. Why, and what is the fix?**

A: Resampling changes the base rate the model implicitly learned. If you train on a
1:1 set drawn from a true 1:500 population, the model's output odds are inflated by
roughly the sampling ratio, so a probability of 0.5 on the resampled model does not
correspond to 0.5 in production. The threshold formula $c_{\text{FP}} / (c_{\text{FP}}
+ c_{\text{FN}})$ assumes true probabilities, so it lands at the wrong precision.
The fix is a prior correction: shift the output log-odds by $\ln(\pi_{\text{train}} /
\pi_{\text{prod}})$ (the log ratio of the sampled to the real positive rate), or
equivalently fit isotonic or Platt calibration on a held-out set drawn from the true
base rate before deriving the threshold.

## Commonly answered wrong (the traps)

**Q: Can you improve recall by adding more data with SMOTE?**

A: SMOTE is useful when labeled fraud is very scarce, but it is not a free
recall boost. Interpolating between minority points invents unrealistic examples
and can blur the decision boundary. More importantly, SMOTE distorts the
calibration of the trained model because the synthetic positives shift the
effective class ratio. Always evaluate on the true base rate, never on a
rebalanced eval set. Try class weights or focal loss first; reach for SMOTE
only when recall is critically low despite weighting.

**Q: The threshold is 0.5 by default, I should tune it if needed.**

A: The threshold is always derived from the cost matrix; it is never 0.5 by
default for fraud. At a cost ratio of $c_{\text{FN}} / c_{\text{FP}} = 10$, the
cost-optimal threshold is approximately 0.09. Saying "I would tune the threshold
if needed" misses the point: the threshold is not a hyperparameter to grid-
search, it is a business decision computed from the cost matrix. State the
formula and derive the number.

**Q: ROC-AUC is 0.97, so the model is great.**

A: Not necessarily, and sometimes the opposite is true. At a 0.2 percent base
rate, the huge true-negative mass flatters ROC-AUC. A model can achieve 0.97
ROC-AUC while having a PR-AUC of 0.05 and catching almost nothing at reasonable
precision. Lead with PR-AUC. ROC-AUC is a secondary sanity check, not the
primary gate. Mechanism: ROC's x-axis is the false-positive rate, FP divided by
the total number of negatives. When negatives outnumber positives 500 to 1, even
tens of thousands of false positives barely move that denominator, so the curve
stays near the top-left and the area stays high. Precision, FP divided by (FP plus
TP), has no such large denominator to hide behind, which is exactly why PR-AUC
collapses when the same model floods the block queue with false positives.

**Q: I will just do a random train/test split.**

A: Random splits leak the future for fraud. If a chargeback from November lands
in the training set while its transaction is in the test set, the model sees
the label before the event. Always split by time. Additionally, exclude
transactions inside the maturation window from both train and validation to
avoid mislabeling recent fraud as legitimate.
