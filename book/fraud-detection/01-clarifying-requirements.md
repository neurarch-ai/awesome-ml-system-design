# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question
either removes work or changes the design, and that the first few exchanges
already reveal the two dominant constraints: extreme imbalance (fraud is a tiny
fraction of all transactions, so the two classes are wildly unequal) and
asymmetric costs (the two kinds of mistake do not cost the same amount).

**Candidate:** What kind of fraud are we detecting, and what actions can the
system take?

**Interviewer:** Card-not-present payment fraud. The system can allow the
transaction, block it outright, or route it to a human review queue for a
manual decision.

**Candidate:** What is the base rate? Roughly what fraction of transactions are
fraudulent?

**Interviewer:** Under 1 percent. In practice it is closer to 0.2 percent.

**Candidate:** What does a false positive cost versus a false negative? A false
positive is blocking a legitimate customer; a false negative is letting fraud
through.

**Interviewer:** A false positive means a lost sale, a support call, and
potential churn. A false negative is a realized loss: the chargeback (the card
network reverses the payment and pulls the funds back from the merchant) amount
plus network fees. The chargeback is typically an order of magnitude more
expensive than the friction of a declined good transaction.

**Candidate:** What is the latency budget for the scoring decision?

**Interviewer:** The decision sits inline with payment authorization, so we
need a p99 in the low tens of milliseconds. We cannot block the auth flow for
longer than that.

**Candidate:** When do ground-truth labels arrive? How long between a
transaction and knowing whether it was fraud?

**Interviewer:** Card-network chargebacks can arrive 30 to 120 days after the
transaction. Some customers dispute instantly, but most labels are slow.

**Candidate:** Do we have a corpus of labeled fraud to train a supervised
classifier, or are we starting from scratch with no labels?

**Interviewer:** We have historical labeled fraud. Assume a supervised path is
viable, but we also want to catch novel attack patterns we have not seen before.

Let us summarize. **We are designing a real-time fraud scoring system for
card-not-present payments.** Every transaction must receive a score inline with
authorization (p99 under tens of milliseconds), and that score maps to one of
three actions: allow, block, or send to human review. Fraud is roughly 0.2
percent of transactions. Labels arrive weeks late. We have historical labeled
fraud, and we want to catch novel attacks the classifier has not seen.

## Consequences that fall out immediately

Stating these early, before any architecture, is most of the signal in this
question.

1. **Extreme class imbalance kills accuracy.** At a 0.2 percent base rate, a
   model that predicts "never fraud" scores 99.8 percent accuracy and catches
   zero fraud. Accuracy rewards ignoring the positive class. The right metric is
   precision-recall and PR-AUC (the area under the precision-recall curve, one
number for how well the model ranks fraud above legitimate). This shapes every
design choice downstream:
   sampling, loss function, and evaluation.

2. **Asymmetric costs set the threshold, not the default 0.5.** A missed fraud
   is roughly ten times more expensive than blocking a good customer. That ratio
   directly determines the optimal decision threshold: block when the estimated
   fraud probability exceeds $c_{\text{FP}} / (c_{\text{FP}} + c_{\text{FN}})$.
   The threshold is a business decision derived from the cost matrix (a small
   table of the dollar cost of each correct and incorrect decision), not a
   modeling default.

3. **Label delay means training is always stale.** The most recent transactions
   have no mature label yet. Treating unlabeled-recent as "not fraud" is a
   labeling bug. The system must respect a maturation window, accept that models
   train on weeks-old data, and lean on fast signals (analyst verdicts) as
   leading indicators until chargebacks settle.

4. **An adversary makes drift the default.** Most ML systems drift by accident
   as the world changes slowly. Fraud drifts on purpose: there is a human
   adversary actively probing the defenses and changing behavior the moment a
   tactic stops working. Drift detection is a safety system here, not an
   afterthought, and the retrain cadence must match the adversary's cadence, not
   the calendar.
