# 8. Interview Q&A

The questions an interviewer actually asks about predictive modeling on tabular
data, grouped by how they trip people up. The commonly-missed ones are where
interviews are won or lost.

## Commonly asked

**Q: Why do gradient-boosted trees beat deep neural nets on tabular data?**

A: Three structural reasons. First, trees are invariant to monotone transforms of
features, so they do not care whether income is raw dollars or log-dollars. Neural
nets are sensitive to feature scale and need normalization. Second, trees handle
missing values natively by learning a default branch direction at each split; neural
nets require explicit imputation. Third, trees capture non-smooth thresholds and
interactions automatically: "income below 30k AND utilization above 80%" is a
natural split sequence, but a neural net needs both depth and data to learn it.
The one exception: when the feature space includes very high-cardinality IDs
(millions of user or item IDs), a neural net with learned embeddings beats the
tree, because one-hot encoding those IDs is prohibitive. For already-meaningful
tabular columns, reach for a tree.

**Q: Your credit model has AUC 0.93 offline. Should you ship it?**

A: Not before two checks. First, suspect leakage: an AUC of 0.93 on credit data
with modest features is suspicious. Audit every feature for point-in-time
correctness. A feature computed post-outcome (an account-status column updated
after default, a window aggregate spanning the label period) will inflate AUC and
completely disappear in production. Second, check calibration. AUC measures
ranking; your model must set a credit limit and a price, so a 0.05 score must
genuinely mean 5 percent. Plot the reliability curve and compute ECE, sliced by
segment and vintage. A model that ranks correctly but is miscalibrated will
mis-price the entire portfolio.

**Q: How do you handle the fact that labels arrive 12 months late?**

A: Three options. Restrict training to matured vintages so every training example
has a resolved label; the model is valid but may be 12 months stale on population
shifts. Use a faster-maturing proxy (early delinquency at 30 or 60 days) that
correlates with the 12-month outcome and resolves in weeks; noisier but timelier.
Use survival analysis, which treats unresolved accounts as censored rather than
discarding them; the model can train on recent applicants without waiting for their
labels to resolve, as long as you use the survival framing correctly. The worst
option: count immature accounts as "good." That biases risk downward on exactly
the most recent, least-well-characterized applicants.

**Q: You want to reduce churn with a discount campaign. Which model do you build?**

A: Uplift, not a churn predictor. A churn predictor identifies who will churn; that
includes sure things who will churn regardless of any intervention, and lost causes
who will churn no matter what you offer them. Sending a discount to both wastes
budget. An uplift model estimates the CATE: whose churn probability is actually
reduced by the discount. Target the persuadables. Under a fixed budget, rank
customers by uplift-per-dollar and fill a knapsack. The churn predictor and the
uplift model answer completely different questions.

**Q: Why do you need calibration if the score only drives an approve/decline
decision?**

A: Even for a binary approve/decline, the optimal threshold depends on the cost
matrix: expected profit times p(good) versus expected loss times p(bad). If the
score is not calibrated to the true rate, you are optimizing the threshold against
a distorted probability, and the resulting approval rate will not match what the
cost matrix implies. The moment you multiply the score into a dollar limit or a
price, miscalibration directly translates into mis-pricing. Calibration is always
load-bearing when the decision has a cost model behind it.

## Tricky (the follow-ups that separate people)

**Q: Your offline AUC dropped from 0.93 to 0.88 after you fixed a feature leakage
issue. Your manager wants to know why the model got worse. What do you say?**

A: The model did not get worse. The 0.93 was measuring leakage, not skill. The
real model capability was 0.88 all along; the leaky feature was just telling the
model the answer. The correct question is: at 0.88, is the model useful for the
decision it feeds, and is its calibration correct? An AUC of 0.88 on a clean
credit dataset is often strong and deployable. The misleading number was the 0.93.

**Q: You train on approvals, but tomorrow you must score new applicants who might
look like historical rejects. What breaks, and how do you fix it?**

A: Selection bias breaks the generalization. The model learned a conditional
distribution P(default | approved), which is biased toward the approved region of
feature space. For applicants near the historical cutoff, the model extrapolates
poorly. Three fixes: reject inference (impute outcomes for historical rejects using
bureau data or behavioral priors, then train on the augmented dataset); a
small randomized approval slice below the cutoff that provides unbiased ground
truth for the borderline region; or a model that explicitly represents the
selection mechanism (Heckman correction or propensity-weighted training). The
randomized slice is the only clean solution; it has a real cost in expected losses
on deliberately borderline approvals, and that cost is a business decision.
**Why only randomization is clean:** reject inference and selection-model
corrections both fill the missing labels with assumptions (that bureau outcomes
transfer, that the selection equation is specified correctly), and those
assumptions cannot be tested using the same biased data that created the
problem. A randomized slice is the one mechanism that produces labels whose
distribution does not depend on the old approval policy, which is why it is
ground truth and the others are educated guesses.

**Q: Why does Nubank use survival curves rather than a single 12-month default
probability?**

A: A survival curve $S(t)$ gives the probability the customer is still solvent at
any time $t$, not just at month 12. This is strictly more informative for a credit
issuer who makes incremental limit decisions over the customer's lifetime: you can
read off the 3-month, 6-month, and 12-month risk in one score, drive collections
timing from the hazard peak, and feed LTV calculations directly. A fixed-window
binary label also discards the most recent applicants whose default window has
not resolved, which survival analysis handles by treating them as censored rather
than discarding them. At Nubank's scale of 122M customers, losing a year of recent
data for training would be a significant loss.

**Q: A model calibrated on average looks fine. Why do you always say to slice?**

A: Average calibration hides per-segment failures. A model calibrated at ECE 0.02
overall can have ECE 0.10 on a new product that the model has not seen in training,
or ECE 0.12 on a demographic segment with a different base rate that the global
model washed out. For a credit model, the decision cares about the tails and the
segments: a miscalibrated new product means mis-priced risk on that product's
entire book. Always report reliability curves sliced by segment, vintage, and
protected group, especially for regulated decisions.
**Why the average hides it:** calibration error is signed before it is averaged
in effect, so a model that over-predicts risk on one segment and under-predicts
on another by the same amount can net out to a beautiful aggregate reliability
curve. The aggregate only certifies the mixture you measured it on; every
decision is made on an individual from some segment, where the local error is
what gets priced.

**Q: Your fraud rate is 0.5 percent. Should you SMOTE the training set to balance
the classes first?**

A: Usually no, and knowing why is the point. SMOTE (Chawla et al., 2002)
synthesizes minority rows by interpolating between a minority example and its
minority nearest neighbors in feature space. Two problems bite in tabular risk
settings. First, interpolating across categorical or non-smooth columns invents
rows that live nowhere in the real distribution, and gradient-boosted trees
already absorb imbalance through the loss (a `scale_pos_weight`-style class weight)
without any resampling. Second, resampling changes the base rate, so every
probability the model then emits is calibrated to the resampled 50/50 world, not
the real 0.5 percent, and must be corrected back before it can feed a threshold or
a price. The cleaner levers for a GBDT are class weighting in the loss plus
threshold tuning on the true distribution, which leave calibration intact. Reach
for SMOTE mainly when the model genuinely cannot express class weights and the
features are smooth and continuous.

**Q: Platt scaling and isotonic regression look similar; both are monotone
post-hoc calibration maps fit on a holdout. When does the difference actually
matter?**
A: The difference is capacity. Platt scaling fits a two-parameter logistic
curve, so it assumes the miscalibration has a specific sigmoid shape (scores too
spread out or too compressed, symmetrically); with only two parameters it is
hard to overfit and works on small holdouts. Isotonic regression fits a
free-form monotone step function, so it can correct any monotone distortion,
including lopsided ones (well calibrated in the middle, badly off in one tail,
which is common after negative downsampling); the price is that each step is
estimated from the examples in its score range, so a small holdout gives noisy
steps and the "correction" can encode noise. The difference matters at the
tails of a money-setting model: a credit or fraud threshold lives in a sparse
score region where Platt's rigid shape may not bend enough and isotonic may not
have enough examples, so the practical rule is Platt (or beta calibration) for
small holdouts, isotonic once the holdout has enough events per bin, and always
inspect the reliability curve in the region where the threshold sits.

## Commonly answered wrong

**Q: Should the model and the decision policy be in the same artifact?**

A: No, and mixing them is a common anti-pattern. The model produces a calibrated
probability; the decision policy (a threshold, an expected-value rule, or an
optimizer) consumes that probability and applies the business cost matrix. Keeping
them separate means you can update the threshold when the cost matrix changes
(interest rates shift, loss-given-default improves) without retraining the model,
and you can audit the policy without understanding the model internals. Uber,
Gojek, and Asos all draw these as two explicit boxes.

**Q: Can you use a propensity score for intervention targeting?**

A: No, and this is the most expensive modeling mistake in practice. A propensity
score answers "who will act?" which includes both sure things (who would act
without treatment) and lost causes (who will not act regardless). Sending an
incentive to sure things is pure waste; sending one to lost causes is also waste,
and in some domains (like a discount that trains price sensitivity) it actively
backfires. Wayfair's WayLift platform makes this explicit: propensity models scale
easily but "over-message"; uplift models require RCT data but target only the
persuadables whose behavior the treatment actually changes.
**Why uplift needs randomized data:** uplift is a difference between two
outcomes for the same person (treated versus not), and you only ever observe
one of them. Without randomization, who got treated was decided by past
targeting policy, so treated and untreated customers differ in ways that
correlate with the outcome, and the model cannot separate "the discount changed
them" from "the kind of person we discounted was different." Randomization is
what makes the comparison group a valid stand-in for the unobserved outcome.

**Q: Is a higher AUC always better?**

A: Not in tabular settings where the score sets money. AUC measures ranking; it
says nothing about whether the absolute probability is right. A model with AUC
0.94 but ECE 0.08 will mis-price a portfolio. A model with AUC 0.89 and ECE 0.015
may price more accurately even though it ranks less well. The right metric for
money-setting decisions is calibration plus business value under the cost matrix,
not AUC alone. Report both. The label is also usually the guide here: for ranking-
only tasks (advertiser churn risk tiers, sales-opportunity prioritization), AUC
is the right primary metric; for pricing and limit-setting, calibration is.

**Q: Should you always retrain the model when the calibration drifts?**

A: No. Calibration drift can often be corrected by re-fitting just the calibration
layer (the Platt scaling or isotonic regression) on a fresh holdout, without
retraining the underlying GBDT. This is cheaper, faster, and less risky than a
full retrain. Nubank's architecture explicitly decouples the ranking model (slow
updates) from the survival calibration layer (frequent recalibration), for exactly
this reason. Retrain the base model when the ranking AUC or feature distributions
shift materially; recalibrate when only the score distribution moves.
**Why the two drift separately:** a base-rate shift (a macro downturn raises
default rates across the board) moves every probability while leaving the
relative ordering of applicants nearly intact, so ranking survives while
calibration breaks; that failure lives entirely in the monotone map from score
to probability, which is exactly the piece the calibration layer owns.
Retraining the whole model to fix it risks disturbing a ranking that was never
broken, in exchange for nothing the cheap recalibration does not already give.
