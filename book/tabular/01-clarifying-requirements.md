# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Every question either removes work
or changes the design.

---

**Candidate:** What is the prediction target, and what horizon?

**Interviewer:** We want the probability a credit-card applicant defaults in the
next 12 months. That number sets who we approve, what credit limit they receive,
and what interest rate we charge.

**Candidate:** When you say "default," how exactly is that defined?

**Interviewer:** Ninety or more days past due within 12 months of the decision.
Nail the definition up front; it controls when a label is mature.

**Candidate:** What decision does the score feed? Is it just approve/decline, or
does it also drive a dollar limit or a price?

**Interviewer:** All three. The same probability feeds an approve/decline rule, a
limit-setting formula, and a risk-based pricing model.

**Candidate:** That means calibration is not optional. A score of 0.05 has to mean
a genuine 5 percent default rate, because both the limit formula and the pricing
model multiply the absolute number into money.

**Interviewer:** Exactly. The number is the product, not just a ranking.

**Candidate:** Do we observe repayment outcomes for everyone we score, or only for
the people we approved?

**Interviewer:** Only for approvals. Rejected applicants disappear from the label.

**Candidate:** That is the reject-inference problem (the challenge of learning about
applicants you declined, since you never see how they would have repaid). We will need
to address it or the model is only valid on the approved region of feature space.

**Interviewer:** Correct. How you handle that is part of the design.

**Candidate:** Are there regulatory constraints? Does every decline owe an
adverse-action reason?

**Interviewer:** Yes. We are a regulated credit issuer. Every decline must carry
the top reasons in plain language, and we cannot use protected attributes.

**Candidate:** That constrains the model family toward something explainable, such
as a monotone-constrained gradient-boosted tree (a tree ensemble forced so that, say,
higher income can never raise predicted risk) with SHAP reason codes (a method that
splits each prediction into a signed per-feature contribution). A deep net
would struggle to satisfy the adverse-action requirement.

**Interviewer:** Good. What about latency and volume?

**Candidate:** For credit, I would expect batch scoring at application time, not
sub-millisecond realtime. The hard part is almost never QPS; it is label maturity,
calibration, and explainability. Is that right?

**Interviewer:** Correct. Decisions run in seconds, not microseconds.

---

## Problem statement

**We are asked to design a credit-risk scoring system.** The input is an applicant's
features at decision time; the output is a calibrated default probability that
drives an approve/decline decision, a credit limit, and a risk-based price. Labels
mature over 12 months and are only available for approved applicants. Declines must
carry plain-language adverse-action reasons. The model family must be explainable
and monotone-constrained to satisfy regulatory requirements.

## Consequences

Two consequences fall out of this immediately, and stating them early carries most
of the signal in this question.

**The probability is the product, not the ranking.** Because the score multiplies
directly into a limit formula and a price, calibration is a first-class
requirement, not a post-hoc nicety. An AUC of 0.93 (AUC being a ranking score from 0.5
for random to 1.0 for perfect, measuring only whether risky applicants score above safe
ones) that is badly miscalibrated on
the tails will mis-price risk at scale, even if the ranking looks clean. Say this
before you draw a single box.

**The label is delayed and biased.** A 12-month default label leaves the most
recent year of applications with no resolved outcome. And you only observe
repayment for people you approved, so the model is trained on a self-selected
slice of the population. Both problems are structural, not fixable by engineering
harder, and the design must account for them explicitly. The reject-inference
discussion and the matured-vintages strategy come before any model choice.
