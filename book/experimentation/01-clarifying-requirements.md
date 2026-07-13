# 1. Clarifying the Requirements

Before designing anything, pin down what the experiment is actually deciding.
Here is a typical exchange. Every question either removes work, changes the
design, or surfaces a trap you would otherwise step into later.

---

**Candidate:** What are we deciding? Are we comparing two model versions, a
feature flag, or a UI change?
**Interviewer:** A new ranking model. It beats the current one on every offline
metric.

**Candidate:** What is the one business outcome this should move? Sessions,
revenue, retention?
**Interviewer:** Engagement rate for the current session. The model is a ranker,
so we expect better-ranked content to lift clicks and dwell time.

**Candidate:** What must it not break?
**Interviewer:** Latency, error rate, revenue. And we care about complaint rate,
because a spammy ranker can win engagement and lose trust.

**Candidate:** What is the unit of diversion? User, session, or request?
**Interviewer:** You tell me. The model is a ranker, so think about whether the
effect carries across a user's requests.

**Candidate:** It does. A user who sees better results in one request will behave
differently for the rest of the session. Diverting per request would split the
same user across arms and dilute the measurement. We should divert per user and
hold the assignment for the whole experiment.
**Interviewer:** Agreed.

**Candidate:** What is the smallest lift worth shipping? That sets the sample
size and the duration.
**Interviewer:** A 1% relative increase in engagement is the bar.

**Candidate:** Is there a network effect? If one user's arm changes what another
user sees, a naive user split is biased.
**Interviewer:** It is a standard feed product. Users' feeds do not interact
directly.

**Candidate:** How much traffic can we dedicate to the treatment arm, and is
there a deadline?
**Interviewer:** Up to 50% of users. No hard deadline, but we want a decision in
two to three weeks.

---

Let us summarize the problem. **We are asked to decide whether a new ranking
model ships, by running a randomized controlled experiment that measures its
causal effect on session engagement rate and verifies that latency, error rate,
revenue, and complaint rate are not harmed.**

Four consequences fall out of this immediately:

- **The offline win is not the decision.** AUC and NDCG were measured on the old
  model's logged data, which reflects what the current ranker chose to show. The
  A/B test is the decision because it measures the real metric on the real
  distribution under the real feedback loop. State this out loud; it is the frame
  the whole answer hangs on.
- **Diversion unit is per user.** The effect carries across a user's session, so
  splitting per request contaminates the same user with both arms. Divert by user,
  analyze by user, and account for within-user correlation in the variance
  estimate; otherwise confidence intervals are too narrow and you ship noise.
- **Pre-register the primary metric and the MDE before launching.** One metric
  decides (engagement rate); everything else is a guardrail. Choosing the metric
  after seeing data is how teams rationalize false winners.
- **Run at least one to two full weeks, not just until significance arrives.**
  Weekly seasonality and novelty effects mean an early reading is not trustworthy.
  The planned duration is set by the sample-size calculation, not by when the
  dashboard first crosses the threshold.
