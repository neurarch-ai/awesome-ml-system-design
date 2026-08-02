# A mock interview, end to end

Every chapter's Q&A is a set of isolated answers. A real loop is 45 continuous
minutes where the interviewer picks the next probe based on what you just said.
This is one full transcript, with the interviewer's private scoring notes in quoted
blocks, so you can see what each turn was actually graded on.

The question is a standard senior one because it starts as a design and turns into a
diagnosis: **"Design the ranking system for our home feed. Then: it shipped, offline
metrics were up, and engagement went down."**

Read it once as the candidate, then once as the interviewer.

---

## Minutes 0 to 6: clarify before designing

**Interviewer:** Design the ranking system for our home feed.

**Candidate:** Before I draw anything, four questions. What is the feed optimizing
for, and who decides? "Engagement" usually means several things that trade against
each other.

**Interviewer:** Time spent is the north star. There is pressure to not tank
creator diversity.

**Candidate:** So one primary metric and at least one guardrail, which pushes me
toward a multi-task model rather than a single objective. Second: what is the
inventory size and the latency budget?

**Interviewer:** Tens of millions of eligible items, 100 ms for the whole ranking
call.

**Candidate:** That settles the architecture as a funnel: I cannot score tens of
millions of items in 100 ms, so retrieval narrows to hundreds or low thousands, then
a heavier ranker scores those. Third: what signals do we log today, and how delayed
are the labels?

**Interviewer:** Impressions, clicks, dwell, follows. Dwell is available in seconds;
follows are sparse.

**Candidate:** Fourth: how often can we retrain, and can we run experiments freely?

**Interviewer:** Daily retraining is fine. Experiments go through a shared platform
with limited slots.

**Candidate:** Limited slots matter more than people expect. It means the offline
gate has to be trustworthy enough to filter candidates before they reach an A/B, so
I will spend real design effort on offline-online correlation.

> **Interviewer note.** Four questions, each of which changes the design: objective
> (single vs multi-task), scale plus latency (funnel), label availability and delay
> (what the model can learn from), and experiment throughput (how good the offline
> gate must be). Candidates who ask generic questions without naming the consequence
> get no credit. The last one is rare and is a strong signal.

---

## Minutes 6 to 18: the design

**Candidate:** The funnel has four stages. **Retrieval** brings tens of millions
down to a thousand or so, using several sources: a two-tower embedding model for
personalized recall, plus recency and social-graph sources, unioned. **Filtering**
applies eligibility and safety rules in code, not in the model. **Ranking** scores
the survivors with a heavier model. **Re-ranking** applies diversity, dedup, and
business rules over the ordered list.

**Interviewer:** Why two towers for retrieval?

**Candidate:** Because the item tower can be precomputed and indexed. At serving
time I embed the user once and do an approximate nearest-neighbor lookup, which is
what makes tens of millions of candidates feasible in single-digit milliseconds. The
cost is that the user and item never interact until the ranking stage, so retrieval
cannot model cross features. That is a deliberate division of labor: retrieval buys
recall cheaply, ranking buys precision expensively.

**Interviewer:** How do you train the retrieval model?

**Candidate:** Contrastive, with in-batch negatives so each positive is scored
against the other items in the batch. Two things I would flag. In-batch negatives
are sampled proportional to popularity, so popular items are over-penalized, which
is what the logQ correction fixes. And offline recall is measured against logged
engagements, which are themselves the output of the current ranker, so the metric
flatters anything similar to today's system.

**Interviewer:** Now the ranker.

**Candidate:** Multi-task, one shared bottom and several heads: click, dwell past a
threshold, follow, and a negative head like hide or report. The final score is a
weighted combination, and the weights are a product decision that I would want
written down and versioned rather than tuned quietly. Features are user, item,
context, and crucially user-item cross features, since that is what the retrieval
stage could not represent.

> **Interviewer note.** Looking for the funnel with a reason for each stage, not a
> recital. The two strong signals here are naming what retrieval structurally cannot
> do (cross features) and treating the multi-task weights as a product artifact
> rather than a hyperparameter.

**Interviewer:** What do you watch during training?

**Candidate:** Beyond the task metrics, calibration, because the scores get combined
across heads and compared across surfaces. AUC can look fine while the probabilities
are systematically off, and a mis-calibrated score corrupts any weighted blend. If
we downsample negatives for class balance, the correction is arithmetic, not
optional.

---

## Minutes 18 to 30: it shipped and engagement fell

**Interviewer:** You ship it. Offline AUC is up 1.5 points, NDCG up. The A/B shows
time spent down 2 percent, significant. What is happening?

**Candidate:** Offline and online disagreeing is a class of failure with a short
list of usual causes, and I would work it in order of prior probability.

First, **training-serving skew**. A feature computed one way in the training
pipeline and another way at serving is the most common silent killer. The check is
mechanical: log the feature vector actually used at serving for a sample of
requests, re-score offline with the training pipeline, and diff. If the two disagree
on any feature, that is the bug.

Second, **position bias and the logging policy**. The training labels came from
impressions the old ranker chose and placed. If I did not correct for position, the
model partly learned the old policy's exposure pattern rather than user preference,
which can look better offline (it predicts logged clicks well) and do worse online.

Third, **an objective mismatch**. AUC and NDCG reward ordering; time spent is
driven by what actually gets rendered at the top of a short viewport. A model can
improve mid-list ordering, which moves AUC, without changing the visible top few.

Fourth, **the metric moved for a non-model reason**: a bad rollout, a latency
regression from the heavier model, or an SRM in the experiment.

> **Interviewer note.** This is the core of the question. Weak candidates offer one
> cause. Strong candidates offer an ordered list with a discriminating test for each
> and start with the most probable one. Naming the latency regression as a candidate
> is a good sign that they have shipped something.

**Interviewer:** Start with the fourth. How would you rule it out?

**Candidate:** Check the sample ratio first: if the assignment split deviates from
the intended one beyond chance, the readout is invalid and nothing else is worth
analyzing. Then check p99 latency in the treatment arm. A heavier ranker that adds
30 milliseconds can cost engagement on its own, and that is a capacity problem
masquerading as a modeling result.

**Interviewer:** SRM is clean, latency is flat. Now what?

**Candidate:** Then I would test the skew hypothesis, because it is cheap and it is
the most common. If features match, I would look at position bias by checking
whether the model's gain concentrates in positions the old policy favored. A clean
way to separate preference from exposure is to compare against a small randomized
slice, if we have one, or to add an inverse-propensity weighting to the training
labels and see whether the offline gain survives.

**Interviewer:** Suppose the offline gain does not survive the correction.

**Candidate:** Then the offline metric was measuring agreement with the old policy,
not quality, and the honest conclusion is that our offline gate is miscalibrated
rather than that the model is bad. The fix is to the gate: correct for position in
the labels, and validate the corrected offline metric against the handful of A/B
outcomes we already have. If the corrected metric predicts the online results we
have seen, it earns the right to filter candidates for the scarce experiment slots.

> **Interviewer note.** The move being graded is fixing the measurement rather than
> defending the result. Candidates who propose shipping anyway, or who blame the A/B,
> are ruled out here regardless of how well they designed the funnel.

---

## Minutes 30 to 40: the experiment itself

**Interviewer:** Let us talk about the A/B. It ran a week on 5 percent of users.
How would you have sized it?

**Candidate:** Backwards from the smallest lift worth shipping. For a difference of
means, $n \approx 2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2/\text{MDE}^2$ per arm, and
at 5 percent significance and 80 percent power that constant is about 7.85. The two
inputs are the metric's variance, which I get from historical data, and the MDE,
which is a product decision. The thing I would push back on is choosing the MDE
after seeing the result.

**Interviewer:** The team also looked at the dashboard daily and stopped when it
crossed significance.

**Candidate:** That inflates the false positive rate well above the nominal 5
percent, because each look is another chance to cross the line. The fix is either a
fixed horizon that nobody peeks at, or a sequential test designed for continuous
monitoring, which spends the error budget deliberately. It is worth saying that this
cuts both ways: a peeked-at loss is as untrustworthy as a peeked-at win.

**Interviewer:** How would you get more power without more users?

**Candidate:** CUPED. Regress out a pre-experiment measurement of the same metric,
which reduces variance by a factor of $(1-\rho^2)$; at a typical correlation of 0.6
to 0.8 that removes 35 to 65 percent of the variance and buys the same power with
substantially fewer users. For ranking specifically, interleaving is even more
sensitive, because it compares two rankers within the same user session and removes
between-user variance entirely. The tradeoff is that interleaving measures relative
preference between rankings, not the downstream business metric, so I would use it
as a fast filter and confirm the winner with a conventional A/B.

**Interviewer:** Anything you would watch beyond time spent?

**Candidate:** Guardrails: creator diversity, since that was the stated constraint,
plus latency, error rate, and a hide or report rate as a quality floor. And I would
be skeptical of a first-week win on a UI-visible change because of novelty effects,
so I would look at the trend across the week rather than the pooled average.

---

## Minutes 40 to 45: fast follow-ups

**Interviewer:** Quickly. How often do you retrain this?

**Candidate:** Derived from a staleness curve rather than chosen: train to a date,
evaluate the frozen model at plus one day, plus seven, plus thirty, and retrain where
the decay crosses the smallest lift worth shipping. For a feed with catalog churn
that usually lands at daily. I would also check whether the decay is really in the
model or in the features, because stale counts and recency features degrade faster
than weights do, and refreshing them is cheaper than retraining.

**Interviewer:** New items get no engagement, so the model never learns them.

**Candidate:** Item cold start, and it is a feedback loop rather than a data
problem: the item is not shown, so it gets no engagement, so it is not shown. Two
levers: a content tower so a brand-new item has an embedding from its features
alone, and an explicit exploration budget, contextual bandit or a fixed slot, so
new items get impressions the ranker would not have granted. The cost of exploration
is measured, not free, and I would size it against the measured value of the items it
surfaces.

**Interviewer:** Last one. If you could only keep one metric on the dashboard?

**Candidate:** The one the experiment was powered to detect, with its confidence
interval, next to the guardrails. A single point estimate without an interval is how
teams ship noise.

> **Interviewer note.** Rapid-fire at the end confirms the earlier answers were
> understood rather than recited. Consistency matters more than completeness here.

---

## The scoring rubric

| Dimension | No hire | Hire | Strong hire |
|---|---|---|---|
| Scoping | Starts drawing immediately | Asks about scale, latency, labels | Names the design consequence of each answer, including experiment throughput |
| Architecture | One model over everything | Retrieve-then-rank funnel | Explains what each stage structurally cannot do, and why the split exists |
| Training data | "We train on clicks" | Notes label delay and imbalance | Names position bias and the logging policy as a confound before being asked |
| Metrics | Reports AUC | Uses ranking metrics and calibration | Ties each metric to a decision, and knows calibration is separate from ordering |
| Diagnosis | One hypothesis | Several hypotheses | Ordered by prior probability with a discriminating test for each |
| Experimentation | "Run an A/B" | Sizes it from MDE and variance | Peeking, SRM, CUPED or interleaving, guardrails, novelty effects |
| Handling disagreement | Defends the offline number | Believes the online result | Fixes the offline gate and validates it against past A/B outcomes |
| Communication | Answers what was asked | Structured, clear | Says the boundary conditions out loud and converts an inconclusive result into a decision |

## How to use this

Do it as a drill, not a read. Cover the candidate turns, answer out loud, then
compare. The gap is usually not knowledge, it is **structure**: the transcript
answers in the order the interviewer scores, and states the assumption instead of
silently making it.

Then repeat with a different question from the [question bank](../questions.md)
using the [answer framework](../framework/answer-framework.md), and read that
chapter's Q&A afterwards to find what you missed.
