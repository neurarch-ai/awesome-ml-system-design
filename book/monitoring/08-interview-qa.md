# 8. Interview Q&A

The questions an interviewer actually asks about ML monitoring, grouped by how
they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: How is ML monitoring different from normal service monitoring?**
A: Service monitoring watches latency, error rate, and uptime. Those signals
tell you the serving process is healthy; they say nothing about whether the
predictions are still good. ML monitoring watches the data the model consumes
and the outcomes it produces. A model can serve at 10 ms p99 with a 0.01%
error rate while its AUC has quietly decayed from 0.82 to 0.73. The silent
decay in the prompt is exactly this failure.

**Q: What are the three reasons a model degrades in production?**
A: Data drift (covariate shift): the input distribution moves but the
input-to-label mapping is unchanged. Concept drift: the input-to-label mapping
itself changes, so even identical inputs now deserve different predictions.
Pipeline and data-quality bugs: an upstream schema change, a frozen feature, a
null-returning join. The third is the most common in practice and the least
glamorous. Naming all three, and calling the third the most frequent real cause,
signals operational experience.

**Q: What do you monitor when labels have not arrived yet?**
A: Input drift and prediction drift as leading indicators. If features have
shifted materially from the training reference (PSI, KS), the model is scoring
inputs unlike anything it trained on. If the model's own score distribution has
shifted, that often precedes a measured accuracy drop. Both are label-free and
observable immediately. Once labels land, confirm with the performance metric.

**Q: How do you avoid alert fatigue?**
A: Three levers. Alert on sustained breaches, not single noisy points (a single
PSI spike is noise; three consecutive windows above the threshold is signal).
Set thresholds from historical variation per feature, not from default values.
Tier severity so a feed model earns a dashboard note while a fraud model pages
on-call; not every signal deserves a wake-up.

**Q: How do you close the loop from monitoring to retraining?**
A: Scheduled retraining on a fixed cadence handles slow steady drift. Triggered
retraining fires when a monitor breaches: a drift or performance drop kicks off
a retrain on recent data, which goes through the evaluation gate before
promotion. Rollback is the fast path if the cause is a recently promoted model.
The evaluation gate is non-negotiable: skip it and you may retrain on a
corrupted data window and promote the broken model.

## Tricky (the follow-ups that separate people)

**Q: Feature drift was detected but model quality is unchanged. What happened?**
A: Drift without decay. A feature can shift significantly without affecting
quality if the model barely weights it. PSI and KS answer "did it move?" not
"does it matter?" Check feature importance before acting. Similarly, a feature
can drift from one benign range to another benign range; what matters is whether
the new distribution is outside the model's learned support.
**Why:** predictions only change where the model's decision function changes. A
shifted feature with near-zero weight moves inputs through a flat region of that
function, so nothing downstream moves. Drift metrics see only the inputs; they carry
no information about the function's sensitivity to those inputs, which is why
importance-weighted drift ranking exists in the first place.

**Q: The monitoring dashboard shows no drift, but the product team reports
engagement is down. What do you check?**
A: Concept drift, segmentation, and feedback loops. Concept drift can shift the
input-to-label mapping while the marginals stay flat; a green feature-drift
dashboard does not rule it out. Check whether the engagement drop is in a
specific cohort that the aggregate hides. Check whether the model's predictions
are influencing the data it is now being evaluated on (feedback loop narrowing
the coverage). Finally, check whether the business metric itself changed
definition.
**Why the dashboard stays green:** drift detectors compare input marginals, but
concept drift lives in the conditional relationship between inputs and labels. The
same inputs can keep arriving in the same proportions while the correct answers for
them have changed, so no input-side statistic can catch it even in principle; only
label-based or outcome-based signals can.

**Q: You detect drift in a feature and start a retrain. The retrained model
performs worse. Why?**
A: The drift was probably a pipeline bug, not a real distribution shift. A null-
returning feature or a schema change reads as a distribution shift; retraining
on a data window that includes the corrupted records bakes the bug into the new
model. The fix is to run data-health checks before retraining, identify and
backfill the broken window, and only then retrain.

**Q: You widened the monitoring window from one day to one week and now the KS test
fires on nearly every feature, though nothing about the data actually changed. Why?**
A: The KS test (statistics) returns a p-value, and a p-value's power grows with sample
size. The KS statistic (the maximum gap between two empirical CDFs) may be tiny, but
with seven times the samples the same tiny gap crosses the significance threshold, so
you get "statistically significant" drift that is practically meaningless. This is why
threshold-based measures like PSI (Population Stability Index, from credit-risk
practice) are often preferred for monitoring: PSI measures effect size directly and
does not inflate with sample count the way a significance test does. If you keep the
KS test, gate on the statistic magnitude (or a minimum-detectable-effect floor), not
on the p-value alone.

**Q: Data-quality checks and drift detection look similar; when does the difference
actually matter?**
A: Both watch incoming data, compare it against an expectation, and raise alerts,
so teams often treat one as covering the other. The mechanism differs: a quality
check asserts per-record invariants (schema, types, null rate, allowed ranges) that
are either violated or not, deterministically; drift detection compares a window's
distribution to a training reference statistically, and can only say "this looks
different," never why. The difference matters because the correct responses are
opposites. A broken upstream join surfaces in a drift-only stack as "feature
drifted," and the reflexive response, retraining, bakes the corrupted window into
the next model; the right response is an upstream fix plus a backfill. Genuine
drift has no upstream fix at all; retraining is the right response. Run quality
checks earlier in the pipeline so the bug class is caught and named
deterministically, leaving the statistical layer to flag only changes in the world.

## Commonly answered wrong (the traps)

**Q: If I have fast labels (clicks arrive in seconds), I don't need drift
monitoring; I can just watch accuracy. Right?**
A: Wrong, for two reasons. Accuracy tells you what happened; drift tells you
what is about to happen. Drift monitoring is a leading indicator that gives you
time to act before quality has already degraded. Second, even with fast labels,
aggregate accuracy hides per-segment regressions. A global AUC that is holding
steady can mask a new-user cohort that has fallen off a cliff.
**Why drift leads accuracy:** decay starts in the small slice of traffic that has
moved outside the training distribution. While that slice is small, aggregate
accuracy barely moves, but the input shift on the affected features is already
measurable. By the time the aggregate metric visibly drops, the drifted slice has
grown, and the lead time in which a retrain would have been cheap is gone.

**Q: Monitoring is expensive, so I should only log the features that matter.**
A: Partially right, and easy to overdo. Logging only high-weight features saves
cost but removes your ability to detect problems in features that are currently
low-weight but become important after a distribution shift. A feature that was
irrelevant under the training distribution may become a strong signal after the
world moves. Stratified sampling (log all features for a fraction of requests)
is usually better than dropping features entirely.

**Q: I can just set PSI threshold 0.10 for all features.**
A: The field rule is a starting point, not a universal constant. A feature that
naturally swings by PSI 0.15 over weekends will page on-call every Monday with
that threshold. Calibrate per-feature from historical variation. Uber D3 uses
a Prophet (Meta, 2017) forecast to set dynamic bands precisely to avoid this problem.
Mechanism: PSI sums $(p_i - q_i)\ln(p_i / q_i)$ over fixed reference bins, so its
scale depends on how many bins a feature has and how lumpy its distribution is. A
naturally multimodal or bursty feature accumulates a larger baseline PSI just from its
normal week-to-week churn, so the same 0.10 line that is quiet for a smooth feature is
a hair-trigger for a spiky one. The Prophet-forecast band replaces one global constant
with a per-feature expected range that already accounts for that seasonality.

**Q: To fix concept drift, retrain on more data.**
A: More data does not fix concept drift if the historical data reflects the old
mapping. Concept drift requires fresh data that reflects the new input-to-label
relationship. More data from before the shift reinforces the wrong mapping. The
fix is recent data plus confirmation that the new distribution is actually
different, not just a short-term fluctuation.
**Why more old data actively hurts:** training minimizes average loss over the
dataset, so examples vote in proportion to their count. Padding the set with
pre-shift data outvotes the fresh examples that encode the new mapping and drags
the fit back toward the old relationship; recency weighting or a sliding window is
how you change the vote, not raw volume.
