# 4. Analysis

## The basic comparison: two-sample t-test

At the end of the planned window, compute the per-user mean of the primary metric
in each arm and test whether the difference is statistically distinguishable from
zero.

For large samples (thousands of users), the two-sample t-test is equivalent to a
z-test. The test statistic is:

$$t = \frac{\bar{Y}_{\text{treat}} - \bar{Y}_{\text{ctrl}}}{\sqrt{\hat{\sigma}^{2}_{\text{treat}}/n_{\text{treat}} + \hat{\sigma}^{2}_{\text{ctrl}}/n_{\text{ctrl}}}}$$

Report the 95% confidence interval alongside the p-value. A p-value tells you
whether an effect is distinguishable from zero; the confidence interval tells you
whether it is *large enough to matter*. A statistically significant effect that
is below the MDE is not a ship.

### Variance must be clustered at the diversion unit

If you divert by user but the outcome data is at the request or pageview level,
the rows in your table are not independent: many rows share the same user. Naive
standard errors treat them as independent, which makes the intervals too narrow
and manufactures false winners. Aggregate to one row per user before the
t-test, or compute a clustered variance estimator. The rule: **analyze at the
level you divert.**

## CUPED: variance reduction before the t-test

As derived in the sizing chapter, the CUPED adjustment replaces the raw outcome
$Y$ with a residualized version:

$$Y_{\text{cv}} = Y - \theta \cdot (X - \mathbb{E}[X])$$

You then run the t-test on $Y_{\text{cv}}$ instead of $Y$. The estimate of the
treatment effect is unchanged (the adjustment is mean-zero in both arms), but the
variance of $Y_{\text{cv}}$ is smaller, so the confidence interval is tighter.

![CUPED variance reduction](assets/fig-cuped-variance-reduction.png)

*Left: standard deviation of the per-user metric before and after CUPED
adjustment. Right: the distribution of per-user treatment-minus-control, raw
versus CUPED. The narrower CUPED distribution means the test needs fewer users
to detect the same effect.*

## Sequential testing: looking without peeking

The standard fixed-horizon test assumes you look exactly once, at the planned
sample size. If you monitor a live dashboard and stop the moment the metric
crosses alpha = 0.05, the actual false-positive rate is far higher than 5%,
because you are implicitly running many tests.

Sequential testing methods (mSPRT, always-valid p-values, group-sequential
boundaries) are built for continuous monitoring. They adjust the threshold
dynamically so that the false-positive rate stays controlled at alpha regardless
of when you stop. Uber uses mSPRT with jackknife variance estimation. The cost
is a slightly wider threshold at any given moment, but you pay it once and you
can look whenever you want.

The practical middle ground: if you do not need continuous monitoring, commit to
the fixed horizon and look once. If you need early looks (for example, to catch
a guardrail breach quickly), switch to a sequential method. The mistake is
combining a fixed-horizon threshold with daily peeking.

## Novelty and primacy effects

**Novelty effect:** users click anything new. The lift in the first few days
after a change can be inflated because users are reacting to the novelty itself.
As the novelty wears off, the effect decays. A test that stops at a novelty
spike ships a feature that will underperform.

**Primacy effect:** the mirror image. Users are anchored on the old experience
and initially resist the new one. An early flat result can hide a real win that
only becomes visible after users adapt.

Both effects are why duration is not just "until significant." The discipline:
plot the **daily treatment effect** (not just the cumulative number) and require
it to stabilize before deciding. A real, shippable effect converges; a novelty
spike decays and a primacy dip recovers. Run at minimum one to two full weeks.

## When to use which analysis method

| Reach for | When | Instead of |
|---|---|---|
| t-test on per-user aggregates | default analysis for a fixed-horizon test at scale | analyzing request rows as independent, which manufactures false winners |
| CUPED adjustment | a correlated pre-experiment covariate is available (typically correlation above 0.5 meaningfully reduces variance) | running longer to compensate for high variance |
| Sequential / mSPRT | you need continuous early looks without peeking inflation (Uber) | checking daily and using a fixed-horizon threshold, which inflates the false-positive rate |
| Fixed horizon (commit and look once) | you can pre-register duration and remove the temptation to peek (Booking.com) | ad-hoc stopping at the first significant reading |
| Non-inferiority test for guardrails | you must prove a guardrail is not meaningfully harmed (Airbnb, Spotify) | reading "guardrail not significant" as "guardrail safe," which is only true if the test is powered |
| Daily effect curve visualization | before any final decision, to check for novelty or primacy | reading only the final cumulative number, which averages over any decay or recovery |
| Bonferroni or FDR correction | you are genuinely testing multiple hypotheses (multiple variants, multiple subgroups) | applying no correction and accepting the first significant secondary metric |
