# 8. Interview Q&A

The questions an interviewer actually asks about online experimentation, grouped
by how they are used. The commonly-missed ones are where interviews are won or
lost.

## Commonly asked

**Q: Your model wins on every offline metric. Why might it lose online?**

A: Three reasons. First, the offline metric is a proxy: AUC and NDCG measure
ranking quality on logged data, not the business outcome you care about.
Second, the offline data is the old model's data: the logs reflect what the
current ranker chose to show, so the new model is evaluated on a distribution
it would never have produced (counterfactual bias, offline numbers are
optimistic). Third, training-serving skew: if any feature is computed
differently online than in training, the model that looked great offline meets a
different distribution at serve time. The A/B test is the decision because it
measures the real metric on the real distribution under the real feedback loop.

**Q: What is your unit of diversion and why?**

A: Per user for a ranking change. The effect carries across a user's requests:
a user who sees better results behaves differently for the rest of the session,
so splitting per request contaminates the same user with both arms and dilutes
the effect. Divert and analyze at the user level, and use variance clustered at
the user to avoid confidence intervals that are too narrow.

**Q: How long do you run the experiment and how big does it need to be?**

A: Compute the sample size before launching from the baseline rate, per-user
variance, MDE, significance level, and target power. Divide by daily users
times traffic share to get duration in days, then round up to a multiple of 7
for weekly seasonality. Run at least one to two full weeks regardless of early
significance, to absorb novelty effects. The planned window is a commitment, not
a suggestion.

**Q: What is the minimum detectable effect and how do you set it?**

A: The MDE is the smallest lift worth shipping. Set it by asking the product
team: "what improvement would justify the maintenance and risk of this change?"
For a ranker, 1% relative lift in engagement might be the bar. Then use the MDE
to compute the required sample size before launching. Sample size scales as
$1 / \text{MDE}^{2}$, so halving the MDE roughly quadruples the traffic needed.

**Q: What is sample ratio mismatch and what do you do if you see it?**

A: SRM is when the observed control/treatment split differs significantly from
the intended split. A 50/50 ask that comes back as 50.8/49.2 at millions of
users means randomization or logging is broken. Run a chi-squared test on every
readout; if it fails, the experiment is invalid and you do not read the results,
no matter how good the primary metric looks. Common causes: a CDN serving cached
responses that bypass the assignment layer, a logging bug that drops events
asymmetrically, or a bot-filtering step applied only to one arm.

## Tricky (the follow-ups that separate people)

**Q: The primary metric is significant and above the MDE, but p99 latency is
10 ms higher. Do you ship?**

A: No. A guardrail regression is a kill or iterate, not a ship. The right frame
for a guardrail is non-inferiority: the confidence interval of the latency
increase must exclude a meaningful regression (say, any increase above 5 ms) for
the experiment to pass. "Not statistically significant" is not the same as "not
worse"; an underpowered guardrail can produce a non-significant result even
when the regression is real.
**Why** non-inferiority is the fix: it flips the burden of proof. A standard
significance test only controls the rate of falsely finding harm, so collecting
less data makes it easier to "pass," which rewards under-powering exactly where
safety matters. Non-inferiority demands positive evidence that any harm is
smaller than the margin, so an underpowered guardrail now fails the test
instead of sneaking through it.

**Q: Your ranking test wins in the first week. Why not ship early?**

A: Novelty effect. Users click anything new. The early lift may be inflated by
the novelty of seeing different content, and the effect can decay substantially
as novelty wears off. Plot the daily treatment effect over time: a shippable win
stabilizes; a novelty spike decays. Also, an early read is likely a peek before
the planned sample size, which inflates the false-positive rate.
**Why** peeking inflates it: under the null hypothesis the running p-value
wanders randomly as data accumulates, and each look is another chance to catch
it on a random excursion below 0.05. The nominal 5% rate is only valid for one
pre-planned look; checking daily and stopping at the first significant readout
can push the realized false-positive rate several times higher, which is what
sequential-testing corrections exist to repair.

**Q: You are running a ranker experiment at a marketplace. The treatment shows
item X more, and item X sells out. What breaks?**

A: SUTVA violation (interference). The treatment arm depletes shared inventory,
which changes what the control arm can show. The measured difference between
arms is now biased: the control arm is harmed not by its own ranker but by the
treatment arm's behavior. A naive user split is invalid here. Use a switchback
(time-based alternation of the whole system) or a geo split where the two arms
do not share inventory.
**Why** those designs fix it: the causal estimate assumes each unit's outcome
depends only on its own assignment, and shared inventory couples the arms
through a common resource, so control's outcome partly reflects treatment's
behavior. Switchbacks and geo splits redraw the arm boundary around the
interfering resource: within any one time window or region the entire inventory
pool lives under a single arm, so neither arm can deplete the other's supply
and the comparison becomes a clean counterfactual again.

**Q: What is interleaving and when would you use it?**

A: Interleaving blends both rankers' results into one list per user via
team-draft assignment and attributes each click to whichever ranker contributed
the clicked item. Because every user sees both rankers, the comparison is
within-user: you cancel out per-user variance entirely. Netflix finds it roughly
100 times more sensitive than a standard A/B for ranking comparisons. Use it as
a fast screen to prune a pool of candidate rankers to the few worth a full A/B
test, then run the A/B to measure the actual business metric and guardrails
before shipping. Interleaving alone does not measure the full business metric.
**Why** the sensitivity gain is so large: in a parallel A/B, the noise term
includes the huge between-user spread in baseline click propensity (heavy
versus light users), and the ranker effect has to be detected on top of it.
Interleaving turns the design within-subjects: the same user in the same
session supplies the measurement for both rankers, so each user's baseline
propensity cancels in the comparison and the residual noise is only the
within-session choice variability, a far smaller quantity.

**Q: CUPED reduces variance by subtracting a pre-experiment covariate. Why does
that not bias the treatment effect?**

A: The mechanism is the key. CUPED (Microsoft, 2013) forms an adjusted outcome
$Y_{adj} = Y - \theta (X - \bar{X})$, where $X$ is a pre-experiment measurement
(usually the same metric from the prior period) and $\theta = \text{Cov}(Y, X) /
\text{Var}(X)$ is the regression coefficient of $Y$ on $X$. Two facts make it
unbiased. First, $X$ is measured before treatment is assigned, so its expectation
is identical across arms; subtracting a term built from a pre-treatment quantity
cannot shift the between-arm difference in expectation. Second, subtracting the
same linear function from both arms preserves the difference of means while
removing the shared, predictable component of the variance. The variance of the
adjusted estimator drops by a factor of $(1 - \rho^2)$, where $\rho$ is the
correlation between $X$ and $Y$, which is why a covariate correlated at 0.7 removes
roughly half the variance. The senior point: CUPED buys power for free precisely
because the covariate is pre-treatment; using an in-experiment covariate instead
would bias the estimate.

**Q: A switchback and a geo split look similar; when does the difference
actually matter?**

A: Both contain interference the same way, by redrawing the arm boundary around
the shared resource so that inventory or supply never straddles arms, and both
pay for it in effective sample size (windows or regions instead of users). The
difference is which axis they separate on, and each axis has its own leakage.
A switchback separates in time, so its weakness is carryover: if the treatment
leaves a persistent state behind (drivers repositioned, inventory depleted,
users who learned something), the next window inherits it and control windows
are contaminated by their treated predecessors; the mitigations are longer
windows and burn-in gaps, both of which cost sample. A geo split separates in
space, so its weakness is cross-border flow: demand, supply, or social
influence that crosses region boundaries leaks treatment into control, and
larger, better-isolated regions mean fewer units and higher variance. The
choice turns on where your interference travels: pick switchbacks when effects
decay quickly in time but the resource pool is global, and geo splits when
regions are naturally isolated but treatment effects persist too long for
windows to wash out.

## Commonly answered wrong (the traps)

**Q: If the confidence interval includes zero, the change has no effect. Right?**

A: Partially right but often misapplied. A confidence interval that includes
zero means you cannot rule out zero effect at the stated confidence level. It
does not mean the effect is zero. The interval might be wide because the test
was underpowered. The correct answer after a flat, non-significant result is to
check whether the test had enough power to detect the MDE, and to report the
interval width, not just the binary outcome.

**Q: Can you test multiple variants in one experiment?**

A: Yes, but you must correct for multiple comparisons. A two-treatment (A/B/C)
test at alpha = 0.05 each, comparing each treatment against control only,
does not give you a 5% false-positive rate overall; it gives you roughly
1 - (0.95)^2 = 9.75%. If all pairwise comparisons (A vs B, A vs C, B vs C)
were made, the FWER would rise to 1 - (0.95)^3 ~ 14.3%. Pre-declare which
comparison is the primary, apply Bonferroni or FDR correction for the rest,
and power each variant arm to detect the MDE against control independently.

**Q: A guardrail is flat (not significant). Is it safe to ship?**

A: No. "Not significant" is not "proven safe." A guardrail that is
underpowered produces a non-significant result even when a real regression
exists. Use a non-inferiority test with an explicit margin (Airbnb and Spotify
do this): the confidence interval of the guardrail change must exclude a
meaningful harm before you declare it safe. Power each guardrail metric at the
joint beta-star level when you need all guardrails to pass.
**Why** the joint powering matters: the ship decision requires every guardrail
to be clean simultaneously, and misses compound. If each of five guardrails
individually has only a modest chance of catching its own real regression, the
probability that at least one real regression slips through the whole panel is
much higher than any single metric's miss rate, so each guardrail must be
powered above the level that would suffice for it alone.

**Q: What does pre-registering the experiment actually mean?**

A: Writing down, before launching, the primary metric, the expected direction of
the effect, the guardrail metrics with their non-inferiority margins, the planned
sample size and duration, and the stopping rule. This removes your ability to
rationalize a result after seeing it (HARKing: hypothesizing after results are
known). Booking.com uses pre-registration adherence as the core of their
experimentation quality KPI. Without it, even a well-designed experiment can
produce a misleading ship decision.
**Why** post-hoc freedom corrupts the statistics: with dozens of metrics,
slices, and time windows available after the fact, some comparison will clear
the 5% bar by pure chance, and picking the hypothesis after seeing which one
did converts that expected fluke into an apparently confirmed finding. The
p-value's guarantee assumes the test was chosen before the data were seen; the
implicit search over candidate hypotheses is a multiple-comparisons problem the
final p-value silently ignores.

**Q: Your intended 50/50 split came back 50.4/49.6 across millions of users. That
is close enough to balanced, so proceed with the analysis?**

A: No; that is a sample ratio mismatch (SRM), and at that scale it is a red flag
that usually invalidates the whole test. The check is a chi-square goodness-of-fit
test on the raw assignment counts against the intended ratio: with millions of
users, a 50.4/49.6 split produces a vanishingly small p-value, so the imbalance is
not chance. The reason it matters is that a broken randomization or logging path
(a redirect that drops slow clients on one arm, a bot filter applied
asymmetrically, differential opt-out) means the two arms are no longer exchangeable,
so the treatment-effect estimate is confounded by whatever caused the skew, not by
the treatment. The correct response is to stop, diagnose the pipeline (assignment,
exposure logging, filtering), and rerun, not to analyze the metric. "Close to
50/50" is exactly the intuition SRM detection exists to override.
