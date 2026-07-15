# 3. Sizing and Power

## The two errors you are controlling

Every experiment makes a decision under uncertainty. Two ways to be wrong:

- **Type I error (false positive, alpha):** you ship a change that does nothing.
  Controlled by the significance level, commonly alpha = 0.05.
- **Type II error (false negative, beta):** you miss a real win.
  Controlled by the power (the probability of correctly catching a real effect when one truly exists), commonly 1 - beta = 0.80.

The **minimum detectable effect (MDE)** connects them both to the sample size:
it is the smallest true effect for which your test will correctly reject the null
(the null hypothesis: the default assumption that the change has no effect)
with the specified power. Declare the MDE before launching, because it represents
the smallest lift that is worth shipping.

![Power diagram showing control and treatment distributions](assets/fig-power-diagram.png)

*The control and treatment distributions overlap. The green area is the Type I
error (false alarm rate, alpha): the probability of rejecting the null when the
truth is no effect. The red area is the Type II error (miss rate, beta): the
probability of failing to detect a real effect of size MDE. Power = 1 minus the
red area. A larger MDE or a larger sample pushes the treatment curve further
right (in standardized effect-size space), shrinking the red area.*

## The sample-size formula

For a two-sample difference-of-means test, the required sample size **per arm**
is:

$$n \approx \frac{2 \cdot \sigma^{2} \cdot (z_{1-\alpha/2} + z_{1-\beta})^{2}}{\text{MDE}^{2}}$$

where:
- $\sigma^{2}$ is the per-user variance of the primary metric,
- $\text{MDE}$ is the minimum detectable effect (absolute, in the same units as
  the metric),
- $z_{q}$ is the standard-normal quantile (the value below which a fraction $q$ of a standard normal distribution falls) at probability $q$ (e.g.
  $z_{0.975} \approx 1.96$ for a two-tailed test at alpha = 0.05, and
  $z_{0.80} \approx 0.84$ for 80% power).

At alpha = 0.05 two-tailed and 80% power, $(z_{0.975} + z_{0.80})^{2} \approx
(1.96 + 0.84)^{2} \approx 7.85$.

```python
import math
def sample_size_per_arm(sigma, mde, z_alpha=1.96, z_beta=0.84):
    # z_alpha = 1.96 is the 0.975 standard-normal quantile (alpha=0.05, two-tailed)
    # z_beta  = 0.84 is the 0.80  standard-normal quantile (power=0.80)
    z_sum = (z_alpha + z_beta) ** 2         # ~7.84 at the standard settings
    n = 2 * sigma ** 2 * z_sum / mde ** 2   # per-arm users, diff-of-means test
    return math.ceil(n)                     # round up to whole users
# sample_size_per_arm(0.15, 0.01) -> 3528
```

Solving the same relationship for the other two unknowns gives the **achieved
statistical power** at a chosen sample size, and the **MDE** you can detect with
a fixed budget of users:

```python
import numpy as np, math
def power(n, sigma, mde, z_alpha=1.96):
    # achieved power: probability of rejecting the null when the true effect is mde
    ncp = mde * math.sqrt(n / (2 * sigma ** 2))              # noncentrality, in SE units
    return 0.5 * (1 + math.erf((ncp - z_alpha) / math.sqrt(2)))  # Phi(ncp - z_alpha)
# power(3528, 0.15, 0.01) -> 0.7995458067395504   (~0.80, the n we sized for)

def mde_from_n(n, sigma, z_alpha=1.96, z_beta=0.84):
    # smallest detectable effect given n per arm: invert the sample-size formula
    return np.sqrt(2 * sigma ** 2 * (z_alpha + z_beta) ** 2 / n)
# mde_from_n(3528, 0.15) -> 0.01   (inverts sample_size_per_arm(0.15, 0.01))
```

### Why sample size scales as $1 / \text{MDE}^{2}$

Because $\text{MDE}$ appears squared in the denominator, halving the effect you
want to detect roughly **quadruples** the required traffic and duration. This is
the single most important intuition to carry into any discussion of experiment
power. Name it explicitly; it is what every sizing conversation hinges on.

![Sample size vs MDE curve](assets/fig-sample-size-vs-mde.png)

*Required users per arm as a function of MDE, at sigma = 0.15, alpha = 0.05,
power = 80%. A 1% MDE needs roughly 4x the traffic of a 2% MDE. Annotated
points are reference markers; numbers are illustrative, not exact baselines.*

## Computing duration from sample size

Once you have $n$ per arm, divide by the daily unique users you can expose (your
traffic share times total daily users) to get the duration in days. Then round up
to the nearest multiple of 7 to absorb weekly seasonality. That number is your
commitment. Do not look for significance before the planned duration expires.

## Variance reduction (CUPED)

If a pre-experiment measurement of the primary metric correlates with the
in-experiment outcome, you can use it to remove predictable variance before the
test runs. The CUPED-adjusted metric is:

$$Y_{\text{cv}} = Y - \theta \cdot (X - \mathbb{E}[X]), \quad \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

The adjusted variance is:

$$\text{Var}(\bar{Y}_{\text{cv}}) = \text{Var}(\bar{Y}) \cdot (1 - \rho^{2})$$

where $\rho$ is the correlation between $X$ and $Y$. A correlation of $\rho =
0.7$ removes roughly half the variance, which is equivalent to doubling the
effective sample size without collecting a single extra user. Use the week before
the experiment as the pre-period covariate; it is almost always available and
typically achieves $\rho$ around 0.6 to 0.8 for engagement metrics.

CUPED is a post-hoc adjustment: estimate $\theta$ from both arms pooled after the
experiment, then analyze the adjusted outcomes.

```python
import numpy as np
def cuped(y, x):
    # y: in-experiment outcome per user; x: matching pre-period covariate per user
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)  # theta = Cov(Y,X)/Var(X)
    y_adj = y - theta * (x - x.mean())     # subtract the predictable, pre-period part
    return y_adj                            # same mean as y, but smaller variance
# with rho = corr(x, y), var(y_adj) ~ var(y) * (1 - rho^2). Extreme check:
# y = np.array([1., 2, 3, 4]); cuped(y, y).var() -> 0.0  (x == y, rho = 1, all variance gone)
```

## When to use which approach

| Reach for | When | Instead of |
|---|---|---|
| Standard t-test sizing | baseline rate, variance, and MDE are known; run the formula before launch | picking a duration by gut feel and peeking for significance |
| CUPED variance reduction | a pre-experiment measurement of the same metric is available (typical correlation 0.6-0.8 removes 35-65% of variance) | running longer; CUPED buys the same power with fewer users |
| Relative MDE (percent lift) | communicating with stakeholders ("we can detect a 1% lift") | absolute MDE alone, which is hard to interpret without the baseline |
| Conservative MDE (small effect) | the change is high-risk, or you care about small improvements | a large MDE that would miss real but modest wins |
| Increase traffic share | CUPED is not available and the metric variance is high | extending the experiment window indefinitely, which risks novelty effects |
| Power at the joint level (beta star = beta / (G + 1) for G guardrails) | you require all G guardrail metrics to pass at power beta (Spotify pattern) | powering only the primary and then being surprised when guardrails are underpowered |

**Provenance.** The CUPED variance-reduction approach is from Microsoft (2013).

**Tools.** The per-arm sample-size formula is packaged in statsmodels (statsmodels.stats.power, for example TTestIndPower.solve_power), so you solve for n, MDE, or power directly rather than coding the z-quantiles by hand. Baseline rate and per-user variance are estimated from historical logs with pandas or SQL, and duration is that n divided by daily exposed users, rounded up to a multiple of seven. CUPED is a short ordinary-least-squares residualization in statsmodels using the pre-period covariate; the joint-power correction for G guardrails is a hand-applied tightening of the target power fed back into the same solver.

**Worked example.** A marketplace sizing a checkout experiment first declares the smallest lift worth shipping as its MDE, then solves for users per arm with statsmodels power given the historical variance, remembering that halving the MDE roughly quadruples the traffic and duration because n scales as one over MDE squared. A prior-week measurement of the same metric correlates around 0.7, so it applies CUPED to remove roughly half the variance, buying the same power with far fewer users instead of extending the window into novelty-effect territory. It communicates the target to stakeholders as a relative percent lift rather than an opaque absolute number, and because the launch also guards several metrics, it powers each at the tighter joint level rather than powering only the primary and being surprised when a guardrail comes back underpowered. Only when CUPED is unavailable and variance stays high does it raise traffic share rather than run indefinitely.

## The Spotify joint-power correction

When you require G guardrail metrics to all pass alongside the primary, the
joint false-negative rate rises. Spotify corrects by targeting tighter power per
metric:

$$\beta^{\ast} = \frac{\beta}{G + 1}$$

False-positive rates are not adjusted across guardrails, because requiring all
of them to pass does not compound alpha the way independent tests would.

```python
def joint_beta(beta, G):
    # tighten the per-metric miss rate so the primary plus G guardrails all hold jointly
    return beta / (G + 1)          # the primary metric is the "+1" alongside G guardrails
# joint_beta(0.20, 3) -> 0.05   (target beta 0.20 with 3 guardrails => power 0.95 each)
```

You then feed this tighter $\beta^{\ast}$ back into `sample_size_per_arm` as a
smaller `z_beta` target, which raises the required sample size per arm.
