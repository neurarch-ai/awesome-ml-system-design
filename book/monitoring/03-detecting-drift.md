# 3. Detecting drift

You detect drift by comparing a current production window against a reference,
usually the training distribution or a recent healthy baseline. Three
statistics dominate in practice.

## Population Stability Index (PSI)

PSI is the workhorse for tabular features in industry. Given bin fractions
$p_i$ from the reference window and $q_i$ from the current window:

$$\text{PSI} = \sum_{i} (p_i - q_i) \ln \frac{p_i}{q_i}$$

```python
import numpy as np
def psi(p, q):
    # p: reference bin fractions, q: current bin fractions (each vector sums to 1)
    p, q = np.asarray(p, float), np.asarray(q, float)
    total = 0.0
    for pi, qi in zip(p, q):                 # walk the aligned reference/current bins
        total += (pi - qi) * np.log(pi / qi)  # (p_i - q_i) ln(p_i / q_i)
    return total
# psi([0.4, 0.3, 0.3], [0.2, 0.3, 0.5]) -> 0.2408 (moderate shift, in the 0.10-0.25 band)
```

The field rule for reading PSI:

| PSI range | Interpretation |
|---|---|
| below 0.10 | stable; no action |
| 0.10 to 0.25 | moderate shift; investigate |
| above 0.25 | material move; alert and act |

PSI is symmetric: it does not care which window you call the reference, because
it adds both directions of divergence. This makes it stable across window
choices, which is why it is the default in tools like Evidently AI.

![PSI over time with thresholds](assets/fig-psi-over-time.png)

*PSI stays below 0.10 for the first 13 weeks, breaches the warn threshold
after a traffic shift, and eventually exceeds the alert threshold. The two
dashed lines implement tiered alerting (warn vs page). Illustrative.*

## KL divergence

KL divergence is the directed parent of PSI:

$$D_{\text{KL}}(P \parallel Q) = \sum_{i} P_i \ln \frac{P_i}{Q_i}$$

PSI is just the symmetric version:

$$\text{PSI} = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)$$

Use KL directly when you specifically need a directional measure, for example
when you want to ask "how much harder is it to encode current data using the
training reference?" rather than a symmetric distance. In most monitoring
contexts PSI is the right default because it is window-agnostic.

## Kolmogorov-Smirnov test

The KS test measures the maximum absolute difference between two empirical
cumulative distribution functions:

$$\text{KS} = \max_x \left| F_{\text{ref}}(x) - F_{\text{cur}}(x) \right|$$

It is non-parametric: no assumption about the shape of either distribution.
It returns a p-value, so the decision threshold is statistical significance
rather than a field-rule number.

## Chi-square test

For categorical features, the chi-square test measures whether the observed
cell counts in the current window differ from those expected under the
reference distribution. PSI also applies to categorical features if you use
the categories as bins, but chi-square gives a proper statistical test.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| PSI (0.10/0.25 thresholds) | you want a symmetric, window-agnostic drift check on a continuous or ordinal feature; industry default | directional KL, unless you specifically need asymmetry |
| KL divergence | you need a directional measure (e.g., encoding cost in one direction) | PSI, which is the symmetric default for monitoring |
| KS test | continuous features, no distributional assumption, want a p-value | PSI, when you prefer a threshold-based rule over statistical significance |
| Chi-square test | categorical features (counts per category) | KS, which requires a continuous CDF; one-size tests misfire on categorical data |
| MMD (Maximum Mean Discrepancy) | multivariate joint shift that per-feature tests miss; rare in practice | per-feature tests, when you can afford the computational cost of MMD |

**Tools.** Evidently runs PSI, KS, and chi-square drift tests out of the box, and whylogs logs the feature profiles those statistics run over. Alibi Detect provides KS, chi-square, and MMD detectors, and scipy.stats gives KS and chi-square directly. PSI and KL are a short computation over binned histograms once the reference and current profiles exist.

**Worked example.** A marketplace monitors its ranking features against a healthy baseline. For an ordinal price feature it uses PSI with the field-rule thresholds (Evidently) because it wants a symmetric, window-agnostic check that does not depend on which window it calls the reference. When it specifically needs a directional read, how much harder current data is to encode under the training reference, it switches to KL. For a continuous feature where it would rather have a p-value than a fixed threshold it runs the KS test (scipy), and for a categorical placement feature it uses chi-square instead, since KS needs a continuous CDF. It reserves MMD (Alibi Detect) for the rare case where a joint multivariate shift slips past every per-feature test.

## What drift detection does not tell you

PSI and KS answer "did the distribution move?" They do not answer "does it
matter?" A feature can drift significantly without affecting model quality if
the model barely weights it. Before acting on a drift alert, check the feature's
importance and whether prediction drift or performance decay followed. Drift
without decay is often noise.

Similarly, data-health failures (a broken upstream join, a null-returning
feature) read as sharp distribution shifts. Running layer-1 data-health checks
before drift tests avoids the trap of retraining to "fix drift" that is actually
a broken pipeline.

## Setting thresholds from history

The PSI field rule (0.10/0.25) is a starting point, not a universal constant.
Set your thresholds from the observed historical variation in each feature. If
a feature naturally swings by PSI 0.15 every weekend, a threshold of 0.10 will
page on-call every Monday. Uber's D3 system uses a Prophet time-series model
to learn the expected range for each monitor so that seasonal swings do not
generate false alarms.
