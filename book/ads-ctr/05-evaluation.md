# 5. Evaluation

Ads CTR models need two evaluation lenses: one that measures how well the model
**orders** candidates (standard ranking quality), and one that measures how well
it **prices** them (calibration, that is, whether a predicted 2% click rate
really clicks about 2% of the time). Using only AUC is the single most common
evaluation mistake for this problem. State both lenses early.

## AUC: ranking quality

Area Under the ROC Curve measures the probability that the model scores a random
positive above a random negative. It is an aggregate rank-order metric.

- **Input / output.** Takes model scores paired with binary click labels; returns
  a scalar in $[0, 1]$: 0.5 is random, 1.0 is perfect rank-order separation.

$$\text{AUC} = P(\hat{p}_{\text{click}} \gt \hat{p}_{\text{no-click}})$$

```python
import numpy as np
def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    pos = scores[labels == 1]                    # scores given to positives (clicks)
    neg = scores[labels == 0]                    # scores given to negatives (no-click)
    wins = sum((p > n) + 0.5 * (p == n)          # each pos vs each neg: 1 if ordered, 0.5 on a tie
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))          # fraction of correctly ordered pos/neg pairs
# auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) -> 0.75
```

**What AUC captures:** whether the model orders ads correctly. A higher AUC
means the top-scored ads are more likely to be clicks.

**What AUC misses:** the absolute scale of the probabilities. Two models with
identical AUC but different probability scales will rank ads identically, but
will produce different eCPMs, different prices, and different auction outcomes.
This is the gap that makes AUC alone insufficient for ads.

**A second, subtler AUC trap: global AUC over-counts easy cross-request pairs.**
Global AUC forms every positive-versus-negative pair in the whole eval set,
including pairs from *different* requests: a click shown to a heavy user against a
non-click shown to a light user. Those cross-request pairs are easy to order (the
model can separate them on user-level or context-level features alone) and they
dominate the pair count, so global AUC can look high while the model is weak at
the only comparison the auction ever actually makes: ranking the candidate ads
*within one request*. The fix ads teams use is **grouped AUC** (also called user
AUC or GAUC), popularized in Alibaba's Deep Interest Network work (Zhou et al.,
2018): compute AUC within each request or each user, then average the per-group
values weighted by impressions or clicks. A model can move grouped AUC while
leaving global AUC flat, and vice versa, so report the grouped version when the
decision is about in-request ranking. Note grouped AUC is undefined for any group
that is all clicks or all non-clicks (no valid pair exists), and those
single-label groups are common at ad-level granularity, so the weighting and the
drop rule both need to be stated explicitly.

## Log loss: calibration-aware quality

Log loss (cross-entropy) rewards the model for predicting the correct
*probability*, not just the correct order.

- **Input / output.** Takes predicted probabilities $\hat{p}_i$ and binary click
  labels $y_i \in \{0, 1\}$; returns a positive scalar where lower is better.

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \big[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \big]$$

```python
import numpy as np
def log_loss(probs, labels, eps=1e-15):
    probs = np.clip(np.asarray(probs, float), eps, 1 - eps)  # keep log() finite at 0 and 1
    labels = np.asarray(labels, float)
    terms = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)  # per-example cross-entropy
    return -terms.mean()                                               # average; lower is better
# log_loss([0.9, 0.1, 0.8], [1, 0, 1]) -> 0.14462152754328741
```

Log loss is a **proper scoring rule**: it is minimized by the true probability,
so it simultaneously rewards calibration and ranking quality. AUC rewards only
the latter.

![AUC vs log loss](assets/fig-auc-vs-logloss.png)

*Left: AUC only cares about whether the positive scores exceed the negative
scores. A systematic shift in the scale of all scores leaves AUC unchanged.
Right: log loss penalizes probability scale errors directly. A model whose
predicted rates are 40% of truth incurs high loss even if AUC is the same as a
well-calibrated model. Illustrative.*

## Normalized Entropy (NE): the ads-ranking workhorse

In practice the number ads teams actually track is not raw log loss but
**Normalized Entropy (NE)**, also called normalized cross-entropy. It is the
model's average log loss divided by the log loss of a trivial model that always
predicts the **background CTR** (the overall average click rate). The formula is:

$$\text{NE} = \frac{-\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]}{-\left[\bar{p}\log\bar{p} + (1-\bar{p})\log(1-\bar{p})\right]}$$

where $p_i$ is the predicted click probability, $y_i \in \{0,1\}$ is the click
label, and $\bar{p}$ is the background CTR. The denominator is just the entropy
(the inherent uncertainty) of that background rate.

```python
import numpy as np
def normalized_entropy(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))   # model's average cross-entropy
    pbar = y.mean()                                               # background (average) CTR
    bg = -(pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar))   # entropy of predicting pbar
    return logloss / bg
# NE = 1.0 means the model is no better than always guessing the background rate;
# NE < 1 means it beats that baseline (lower is better). A perfect model approaches 0.
```

Why NE and not raw log loss: raw log loss depends on the background CTR, so a low-CTR
surface (say 0.5% clicks) and a high-CTR surface (say 8%) produce very different log
loss numbers even for equally good models. Dividing by the background entropy
**removes that dependence**, so NE is comparable across placements, days, and traffic
mixes. That property is exactly why it is the offline north-star for CTR models: a
relative NE improvement (for example a 1% NE reduction) tends to translate into a
predictable direction on the online business metrics, so teams read NE to reason
about where ads revenue (RPM) and engagement proxies (like value-per-view) will move
before running the A/B test. NE is usually paired with **calibration** (next
section): NE tells you the ranking-and-pricing quality, calibration tells you whether
the price level is right. NE was introduced in Facebook's ads-prediction work (He et
al., 2014), and remains the standard CTR-model metric across ads-ranking teams.

## Calibration metrics

Calibration measures whether predicted probabilities match observed rates.

**Reliability curve (calibration plot).** Plot the mean predicted probability on
the x-axis against the observed click rate on the y-axis, across equal-width bins
of predicted probability. Input: predicted probabilities and binary click labels.
Output: a 2-D curve; a perfectly calibrated model lies on the diagonal.

**Expected Calibration Error (ECE).** Summarizes the reliability curve into a
scalar.

- **Input / output.** Takes predicted probabilities bucketed into $B$ bins,
  paired with binary click labels; returns a positive scalar in $[0, 1]$ where 0
  is perfect calibration.
- **How it is computed.**

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \big|\text{acc}(b) - \text{conf}(b)\big|$$

where $n_b$ is the count in bin $b$, $\text{acc}(b)$ is the observed click rate,
and $\text{conf}(b)$ is the mean predicted probability.

```python
import numpy as np
def ece(probs, labels, n_bins=10):
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)     # bin boundaries on the probability axis
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)      # predictions falling in this bin
        if m.sum() == 0: continue
        conf = probs[m].mean()                # mean predicted prob in the bin
        acc = labels[m].mean()                # observed click rate in the bin
        e += m.mean() * abs(acc - conf)       # weight the gap by the bin's fraction of data
    return e
# ece([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) -> 0.15
```

![Calibration reliability curve](assets/fig-calibration-reliability.png)

*A raw DNN head (red) over-predicts probabilities (over-confident). After Platt
scaling (green) the curve tracks the diagonal closely. The shaded zone is the
calibration gap: every point in it is an auction that prices the slot incorrectly.
Illustrative.*

## Why calibration matters for bidding: the full chain

Repeat the chain because it is the crux. eCPM = bid times pCTR. The auction
ranks ads by eCPM and charges second-price: the winning advertiser pays roughly
the eCPM of the runner-up divided by their own pCTR.

$$\text{price} = \frac{\text{eCPM}_{\text{runner-up}}}{1000 \cdot \hat{p}(\text{click})}$$

If $\hat{p}$ is systematically high (over-confident model), the platform
over-values every ad. Advertisers overpay, or the wrong ad wins because its
pCTR is inflated relative to competitors. If $\hat{p}$ is systematically low,
real revenue is left unbilled and good ads lose to weaker ones. A model with
great AUC but 20% upward calibration drift silently mis-prices every slot, every
second.

This is why you must:

- Monitor calibration **sliced** by placement, device, ad type, and time, not
  just globally. A model calibrated on average can be badly miscalibrated on the
  high-value slices the auction cares about most.
- Monitor it **continuously in production**, not just at release.
- Treat ECE as a first-class production metric, not a one-time eval step.

## The online gate: A/B test on revenue, not just AUC

Offline metrics are necessary but not sufficient. The real launch decision is an
online A/B test. What to measure:

- **Revenue per thousand requests (RPM).** The ultimate metric. A calibration
  improvement that correctly prices more slots will show up here.
- **Advertiser ROI.** Are advertisers getting value? A model that over-bids
  (high pCTR) produces clicks but at inflated cost to advertisers, which reduces
  long-term spend.
- **Calibration stability over time.** Does the model stay calibrated as
  campaigns change, or does ECE drift up within days of launch?

Offline and online metrics can diverge significantly; the online gate is mandatory.

## Compare and contrast: AUC vs NE vs calibration (ECE)

The three offline metrics are often treated as interchangeable "model quality"
numbers, but each is blind to something the others measure; the table separates
what they genuinely share from where they diverge.

| Dimension | AUC | NE | Calibration (reliability + ECE) |
|---|---|---|---|
| Computed from | The same offline pairs of (predicted probability, click label); no extra data or labels needed | Same | Same |
| Role in the launch process | Offline diagnostic feeding a go/no-go decision | Same | Same |
| What it measures | Rank order only: probability a random click outscores a random non-click | Probability quality relative to always predicting the background CTR; mixes ranking and scale | Whether predicted rates match observed rates, bin by bin; scale only, no ranking signal |
| Monotone rescaling of all scores | Invariant: a 20% uniform inflation changes nothing | Penalized: the numerator's log loss rises | Penalized directly: the whole reliability curve leaves the diagonal |
| Sensitivity to background CTR | Largely insensitive; comparable across surfaces by construction | Normalized out by dividing by the background entropy; that is its whole point | Absolute by design: ECE is read within a surface and must be sliced |
| Failure mode it catches | Rank inversions (the wrong ad wins on order) | A model no better than the base-rate guess, or regression in overall probability quality | Systematic scale drift that mis-prices every auction while order is unchanged |

The distinction changes the design decision the moment a candidate model shifts
its probability scale without changing its ordering: AUC passes it, NE degrades
mildly, and only calibration flags the shift that would repay itself as
mis-priced auctions, so the launch gate must include all three rather than any
one.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| AUC | measuring rank-order quality in isolation; comparing two models trained with the same loss | AUC alone as the final eval, which misses calibration |
| Log loss | training objective and offline quality score; rewards both ranking and calibration simultaneously | accuracy or AUC as the training loss, which ignores the probability scale |
| Reliability curve + ECE | diagnosing whether raw scores are trustworthy as probabilities before feeding the auction | AUC, which is blind to probability scale errors |
| Sliced ECE | catching calibration rot in specific segments (placement, device, ad category) | a single global ECE that hides local mis-pricing |
| Online A/B on RPM and advertiser ROI | the final launch decision and assessing whether the improvement actually moves money | offline log-loss or AUC alone, which misses the closed-loop auction dynamics |

**Tools.** scikit-learn computes AUC (roc_auc_score), log loss, and calibration curves. ECE is a small bucketing over predicted probabilities, available from netcal or a short custom routine, and sliced ECE is the same computation grouped by segment. The online A/B on RPM and advertiser ROI runs through the in-house experiment platform, with significance testing via scipy.stats or statsmodels.

**Worked example.** An ad network evaluates a new pCTR model with both lenses. It reports AUC (scikit-learn) to compare rank-order quality against the current model, but never as the final word, because a systematic scale shift leaves AUC unchanged while it moves every price. It tracks log loss as a proper scoring rule that rewards ranking and calibration together, and it plots a reliability curve with ECE (netcal) to confirm the raw scores are trustworthy before they feed the auction. Since a model calibrated on average can be badly miscalibrated on high-value slices, it slices ECE by placement and device rather than trusting one global number. The launch itself is gated on an online A/B test measuring revenue per thousand requests and advertiser ROI, which offline metrics alone cannot capture.
