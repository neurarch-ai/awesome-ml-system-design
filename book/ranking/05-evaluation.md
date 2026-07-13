# 5. Evaluation

Ranking is judged by different metrics for different purposes. Using the wrong
one is a classic mistake, and being able to explain which metric earns its place
when is a strong interview signal.

## NDCG: quality of the ordered list

**NDCG@k** (Normalized Discounted Cumulative Gain) measures how well the top-k
returned items match the ground-truth relevance, weighted by position. Hits near
the top count more than hits further down.

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \qquad \text{DCG@k} = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i + 1)}$$

where $r_i$ is the relevance label of the item at position $i$ and IDCG@k is
the DCG of the ideal (perfect) ranking. NDCG is the right offline metric when
you care about both recall (were the good items retrieved?) and order (are they
near the top?). Yelp and Airbnb use NDCG as the offline signal for their
learning-to-rank models because NDCG directly rewards putting the most relevant
item at rank 1.

![NDCG@k curves for different optimization approaches](assets/fig-ndcg-k.png)

*LambdaMART (blue) scales each training gradient by how much swapping two items
would move NDCG, so it directly optimizes the metric. Pointwise log-loss (red)
treats items independently and lags behind. Illustrative values.*

## AUC: ranking quality of a binary objective

**AUC** (area under the ROC curve) measures whether the model ranks positive
examples above negative examples across all thresholds. It is the standard
offline metric for binary classification objectives (click vs. no-click) at
scale, because it is threshold-free and easy to compute over billions of rows.
AUC is insensitive to the number of positives, which matters when the click rate
is 1-2%. The typical failure mode: AUC can improve while calibration worsens, so
never use AUC alone when scores feed an auction.

## Logloss: sharpness of the probability estimate

**Logloss** (binary cross-entropy) penalizes confident wrong predictions harder
than uncertain ones. It is complementary to AUC: AUC captures ranking quality,
logloss captures sharpness of the probability estimates. A model can gain AUC
while its logloss worsens if it is shifting confident scores in the right
direction but over- or under-estimating magnitudes.

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]$$

## Calibration: does 0.1 mean 10%?

When ranking scores feed a downstream auction, a bid, a threshold, or a utility
blend, **the predicted probability must mean what it says**. A calibration
reliability diagram plots mean predicted confidence on the x-axis against actual
fraction of positives on the y-axis. The diagonal is perfect calibration.

$$\text{ECE} = \sum_{b=1}^{M} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|$$

where $M$ is the number of bins, $n_b$ the count in bin $b$, $\text{acc}(b)$ the
fraction of actual positives in the bin, and $\text{conf}(b)$ the mean predicted
probability. ECE summarizes the area between the curve and the diagonal.

Training on downsampled negatives and stratified samples distorts calibration,
so apply a post-hoc step (Platt scaling or isotonic regression) and monitor ECE
as a first-class metric. Spotify monitors ECE live because it directly drives
auction pricing: a 10% over-confident prediction means a 10% over-bid.

![Calibration reliability diagram](assets/fig-calibration.png)

*An over-confident model bows toward the x-axis (red): it predicts 0.8 but only
30% of those actually click. An under-confident model bows above the diagonal
(orange). A well-calibrated model tracks the diagonal (green). Illustrative.*

## Offline-online gap

An offline metric gain (AUC, NDCG, logloss) does not guarantee an online
engagement win. The most common reasons for the gap:

- **Training-serving skew.** A feature computed one way in the training pipeline
  and another way in the serving code means the model operates on a distribution
  it never trained on. This is the single most common silent failure in deployed
  rankers.
- **Label leakage.** A feature that encodes the outcome (for example an item
  engagement rate including the current impression) inflates offline metrics and
  collapses online.
- **Position bias not corrected.** The offline labels carry position signal not
  present at serving, so the offline metric is flattering.

The guardrail to state out loud: a positive offline metric is a pre-gate, not a
ship decision. The ship decision is an online A/B test on the business metric.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| AUC | Binary engagement objective, billions of training rows, no calibration requirement | NDCG when order and position matter more than binary discrimination |
| NDCG@k | The order of the top-k items matters (search, LTR); you want to reward the best item at rank 1 | AUC when you only care about binary separation, not ranked position |
| Logloss | You need to track sharpness of probability estimates alongside AUC | AUC alone when downstream uses need calibrated probabilities |
| ECE + calibration plot | Score feeds an auction, a bid, a threshold, or a cross-task blend | Shipping raw scores as probabilities before checking calibration |
| Online A/B on business metric | The final ship decision | Offline metric alone, which misses the training-serving seam |
