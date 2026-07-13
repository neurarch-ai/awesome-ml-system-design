# 5. Evaluation

Forecasting evaluation has more traps than almost any other ML domain. Three mistakes dominate interviews and real systems: using MAPE, using a random split, and reporting a single aggregate number that hides horizon-dependent decay.

## Why MAPE is broken

Mean absolute percentage error:

$$\text{MAPE} = \frac{1}{n}\sum_{t=1}^{n}\frac{\left|y_t - \hat{y}_t\right|}{y_t}$$

Three problems. First, it is **undefined when $y_t = 0$**, which happens constantly at item-store granularity for intermittent demand. Second, it is **asymmetric**: a 50 percent under-prediction contributes 0.5 while the equivalent over-prediction can contribute far more when the denominator is small. Third, it **explodes on small denominators**, making a SKU with 2 units of demand dominate the average and distort model selection. Say this plainly in an interview; it is the most commonly missed point.

## MASE: the scale-free point accuracy metric

Mean absolute scaled error normalizes by the in-sample naive seasonal forecast error, making it **unit-free and comparable across series** of different scales, and it is **defined at zero demand**:

$$\text{MASE} = \frac{\frac{1}{n}\sum_{t=1}^{n}\left|y_t - \hat{y}_t\right|}{\frac{1}{T - m}\sum_{t=m+1}^{T}\left|y_t - y_{t-m}\right|}$$

where $m$ is the seasonal period (52 for weekly annual seasonality, giving the seasonal naive baseline). When there is no meaningful seasonal period, set $m = 1$, which reduces to the non-seasonal MASE normalized by the one-step naive error. A MASE $\lt 1$ means the model beats the naive baseline; a MASE $\gt 1$ means it is worse. Use MASE as the headline point-accuracy metric.

## Pinball loss and WQL: scoring the distribution

Pinball loss (defined in section 4) scores a single quantile. To score the **whole predictive distribution**, sum or integrate pinball loss across all target quantile levels.

The **volume-normalized discrete form** (used by most production forecasting tools) is:

$$\text{WQL} = \frac{2}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{\sum_{t=1}^{n} L_\tau\!\bigl(y_t, \hat{q}_t(\tau)\bigr)}{\sum_{t=1}^{n} |y_t|}$$

where $\mathcal{T}$ is the set of target quantile levels and the denominator normalizes by total volume so high-volume series receive proportionally more weight. An alternative in the literature is the **unnormalized continuous integral form**: $\text{WQL} = 2 \int_0^1 L_\tau(y, \hat{q}(\tau))\,d\tau$; the two are consistent in purpose but differ in normalization.

In the continuous limit, the **unnormalized integral form** approximates the **continuous ranked probability score (CRPS)**:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty}\!\bigl(F(z) - \mathbf{1}[z \ge y]\bigr)^2\,dz$$

CRPS is a proper scoring rule: the forecaster minimizes it by reporting the true predictive distribution, not a strategically sharpened or widened one. The volume-normalized WQL above does not share the same limiting relationship because the per-series volume scaling changes the integrand. Use volume-normalized WQL for practical evaluation across heterogeneous series; CRPS as the theoretical ideal.

## Calibration: the check that intervals mean what they say

Reporting WQL or pinball loss is not sufficient; you must also check that the intervals are **calibrated**. A P90 forecast should be exceeded about 10 percent of the time:

$$\widehat{\text{cov}}(\tau) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\bigl[y_i \le \hat{q}_i(\tau)\bigr] \approx \tau$$

If the empirical coverage of the P90 forecast is 60 percent rather than 90 percent, the intervals are too narrow and the optimizer will stock out far more than intended. Always report empirical coverage alongside the loss value.

## Rolling-origin backtesting

A single train/test split on a time series is not a valid evaluation: it measures accuracy at one horizon from one cutoff, hides how performance degrades with distance, and is sensitive to what happened to be in the test window. The correct approach is **rolling-origin evaluation**: fix an origin, forecast forward the full production horizon (12 weeks), score each week's forecast against realized demand, roll the origin forward by one period, repeat.

![Rolling-origin backtesting windows](assets/fig-backtesting-windows.png)

*Each origin (row) expands the training window and evaluates on the next 12 weeks. Rolling forward and scoring each horizon distance separately reveals the accuracy decay curve: week-1 forecasts are almost always much more accurate than week-12 forecasts, and hiding that behind a single aggregate number hides where the model degrades.*

Report error **by horizon distance** (week 1 through week 12), not only as an average across all 12 steps. Horizon-dependent decay is structural and informs whether a 12-week forecast is reliable enough to act on.

## Weighted evaluation across series

Five million SKUs are not all equally important. Weight the evaluation by business value (revenue, volume, or margin) so that a million low-volume tail SKUs with noisy demand do not dominate the average and distort model selection. A weighted MASE or weighted WQL is the correct aggregate.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| MASE | scale-free point accuracy, defined at zero demand, comparable across heterogeneous series | MAPE, which is undefined at zero and explodes on small denominators |
| Pinball loss at specific quantiles | checking whether the P10, P50, or P90 is well-calibrated individually | only WQL, which hides per-quantile problems |
| WQL (weighted quantile loss) | grading the whole distribution efficiently and weighting by volume | a single-quantile pinball number that ignores the rest of the distribution |
| CRPS | theoretical ideal for evaluating a continuous predictive distribution | WQL, which approximates it and is easier to compute in practice |
| Empirical coverage check | confirming the interval is calibrated, not just reporting a loss value | trusting loss alone, which does not reveal a systematically overconfident or underconfident model |
| Rolling-origin backtest at the production horizon | all offline evaluation of a time-series model | a single hold-out period, which misses horizon-dependent decay and variability across time |
| Business-weighted aggregate | final model selection where high-value series matter more | an unweighted average dominated by tail SKUs |
