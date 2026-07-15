# 4. Model development

## The honest ordering of model families

Reaching for a Transformer before benchmarking a simpler baseline is a red flag. The mature answer goes from simple to complex, stopping at whichever level justifies the extra cost.

**Classical (ARIMA, ETS, Prophet, Theta).** One model per series. ARIMA models autocorrelation and differencing. ETS (exponential smoothing state space) models level, trend, and seasonality in a principled probabilistic form. Prophet is a robust additive decomposition with a piecewise linear or logistic trend plus Fourier-series seasonality and explicit holiday regression. These win when you have **few series with long, clean, stable history**, and they are the fast baseline you must benchmark against before claiming a deep model helps. At millions of series, fitting one model per series does not scale and cannot borrow strength across related items.

**Global gradient-boosted trees (LightGBM, XGBoost) on lag and calendar features.** Reframe forecasting as tabular regression: target is future demand, features are lags, rolling statistics, and calendar signals. One global model shares weights across all series, borrows strength, and generalizes to new items via attribute features. This is the **workhorse for many related series** and is cheap to iterate and retrain. Zalando adopted LightGBM with Nixtla's MLForecast for demand forecasting after evaluating deep models including the Temporal Fusion Transformer, citing faster iteration and a robust open-source ecosystem.

**Deep learning (DeepAR, N-BEATS, TFT, PatchTST).** Global neural models that emit distributions natively. DeepAR parameterizes a likelihood (negative binomial for counts) and samples paths. TFT emits quantiles via a multi-horizon quantile head with attention over covariates. N-BEATS is a pure time-series model without covariates, based on residual stacks of basis expansions. PatchTST splits the series into patch tokens before a Transformer, which lets it handle long horizons cheaply and channel-independently. These earn their keep at **large scale, long horizons, or rich covariate structures**, but do not reliably beat a well-tuned global GBT on short-horizon tabular demand. The cost in training, serving, and iteration speed is real; justify it explicitly.

**Time-series foundation models (the zero-shot option, 2024 onward).** The newest
family is pretrained forecasting models, the time-series analog of an LLM: train one
large model on billions of time points across many domains, then forecast a new
series **zero-shot** (with no per-series training) or after light fine-tuning. The
reference points are TimesFM (Google, a decoder-only patched transformer, Das et al.,
[arXiv:2310.10688](https://arxiv.org/abs/2310.10688)), Chronos (Amazon, which
tokenizes scaled-and-quantized values and trains a language-model-style cross-entropy
objective, Ansari et al., [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)),
Moirai (Salesforce, a universal any-frequency encoder, Woo et al.,
[arXiv:2402.02592](https://arxiv.org/abs/2402.02592)) and its mixture-of-experts
successor Moirai-MoE ([arXiv:2410.10469](https://arxiv.org/abs/2410.10469)), plus the
hosted TimeGPT (Nixtla). They shine as a **strong instant baseline and for cold-start
series** with little history, since they bring prior knowledge from pretraining. Where
they do not yet win: a well-tuned global GBT on your own large, covariate-rich panel
usually still beats a zero-shot foundation model, because the GBT is trained on
exactly your distribution and can consume your promotions and price features, which
most current foundation models handle weakly. Treat a foundation model as the new
"what does an off-the-shelf model get before I build anything" bar, not an automatic
replacement for a fitted model.

### Why a global tree cannot extrapolate a trend

The global GBT workhorse hides one structural limitation worth stating before you
ship it: a decision tree predicts a piecewise-constant function, and every leaf
value is an average of training targets. Its output is therefore bounded by the
range of targets it saw in training. A tree literally cannot predict a value larger
than the largest one in its training data. For a series with a persistent upward
trend (growing demand, an inflating price level, a metric that only climbs), this
means the model flatlines the moment the horizon leaves the historical range: it
clamps to the maximum leaf value instead of continuing the trend. Linear models and
classical ETS/ARIMA extrapolate a trend by construction; trees do not.

The standard fix is to remove the trend before the tree sees it and add it back
afterward: difference the target (predict $y_t - y_{t-1}$ or $y_t - y_{t-m}$), or
model the target as a ratio or residual against a simple trend or seasonal-naive
baseline, so the tree only has to fit the stationary part it is good at. This is why
lag-and-calendar GBT pipelines almost always predict a differenced or deflated
target rather than the raw level, and why a GBT that looks strong in backtest can
still degrade on a strongly trending SKU when that detrending step is skipped.

## Producing the probabilistic output

A probabilistic forecast is the output the decision layer needs. There are three practical paths to get there.

**Quantile regression / pinball loss.** For each target quantile $\tau \in \{0.1, 0.5, 0.9, \ldots\}$, minimize the **pinball loss**:

$$L_\tau(y, \hat{q}) = \max\!\bigl(\tau\,(y - \hat{q}),\;(\tau - 1)\,(y - \hat{q})\bigr)$$

When $y \gt \hat{q}$ (under-prediction), the penalty is $\tau \cdot (y - \hat{q})$. When $y \lt \hat{q}$ (over-prediction), the penalty is $(\tau - 1) \cdot (y - \hat{q})$. For $\tau = 0.9$, under-predictions are penalized 9x more than over-predictions, which correctly encodes the cost of stocking below the service-level target.

```python
import numpy as np
def pinball_loss(y, q_hat, tau):
    y, q_hat = np.asarray(y, float), np.asarray(q_hat, float)
    diff = y - q_hat                        # positive when we under-predict
    # penalize under-prediction by tau, over-prediction by (1 - tau)
    loss = np.maximum(tau * diff, (tau - 1) * diff)
    return loss.mean()                      # mean pinball loss over all points
# pinball_loss([10, 8], [7, 9], 0.9) -> 1.4
```

![Pinball loss shape](assets/fig-pinball-loss.png)

*Pinball loss for three quantile levels. At P10 (red, left), over-prediction is punished heavily; the model is incentivized to forecast a low value that is only undercut 10 percent of the time. At P90 (blue, right), under-prediction is punished heavily. P50 (gray, center) is symmetric and recovers MAE.*

GBT models grow separate trees per quantile head; TFT adds quantile output heads on top of its shared encoder. Both fit cleanly into the tabular regression framework.

**Parametric likelihood (DeepAR-style).** Instead of fixed quantiles, predict the parameters of a distribution (mean and dispersion of a negative binomial for count demand). Sample paths from that distribution for Monte Carlo downstream use. The advantage is a continuous density; the risk is distributional misspecification if the true shape changes across seasons.

**Conformal prediction.** A distribution-free calibration wrapper on any point model. Train the point model, compute residuals on a held-out calibration set, and widen intervals to achieve nominal empirical coverage. Cheap honest intervals for a model that is already in production.

```python
def conformal_interval(cal_residuals, point_pred, alpha=0.1):  # split conformal, 1-alpha coverage
    import numpy as np
    r = np.abs(np.asarray(cal_residuals, float))   # errors on a held-out calibration set
    q = np.quantile(r, 1 - alpha)                  # the (1-alpha) empirical error quantile
    return point_pred - q, point_pred + q          # symmetric calibrated interval
# conformal_interval([1, -2, 3, -1, 2], 10, 0.1) -> (7.4, 12.6)
```

## Hierarchical reconciliation

Forecasting each level independently (item, store, region, national) produces incoherent numbers: the item-level forecasts will not sum to the store total, and the business cannot act on levels that contradict each other. There are three reconciliation strategies:

- **Bottom-up.** Forecast the most granular level (item-store) and sum upward. Coherent by construction. Propagates leaf noise upward.
- **Top-down.** Forecast the aggregate, split by historical proportions. Stable aggregate but misses leaf dynamics.
- **Optimal reconciliation (MinT).** Forecast all levels independently, then project the entire set onto the coherent subspace using a trace-minimizing step derived from the estimated residual covariance matrix. Provably reduces total error by incorporating signal from every level. Amazon's hierarchical forecasting paper embeds this step as a differentiable layer, enabling end-to-end learning that emits coherent probabilistic forecasts without a post-hoc reconcile step.

Concretely, MinT takes the stacked base forecasts $\hat{y}$ (all levels), the summing matrix $S$ (which rows sum up the hierarchy), and the residual covariance $W$, and returns the coherent set $\tilde{y}$:

$$\tilde{y} = S\,(S^\top W^{-1} S)^{-1} S^\top W^{-1}\,\hat{y}$$

This is a covariance-weighted least-squares projection: levels whose residuals are small and reliable (low variance in $W$) pull the reconciled numbers toward themselves, and noisy levels are trusted less. Bottom-up and top-down are the special cases where all trust is placed on a single level.

## When to use which model family and output

| Option | Reach for it when | Skip it when |
|---|---|---|
| Classical (ARIMA, ETS, Prophet, Theta) | few series (dozens to hundreds), long and clean history, stable seasonality; the benchmark baseline | millions of related series where one-model-per-series does not scale and cannot borrow strength |
| Global GBT on lag and calendar features | many related series, short to medium horizons, cheap iteration and serving; the production workhorse | very long horizons or rich covariate structures where a deep model pulls ahead at acceptable cost |
| Deep (DeepAR, N-BEATS, TFT, PatchTST) | large scale, long horizons, rich covariates, or cold-start via learned series embeddings | short-horizon tabular demand where a tuned global GBT wins for less cost and iteration effort |

| Probabilistic output | Reach for it when | Skip it when |
|---|---|---|
| Quantile regression / pinball loss | you need specific operating points (P10, P50, P90) directly, assumption-free | you need full sampled paths for Monte Carlo downstream use, not just fixed quantiles |
| Parametric likelihood (negative binomial) | demand is count-valued and over-dispersed; sampled paths are useful | distributional assumption misfits (e.g., demand has two modes); quantile regression is assumption-light |
| Conformal prediction | you have a reliable point model and want calibrated intervals with minimal extra work | you need per-quantile asymmetric shape, not just nominal coverage symmetric around a point |

| Reconciliation | Reach for it when | Skip it when |
|---|---|---|
| Bottom-up | leaf-level forecasts are trustworthy and coherence by construction is the priority | leaf noise is high and propagates upward to degrade aggregates |
| Top-down | the aggregate is stable and historical split proportions are reliable | leaf dynamics shift over time and the top-down splits miss them |
| Optimal (MinT) or end-to-end coherent | all levels are forecast and you want provably lower total error using the full covariance | extra compute and covariance estimation overhead are prohibitive; an end-to-end coherent model can replace it |

**Provenance.** ARIMA and ETS are classical statistics; the Theta method (2000) won
the M3 competition as a strong simple baseline. Prophet (Meta, 2017) added a
decomposable trend/seasonality/holiday model. The deep families originate from
DeepAR (Amazon, 2017, autoregressive RNN with parametric likelihood), N-BEATS
(Element AI, 2019), Temporal Fusion Transformer (Google, 2019), and PatchTST (2022).
The global GBT workhorse rests on XGBoost (Chen and Guestrin, 2016) and LightGBM
(Microsoft, 2017). Optimal reconciliation is MinT (Wickramasuriya et al., 2019).

**Tools for each family.** Classical: statsmodels and the fast statsforecast
(ARIMA, ETS, Theta), plus Prophet (Meta). Global GBT: LightGBM or XGBoost on
lag-and-calendar features, wired up with mlforecast or skforecast. Deep: GluonTS and
PyTorch-Forecasting (DeepAR, TFT, N-BEATS) and neuralforecast (PatchTST).
Reconciliation: hierarchicalforecast implements bottom-up, top-down, and MinT.

**Worked example.** A retailer forecasts daily demand for 500k SKU-store pairs over
an 8-week horizon, with promotions and holidays as covariates. Per-series ARIMA does
not scale to 500k fits and cannot borrow strength across related SKUs, so the
production choice is a single global GBT (mlforecast + LightGBM) on
lag/calendar/promotion features, emitting P10/P50/P90 through quantile (pinball) loss
so the safety-stock decision gets the operating points it needs directly. A deep
model such as TFT only earns its extra training and serving cost if the horizon
grows long or the covariate structure gets rich enough that the GBT plateaus. For
coherence across the SKU to category to region hierarchy, reconcile the leaf
forecasts with MinT (hierarchicalforecast) rather than trusting bottom-up sums when
leaf noise is high.

## Implementation and training pitfalls

Forecasting models fail less on the algorithm than on how time is handled around
them: features that peek past the forecast origin, validation that shuffles the
future into the past, and scaling statistics fit on data the model should not have
seen yet.

| Problem | Symptom | Fix |
|---|---|---|
| Forecast horizon leakage | eval strong, live accuracy drops at longer horizons | build lags and rolling stats only from data available at the forecast origin, respect the horizon gap |
| Per-series normalization leakage | validation error looks great, generalization poor | fit each series scaler on the train window only and apply it forward |
| Random train/val split | optimistic error, model has effectively seen the future | rolling-origin (time-series) cross-validation |
| Quantile crossing | P90 forecast falls below P50 for some points | fit monotone quantile heads or sort the predicted quantiles post-hoc |
| Intermittent, zero-inflated demand | RMSE minimized by predicting near zero everywhere | use a count likelihood (negative binomial) or Croston-style handling and evaluate with pinball, not RMSE |
| Concept drift, regime change | accuracy drops after a promotion or price change | include promo, price, and holiday covariates, retrain on recent windows, track error by segment |
| Hierarchy incoherence | store forecasts do not sum to the region forecast | reconcile with MinT rather than trusting bottom-up sums |
| New-item cold start | no history, the global model emits garbage | fall back to attribute-based analogs or category priors until history accrues |

The through-line: a forecast that scores well under a random split is almost always
leaking the future, so validate on a rolling origin before trusting any number.
