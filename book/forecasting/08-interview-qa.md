# 8. Interview Q&A

The questions an interviewer actually asks about demand forecasting and time series, grouped by how they are used. The commonly-answered-wrong section is where interviews are won or lost.

## Commonly asked

**Q: Why not just minimize MAPE?**

MAPE is undefined when demand is zero, which happens constantly at the item-store leaf for intermittent demand. It is also asymmetric (over-prediction and under-prediction of the same magnitude contribute different values) and explodes on small denominators. Use MASE for scale-free point accuracy (it is defined at zero, comparable across series, and measures skill against a seasonal naive baseline) and pinball/WQL for probabilistic forecast quality.

**Q: When do you use classical models vs global GBT vs deep learning?**

Classical per-series models (ARIMA, ETS, Prophet) are the baseline for few series with long, clean, and stable history. A global GBT on lag and calendar features is the workhorse for many related series: it borrows strength, scales to millions of SKUs, and iterates cheaply. Deep models (DeepAR, TFT, PatchTST) earn their keep at very large scale, long horizons, or when rich covariate structure or cold-start via learned embeddings is needed. Baseline before going deep; a well-tuned GBT usually wins on short-horizon tabular demand at a fraction of the cost.

**Q: Why does the optimizer need a distribution, not a point forecast?**

A replenishment optimizer stocks to the quantile set by the cost ratio of understocking to overstocking (the critical fractile). Safety stock is computed from the spread of the demand distribution, not its mean. Handing the optimizer a single number makes safety stock uncomputable and results in stocking out at the service-level target. The distribution is not optional; it is what the decision layer consumes.

**Q: How do you evaluate a time-series model?**

Rolling-origin (walk-forward) backtest at the production horizon: fix an origin, forecast forward the full horizon, score against realized demand, roll the origin forward, repeat. Score each horizon distance separately (week 1 through week 12) using MASE for point accuracy and WQL plus empirical coverage for the probabilistic output. Weight by business value so high-volume series drive model selection. Never use a random split; it leaks the future.

**Q: What is hierarchical reconciliation and when does it matter?**

A hierarchy is a tree of aggregates: item rolls up to category, store to region to national. Forecasting each level independently produces incoherent numbers that do not sum, and the business cannot place inventory orders using levels that contradict each other. Reconciliation projects the forecasts onto the coherent subspace: bottom-up (sum leaves), top-down (split aggregate), or optimal MinT (use the full residual covariance to provably minimize total error). It matters any time a decision consumes more than one level of the hierarchy.

## Tricky (the follow-ups that separate people)

**Q: You fixed MASE, but now WQL is great and the optimizer still stocks out. What happened?**

Most likely a calibration failure: the empirical coverage of your P90 forecast is 60 percent rather than 90 percent. The model is overconfident, the intervals are too narrow, and the optimizer stocks to a quantile that is actually around P60 of realized demand. Always report empirical coverage alongside WQL, and verify it level by level (P10 exceeded 10 percent of the time, P90 exceeded 10 percent of the time). Loss alone does not catch a systematically overconfident model.

**Why:** pinball loss rewards sharpness as well as calibration, and it is averaged over many series. A model can trade slightly worse tail coverage on the volatile series for much sharper intervals everywhere else, which improves aggregate WQL while breaking the per-quantile coverage guarantee that the stocking decision actually depends on. The optimizer consumes the quantile as a promise ("demand exceeds this only 10 percent of the time"), so a broken promise turns directly into stockouts even when the loss looks good.

**Q: New items have no history. How do you forecast them?**

A global model with learned attribute and category embeddings can forecast a brand-new item by borrowing the pattern from similar items. Hierarchical shrinkage blends the parent-category pattern with the item's own accumulating history. Lag features are unavailable at day zero; the model falls back to attributes (price point, category, image embeddings for visual similarity). Keep the prediction intervals wide until several weeks of history have accumulated; the model is guessing from priors, not from the item's own signal.

**Why:** this works because the global model has learned a mapping from attributes to demand pattern across every existing item, so a new item's attributes place it near similar items in embedding space and its forecast inherits their level and seasonal shape as a prior. The item's own history then refines that prior as it accumulates, which is also why the intervals should start wide and narrow over time: early on, all the uncertainty is in whether the prior actually fits this item.

**Q: A Transformer is the state of the art for sequences. Should you default to it for forecasting?**

No. For short-horizon tabular demand forecasting, a well-tuned global GBT consistently matches or beats deep models at a fraction of the training and serving cost. Transformers (TFT, PatchTST) earn their keep when the horizon is long (PatchTST handles long sequences cheaply via patch tokens), when there are many correlated series with rich covariate structure, or when cold-start via learned embeddings is required. Zalando explicitly chose LightGBM over TFT because iteration speed and cost dominated at their horizon. Justify the deep model; do not default to it.

**Why:** the predictive signal in short-horizon demand lives mostly in a handful of lag and calendar features with tabular interactions, which is exactly the regime gradient-boosted trees are built for. The Transformer's advantage is learning representations of long raw sequences, and that capacity buys nothing when the useful context is short and already summarized into lag features; you pay its training and serving cost without touching the part of the problem that limits accuracy.

**Q: The item-store series are very sparse and have many zero weeks. How do you handle intermittent demand (demand that is zero in most periods with occasional bursts)?**

First, confirm MASE and pinball loss are the metrics (MAPE and RMSE are broken here). Second, use a model that handles zero-inflated or count-valued demand: a negative binomial likelihood (DeepAR-style) is a natural fit, as is a Tweedie distribution for compound Poisson demand. Third, recognize that borrowing strength from the parent level (category or store) via the global model or MinT reconciliation is often what rescues sparse leaves. A Croston decomposition (classical, splits frequency from size) is a useful baseline for very sparse series. Mechanism note: the negative binomial fits over-dispersed counts because it carries a separate dispersion parameter, so its variance can exceed its mean (a plain Poisson forces variance equal to mean and therefore understates the spread of bursty demand, which is exactly what makes safety stock too low).

**Q: Your global model backtests beautifully but degrades in production. The features look causal and the split is time-based. What subtle leak is left?**

Global models usually scale each series (divide by its own mean or median, or z-score it) so that high-volume and low-volume SKUs are comparable to one model. If that scaling statistic is computed over the whole series including the backtest horizon, every backtest row is silently normalized using future demand, which leaks the target's own level. The fix is to compute the per-series scale only from data strictly before each roll origin (the same point-in-time discipline used for lags), and recompute it as the origin rolls forward. This leak is invisible in the feature list because the leaked information rides in the normalization constant, not in any named column.

**Q: A prediction interval and a confidence interval look similar; when does the difference actually matter?**

A: Both are ranges attached to a coverage percentage, which is why they get conflated. A confidence interval quantifies uncertainty about a fitted quantity such as mean weekly demand; it shrinks toward a point as history grows, because estimation error vanishes with data. A prediction interval quantifies where an individual future observation will land; it contains the irreducible week-to-week demand noise on top of estimation error, so it does not shrink to zero no matter how much history you have. The difference matters the moment you compute safety stock: with two years of history, the 95 percent confidence interval on the mean can be a few units wide while actual weekly demand still swings by hundreds, so stocking from the confidence interval produces intervals that are far too narrow and a stockout rate far above target. The replenishment optimizer must consume the prediction interval (the quantiles of the demand distribution itself), never the confidence interval on a parameter.

## Commonly answered wrong (the traps)

**Q: Can I use the same lag features for a 1-week-ahead and a 12-week-ahead forecast?**

No. A 12-week-ahead forecast cannot use the $t - 1$ lag because that value is not yet observed at forecast time. Only lags of distance 12 or more are available at a 12-week horizon. Using a shorter lag in the feature set inflates offline metrics and collapses live. The correct approach is either to build per-horizon feature sets (lag selection varies by target horizon) or to use a direct multi-horizon model that learns different output heads for each horizon distance. **Why:** the backtest dataset already contains every historical value, so the $t - 1$ column can always be filled in and looks strongly predictive; at serving time that value has not been observed yet, so it must be imputed or replaced with the model's own forecast, and all the accuracy the backtest attributed to it evaporates. The offline number was measuring a feature the production system can never have.

**Q: The model beats the baseline on MASE, so it is ready to ship.**

Not yet. MASE is a point-accuracy metric. A model can improve MASE while producing miscalibrated intervals, incoherent hierarchy levels, or a worse downstream decision cost (stockouts and waste). The full gate is: MASE improvement holds on a rolling-origin backtest, WQL and empirical coverage are acceptable, reconciled levels are coherent, and ideally the decision cost (realized stockouts and waste on a held-out period) is better than the incumbent. **Why:** the downstream optimizer never consumes the point forecast directly; it consumes quantiles and reconciled aggregates. An improvement in a metric the decision layer never sees cannot guarantee an improvement in the decisions it makes, so the gate has to include the quantities that are actually consumed.

**Q: Should I run a separate retrain for every SKU?**

No, at millions of SKUs that is infeasible and unnecessary. A global model trains once over all series, produces forecasts for every SKU without per-series retraining, and borrows strength across related series. Per-series models (classical ARIMA, ETS) are reserved for small-scale baselines or high-value series where per-series nuance justifies the cost. For the full population, global training is the only tractable path and usually produces better accuracy on sparse series anyway, because those series benefit most from borrowing strength. **Why:** a per-series model must estimate trend and seasonality from that one series' few observations, so its parameter estimates carry huge variance; a global model estimates the shared patterns from millions of pooled series and learns only a small per-series adjustment locally, which slashes estimation variance exactly where data is scarcest.

**Q: Hierarchical forecasting means forecasting the total and then splitting it down.**

That is one strategy (top-down), but it is often the worst one. Top-down discards all the leaf-level dynamics and assumes proportions are stable over time, which fails whenever demand mix shifts (new products, regional trends, promotions). Bottom-up aggregates leaf noise upward. Optimal MinT (Wickramasuriya et al., 2019) uses the full residual covariance to reconcile all levels simultaneously and provably reduces total error compared to either top-down or bottom-up alone. Mechanism: MinT projects the stacked base forecasts onto the coherent subspace (the set of forecasts that actually sum up the hierarchy) using a covariance-weighted generalized least squares step, so a level whose historical residuals are small and reliable pulls the reconciled numbers toward itself, and a noisy level is trusted less. Top-down and bottom-up are the degenerate special cases where all the trust is placed on a single level regardless of its residual variance.
