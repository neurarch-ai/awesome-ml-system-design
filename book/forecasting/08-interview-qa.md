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

**Q: New items have no history. How do you forecast them?**

A global model with learned attribute and category embeddings can forecast a brand-new item by borrowing the pattern from similar items. Hierarchical shrinkage blends the parent-category pattern with the item's own accumulating history. Lag features are unavailable at day zero; the model falls back to attributes (price point, category, image embeddings for visual similarity). Keep the prediction intervals wide until several weeks of history have accumulated; the model is guessing from priors, not from the item's own signal.

**Q: A Transformer is the state of the art for sequences. Should you default to it for forecasting?**

No. For short-horizon tabular demand forecasting, a well-tuned global GBT consistently matches or beats deep models at a fraction of the training and serving cost. Transformers (TFT, PatchTST) earn their keep when the horizon is long (PatchTST handles long sequences cheaply via patch tokens), when there are many correlated series with rich covariate structure, or when cold-start via learned embeddings is required. Zalando explicitly chose LightGBM over TFT because iteration speed and cost dominated at their horizon. Justify the deep model; do not default to it.

**Q: The item-store series are very sparse and have many zero weeks. How do you handle intermittent demand?**

First, confirm MASE and pinball loss are the metrics (MAPE and RMSE are broken here). Second, use a model that handles zero-inflated or count-valued demand: a negative binomial likelihood (DeepAR-style) is a natural fit, as is a Tweedie distribution for compound Poisson demand. Third, recognize that borrowing strength from the parent level (category or store) via the global model or MinT reconciliation is often what rescues sparse leaves. A Croston decomposition (classical, splits frequency from size) is a useful baseline for very sparse series.

## Commonly answered wrong (the traps)

**Q: Can I use the same lag features for a 1-week-ahead and a 12-week-ahead forecast?**

No. A 12-week-ahead forecast cannot use the $t - 1$ lag because that value is not yet observed at forecast time. Only lags of distance 12 or more are available at a 12-week horizon. Using a shorter lag in the feature set inflates offline metrics and collapses live. The correct approach is either to build per-horizon feature sets (lag selection varies by target horizon) or to use a direct multi-horizon model that learns different output heads for each horizon distance.

**Q: The model beats the baseline on MASE, so it is ready to ship.**

Not yet. MASE is a point-accuracy metric. A model can improve MASE while producing miscalibrated intervals, incoherent hierarchy levels, or a worse downstream decision cost (stockouts and waste). The full gate is: MASE improvement holds on a rolling-origin backtest, WQL and empirical coverage are acceptable, reconciled levels are coherent, and ideally the decision cost (realized stockouts and waste on a held-out period) is better than the incumbent.

**Q: Should I run a separate retrain for every SKU?**

No, at millions of SKUs that is infeasible and unnecessary. A global model trains once over all series, produces forecasts for every SKU without per-series retraining, and borrows strength across related series. Per-series models (classical ARIMA, ETS) are reserved for small-scale baselines or high-value series where per-series nuance justifies the cost. For the full population, global training is the only tractable path and usually produces better accuracy on sparse series anyway, because those series benefit most from borrowing strength.

**Q: Hierarchical forecasting means forecasting the total and then splitting it down.**

That is one strategy (top-down), but it is often the worst one. Top-down discards all the leaf-level dynamics and assumes proportions are stable over time, which fails whenever demand mix shifts (new products, regional trends, promotions). Bottom-up aggregates leaf noise upward. Optimal MinT uses the full residual covariance to reconcile all levels simultaneously and provably reduces total error compared to either top-down or bottom-up alone.
