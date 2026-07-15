# 6. Serving and scaling

## Batch, not online

Demand forecasting at the scale we scoped (5 million SKUs, 12-week horizon, weekly cadence) is a **batch inference problem**, not an online serving problem. You do not need a forecast in under 50 ms in response to a user request. You need 5 million forecasts to land in a database before the Monday morning replenishment run, every week.

This changes the design significantly compared to, say, an ETA or a feed recommendation. The batch path allows:
- Heavier models (TFT, DeepAR) that would miss a per-request latency budget
- Full feature pipelines that pull 2 years of history and precompute rolling statistics
- Post-hoc hierarchical reconciliation that is too expensive to run inline
- Quality checks (coverage audit, MASE drift) that run before the forecasts reach the optimizer

The constraint that replaces latency is **throughput within the weekly cadence**: all 5 million SKUs must be forecast, reconciled, quality-checked, and loaded before the optimizer runs. At weekly granularity with modern infrastructure (PySpark preprocessing, distributed GBT inference), this is tractable in under 2 hours for most ML teams.

## The forecast-then-optimize pattern

The forecast is not the product; it is the input to the **optimizer**. For replenishment, the canonical optimizer is the newsvendor model: stock to the quantile $q^{\ast}$ set by the critical fractile,

$$q^{\ast} = F^{-1}\!\!\left(\frac{c_u}{c_u + c_o}\right)$$

where $c_u$ is the underage cost (lost sale, churn) and $c_o$ is the overage cost (holding cost, waste). For a 3:1 underage-to-overage ratio, $q^{\ast} = F^{-1}(0.75)$: stock to the P75 of forecasted demand. A more general optimizer (Zalando uses gradient-free Monte Carlo over a full replenishment policy) consumes the entire demand distribution, not just one quantile, and also folds in lead-time uncertainty and current stock state.

Two things candidates miss: the optimizer needs the **distribution** (not the mean), and the metric that ultimately matters is the **downstream decision cost** (realized stockouts and waste), not the forecast error. Where feasible, evaluate model changes against the downstream cost on a held-out period.

## Hierarchical reconciliation at scale

Reconciliation runs once per forecast cycle, after all series are forecast. MinT requires estimating the residual covariance matrix of all levels, which is a large matrix at millions of series. In practice:
- **Sparse covariance estimation** (only within-branch covariances, not across branches) reduces the matrix to block-diagonal.
- An **end-to-end coherent model** (Amazon-style) eliminates the post-hoc step by enforcing coherence in the loss, but requires re-training the whole model to change the hierarchy.
- **Approximate MinT** (shrinkage estimators or diagonal-only covariance) scales to millions of series with modest accuracy loss.

## Feature store and training/serving skew

Lag and rolling features computed at training time must match exactly what is available at serving time. A feature store with **point-in-time lookup** (where each row's feature is computed from data up to but not including the forecast origin) is the standard solution. Without it, subtle skew builds up: training used an actual $t - 1$ value while serving uses an estimated or delayed one, and offline metrics inflate while live accuracy degrades. This is the most common silent failure mode in large-scale forecasting systems.

## Bottlenecks and how to address them

| Bottleneck | First sign | Fix | Tradeoff |
|---|---|---|---|
| Nightly retrain overruns cadence | new week's run starts before last week's finishes | global model over all series (one train job, not 5M), or distributed training | loses some per-series nuance; adds infrastructure complexity |
| Feature assembly is the slowest step | preprocessing takes longer than model training | precompute lags and rolling statistics in a feature store; incremental updates only | storage cost; adds a freshness budget to maintain |
| Long-horizon accuracy decay makes week-12 forecasts unusable | error by horizon distance shows steep rise after week 4 | direct multi-horizon head instead of recursive auto-regression; or separate models per horizon bucket | more parameters; harder to train and maintain |
| Hierarchy incoherence surfaces in the optimizer | item forecasts sum to more than the store total, triggering allocation conflicts | MinT reconciliation post-hoc, or an end-to-end coherent model | covariance estimation cost; full model retrain for hierarchy changes |
| Cold-start SKUs have no lag features | new items show wide intervals and poor P50 accuracy | global model with learned category/attribute embeddings; hierarchical shrinkage toward parent | intervals stay wide until history accrues; embedding training adds warmup time |
| Non-stationary drift degrades forecasts silently | residual bias widens over weeks without triggering an alert | sliding-window retrain, change-point detection, residual monitoring with MASE/coverage thresholds | retrain cost; risk of over-fitting to a recent short window |
| ETA serving latency exceeds inline budget | p99 ETA response time exceeds quota | residual-on-routing-baseline model, precomputed feature lookups, linear-attention Transformer | model capacity limited; baseline quality sets the accuracy ceiling |

Two details worth pinning down. First, the hierarchy-incoherence fix, MinT
reconciliation (Wickramasuriya et al., 2019), needs the full residual covariance
across every series in the hierarchy; at millions of series that matrix is infeasible
to estimate or invert densely, so production systems fall back to a shrinkage or
diagonal-only covariance (approximate MinT), trading a little accuracy for a solve
that actually finishes. Second, recursive multi-step forecasting is what makes
long-horizon decay steep: each step feeds its own prediction back in as the next lag,
so errors compound multiplicatively across the horizon. A direct multi-horizon head
predicts each future step from the observed history only, which breaks that feedback
loop at the cost of more output parameters and no shared error correction between
adjacent horizons.
