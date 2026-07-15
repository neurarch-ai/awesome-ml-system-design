# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Every question below either removes work, changes the model family, or changes the output format. Notice that framing the output as a distribution rather than a point appears as soon as you ask what the forecast actually feeds.

---

**Candidate:** What decision does this forecast feed? Replenishment, driver positioning, ETA, or something else?

**Interviewer:** Inventory replenishment for a marketplace: millions of SKUs across thousands of stores. The planner decides how much to order and when.

**Candidate:** Then the forecast needs to be a distribution, not a mean. A replenishment optimizer stocks to a quantile (the demand level below which a given fraction of outcomes fall; the P90 is exceeded only about 10 percent of the time) set by the service-level target. If I give you only the mean, safety stock is uncomputable and you stock out roughly half the time. Is that the frame you want?

**Interviewer:** Yes. Probabilistic output is required. Continue.

**Candidate:** What horizon and granularity? Next-day per store, or a multi-week horizon per SKU?

**Interviewer:** A 12-week horizon, weekly granularity, per SKU per store. That is the lead-time window for ordering.

**Candidate:** How many series? Millions of related series, or a handful?

**Interviewer:** About 5 million SKUs times a few thousand stores, so potentially billions of series at the most granular level, though many have sparse history.

**Candidate:** At that scale, fitting one model per series is infeasible. We need a global model (one model trained jointly across all series, instead of a separate model fit to each series) that learns across all series and borrows strength. I will come back to which family suits this, but the scale already rules out classical per-series models. What covariates (external input variables such as price, promotions, or holidays that help explain demand) are available, and are they known in the future?

**Interviewer:** Calendar and holidays are known. Planned promotions and prices are known a few weeks out. Weather is available but it is itself a forecast.

**Candidate:** Good. Calendar and holidays go straight into the model as future covariates. Promotions and price are the single strongest demand driver and must be included with lead and lag windows. Weather adds noise from its own forecast error, so we include it but track its contribution separately.

**Candidate:** What is the cost asymmetry between overstock and understock?

**Interviewer:** Understock is lost sale plus churn: call it 3x the holding cost. The target service level is 90 percent.

**Candidate:** That gives us two things to pin down. First, the cost ratio sets the critical fractile directly: c_u / (c_u + c_o) = 3 / (3 + 1) = 0.75, so the newsvendor optimizer stocks to the P75 of demand, not the mean. Second, the business has separately named a 90 percent service-level target. Those are two different numbers: the 3:1 cost ratio implies a P75 stocking quantile, while the P90 service-level target is an independent business specification. We should emit at minimum P10, P50, P75, and P90 quantiles and verify empirical coverage on each.

---

Let us summarize the problem statement. **We are asked to design a demand forecasting system for a large marketplace.** The input is a history of per-SKU-per-store sales plus known-future covariates (calendar, holidays, planned promotions, price). The output is a probabilistic forecast (at minimum P10, P50, and P90 quantiles) for a 12-week horizon at weekly granularity, for roughly 5 million SKUs. The forecast feeds an inventory replenishment optimizer, not a human reader.

Three consequences fall out of this immediately:

- **The output must be a distribution.** The 3:1 underage-to-overage cost ratio sets the critical-fractile stocking quantile at P75 (c_u / (c_u + c_o) = 3/4); the business also named a 90 percent service-level target as a separate specification. Either way, a single point estimate is a broken input to the optimizer and is not acceptable.
- **The scale rules out one model per series.** At millions of series, a global model that shares weights across all series is the only tractable path. Classical per-series methods (ARIMA, ETS) are a fast baseline for small-scale validation, not the production path.
- **Evaluation must be a rolling-origin backtest (backtesting is walk-forward testing on historical data: train up to a cutoff date, forecast forward, then slide the cutoff and repeat), scored on proper metrics.** Random train/test splits leak future demand. MAPE is undefined on zero-demand SKUs, which are common. The correct gate is MASE and pinball/WQL on a walk-forward backtest at the 12-week horizon, plus coverage checks to confirm the P90 is actually exceeded about 10 percent of the time.
