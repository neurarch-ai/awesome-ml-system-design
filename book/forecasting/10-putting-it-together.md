# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed, and it shows how the
same decisions flip when the constraints change. It closes with the smallest
runnable backtest, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing model families before forecasting a single
series. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Libraries
change yearly, but the interface of each stage (frame the output, engineer
features, baseline, model, reconcile, backtest, retrain, serve) does not, so
pick per stage by interface and treat any specific tool as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Output framing | Quantile forecast (P10/P50/P90) via pinball loss | Downstream decision consumes only a central estimate (a plain ETA quote): point forecast | [2](02-frame-as-ml-task.md) |
| Baseline models | Seasonal-naive plus a classical per-series fit (ETS or Theta) on a sample; a zero-shot foundation model as the instant bar | Never skip the baseline; every candidate must beat it in the same backtest | [4](04-model-development.md), [5](05-evaluation.md) |
| Feature engineering | Lags at the seasonal period, 28/90-day rolling stats, cyclic calendar encoding, known-future promo/holiday/price with lead windows | A lag falls inside the forecast horizon: per-horizon feature sets, only lags >= h | [3](03-data-preparation.md) |
| Production model | Global GBT (LightGBM class) with quantile heads on a differenced or deflated target | Few series with long clean history: classical per-series. Long horizons, rich covariates, or embedding-based cold-start: deep (DeepAR, TFT, PatchTST) | [4](04-model-development.md) |
| Hierarchy / reconciliation | Approximate MinT (shrinkage or block-diagonal covariance) after the forecast cycle | Only one level is ever consumed: skip. Leaf forecasts trustworthy and coherence is all you need: bottom-up | [4](04-model-development.md), [6](06-serving-and-scaling.md) |
| Backtesting protocol | Rolling-origin, 3 to 5 origins, scored per horizon step with MASE, WQL, and empirical coverage, business-weighted | Never. Build the backtest before tuning anything | [3](03-data-preparation.md), [5](05-evaluation.md) |
| Retraining cadence | Retrain every forecast cycle (weekly) on a sliding window, with MASE and coverage drift monitors | Stable series and stable error: stretch the cadence. Promo or regime shift detected: trigger an off-cycle retrain | [6](06-serving-and-scaling.md) |
| Serving | Batch inference into a database before the optimizer run; feature store with point-in-time lookup | The forecast is consumed inline per request (ETA): residual on a strong baseline under a latency budget | [6](06-serving-and-scaling.md) |

The backtest row is the one beginners skip and regret: without a rolling-origin
harness, every feature and model decision is scored on a split that leaks the
future, and you cannot tell whether a change helped. One afternoon of building
the harness pays for itself the first time you swap a component.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): roughly
5 million SKUs across a few thousand stores, a 12-week horizon at weekly
granularity, probabilistic output feeding an inventory replenishment optimizer,
a 3:1 underage-to-overage cost ratio, a 90 percent service-level target, and
calendar, holidays, planned promotions, and price known ahead of the origin.
Here is the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Output | Quantiles P10, P50, P75, P90 per series per horizon step, trained with pinball loss | The 3:1 cost ratio sets the P75 stocking quantile (c_u / (c_u + c_o) = 3/4); the 90 percent service target is checked at P90; a mean is a broken optimizer input |
| Multi-step strategy | Direct multi-horizon head, one output per horizon distance | Error compounds badly over 12 recursive steps; direct heads break the feedback loop |
| Model | One global LightGBM with quantile heads on lag, rolling, calendar, and promo features; differenced target | Billions of leaf series rule out per-series fits; trees cannot extrapolate a trend, so the trend is removed before the tree and added back after |
| Cold start | Attribute and category priors with hierarchical shrinkage toward the parent; intervals kept wide | Lag features are zero at day zero; the category average is the warm prior until history accrues |
| Features | Only lags >= 12 weeks, 28/90-day rolling stats closed before the origin, cyclic week-of-year, promo/holiday/price lead windows | Point-in-time discipline per horizon; promotions are the strongest single demand driver |
| Reconciliation | Approximate MinT with shrinkage covariance, run once per cycle after all forecasts land | Dense covariance is infeasible at millions of series; incoherent levels trigger allocation conflicts in the optimizer |
| Evaluation | Rolling-origin backtest, 5 origins, scored per horizon week with MASE, WQL, and coverage; business-weighted aggregate | Random splits leak the future; MAPE is undefined on the zero-demand weeks that dominate the leaf level |
| Serving | Weekly batch: forecast, reconcile, quality-gate, load before the Monday replenishment run | No per-request latency exists; the constraint is throughput within the weekly cadence |
| Optimizer handoff | Newsvendor stocking at the P75 critical fractile, with coverage verified at P90 | The forecast is the intermediate; the ordering decision is the product |

**Series count.** 5 million SKUs times a few thousand stores is billions of
leaf combinations on paper ([section 1](01-clarifying-requirements.md)); even
after pruning stores that never carry an item, the forecastable set stays in the
hundreds of millions (Illustrative). At even 1 second per classical fit, 1
billion per-series fits would consume about 31 CPU-years every weekly cycle
(Illustrative), which is why the scale decision is not a preference: one global
training job over the pooled panel is the only tractable path, and it also wins
on sparse leaves because they borrow strength from related series.

**Backtest windows.** With 3 years of weekly history, 156 weeks (Illustrative),
place 5 roll origins 4 weeks apart at weeks 128 through 144: every origin trains
on at least two full annual cycles, so the lag-52 features and the in-sample
seasonal-naive MASE denominator are both defined, and the latest origin still
leaves 12 realized weeks to score against. That is 5 origins times 12 horizon
steps, 60 scored cells per series per candidate model, reported by horizon
distance because week-12 error is structurally worse than week-1 error and an
aggregate hides where the model degrades ([section 5](05-evaluation.md)).

**The error metric is a cost statement.** MASE below 1 means the model beats
the seasonal-naive forecast it must displace. Pinball loss at the P75 penalizes
under-prediction 3 times as hard as over-prediction (tau / (1 - tau) = 0.75 /
0.25), which is exactly the stated 3:1 lost-sale-to-holding cost ratio, so
minimizing pinball at the critical fractile is minimizing expected inventory
cost, not a proxy for it. Calibration closes the loop: if the P90 forecast is
exceeded 40 percent of the time instead of 10, the optimizer is silently
stocking to something like a P60 and the stockout rate blows through the
service target while WQL still looks fine ([section 5](05-evaluation.md)).

**Throughput.** The whole batch, forecast, reconcile, quality-check, load, must
land before the optimizer runs; with PySpark preprocessing and distributed GBT
inference this is tractable in under 2 hours at this scale
([section 6](06-serving-and-scaling.md)), which leaves headroom in a weekly
cadence for a rerun when a data feed arrives late.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: coverage drift (empirical P90 coverage
sliding below nominal on volatile categories means the intervals are too
narrow and stockouts arrive before any point metric moves), offline-live gap at
short horizons (live week-1 error far above backtest week-1 error means a
feature or a per-series scaling constant is peeking past the origin, the
normalization leak from [section 8](08-interview-qa.md) being the usual
culprit), and hierarchy incoherence surfacing downstream (item forecasts
summing past the store total and triggering allocation conflicts in the
optimizer, which typically means the reconciliation step silently skipped a
branch after a hierarchy change).

## The same techniques under different constraints

The review question that matters in practice is not "which model family is
best" but "which model family is best under my constraints." Here is the same
skeleton built three times. Only the marketplace column is the build above; the
other two keep the identical stage interfaces and swap nearly every
implementation choice.

| | Single-warehouse brand | Marketplace replenishment (this chapter) | Inline ETA |
|---|---|---|---|
| Series / cadence | ~200 SKUs, one location; monthly planning | Millions of SKUs, thousands of stores; weekly batch | One prediction per request, spatiotemporal; millions of requests per day |
| Output | Quantiles from classical fits or conformal intervals | P10/P50/P75/P90 per series per week | Calibrated point estimate with asymmetric loss |
| Model | Per-series ETS or Theta; Prophet if holidays dominate | Global LightGBM, quantile heads, differenced target | Residual on a routing baseline, linear-attention Transformer class |
| Features | The series' own history plus holiday flags | Lags >= horizon, rolling stats, cyclic calendar, promo/price lead windows | Precomputed spatiotemporal lookups; no time to assemble history per request |
| Reconciliation | None; one level is consumed | Approximate MinT across item, store, region | None; there is no hierarchy |
| Backtest | Rolling-origin on a laptop, minutes | Rolling-origin, 5 origins x 12 weeks, distributed scoring job | Time-based holdout plus online calibration monitoring |
| Retraining | Monthly, or when a fit visibly drifts | Weekly, sliding window, drift-triggered off-cycle | Continuous or daily; the baseline absorbs most drift |
| What would be over-engineering | A global model, a feature store, MinT, any deep model | Per-series retrains, a full dense MinT covariance | Batch anything, hierarchies, quantile heads the quote never consumes |

Two lessons fall out. First, the single-warehouse column is mostly deletions:
at 200 series a per-series classical fit is the strong option
([section 4](04-model-development.md)), the whole backtest runs on a laptop,
and every piece of scale machinery is dead weight. Second, the ETA column shows
the latency constraint flipping the architecture: when the forecast is consumed
inline, the heavy model moves out of the request path into a baseline plus a
learned residual, the output collapses to a calibrated point because no
safety-stock decision exists, and the feature store becomes precomputed lookups
([section 6](06-serving-and-scaling.md), [section 7](07-how-teams-do-it-in-production.md)).

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Series count | Model family | Dozens to hundreds: per-series classical. Thousands and up: global GBT. Millions: global only; per-series retrains are infeasible |
| Downstream decision | Output format | Optimizer with asymmetric costs: quantiles. Central estimate only: point forecast. Monte Carlo policy: sampled paths from a likelihood |
| Cost asymmetry | Stocking quantile | Critical fractile = c_u / (c_u + c_o); a 3:1 ratio means stock to the P75, and emit whatever quantiles the service target is checked at |
| Horizon length | Lag availability and multi-step strategy | Only lags >= h are real features at horizon h; beyond a few steps, direct multi-horizon beats recursive because error compounds |
| Intermittent zero-heavy demand | Metric and likelihood | MAPE and RMSE are broken; use MASE and pinball, and a negative binomial, Tweedie, or Croston-style treatment |
| Multiple hierarchy levels consumed | Reconciliation | One level: skip. Trusted leaves: bottom-up. Otherwise approximate MinT; dense MinT does not scale past thousands of series |
| Latency position | Serving architecture | Batch: heavy models, post-hoc reconciliation, pre-run quality gates. Inline: residual on a strong baseline, precomputed features |
| New-item rate | Cold-start machinery | High launch rate: attribute priors, category shrinkage, wide intervals until history accrues; a foundation model is a credible zero-shot fallback |
| Promo intensity and drift | Covariates and retrain cadence | Promo-heavy demand makes known-future covariates mandatory; drift monitors (MASE, coverage) trigger retrains instead of a fixed calendar |

## The smallest runnable backtest

The chapter's core mechanism is not a model; it is the evaluation harness that
decides whether any model earns deployment. So here is a complete
rolling-origin backtest in one file with zero installs: a seeded weekly demand
series with trend, annual seasonality, and noise, three forecasting methods,
MASE scored per fold exactly as [section 5](05-evaluation.md) defines it, and
then the same three methods scored the wrong way, one random split that ignores
time order. Every production component is swapped for the smallest thing with
the same interface: the panel of millions of series becomes one list, the
global GBT becomes a drift adjustment, and the distributed scoring job becomes
a loop. The shape is the lesson; every section of this chapter upgrades one
piece of this file.

```python
"""A rolling-origin backtest against three baselines, runnable with no installs."""
import math, random

random.seed(2)
N, SEASON, H = 208, 52, 12          # 4 years of weekly demand, 12-week horizon
y = [60 + 0.15 * t                                  # slow upward trend
     + 25 * math.sin(2 * math.pi * t / SEASON)      # annual seasonal cycle
     + random.gauss(0, 2)                           # demand noise
     for t in range(N)]

def naive(train, h):                # hold the last observed value flat
    return [train[-1]] * h

def snaive(train, h):               # same week last year
    return [train[-SEASON + i] for i in range(h)]

def snaive_trend(train, h):         # same week last year + estimated drift
    drift = (train[-1] - train[0]) / (len(train) - 1)
    return [train[-SEASON + i] + drift * SEASON for i in range(h)]

METHODS = [("naive last-value", naive), ("seasonal-naive", snaive),
           ("seasonal-naive+trend", snaive_trend)]

def mae(actual, pred):
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)

print("== right way: rolling-origin backtest, 3 folds x 12-week horizon ==")
mean_mase = {name: [] for name, _ in METHODS}
for fold, origin in enumerate([N - 3 * H, N - 2 * H, N - H], start=1):
    train, actual = y[:origin], y[origin:origin + H]
    scale = mae(train[SEASON:], train[:-SEASON])   # in-sample seasonal-naive MAE
    for name, forecast in METHODS:
        m = mae(actual, forecast(train, H))
        mean_mase[name].append(m / scale)
        print(f"  fold {fold}  {name:<21} MAE {m:6.2f}   MASE {m / scale:5.2f}")
for name, v in mean_mase.items():
    print(f"  mean MASE  {name:<21} {sum(v) / len(v):5.2f}")

print("\n== wrong way: one random 80/20 split, time order ignored ==")
test = sorted(random.sample(range(SEASON, N), (N - SEASON) // 5))
drift = (y[-1] - y[0]) / (N - 1)    # even the trend estimate has seen the future
wrong = {"naive last-value":     [y[t - 1] for t in test],
         "seasonal-naive":       [y[t - SEASON] for t in test],
         "seasonal-naive+trend": [y[t - SEASON] + drift * SEASON for t in test]}
actual = [y[t] for t in test]
for name, preds in wrong.items():
    print(f"  {name:<21} MAE {mae(actual, preds):6.2f}")
print("  (every 'forecast' above reads a neighbor the production system"
      "\n   would not have observed at a 12-week lead time)")
```

Run it and the two halves demonstrate the chapter's two central claims in about
fifty lines. The honest rolling-origin half ranks the methods the way the
theory says it must: naive last-value is worst in every fold (mean MASE 1.53,
because holding one value flat misses both the seasonal swing and the trend
over 12 weeks), seasonal-naive lands almost exactly at MASE 1.0 (0.97, which
is the definition of the metric: it is the baseline the denominator measures),
and the trend adjustment is the only method that earns deployment (mean MASE
0.63). The wrong-way half then reverses the verdict: the shuffled split scores
naive last-value at MAE 2.54, three times better-looking than seasonal-naive
at 8.20 and statistically tied with the best method, because ignoring time
order hands every held-out week its true neighboring observation, a value the
production system will not have at a 12-week lead time. The worst method looks
deployable under the broken protocol; that is how leaking evaluations ship bad
models. Swap the list for a panel of millions of series, the drift adjustment
for a global quantile GBT, MAE for pinball and WQL with a coverage check, and
the three-origin loop for a distributed scoring job, and you have rebuilt this
chapter's evaluation spine.
