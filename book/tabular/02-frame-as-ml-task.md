# 2. Framing it as an ML task

## The four modeling families

Before touching a dataset, pick the modeling family from the decision the score
must answer. Getting this wrong is the single most common design error, and it is
invisible in an offline AUC.

```mermaid
flowchart TD
  Q{"What must the<br/>score answer?"}
  Q -->|"Rank risk or value"| CLS["Classification or regression<br/>(approve/decline, LTV point estimate)"]
  Q -->|"WHEN will it happen"| SURV["Survival analysis<br/>(churn timing, time-to-default)"]
  Q -->|"WHETHER to intervene"| UP["Uplift / causal model<br/>(discount, retention offer, incentive)"]
  CLS --> CAL{"Does absolute value<br/>set money?"}
  SURV --> CAL
  UP --> CAL
  CAL -->|"yes"| CALIBRATE["Calibrate: Platt / isotonic"]
  CAL -->|"no, ranks only"| THRESH["Threshold / risk tiers"]
```

**Classification** answers "will this customer default / churn / convert in a
fixed window?" The output is a probability. Use this when the horizon is fixed, the
label is clean, and ranking is all you need (Pinterest 14-day churn, Gousto 4-week
subscription churn, PayPal pipeline propensity).

**Regression** answers "what is the expected value of this customer or listing?"
The output is a point estimate. Use this for LTV, home value, or demand forecasting
when the target is a continuous number (Airbnb home value, Expedia CLV, Asos
markdown).

**Survival analysis** answers "when will this event happen, for customers who
have not yet experienced it?" The output is a survival curve, one per customer,
giving the probability they are still active (or still-paying) at every horizon.
Use this when the timing matters and some rows are censored (the event has not
happened yet, so you know only that the customer survived at least this long, not the
final outcome): still-active
subscription customers, applicants whose default window has not yet resolved
(Nubank, Block Square).

**Uplift / causal models** answer "whose behavior changes if I act?" The output is
a conditional average treatment effect (CATE): the difference in outcome if the
customer receives an intervention versus not. Use this for any pricing, discount,
or retention offer decision, where spending budget on people who would have acted
anyway (sure things) or people who never will (lost causes) is pure waste (Wayfair,
Uber, Gojek).

## Specifying the input and output

| Surface | Input | Output |
|---|---|---|
| Credit risk scoring | applicant features at decision time (income, bureau tradelines, application attributes) | calibrated default probability; adverse-action reason codes |
| Subscription churn | customer behavior at snapshot date (usage, spend, pauses, trends) | churn probability within a horizon, or a survival curve |
| Customer LTV | booking or purchase history, demographics, product mix | expected discounted revenue over a horizon |
| Incentive / discount targeting | customer features plus treatment indicator | CATE (uplift per customer); optimizer allocates budget |

## When to use which framing

| Reach for | When | Instead of |
|---|---|---|
| Classification (fixed-window binary) | clean horizon, label resolves completely, ranking is enough | survival machinery you do not need |
| Regression (point estimate) | target is continuous, no censoring, point value is consumed directly | a binary label that throws away the magnitude |
| Survival analysis | WHEN the event happens matters and some rows are censored (still-active accounts) | a fixed-window binary that discards censored rows and loses timing |
| Uplift / CATE | the question is WHETHER an intervention changes behavior (pricing, discount, retention) | a churn or propensity model that targets sure things and lost causes too |
| Calibrated probability | the absolute number feeds a threshold, a limit formula, or a price | a raw ranking score that only needs to sort correctly |

The choice is driven by the decision downstream, not by what is easiest to label.
Framing churn as classification when you need survival, or framing an intervention
question as propensity when you need uplift, is the most expensive early mistake
in a tabular design. Pin the decision first, then pick the row.

**Provenance.** The default classification and regression engines are the boosting libraries XGBoost (Chen and Guestrin, 2016), LightGBM (Microsoft, 2017), and CatBoost (Yandex, 2017). The calibrated-probability framing, the one that must hold when a score feeds a threshold, limit, or price, relies on Platt scaling (Platt, 1999) (fitting a simple sigmoid that maps raw scores to true probabilities) or isotonic regression (a more flexible non-decreasing step-function version of the same idea) rather than any change to the model.

**Tools for each framing.** Classification and regression on tabular data are
dominated by the gradient-boosting libraries XGBoost, LightGBM, and CatBoost.
Survival analysis: lifelines and scikit-survival, or XGBoost's AFT objective for a
boosted survival model. Uplift/CATE: EconML (Microsoft) and CausalML (Uber), or the
generalized random forest in grf. Calibration: scikit-learn's
CalibratedClassifierCV (Platt scaling or isotonic regression).

**Worked example.** A subscription business wants to reduce churn. If the question
is "who cancels in the next 30 days" and every account resolves inside that window,
frame it as calibrated binary classification (LightGBM followed by isotonic
calibration) because the probability feeds a save-offer threshold, so the absolute
number has to be trustworthy, not just the ranking. But if leadership actually means
"who would a discount change the mind of," that is an uplift/CATE question (EconML or
CausalML): a plain churn model would spend the retention budget on customers who
were going to stay anyway and on those who will leave regardless. And if many
accounts are still active at analysis time (censored), survival analysis
(scikit-survival) keeps that timing information a fixed 30-day binary would throw
away.
