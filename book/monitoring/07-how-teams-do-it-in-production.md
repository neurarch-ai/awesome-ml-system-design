# 7. How teams do it in production

Every large ML system converges on the same skeleton: log the prediction and the
served features, run cheap label-free checks immediately, wait for labels to
confirm the slow performance signal, and tier the response by severity. What
actually differs between companies is two decisions: **how far they go to close
the loop automatically**, and **how they handle label delay and seasonality**.
The architecture everyone shares; the leverage is in the closing and the
thresholds.

## Where real designs diverge

| System | Drift detected | Detection method | Alerting | Action | When it wins | Watch out |
|---|---|---|---|---|---|---|
| Evidently AI | feature + prediction drift | PSI, KS, chi-square, 20+ tests | report or dashboard driven | feeds retrain decision (tooling) | stand up drift metrics fast; rich per-feature reports | detects but does not act; you still wire alerting and retrain around it |
| Uber D3 | column-level data drift | Spark column stats vs Prophet dynamic bands | PagerDuty on breach | detect + alert; manual response | seasonal data; 100,000+ monitors; $0.01 per dataset | stops at detect and page; Prophet baselines need tuning |
| Uber MES | composite quality score across lifecycle | SLA-style scoring across prototype, train, deploy, prediction | scorecard breach | quality gating and prioritization | standardizing a quality bar across many models and teams | composite score can mask which specific signal moved |
| Uber deploy-safety | feature drift + online-offline skew | statistical tests, schema validation, shadow testing | can block promotion | auto-rollback, gradual rollout, shadow | high-stakes promotions where a bad model must never reach users | shadow traffic and gradual rollout cost real infra |
| Lyft | score + performance drift | feature validation, anomaly and drift detection | anomaly-based alerts | retrain trigger | one framework spanning features, scores, and performance | anomaly detection needs careful calibration |
| Netflix | prediction + data drift | logging, monitoring, explainability layer | observability dashboards | diagnose, then retrain | high-stakes domains (payments) that must explain why a prediction moved | surfaces diagnosis but a human still closes the loop |
| Shopify | feature drift, concept drift | distribution monitoring (fraud example) | monitoring surfaces | retrain on drift | adversarial, concept-drift-prone domains where retraining is routine | narrower single-signal focus |

The dividing line is simple: **detection tooling stops at the page; a platform
closes the loop with auto-rollback or triggered retraining.** Detection is
enough for low-stakes models; platforms are worth the infra cost for high-stakes
ones where human response time matters.

## The systems (first-party write-ups)

- **Chip Huyen** [Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html): the clearest single read on covariate vs concept drift, label delay, and what to actually monitor. The four-layer hierarchy and the window-length framing come from here. *(foundations)*
- **Google** [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml): the production discipline, including watching for silent failures in the data feeding the model. Found that 60 of 96 failures were data and pipeline bugs, not model quality. *(discipline)*
- **Evidently AI** [open-source drift detection](https://github.com/evidentlyai/evidently): PSI, KS, chi-square, and 20+ other tests in a runnable library. Reports and Test Suites for point-in-time and continuous monitoring. *(tooling)*
- **"Hidden Technical Debt in Machine Learning Systems"** (Sculley et al., NeurIPS 2015): the classic paper on why ML systems rot in production: entanglement, feedback loops, the CACE principle. *(foundations)*
- **Uber** [D3: an automated system to detect data drifts](https://www.uber.com/blog/d3-an-automated-system-to-detect-data-drifts/): Prophet-based dynamic thresholds across 300+ datasets; 20x faster time-to-detect (45 days to 2), 100x query cost reduction. *(deployment)*
- **Uber** [Model Excellence Scores: enhancing ML quality at scale](https://www.uber.com/en-GB/blog/enhancing-the-quality-of-machine-learning-systems-at-scale/): SLA-style scoring across lifecycle phases; 60% improvement in prediction performance reported. *(eval bar)*
- **Uber** [Raising the Bar on ML Model Deployment Safety](https://www.uber.com/us/en/blog/raising-the-bar-on-ml-model-deployment-safety/): Shadow testing, auto-rollback, real-time data-quality checks, and a readiness score that gates promotion. *(deployment)*
- **Lyft** [Full-Spectrum ML Model Monitoring at Lyft](https://eng.lyft.com/full-spectrum-ml-model-monitoring-at-lyft-a4cdaf828e8f): Feature validation, score monitoring, anomaly and performance-drift detection in one framework. *(eval bar)*
- **Netflix** [ML Observability: transparency for payments and beyond](https://netflixtechblog.com/ml-observability-bring-transparency-to-payments-and-beyond-33073e260a38): A logging, monitoring, and explaining framework; surfaces why a prediction moved, not just that it did. *(deployment)*
- **Shopify** [Shopify's Playbook for Scaling Machine Learning](https://shopify.engineering/shopify-playbook-scaling-machine-learning): Scaling playbook with a mobile-fraud concept-drift example; Airflow-scheduled continuous monitoring. *(who it serves)*

For the full comparison (math, quadrant plot): [tools/comparisons/11.md](../../tools/comparisons/11.md).
For per-system teardowns: [tools/teardowns/11.md](../../tools/teardowns/11.md).
