# Numbers to know

The quantities you should be able to produce from memory in an interview, with the
formula that generates them so you can re-derive rather than recall. Each row links
to the chapter that develops it properly.

Order-of-magnitude figures for illustration, not benchmarks. The habit that matters
is stating the formula, plugging in, and saying what would change the answer.

## Experiment design

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Sample size per arm | $n \approx \dfrac{2\sigma^{2}(z_{1-\alpha/2} + z_{1-\beta})^{2}}{\text{MDE}^{2}}$ | [experimentation](experimentation/03-sizing-and-power.md) |
| The constant at 5 percent and 80 percent power | $(1.96 + 0.84)^2 \approx 7.85$ | [experimentation](experimentation/03-sizing-and-power.md) |
| Halving the MDE | costs 4x the users, because sample size goes as $1/\text{MDE}^2$ | [experimentation](experimentation/03-sizing-and-power.md) |
| CUPED variance reduction | $\text{Var} \cdot (1-\rho^{2})$; a correlation of 0.6 to 0.8 removes 35 to 65 percent of variance | [experimentation](experimentation/03-sizing-and-power.md) |
| Proportion interval | $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$ | [experimentation](experimentation/04-analysis.md) |
| Ratio metric variance | delta method, because the ratio of means is not the mean of ratios | [experimentation](experimentation/04-analysis.md) |
| Clustered variance | cluster at the randomization unit, not the event; effective n is closer to the unit count | [experimentation](experimentation/04-analysis.md) |
| Peeking | fixed-horizon tests looked at daily inflate the false-positive rate well above the nominal 5 percent; use a sequential test | [experimentation](experimentation/04-analysis.md) |
| Multiple metrics | control the false discovery rate (Benjamini-Hochberg) across the family you report | [experimentation](experimentation/05-pitfalls.md) |
| Sample-ratio mismatch | a chi-square on the assignment split; a failing SRM invalidates the readout, it does not "roughly balance out" | [experimentation](experimentation/05-pitfalls.md) |

## Offline metrics

| Quantity | Formula or figure | Chapter |
|---|---|---|
| AUC | probability a random positive scores above a random negative; 0.5 is random, and it is threshold-free | [ranking](ranking/05-evaluation.md) |
| NDCG@k | $\sum_i \dfrac{r_i}{\log_2(i+1)}$ normalized by the ideal ordering | [ranking](ranking/05-evaluation.md) |
| F1 | harmonic mean of precision and recall, so it punishes a lopsided pair | [content moderation](content-moderation/05-evaluation.md) |
| Expected calibration error | binned average of the gap between confidence and accuracy; 0 is perfectly calibrated | [ranking](ranking/05-evaluation.md) |
| Negative downsampling correction | $p = \dfrac{q}{q + (1-q)/w}$ for downsampling rate $w$; AUC hides the shift, every price shows it | [ads CTR](ads-ctr/) |
| Offline delta significance | paired per query, bootstrapped by resampling queries rather than rows | [ranking](ranking/05-evaluation.md) |
| Label noise floor | score the same set against two independent annotation passes; that gap is the smallest delta you can discuss | [ranking](ranking/05-evaluation.md) |

## Serving

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Little's law | $\bar{Q} = \lambda \cdot \bar{W}$: queue length equals arrival rate times wait | [realtime serving](realtime-serving/03-batching-and-throughput.md) |
| Fleet sizing | replicas $\approx$ (arrival rate x service time) / target utilization, with headroom for the tail | [realtime serving](realtime-serving/03-batching-and-throughput.md) |
| Why p99 and not the mean | with many calls per page view, the user experiences the tail, not the average | [realtime serving](realtime-serving/02-the-serving-problem.md) |
| Batching tradeoff | larger batches raise throughput and add queueing delay; the knee is where p99 starts climbing | [realtime serving](realtime-serving/03-batching-and-throughput.md) |
| Retrieval stage budget | thousands of candidates from ANN in single-digit milliseconds, then a heavier ranker over a hundred | [candidate retrieval](candidate-retrieval/) |
| Re-ranking budget | a cross-encoder or heavy model over the top 50 to 100, never over the corpus | [search ranking](search-ranking/) |
| Embedding index size | vectors x dimension x bytes per component; product quantization is the main lever | [embeddings](embeddings/) |
| Feature freshness | a stale count feature degrades quality faster than stale weights do | [feature store](feature-store/) |

## Training and data

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Retraining cadence | derive it from a staleness curve: train to T, evaluate at T+1, T+7, T+30, retrain where decay crosses the smallest shippable lift | [realtime serving](realtime-serving/04-deployment-strategies.md) |
| Decay by domain | adversarial (fraud, abuse) hours to days; catalog churn days; preference drift weeks; physical processes weeks to months | [realtime serving](realtime-serving/04-deployment-strategies.md) |
| Class imbalance | fraud and CTR positives are often well under 1 percent, which is why AUC and PR-AUC diverge and why calibration needs its own check | [fraud detection](fraud-detection/) |
| Delayed labels | conversions arrive hours or days later, so a fresh window is biased unless the delay is modelled | [ads CTR](ads-ctr/) |
| Training-serving skew | the most common silent failure in deployed rankers; point-in-time correctness is the fix | [feature store](feature-store/) |
| Position bias | logged clicks are confounded by rank; correct it or your model learns the old policy | [ranking](ranking/) |

## The five sentences worth memorizing

1. **Sample size goes as $1/\text{MDE}^2$.** Halving the effect you want to detect
   quadruples the users, which is why MDE is declared before the test, not after.
2. **Cluster the variance at the randomization unit.** Analyzing events when you
   randomized users makes every test look significant.
3. **A positive offline metric is a pre-gate, not a ship decision.** The seam
   between offline and online is where training-serving skew and position bias live.
4. **Calibration is separate from ranking.** A model can order perfectly and price
   terribly, and only one of those shows up in AUC.
5. **The user experiences p99.** Averages are for capacity planning; tails are for
   product decisions.
