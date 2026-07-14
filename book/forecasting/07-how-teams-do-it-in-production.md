# 7. How teams do it in production

Every large-scale forecasting system converges on the same skeleton: assemble history plus known-future covariates into features, fit a model that emits a probabilistic or quantile forecast, reconcile levels so they are coherent, hand the distribution to a decision step, and close the loop with a rolling-origin backtest. What actually differs between companies is three decisions: **the data structure** (plain series, a hierarchy that must reconcile, or a spatial graph), **whether the decision consumes a point estimate or a distribution**, and **where the latency constraint sits** (batch vs inline per request). Architecture is shared; the leverage is in those three choices.

```mermaid
flowchart LR
  HIST["historical series"] --> FEAT["features + calendar / covariates"]
  FEAT --> MODEL["forecast model (classical / global GBT / deep)"]
  MODEL --> DIST["probabilistic / quantile forecast"]
  DIST --> OPT["forecast-then-optimize decision"]
  OPT --> ACT["action (stock / position / ETA)"]
  ACT -.->|"realized outcome"| BT["rolling backtest"]
  BT -.->|"retrain"| MODEL
```

## Where the real designs diverge

| System | Model class | Point vs probabilistic | Decision it feeds | Structure | When this shape wins |
|---|---|---|---|---|---|
| Uber (intro stack) | Classical + GBT + deep (portfolio) | Both (prediction intervals stressed) | Driver positioning, capacity, marketing | Plain series | Broad portfolio of business series where no single model family dominates; pick per-problem |
| Uber (uncertainty) | Bayesian neural net | Probabilistic (model + misspecification + noise decomposed) | Capacity reserves, anomaly flagging | Plain series | High-stakes calls that need to know why the model is uncertain, not just how much |
| Uber (DeepETA) | Linear-attention Transformer residual on routing baseline | Point (calibrated, asymmetric loss) | ETA quote inline per request | Spatiotemporal | A strong routing baseline to correct under a tight global inline latency budget |
| Amazon (hierarchical) | Deep, end-to-end coherent probabilistic | Probabilistic, coherent across levels | Supply-chain and resource planning | Hierarchy | Many levels that must reconcile; end-to-end learning removes the post-hoc reconcile step |
| Google DeepMind (Maps ETA) | Graph neural network + temporal | Point | Google Maps routing / ETA | Spatiotemporal graph | Travel time diffuses across a connected road graph; geography carries signal a per-series model misses |
| Instacart (availability) | Layered general / trending / real-time | Probability of availability (0 to 1) | Item availability surfacing for customers and logistics | Plain series | Hundreds of millions of items under a tight cost-per-prediction budget; stratified cadence cuts cost 80 percent |
| Instacart (balance) | Unified supply vs demand engine | Probabilistic (supply and demand separately) | Shopper dispatch interventions, market balance | Plain series | Two-sided marketplace where the gap between supply and demand drives the action |
| Zalando (ZEOS) | LightGBM + MLForecast; Monte Carlo optimizer | Probabilistic (12-week distribution per SKU) | Replenishment optimization | Plain series | The optimizer needs the full distribution to set safety stock; GBT beats TFT on iteration speed and cost |
| Grab (supply-demand) | Geo-temporal ratio computation | Point ratios (supply/demand per cell) | Matching and spatial rebalancing | Spatiotemporal (geohash cells) | Matching where the supply-over-demand ratio per geo-cell is the signal; ratio plus absolute gap restores magnitude |
| Wayfair (Predicted Winners) | NN with LSTM feature extraction; distributional heads | Probabilistic (Bernoulli, Log-Normal, Negative Binomial) | Cold-start product surfacing and supplier partnerships | Plain series (but day-zero has no history) | Cold-start demand prediction from content embeddings before any sales history exists |
| Lyft (causal) | Causal DAG forecasting | Point under a causal model | Marketplace policy decisions under confounding | Plain series | Policy changes confound the observed metric; correlation misleads; causal model forecasts under intervention |
| Mercado Libre | Global LSTM, two separate models | Point (MAE, robust to zeros) | Inventory and replenishment planning | Plain series (item-level, global) | Observed sales are censored by stock; separate models for sales and latent demand reveal true interest after stockouts |

The core dividing line is the data structure (plain series, a hierarchy that must reconcile, or a spatial graph) crossed with whether the decision consumes a point estimate or a full distribution.

## The systems

- **Uber** [Forecasting at Uber: An Introduction](https://www.uber.com/blog/forecasting-introduction/): An overview of classical, ML, and deep learning forecasting with prediction intervals, validated via Omphalos (a chronological parallel backtesting framework). *(product design)*
- **Uber** [Engineering Uncertainty Estimation in Neural Networks for Time Series](https://www.uber.com/blog/neural-networks-uncertainty-estimation/): A Bayesian neural network decomposing model uncertainty, misspecification, and noise uncertainty into separate terms. *(eval bar)*
- **Uber** [DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/us/en/blog/deepeta-how-uber-predicts-arrival-times/): A linear-attention Transformer learning a residual on a routing baseline under global inline latency, with asymmetric Huber loss and quantile-bucketed embeddings. *(deployment)*
- **Amazon Science** [End-to-end learning of coherent probabilistic forecasts for hierarchical time series](https://www.amazon.science/publications/end-to-end-learning-of-coherent-probabilistic-forecasts-for-hierarchical-time-series): One neural network emitting coherent probabilistic forecasts across all hierarchy levels via a differentiable reconciliation layer. *(product design)*
- **Google DeepMind** [Traffic prediction with advanced Graph Neural Networks](https://deepmind.google/blog/traffic-prediction-with-advanced-graph-neural-networks/): GNNs over road Supersegments improving Google Maps ETA accuracy up to 50 percent, trained with MetaGradients to handle variable graph sizes. *(deployment)*
- **Instacart** [Building for Balance](https://company.instacart.com/how-its-made/building-for-balance): A unified engine forecasting shopper supply versus customer demand to guide interventions on the two-sided marketplace. *(product design)*
- **Instacart** [Modernizing real-time availability prediction for hundreds of millions of items](https://company.instacart.com/tech-innovation/how-instacart-modernized-the-prediction-of-real-time-availability-for-hundreds-of-millions-of-items-while-saving-costs): A hierarchical general, trending, and real-time model stack cutting cost approximately 80 percent via stratified scoring cadence. *(deployment)*
- **Zalando** [Building a dynamic inventory optimisation system](https://engineering.zalando.com/posts/2025/06/inventory-optimisation-system.html): Probabilistic LightGBM forecasts for millions of SKUs feeding a gradient-free Monte Carlo replenishment optimizer. *(product design)*
- **Grab** [Understanding Supply and Demand in Ride-hailing Through Data](https://engineering.grab.com/understanding-supply-demand-ride-hailing-data): Geo-temporal supply-demand ratios with distance-weighted neighbor assignment for matching and spatial rebalancing. *(eval bar)*
- **Lyft** [Causal Forecasting at Lyft (Part 1)](https://eng.lyft.com/causal-forecasting-at-lyft-part-1-14cca6ff3d6d): Causal-DAG-based forecasting of marketplace metrics for policy decisions under confounding. *(product design)*
- **Wayfair** [How Wayfair uses "Predicted Winners" Models to Accelerate Success for New Products](https://www.aboutwayfair.com/careers/tech-blog/how-wayfair-uses-predicted-winners-models-to-accelerate-success-for-new-products): Cold-start demand models predicting which new products will sell from image and text embeddings before any sales history exists. *(product design)*
- **Mercado Libre** [Marketplace Forecasting: Sales or Demand?](https://medium.com/mercadolibre-tech/global-time-series-forecasting-models-for-item-level-demand-and-sales-forecasts-in-our-marketplace-aee2956957ae): Separate global LSTM models for observed sales (censored by stock) and latent demand (unconstrained), revealing true interest after stockouts. *(eval bar)*
- **Ocado** [Finding the sweet spot](https://careers.ocadogroup.com/blogs/careers-blogs/our-technologies/finding-the-sweet-spot): A tiered stack from heuristics to deep sequence models balancing grocery availability against perishable waste per retailer. *(product design)*
- **Oda** [How we went from zero insight to predicting service time](https://medium.com/oda-product-tech/how-we-went-from-zero-insight-to-predicting-service-time-with-a-machine-learning-model-part-2-2-ad8b0c3e4838): LightGBM per-stop service-time prediction feeding grocery delivery route planning, with a 23 percent per-stop MAE improvement but only 10 percent route-level delay improvement because errors partly cancel over 30 stops. *(deployment)*
