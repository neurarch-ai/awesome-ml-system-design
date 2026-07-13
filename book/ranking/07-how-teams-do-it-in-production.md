# 7. How teams do it in production

Every large ranking system converges on the same skeleton: retrieve candidates,
assemble features (user once, item and cross per candidate), batch-score through
a DNN or GBDT, optionally calibrate, and sort by a utility that blends per-
objective scores. What actually differs is three decisions: **how interactions
are modeled**, **how many objectives are trained jointly**, and **whether
calibration is treated as first-class**. The architecture everyone shares; the
leverage is in those three choices.

## Where the real designs diverge

| System | Interaction model | Objectives | Calibration | Why this shape |
|---|---|---|---|---|
| Google Wide and Deep | Wide linear plus deep MLP | Single (install) | Implicit | Memorization for frequent rules, generalization for the long tail; shipped in TensorFlow for Google Play |
| Meta DLRM | Explicit pairwise dot products after embedding tables | Single or multi | Not central | Many sparse ids; second-order crosses modeled structurally; embedding tables sharded for memory |
| Instacart pCTR | Wide-and-deep with FM wide side | Single (pCTR) | Explicit per-surface | Consolidating five surface-specific XGBoost models into one; 64-77% calibration improvement |
| Pinterest home feed | Shared-bottom DNN with per-action heads | Multi (click, long-click, close-up, repin) | Logistic regression per head, 80+ features | Several engagement actions to calibrate independently and combine; utility weights tunable without retraining |
| Pinterest related products | Shared body, four engagement heads | Multi | Downsampling correction | Collapsing distinct actions into one binary label lost signal; four heads beat the binary classifier |
| LinkedIn home feed | Passive and active towers over XGBoost leaf features | Multi (click, dwell, comment, reshare) | Not central | Tree leaf indices bridged into DNN; dense-tensor encoding cut gRPC overhead by about two-thirds |
| Airbnb search | GBDT then neural net (pairwise LambdaRank) | Single (booking) | Not central | GBDT baseline first, then neural; listing-id embeddings overfit (365 bookings/year limit) |
| DoorDash ads | Shared DNN body, click and conversion heads | Multi (pCTR, pCVR) | Calibrated for auction | Conversion labels arrive delayed; multi-task DNN captures user behavior tree models cannot |
| Spotify ads (CAMoE) | MMoE with DCN-v2 inside each expert; modality-specific gates | Multi (audio CTR, video CTR) | ECE-monitored, drives auction pricing | Modality imbalance suppressed video without adaptive loss masking; ECE drift means over-bid |
| Pinterest lightweight | XGBoost with blended objective | Single (engagement+funnel) | Not central | Cheap early-funnel stage between retrieval and full ranker; logs at serving not frontend |
| Wayfair | Post-hoc monotonic calibration over ranking scores | Ranking scores in | Time-aware purchase probability | Raw scores order well but are not probabilities; seasonality and holidays require time-aware curve |
| Walmart search | Two-round ranker: product-type matching then relevance+engagement re-ranker | Multi | Not central | Pure engagement ranking drifted toward popular-but-irrelevant items; 4.5% relevance lift |
| Snap ads | User tower plus ad tower; DCN-v2; MMoE/PLE; hourly warm-start | Multi (install, purchase, sign-up) | Platt or isotonic, drives auction | User tower computed once per request; ad ids churn constantly; calibration is auction revenue |

The core dividing line: systems with sparse id-heavy feature sets (Meta, Snap,
Spotify) invest in structured second-order interactions (DLRM, DCN-v2). Systems
that need to optimize a list quality metric directly (Airbnb, Yelp) use
LambdaMART. Systems with multiple engagement objectives (Pinterest, LinkedIn,
DoorDash, Spotify, Snap) share a body across tasks. Calibration is only first-
class when a score leaves pure sorting and feeds an auction, a price, or a
cross-task utility blend.

## The systems (first-party write-ups)

- **Google** [Wide and Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792): joint wide linear (memorization) plus deep net (generalization) for Google Play ranking.
- **Meta** [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091): dense MLP plus sparse embedding tables with explicit pairwise feature interactions, sharded for scale.
- **Instacart** [One Model to Serve Them All](https://company.instacart.com/how-its-made/one-model-to-serve-them-all-how-instacart-deployed-a-single-deep-learning-pctr-model-for-multiple-surfaces-with-improved-operations-and-performance-along-the-way): consolidating per-surface XGBoost into one wide-and-deep pCTR model; calibration and ops wins.
- **Pinterest** [Multi-task Learning and Calibration for Utility-based Home Feed Ranking](https://medium.com/pinterest-engineering/multi-task-learning-and-calibration-for-utility-based-home-feed-ranking-64087a7bcbad): per-action heads with logistic calibration; utility weights kept outside the loss.
- **Pinterest** [Multi-task Learning for Related Products](https://medium.com/pinterest-engineering/multi-task-learning-for-related-products-recommendations-at-pinterest-62684f631c12): four engagement heads beat a binary classifier; tune utility weights post-training.
- **LinkedIn** [Homepage Feed Multi-task Learning](https://www.linkedin.com/blog/engineering/feed/homepage-feed-multi-task-learning-using-tensorflow): jointly trains passive and active objectives; XGBoost leaf indices as DNN inputs.
- **Airbnb** [Applying Deep Learning to Airbnb Search](https://medium.com/airbnb-engineering/applying-deep-learning-to-airbnb-search-7ebd7230891f): the journey from GBDT to neural-network ranking; listing-id embeddings overfit at booking sparsity.
- **DoorDash** [Deep Learning for Ads Conversion in Last-mile Delivery](https://arxiv.org/abs/2502.10514): homepage ads moving from trees to multi-task DNNs; conversion labels arrive delayed.
- **Spotify** [Modality-aware Multi-task Learning for Ad Targeting (CAMoE)](https://research.atspotify.com/2025/8/modality-aware-multi-task-learning-to-optimize-ad-targeting-at-scale): MMoE with DCN-v2 and adaptive loss masking for modality imbalance; ECE monitored live.
- **Pinterest** [Improving Recommended Pins with Lightweight Ranking](https://medium.com/pinterest-engineering/improving-the-quality-of-recommended-pins-with-lightweight-ranking-8ff5477b20e3): XGBoost between retrieval and full ranker; blended engagement plus funnel-efficiency objective wins.
- **Wayfair** [Time Informed Calibration](https://www.aboutwayfair.com/careers/tech-blog/time-informed-calibration): monotonic post-hoc calibration with Prophet seasonality for 300-plus models.
- **Walmart** [Improving Walmart Search](https://medium.com/walmartglobaltech/improving-walmart-search-to-help-our-customers-save-time-e9fcd1f03e94): two-round ranker balancing relevance and engagement; 4.5% relevance lift validated by editorial evaluation.
- **Snap** [Machine Learning for Snapchat Ad Ranking](https://eng.snap.com/machine-learning-snap-ad-ranking): four-stage funnel; user tower computed once per request; MMoE/PLE; Platt or isotonic calibration drives auction.
- **Asos** [Transforming Recommendations at Asos](https://medium.com/asos-techblog/transforming-recommendations-at-asos-254b95c6a07a): transformer sequence ranker over interaction history; over 20% offline lift versus matrix factorization.
- **Yelp** [Beyond Matrix Factorization](https://engineeringblog.yelp.com/2022/04/beyond-matrix-factorization-using-hybrid-features-for-user-business-recommendations.html): XGBoost LambdaMART with hybrid content and interaction features; user-business cosine similarity the strongest single feature; doubled user coverage.
