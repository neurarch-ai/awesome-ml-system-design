# 7. How Teams Do It in Production

Every large platform runs the same skeleton: state a hypothesis, hash a
diversion unit into stable arms, log a pre-declared success metric next to
guardrails, squeeze variance, then make an explicit ship-or-hold call. All real
divergence lives in three choices: **how they reduce variance, how they contain
interference, and how they pull the ship trigger.** The architecture everyone
shares; the leverage is in the details.

## Where the real designs diverge

| System | Randomization unit | Variance reduction | Guardrail approach | Interference handling | When it wins | Watch out |
|---|---|---|---|---|---|---|
| Uber (XP) | Per user; flicker users excluded | CUPED (pre-period covariate) | App crash rate, trip frequency; sequential auto-pause | mSPRT sequential monitoring; flicker exclusion | Single-app product changes needing early looks without peeking inflation | Flicker users must be detected and removed; per-user split does not contain rider-driver marketplace spillover |
| Airbnb | Per user | Power gating (SE bound) before launch | Impact, Power, Stat-Sig-Negative guardrails; escalation review | Guardrails flag harmful tests pre-launch; auto-approve confidently safe ones | Stopping harmful or underpowered tests before they reach scale | Coverage-adjusted thresholds needed; over-escalation can hold real wins |
| Booking.com | Per user | Quality rules in Design phase | Experimentation quality (protocol adherence) as platform KPI | Quality checks catch broken or invalid experiments | Keeping a very large experiment program trustworthy end to end | Optimizing process adherence can diverge from business outcomes if rules are stale |
| Spotify | Per user | Power correction for G guardrails (beta star = beta / (G+1)) | Four metric roles: success, guardrail, deterioration, quality | Quality tests (SRM, pre-exposure bias) hard gate before reading | Ship calls balancing multiple competing metrics | Combining metric roles needs pre-set margins and weights; missed margin calibration masks regressions |
| LinkedIn | Per user (individual arm) vs per cluster (cluster arm) | CUPED; stratified cluster assignment via reLDG | Network-effect metrics flagged when delta between designs is significant | Dual-design test detects interference; egoClusters/ELEMENT measures it | Social and network features where treatment leaks across connected users | Cluster randomization cuts effective units; must apply CUPED and stratification to keep power |
| Lyft | Session, geo, or time (switchback) | Not the focus | Marketplace health metrics | Geo and time switchbacks contain marketplace spillover | Two-sided marketplaces where supply and demand shift under treatment | Coarse switchback windows add temporal variance; effective units are windows, not users |
| Netflix | Per user (both rankers blended into one list for interleaving) | Interleaving (~100x fewer subscribers than A/B to prune rankers) | Business metrics confirmed in follow-up A/B | Within-user comparison sidesteps cross-user leakage | Pruning many ranking algorithms fast when subscribers are scarce | Interleaving measures only ranking preference; A/B still needed for business metric and guardrails |

The core dividing line: per-user assignment is simplest and highest-power, but
leaks whenever treatment spills across users. Network and marketplace products
pay power for interference safety via cluster, switchback, or within-user
interleaving designs.

## The systems (first-party write-ups)

- **Uber** [Under the Hood of Uber's Experimentation Platform](https://www.uber.com/blog/xp/): XP with CUPED variance reduction, mSPRT sequential monitoring, flicker exclusion, and jackknife/block-bootstrap variance. The canonical reference for combining sequential testing with CUPED.

- **Airbnb** [Designing Experimentation Guardrails](https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669): Three guardrail types (Impact, Power, Stat-Sig-Negative), coverage-adjusted thresholds, and non-inferiority-based auto-approval. About 25 experiments per month escalated; 80% still launch after review.

- **Booking.com** [Experimentation quality as the main platform KPI](https://medium.com/booking-product/why-we-use-experimentation-quality-as-the-main-kpi-for-our-experimentation-platform-f4c1ce381b81): Protocol adherence across Design, Execution, and Shipping phases as the north-star metric, rolling up to team and department level.

- **Spotify** [Risk-Aware Product Decisions in A/B Tests with Multiple Metrics](https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics): Four metric roles with joint error-rate control; the beta-star power correction for multiple guardrails.

- **LinkedIn** [Detecting interference: an A/B test of A/B tests](https://www.linkedin.com/blog/engineering/ab-testing-experimentation/detecting-interference-an-a-b-test-of-a-b-tests): Dual-design (individual plus cluster) with reLDG balanced partitioning and CUPED to detect and bound network-effect interference.

- **Lyft** [Experimentation in a Ridesharing Marketplace](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-b39db027a66e): Statistical interference in two-sided markets; session, geo, and time-based switchback randomization as remedy.

- **Netflix** [Innovating faster on personalization using Interleaving](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55): Team-draft interleaving prunes ranking algorithms with roughly 100x fewer subscribers before a full A/B confirmation.

- **Netflix** [Reimagining Experimentation Analysis at Netflix](https://netflixtechblog.com/reimagining-experimentation-analysis-at-netflix-71356393af21): Modular analysis infrastructure letting scientists plug in custom metrics and causal models on top of a shared experiment backbone.

- **Kohavi, Tang, Xu** *Trustworthy Online Controlled Experiments* (Cambridge University Press): the canonical book-length reference on OEC choice, sample ratio mismatch, peeking, interference, and running experiments at scale.

- **Google** [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml): emphasizes measuring real online impact rather than trusting offline metrics; the perspective that motivates why this whole chapter exists.

For additional production case studies, the [Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design) indexes experimentation writeups from over 150 companies.
