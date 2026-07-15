# 7. How teams do it in production

Every system that ships exploration converges on the same four-field log
(context, action, propensity, reward) and the same two-stage funnel
(retrieval cuts millions to hundreds, then the exploration policy operates
over the candidate set). What actually differs is which exploration policy
they chose, how they bound the arm set, and how they run offline evaluation.
These are the decisions worth understanding from each writeup.

## Where the real designs diverge

| System | Exploration policy | Arm set | Cold-start approach | Off-policy eval | Why this shape |
|---|---|---|---|---|---|
| Netflix (artwork) | contextual bandit | small: artwork variants per title | not the primary problem | production infra for reward computation and logging | few arms per title; action space is bounded; cache-served at scale |
| Spotify (homepage) | epsilon-greedy exploration branch | content-type distribution, not raw items | not the primary problem | offline accuracy vs historical baseline | high-traffic homepage; explore rate must stay small and bounded |
| Spotify (podcasts) | pure-exploration best-arm ID, infinitely-armed | reservoir of new podcasts | new podcast = cold arm from reservoir | fixed-budget evaluation before feeding winners to exploit system | goal is finding broadly-appealing new shows, not cumulative reward; decoupled from exploit feed |
| Spotify (impatient bandits) | bandit over items, delayed reward | warm catalog | not the primary problem | Bayesian filter over partial observations; validated against held-out 2-month labels | true reward arrives weeks later; myopic proxy is insufficient |
| Yahoo (LinUCB) | LinUCB contextual bandit | dynamic article pool | feature-parameterized model gives cold articles uncertainty from features | replay on 33M events with logged uniform-random traffic | news article pool changes every few hours; collaborative filtering fails |
| Stitch Fix | Thompson sampling (Beta posterior) | small: content type variants, styling choices | not the primary problem | posterior draws; propensities from Thompson for IPS | bandits as a first-class experiment type; reuses existing experiment platform |
| Instacart | contextual bandit, arms are ranking strategies not items | small: 8 objective-blend strategies | feature-shared model scores never-seen products from features | IPS and doubly-robust before A/B | millions of products; per-product arms do not scale; make arms ranking strategies |
| DoorDash (cuisine filter) | multi-armed bandit with geo-hierarchy priors | cuisine categories | new user or district inherits geo-hierarchy prior | not detailed in writeup | cold-start on both user and geo dimension; priors backed off from district to city to region |
| Duolingo (recovering bandit) | sleeping and recovering bandit | notification time slots | not the primary problem | offline simulation with recency decay model | reward regenerates over time; rested arm should be re-explored |
| Google (neural-linear) | neural-linear bandit | large catalog, billions of users | linear head gives cold items uncertainty from learned features | long-horizon experiment design measuring corpus growth, not session clicks | standard A/B hides exploration's value; corpus growth is the real signal |

The core dividing line: teams with small arm sets (Netflix artworks, Stitch
Fix variants) can maintain per-arm posteriors. Teams with large catalogs
(Instacart, Google) must either make the arms ranking strategies, or use a
parametric model shared across arms so a never-seen item still gets
uncertainty from its features.

## The systems (first-party writeups)

- **Netflix** [Artwork Personalization at Netflix](https://netflixtechblog.com/artwork-personalization-c589f074ad76): contextual bandits pick per-member title artwork; small action space, cache-served at scale. *(product design)*
- **Netflix** [Infra for Contextual Bandits and Reinforcement Learning](https://netflixtechblog.com/ml-platform-meetup-infra-for-contextual-bandits-and-reinforcement-learning-4a90305948ef): production infra for reward computation, logging, and offline policy evaluation of bandits. *(deployment)*
- **Spotify** [Identifying New Podcasts with a Pure-Exploration Infinitely-Armed Bandit](https://research.atspotify.com/publications/identifying-new-podcasts-with-high-general-appeal-using-a-pure-exploration-infinitely-armed-bandit-strategy): a pure-exploration bandit surfaces broadly-appealing new podcasts without popularity bias. The mechanism: it draws candidate shows from a reservoir of new podcasts and spends a fixed exploration budget to identify the high-appeal ones, decoupled from the exploit feed that then serves the winners. *(who it serves)*
- **Spotify** [Calibrated Recommendations with Contextual Bandits on the Homepage](https://research.atspotify.com/2025/9/calibrated-recommendations-with-contextual-bandits-on-spotify-homepage): a contextual bandit balances the music, podcast, audiobook mix per user context. *(product design)*
- **Spotify** [Impatient Bandits: Optimizing for the Long-Term Without Delay](https://research.atspotify.com/publications/impatient-bandits-optimizing-for-the-long-term-without-delay): a Bayesian filter fuses partial observations so the bandit can act on delayed reward without waiting. *(eval bar)*
- **Yahoo** [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146): the LinUCB news bandit plus offline replay evaluation on 33 million events. *(eval bar)*
- **Stitch Fix** [Multi-Armed Bandits and the Experimentation Platform](https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/): Thompson-sampling bandits as a first-class experiment type with a reward service. *(deployment)*
- **Instacart** [Contextual Bandit Models in Large Action Spaces](https://company.instacart.com/tech-innovation/using-contextual-bandit-models-in-large-action-spaces-at-instacart): contextual bandits for product recs when the catalog action space is very large. The mechanism: the arms are a small set of objective-blend ranking strategies rather than individual products, and a feature-shared model scores never-seen products from their features. *(deployment)*
- **DoorDash** [Personalized Cuisine Filter](https://careersatdoordash.com/blog/personalized-cuisine-filter/): a multi-armed bandit with geo-hierarchy priors handles new-user and new-district cold start. *(who it serves)*
- **Duolingo** [A Sleeping, Recovering Bandit for Optimizing Recurring Notifications](https://research.duolingo.com/papers/yancey.kdd20.pdf): a recovering bandit picks the daily reminder with a recency penalty, lifting retention. *(product design)*
- **Google** [Long-Term Value of Exploration](https://arxiv.org/abs/2305.07764): neural-linear bandit exploration grows the content corpus; long-horizon experiment design makes the gain visible. *(eval bar)*
