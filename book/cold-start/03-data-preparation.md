# 3. Data preparation

## Content features for cold items

The root cause of item cold start is using an ID embedding as the item
representation. A just-uploaded item has an untrained ID embedding: it maps
to an effectively random vector, so retrieval and ranking treat it as noise
and it never surfaces. The fix is a content tower: the item is represented
as a function of its features rather than its ID.

| Feature category | Examples | Why it helps cold items |
|---|---|---|
| Taxonomy and category | top-level category, subcategories, tags | places the item near semantically similar items from day zero |
| Creator signal | creator ID or creator embedding, creator quality score | cold items from a high-quality creator inherit a prior from that creator's track record |
| Text content | title embedding, description tokens, hashtags | semantic similarity to warm items that share topics or vocabulary |
| Visual content | thumbnail embedding from a pretrained vision model | places visual items near visually similar warm items |
| Structured attributes | price, language, duration, format | hard constraints a user filter can match before any neural scoring |
| Freshness | upload timestamp, age in hours | lets the system apply an explicit recency bonus or freshness bucket |

A new item runs through the content tower the moment it is uploaded, gets a
vector, and is inserted into the ANN index. No training cycle required.

## Context features for cold users

A new user has no interaction history. Their user-tower vector must come
entirely from what is available at request time.

| Feature category | Examples | Why it helps cold users |
|---|---|---|
| Acquisition channel | organic search, referral, paid campaign, social | strong prior on intent and taste |
| Device and platform | mobile vs web, OS, screen size | correlates with content format and session length |
| Locale and geography | country, language, coarse city or region | powers geo-hierarchy priors (district to city to global) |
| Time of day and day of week | morning, weekend, holiday | shifts content-type priors even for warm users |
| Onboarding signals | declared interests from signup wizard, category selections | the most direct intent signal on record |
| Session signals | first query, first click in this session | even a single in-session interaction provides strong signal |

The DoorDash cuisine-filter design is the clean example: a new user in a
new district has no history, so the system backs off to district-level priors,
then city-level, then regional. As interactions arrive, the prior blends
toward the specific user. Geo-hierarchy priors are a content feature for
the user, just as text embeddings are content features for an item.

## Feedback logging: the four-field contract

Every served impression must be logged with four fields. Missing any one of
them breaks either the reward model or offline policy evaluation.

| Field | What it is | Why it is load-bearing |
|---|---|---|
| Context `x` | full feature vector at serve time: user, item, session | the input the policy acted on; needed to replay or re-evaluate |
| Action `a` | which item or strategy was chosen | identifies the arm for reward attribution |
| Propensity `pi0(a given x)` | the probability the serving policy assigned to `a` | the denominator in every importance-sampling estimator; wrong or missing propensity silently corrupts all offline evaluation |
| Reward `r` | the observed outcome, possibly delayed | the signal that updates the reward model |

The propensity field is where most teams go wrong. A deterministic argmax
policy assigns propensity 1 to the chosen action, which makes the
importance-sampling ratio either 1 or infinite and breaks all offline
estimators. You must use a stochastic policy (epsilon-greedy, Thompson,
or UCB with logged randomness) and log the exact probability the policy
used, not a reconstructed one.

## Choosing the reward: immediate proxy vs long-term signal

| Reward | Latency | Risk |
|---|---|---|
| Click | immediate | clickbait; does not predict retention |
| Dwell time | seconds to minutes | games by autoplay, misleading for short content |
| Completion | minutes to hours | slower to observe; more aligned with content quality |
| Return visit or retention | days | the true objective; too slow to use directly for bandit updates |
| Learned early proxy | configurable | the Spotify Impatient Bandits approach: a Bayesian filter fuses partial short-term observations into a probabilistic belief about the eventual delayed reward, so the bandit can act without waiting |

The design choice is: pick a proxy that predicts long-term value, model
the delay if possible, and hold out long-horizon cohorts to detect proxy drift.

With data and features designed, the next section builds the exploration
policies that use them.
