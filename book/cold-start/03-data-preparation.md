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
| Creator signal | creator ID or creator embedding, creator quality score | cold items from a high-quality creator inherit a prior (a starting belief used before the item has data of its own) from that creator's track record |
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

Concretely, the backoff is a Bayesian blend. Let $n$ be the number of
interactions observed for the user and $\bar{r}_{\text{user}}$ their observed
rate; the estimate shrinks toward the coarsest level that has enough data:

$$\hat{\theta} = \frac{n\,\bar{r}_{\text{user}} + k\,\theta_{\text{prior}}}{n + k}, \qquad \theta_{\text{prior}} = \theta_{\text{district}} \rightarrow \theta_{\text{city}} \rightarrow \theta_{\text{region}}$$

where $k$ is the prior strength (how many interactions the prior is worth). At
$n = 0$ the estimate equals the prior exactly; as $n$ grows it blends toward
the user's own signal.

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

## Time-boxed items: cold start with no second chance

Standard item cold start is a race against time: the item is new, it accumulates
signal, and eventually it graduates. **Time-boxed items never graduate.** An event
next Saturday, a limited-time offer, a live stream, a seat on a specific flight, a
same-day delivery slot: by the time enough interaction data exists, the item is
worthless.

Three properties change the design.

**The item is always cold, so content features are not a bootstrap, they are the
model.** There is no "eventually the collaborative signal takes over" phase. Features
come from what the item *is* (category, price, venue, host, description embedding)
and from where and when it is (distance from the user, hours until it starts, day of
week, whether it falls on a weekend).

**Relevance decays on a clock, and not monotonically.** A concert three months out
is interesting, three days out is urgent, three hours out is unreachable for most
users, and one minute after it starts it is unrankable. That is not a decay curve
you can learn as a single feature: encode time-to-event explicitly, and expect the
optimal recommendation to depend on the interaction of time-to-event with distance
and with the user's planning horizon.

**Supply is finite and consumption is exclusive.** Recommending a sold-out event or
an unavailable slot is worse than a mediocre recommendation, because it costs the
user a click and the platform a slot. This pushes availability into a hard filter
before ranking rather than a feature inside it, and it makes the candidate set
change minute to minute.

| Property | Standard item cold start | Time-boxed items |
|---|---|---|
| Interaction data | Sparse, then grows | Sparse, then the item expires |
| What carries the model | Content features until the collaborative signal arrives | Content, time, and location features, permanently |
| Label volume | Grows with item age | Bounded by the item's lifetime |
| Failure mode | New items never get exposure | Items get exposure too late to convert |
| Candidate set | Slowly changing | Changes by the minute as items expire or sell out |
| Exploration cost | Recoverable; a bad impression today is a better model tomorrow | Partly unrecoverable; the item may be gone before the learning pays off |

The design consequences are concrete. Retrieval becomes **filter-first**: eligible by
time, location, and availability, and only then ranked, which keeps the candidate
set small enough that a heavier ranker is affordable. Features are computed at
request time relative to *now* rather than materialized (hours-until-start is not a
column you can precompute). And evaluation must be **time-aware**: a random train and
test split leaks the future, so backtest by time and score each request against the
items that were actually available at that moment, which is the same discipline as
[forecasting](../forecasting/).

Exploration also needs re-pricing. In a standard catalog, exploration buys
information you will use for months. On a perishable item it buys information that
expires with the item, so the budget should be spent on **item classes** (this venue,
this category, this price band) rather than on individual items, which is where the
learning transfers.

## Where the labels come from

Cold-start is defined by the absence of the cheap label, so knowing the three sources
and their biases is the whole game.

| Label source | What it gives you | The bias, and the fix |
|---|---|---|
| Implicit feedback (interaction logs) | Abundant and cheap once traffic exists, but nearly empty for a brand-new user or item | Exposure and selection bias, and worse, sparsity: a new arm gets almost no signal unless you show it. Log propensity and use a stochastic exploration policy so new arms accumulate unbiased reward. |
| Human raters | High-quality judgments that need no interaction history, so they work on day zero | Slow, costly, low volume. Use for the golden eval set and to calibrate a cheaper content-based model that scores cold items. |
| Targeted collection / synthesis | The cold-start workhorse: open datasets, content-feature augmentation, and a stronger model as a teacher to score items before any clicks exist | Not organic, so treat it as a warm-start prior that the exploration policy corrects once real feedback arrives. |

Because feedback is time-ordered, split train and test by time (train on earlier
cohorts, evaluate on later ones), never by a random shuffle, so a warm item's future
engagement cannot leak into the cold-start evaluation.
