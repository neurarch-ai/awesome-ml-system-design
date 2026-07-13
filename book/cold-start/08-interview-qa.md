# 8. Interview Q&A

The questions an interviewer actually asks about cold start and exploration,
grouped by how they are used. The commonly-missed ones are where interviews
are won or lost.

## Commonly asked

**Q: What is the cold-start problem and what is the standard fix?**

A: A model that represents items (or users) by learned ID embeddings has
nothing to say about an entity that has never been seen before. The embedding
is untrained, so retrieval and ranking treat the entity as noise. The fix is
to represent the entity by its content and metadata instead of its ID, using
a content tower: the item vector is a function of category, text, and visual
features. A new item gets a vector the moment it is uploaded and inherits a
location in embedding space from similar warm items. This makes it retrievable
and rankable on day zero without waiting for interactions.

**Q: Why does pure exploitation cause the feed to ossify?**

A: A greedy policy only collects labels for what it already ranks highly.
Items it ranks low get zero fresh impressions, so their reward estimates
are frozen at whatever they were when the system last showed them. If tastes
shift or an item improves, the model has no way to learn this. The corpus
narrows to the items the model was already confident about, which may be
stale or unrepresentative. The only escape is to deliberately show uncertain
items, so the label distribution stays broad enough that the model can
correct itself.

**Q: What is epsilon-greedy and when would you use it?**

A: With probability epsilon, serve a uniformly random item. With probability
1 minus epsilon, serve the argmax. The explore branch has a clean, trivially
computable propensity, making it the baseline of choice when you need a
stochastic policy with known propensities for off-policy evaluation. It is
the right choice on a high-traffic surface where the explore rate must stay
very small. The weakness is that exploration is blind: it spends as many
impressions on obviously-bad items as on promising-but-uncertain ones.

**Q: How do you evaluate a new bandit policy without a live A/B?**

A: Off-policy evaluation. If the log contains uniformly-random exploration
traffic, use replay: replay the stream against the new policy, score only
events where the new policy's choice matches the logged action, and compute
the win rate. The resulting estimate is unbiased. If the log contains known
propensities from any stochastic policy (not necessarily uniform), use
inverse-propensity scoring (IPS) or doubly-robust (DR). All three require
that propensities were logged at serve time and match the policy that
actually ran. A deterministic argmax with no logged randomness breaks all
three.

**Q: How do you handle cold start for a brand-new user?**

A: The user-side fix is symmetric to the item-side fix: use context as
the metadata. On a brand-new user's first request, the user tower keys off
whatever is available: device, locale, time of day, acquisition channel,
and any onboarding preferences they declared. That places the user in a
plausible region of the embedding space on the very first request. As
interactions arrive, the user's specific signal blends in and dominates.
The DoorDash cuisine-filter design is a clean example: new users and new
districts start with geo-hierarchy priors (district to city to region) and
personalize as data accumulates.

## Tricky (the follow-ups that separate candidates)

**Q: Exploration lowers short-term engagement. How do you justify it?**

A: You frame it as an investment with a long-horizon payoff. Exploration
costs a small number of clicks now. In return it: (1) discovers good content
that the greedy policy would never promote, growing the effective catalog;
(2) keeps reward estimates fresh, so the model can correct itself when tastes
shift; (3) breaks the ossification loop, preventing the corpus from collapsing
onto a shrinking set. The Google long-term-value-of-exploration work makes
this explicit: measuring only session-level engagement makes exploration look
neutral or negative. Measuring corpus growth and long-horizon retention shows
the payoff. You should name this explicitly and propose measuring both a
short-term engagement metric and a long-horizon retention or diversity metric,
with the understanding that a small short-term dip is acceptable if the
long-horizon metric rises.

**Q: How would you make a bandit work at a catalog of millions of items?**

A: You cannot enumerate millions of items as arms with per-arm posteriors;
the memory and update cost is infeasible. Three moves:
First, a two-stage funnel: retrieval (ANN over content tower vectors) cuts
millions to hundreds, and the bandit only operates over the candidate set,
never the whole catalog.
Second, a parametric, feature-shared reward model: the reward is a function
of features, not an ID lookup. A never-seen item gets an uncertainty estimate
from its features by generalization from similar items. This is how LinUCB
and neural-linear bandits scale.
Third, make the arms ranking strategies instead of raw items (the Instacart
approach): the bandit picks among a small set of ranking objectives or formula
variants, not among millions of individual products.

**Q: Why does the reward proxy matter, and how do you handle a delayed reward?**

A: Optimizing an immediate proxy (click) that does not predict long-term
value produces clickbait: the model learns to generate curiosity gaps or
misleading thumbnails, which drives the immediate signal but destroys
long-term retention. The resolution: choose a proxy that predicts long-term
value (completion rate, saves, or a learned proxy trained to predict
retention), and model the delay if needed. The Spotify Impatient Bandits
design is the reference: a Bayesian filter fuses partial short-term
observations into a belief about the eventual delayed reward, so the bandit
can act immediately without waiting weeks for the true signal to arrive.
You still need occasional full long-term labels to anchor the filter.

**Q: When would you pick Thompson sampling over UCB?**

A: For most production systems, Thompson sampling is the better default.
It gives naturally stochastic actions, so logged propensities are clean and
off-policy evaluation is tractable. It is also more forgiving to tune:
UCB requires choosing the alpha coefficient, and the right value depends
on the reward scale and action space. Thompson sampling adapts: if the
posterior is wide (uncertain arm), it wins draws often; if narrow (confident
arm), it wins draws only when genuinely good. UCB is preferable when infra
requires a deterministic choice (some caches and logging systems expect a
fixed action per context hash), or when the uncertainty bonus needs to be
a specific closed form for latency reasons.

## Commonly answered wrong

**Q: Should exploration be baked into the ranking model's objective function?**

A: No. The ranking model's job is to produce an accurate point estimate of
the reward. Exploration is a separate decision layer that reads that estimate
and an uncertainty estimate and decides whether to exploit or explore. Baking
exploration into the ranking loss conflates two concerns: the ranker gets
worse at estimating reward (because it is simultaneously trying to encourage
exploration), and the exploration policy loses the ability to be tuned or
disabled independently. Keep them separate: the ranker estimates, the
exploration layer decides.

**Q: Can you just boost new items manually to solve cold start?**

A: Manual boosts are a blunt instrument with serious problems. They require
human curation to determine which items deserve a boost and by how much.
They do not generalize: every new item needs a separate decision. They
introduce a quality floor problem: a poorly-produced new item that gets
boosted can harm the user experience. And they do not solve the model problem
underneath: after the boost expires, the item has whatever signal it earned
during a potentially non-representative promo period. The right answer is
to build a content tower that places new items in embedding space automatically
from metadata, so they earn impressions proportional to their content quality
from day zero, with no manual intervention.

**Q: Is it enough to log only the chosen action for off-policy evaluation?**

A: No. You must log the propensity (the probability the policy assigned to
the chosen action). Without the propensity, importance-sampling estimators
cannot reweight the log to simulate a different policy. If you use a
deterministic argmax (propensity always 1), there is no randomness to exploit
and every off-policy estimator degenerates. A common mistake is to add
exploration (epsilon-greedy or Thompson) but forget to log the propensity,
leaving the team able to explore but unable to evaluate the exploration policy
offline. Log the propensity at serve time, version it with the policy model,
and test it by replaying known-propensity traffic and verifying the estimator
recovers the expected outcome.

**Q: Isn't exploration just A/B testing with extra steps?**

A: No; they differ on the goal and the mechanism. A/B testing allocates
traffic to variants statically and waits for significance before acting.
Exploration shifts traffic toward better arms as evidence accumulates, without
waiting. A/B testing finds the best variant in a small, pre-specified set.
Exploration scales to large or dynamic arm sets (millions of items) by
sharing parameters across arms. A/B testing has no notion of uncertainty per
item; exploration uses per-arm uncertainty to direct where to spend impressions.
Stitch Fix's design of bandits-as-a-first-class-experiment-type is the cleanest
synthesis: the experiment platform handles logging and assignment, and a
Thompson-sampling allocator replaces the static traffic split with an adaptive
one, reusing all the existing infrastructure while adding the exploration
benefit.
