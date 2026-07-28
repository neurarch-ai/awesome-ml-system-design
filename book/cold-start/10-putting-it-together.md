# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable exploration loop, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing bandit papers before serving a single
exploratory impression. Skip that. The stack below is a sane default for a
first production build; each row names when to deviate and which section
explains why. Policies and libraries change, but the interface of each stage
(represent, retrieve, estimate, explore, log, evaluate) does not, so pick per
stage by interface and treat any specific algorithm as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Cold-item representation | Content tower over category, creator, text, and thumbnail; upsert into the ANN index at upload | Items are long-lived and metadata is thin or gameable: hybrid ID-plus-content | [3](03-data-preparation.md), [4](04-model-development.md) |
| Cold-user onboarding | Context features (acquisition channel, device, locale, time) plus declared interests, with hierarchical prior backoff | No onboarding wizard exists: lean on session signals; even one in-session click is strong | [3](03-data-preparation.md) |
| Exploration policy | Thompson sampling; a neural-linear head once per-arm posteriors stop scaling | Infra demands a deterministic action per context: UCB plus a small epsilon perturbation | [4](04-model-development.md) |
| Exploration budget | A few percent on the discovery feed behind a content-tower quality floor; near-zero on checkout and notifications | Discovery is the product goal itself: split out a pure-exploration bucket (best-arm identification) | [6](06-serving-and-scaling.md), [2](02-frame-as-ml-task.md) |
| Fallback ladder | Popularity floor, then geo or segment prior, then content-tower personalization, then earned estimates | The popularity loop is the disease you were hired to cure: keep the floor a floor, never a driver | [4](04-model-development.md), [3](03-data-preparation.md) |
| Logging | Four fields at serve time (context, action, propensity, reward), versioned with the policy | Never. A wrong propensity silently corrupts every downstream evaluation | [3](03-data-preparation.md), [5](05-evaluation.md) |
| Evaluation | Replay on a small uniform-random bucket; IPS and doubly-robust on general logs; online: new-item share, diversity, long-horizon retention | Cannot afford a random bucket: SNIPS or doubly-robust on Thompson propensities | [5](05-evaluation.md) |

The last two rows are the ones beginners skip and regret: teams add exploration
but forget the propensity, and end up able to explore yet unable to evaluate
the exploration offline. The four-field log costs one schema change before
launch and is nearly unrecoverable after.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
catalog of millions of items, thousands of new listings ingested per day,
about 20% of impressions going to items less than 24 hours old, metadata of
category, creator, free-text description, and thumbnail, click available
immediately with completion and retention arriving hours or days later, an
existing experiment platform to reuse, and guardrails that let the main feed
absorb exploration cost while checkout and notifications cannot. Here is the
whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Item representation | Content tower over category, creator embedding, text, and thumbnail; vector upserted into the ANN index at upload | 20% of impressions go to day-old items; an untrained ID embedding makes every one of them noise |
| User cold start | Context features plus onboarding interests, hierarchical prior backoff (district to city to region) | Signup spikes must see a plausible feed on the first request, before any interaction exists |
| Funnel | Two-stage: ANN retrieval cuts millions to a few hundred; the policy operates only on the candidate set | Per-arm anything over millions of items is infeasible in memory and update cost |
| Reward model | Neural-linear: shared deep encoder, Bayesian linear head | A never-seen listing gets uncertainty from its features; the closed-form bonus fits a tens-of-ms budget |
| Exploration policy | Thompson sampling on the linear head | Stochastic propensities come free; no alpha exchange-rate tuning between count units and reward units |
| Budget and floor | ~5% of feed impressions (Illustrative), content-tower quality screen; epsilon near zero on checkout and notifications | The stated guardrail: the feed absorbs exploration cost, sensitive surfaces cannot |
| Warm-up | Guaranteed per-item impression budget served to well-matched users from the content-tower neighborhood | Triage dead versus viable listings quickly, then let earned estimates take over |
| Reward | Click as the immediate signal; completion blended in when it lands; long-horizon holdout cohorts | Click is available now, completion hours later; the holdout catches clickbait drift |
| Logging | Four-field contract, propensity versioned with the policy that ran | One wrong field breaks replay, IPS, and doubly-robust all at once |
| Offline eval | 1% uniform-random bucket (Illustrative) for replay; IPS plus doubly-robust on general logs | Replay on random traffic is unbiased; doubly-robust hedges a bad reward model or bad propensities |
| Launch metric | New-item impression share, corpus diversity, 7- and 30-day retention; session CTR as a guardrail, not the judge | Exploration structurally loses on session CTR; its payoff lands in future sessions and other users |

**New-item inflow and warm-up spend.** Take 5,000 new listings per day
(Illustrative; the interviewer said "thousands") with a warm-up budget of 200
impressions each: 1M guaranteed exploratory impressions per day. On a surface
serving 50M impressions per day (Illustrative), that is 2% of traffic spent on
warm-up alone. The chapter's 20% share for day-old items is not all granted:
the guaranteed 200 are the floor, and the rest of a young item's impressions
are earned through its exploration-adjusted score. The warm-up spend fits
inside the ~5% exploration budget with room left for re-exploring stale warm
items.

**Exploration budget and its engagement cost.** Suppose exploit-path CTR is 4%
and content-screened exploratory impressions land around 2% (Illustrative).
At a 5% budget the blended rate is 0.95 x 4% + 0.05 x 2% = 3.9%, a 0.1-point
absolute dip, 2.5% relative. Uniform-random exploration at ~1% CTR would cost
0.15 points; the content-tower quality floor is what halves the bill. This is
the number to defend in the launch review, and [section 5](05-evaluation.md)
says how: against corpus diversity and long-horizon retention, never against
session CTR alone.

**Time-to-first-signal.** Retrievability is minutes: content-tower embedding
plus an online ANN upsert, no training cycle ([section 6](06-serving-and-scaling.md)).
Statistical signal is slower and the warm-up budget is sized for triage, not
ranking. At a true rate of 4%, 200 impressions yield about 8 clicks and a Beta
posterior with a standard deviation near 1.4 points: enough to separate dead
listings (zero or one click) from viable ones, not enough to rank 4% against
5%. Resolving a 1-point gap takes roughly 1,600 impressions, so only survivors
of triage keep earning them, which is exactly the annealing Thompson sampling
provides for free.

**What breaks in month one.** Three failure modes dominate early operations,
so wire their signals before launch: new-item impression share falling (either
the ANN upsert path silently degraded to batch cadence and fresh items are
dark, or exploration collapsed because the epsilon floor or prior variance was
tuned away); off-policy estimates turning implausibly good after a retrain
(propensity mismatch, the logged value no longer matches the policy that ran;
the fix is the policy-versioned propensity log and a replay test on random
traffic); and session metrics rising while completion falls (reward-proxy
gaming, clickbait climbing the click objective; the long-horizon holdout
cohort is the alarm that fires).

## The same techniques under different constraints

The review question that matters in practice is not "which bandit is best" but
"which bandit is best under my constraints." Here is the same explore-exploit
machinery built three times. Only the middle column is the build above; the
other two keep the identical stage interfaces and swap nearly every
implementation choice.

| | Artwork or variant picker | UGC marketplace feed (this chapter) | Notification timing |
|---|---|---|---|
| Arm set | Dozens of variants per decision (Netflix artwork, Stitch Fix class) | Millions of items, funneled to hundreds of candidates | A handful of time slots per user (Duolingo class) |
| Item churn | Slow; variants live for months | Thousands of new arms per day; cold start is the steady state | None; the arms never change |
| Cold-start representation | None needed; every arm accumulates its own history fast | Content tower plus online-upsert ANN; hybrid ID-plus-content for long-lived listings | None; priors per slot suffice |
| Exploration policy | Per-arm Beta Thompson sampling, two counters per arm | Neural-linear Thompson; uncertainty from features, not history | Sleeping and recovering bandit; a rested arm regains value |
| Exploration budget | Generous; a bad artwork impression is cheap | ~5% behind a quality floor on the feed; near-zero on sensitive surfaces | Near-zero tolerance; one bad notification burns trust and an unsubscribe is forever |
| Reward | Click or play, immediate | Click now, completion blended late, retention cohorts held out | Open and next-day retention, delayed and recovering over time |
| Offline eval | Posterior draws give propensities; IPS on the experiment platform's logs | Replay on a 1% random bucket plus IPS and doubly-robust | Offline simulation with a recency-decay model |
| What would be over-engineering | Content tower, ANN index, neural-linear head | Per-arm posteriors over raw items; manual boost curation | Contextual neural model, content tower, any ANN infra |

Two lessons fall out. First, the artwork column is mostly deletions: with
dozens of long-lived arms, per-arm Beta counters are exact, cheap, and remove
the entire representation problem, which is why [section 7](07-how-teams-do-it-in-production.md)
shows small-arm-set teams shipping bandits as a first-class experiment type on
existing platforms. Second, the notification column shows the guardrail
flipping from a tuning knob to the design driver: when a bad exploratory
impression is nearly unrecoverable, the budget goes to near-zero, the arm set
stays tiny, and the interesting modeling moves into the reward (delay and
recovery), not the policy.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any algorithms.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Arm-set size | Policy family | Dozens: per-arm Beta posteriors. Large or shifting: feature-shared LinUCB or neural-linear. Millions with weak features: make the arms ranking strategies |
| Item churn rate | Index infrastructure | Cold start as steady state: online-upsert ANN is mandatory; batch rebuild cycles leave new items dark for hours |
| Reward delay | Proxy choice | Immediate click plus a long-horizon holdout when delay is hours; a learned early proxy (Impatient-Bandits style) when delay is weeks |
| Surface sensitivity | Budget and quality floor | Discovery feed: a few percent behind a content screen. Checkout and notifications: epsilon near zero, hard floor, greedy behind it |
| Off-policy eval on the roadmap | Policy stochasticity | Thompson gives propensities natively; UCB needs an epsilon perturbation; fund a small uniform-random bucket if you want replay |
| Metadata richness | Cold representation | Rich: content tower. Long-lived items: hybrid ID-plus-content so earned signal can override the prior. Thin: popularity floor and earn signal fast |
| Traffic volume | Explore rate and learning speed | High traffic: a tiny epsilon still learns fast. Low traffic: exploration must be directed (Thompson or UCB); a uniform tax is unaffordable |
| What you can measure at launch | Whether exploration ships at all | If only session CTR is measurable, exploration will always look bad; secure new-item share, diversity, and retention cohorts first |

## The smallest runnable exploration loop

The review of every bandit tutorial is the same: the reader sees the regret
curves but never the mechanism that produces them. So here is the chapter's
core claim in one file with zero installs. A warm catalog serves impressions
while new items keep arriving with unknown true engagement rates, some of them
genuinely better than anything warm. Three policies face the same stream: pure
exploitation, epsilon-greedy, and Thompson sampling. Every production
component is swapped for the smallest thing with the same interface: the
catalog is a list of click probabilities, the reward model is a click counter
per arm, and the upload pipeline is an append.

```python
"""Cold start in one file: three policies meet a stream of new items, no installs."""
import random

random.seed(7)
T = 30000                        # impressions served over the run
ARRIVAL_EVERY = 300              # a new item is uploaded every N impressions

# Warm catalog: 20 items the system already has labels for.
warm_rates = [random.uniform(0.01, 0.05) for _ in range(20)]
best_warm = max(warm_rates)
# Upload stream: most new items are mediocre, ~15% are genuinely better than
# anything warm. Nobody knows which is which at upload time.
new_rates = [random.uniform(0.06, 0.12) if random.random() < 0.15
             else random.uniform(0.005, 0.04) for _ in range(T // ARRIVAL_EVERY)]

def simulate(policy, seed):
    rng = random.Random(seed)
    rates = list(warm_rates)
    pulls = [30] * len(warm_rates)                      # warm items arrive pre-labeled
    clicks = [round(30 * r) for r in warm_rates]
    total = early = 0
    for t in range(T):
        if t % ARRIVAL_EVERY == 0 and t // ARRIVAL_EVERY < len(new_rates):
            rates.append(new_rates[t // ARRIVAL_EVERY])  # upload: retrievable at once,
            pulls.append(0); clicks.append(0)            # but zero interaction history
        a = policy(rng, pulls, clicks)
        r = 1 if rng.random() < rates[a] else 0
        pulls[a] += 1; clicks[a] += r; total += r
        if t == T // 20 - 1: early = total              # snapshot: short-term cost
    good = [i for i in range(len(warm_rates), len(rates)) if rates[i] > best_warm]
    found = sum(1 for i in good if pulls[i] >= 30)      # surfaced enough to be learned
    return total, early, found, len(good)

def greedy(rng, pulls, clicks):
    means = [c / p if p else 0.0 for p, c in zip(pulls, clicks)]
    return means.index(max(means))

def eps_greedy(rng, pulls, clicks):
    if rng.random() < 0.05:                              # 5% exploration budget
        return rng.randrange(len(pulls))
    return greedy(rng, pulls, clicks)

def thompson(rng, pulls, clicks):
    draws = [rng.betavariate(1 + c, 1 + p - c) for p, c in zip(pulls, clicks)]
    return draws.index(max(draws))

for name, pol in [("pure exploit", greedy), ("epsilon-greedy 5%", eps_greedy),
                  ("thompson", thompson)]:
    total, early, found, n_good = simulate(pol, seed=1)
    print(f"{name:18s} clicks: first-1500={early:4d} total={total:5d}   "
          f"good new items surfaced: {found}/{n_good}")
```

Run it and the three lines are the chapter in miniature. Pure exploit banks 55
clicks in the first 1,500 impressions and then never surfaces a single one of
the 19 genuinely-good new items: it finishes at 1,106 total clicks, locked
onto the best warm item forever. Thompson sampling posts the worst early
number, 41, because it is busy spending impressions on unknown arrivals; that
14-click short-term cost buys all 19 good items and a finish at 2,372 clicks,
more than double pure exploit. Epsilon-greedy lands between (2,005 total, 5 of
19 found; its early 71 comes from a lucky uniform draw landing on a good early
arrival): the flat 5% tax does explore, but blindly, so most exploratory
impressions are wasted on arms already known to be bad and most good arrivals
are never met. The mapping to production: `warm_rates` is the
labeled catalog, the append inside the loop is the content-tower upsert that
makes an upload retrievable at once, the Beta counters are the reward model's
posterior, the `pulls >= 30` threshold is the warm-up budget, and the `early`
snapshot is the session-CTR dip the launch review will argue about. Swap the
Beta counters for a neural-linear head, the list scan for ANN retrieval over
hundreds of candidates, and log a propensity next to every increment, and you
have rebuilt this chapter.
