# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it shows
how the same decisions flip when the constraints change. It closes with the
smallest runnable demonstration of the chapter's sharpest lesson, one file, no
installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing architectures before scoring a single
candidate. Skip that. The stack below is a sane default for a first production
ranker; each row names when to deviate and which section explains why. Model
families change yearly, but the interface of each stage (funnel, features,
labels, model, debiasing, blending, eval, serving) does not, so pick per stage by
interface and treat any specific architecture as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Candidate funnel | Full ranker directly over retrieval's few hundred survivors; no pre-ranking stage | Candidates grow into the thousands: insert a lightweight distilled pre-ranker graded on recall of the full ranker's top-k | [4](04-model-development.md) |
| Features | All three families: user (fetched once), item, and user-item cross features from a feature store | Never drop cross features; they are the biggest accuracy lever | [3](03-data-preparation.md) |
| Objective and labels | Pointwise binary cross-entropy per engagement signal over impression logs, point-in-time joined, time-based split | Order matters more than probabilities and (winner, loser) pairs are clean: pairwise LambdaMART | [2](02-frame-as-ml-task.md), [3](03-data-preparation.md) |
| Model family | GBDT (LightGBM/XGBoost) baseline first; migrate to a deep ranker (DCN-v2 or DLRM) when sparse ids and embedding tables earn their keep | Feature set stays compact and tabular: stay on GBDT, as Airbnb and Yelp did for years | [4](04-model-development.md) |
| Position bias | Position as a training feature, fixed to a constant at serving | You can estimate examination propensities: inverse-propensity weighting; add an exploration slice either way | [3](03-data-preparation.md) |
| Multi-objective blending | Shared-bottom multi-task heads, one per signal; calibrate each head; utility = weighted sum with weights outside the loss | Heads are negatively correlated and seesaw: MMoE or PLE gating | [4](04-model-development.md) |
| Evaluation | AUC per head offline plus ECE on any head that feeds a threshold; NDCG@k at the k you render; online A/B gates the ship | Never skip the online A/B; offline metrics are a pre-gate, not a ship decision | [5](05-evaluation.md) |
| Serving | One batched forward pass over all candidates; user context fetched once and broadcast; feature store for precomputed signals | Almost never; the batched pass and the shared user fetch are what make the budget reachable | [6](06-serving-and-scaling.md) |

The last row is the one that quietly decides everything above it: the 20 ms
budget divided by hundreds of candidates leaves a fraction of a millisecond per
item, and every architecture choice in section 4 was made under that ceiling.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): the
ranking stage of a multi-objective personalized feed. A few hundred to around a
thousand candidates arrive from retrieval, the list shows the top tens, ranking
gets roughly 20 ms at p99, labels are billions of impression rows blending
clicks, long dwell, and saves, at least one consumer feeds a threshold so
calibration matters, and cold-start items are real. Here is the whole system
with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Funnel | Full ranker over all candidates, no pre-ranking stage | Retrieval already returns a few hundred; section 4's skip rule says pre-ranking earns its place only when the full ranker cannot score the set in time |
| Features | User and context fetched once; item and cross features per candidate from the feature store, point-in-time joined in training | Cross features are the biggest accuracy lever; the shared user fetch is what keeps the fan-out flat |
| Objective | Pointwise binary cross-entropy, one head each for click, long dwell, save | Billions of rows favor pointwise scale; per-head probabilities are what the utility blend and the threshold consumer need |
| Model | DCN-v2 cross blocks over embedding tables, shared bottom, three task heads | Learned bounded-order crosses without maintaining a hand-crafted wide side; sparse ids rule out a pure GBDT long-term |
| Position bias | Position as a training feature, set to 1 at serving; a small exploration slice in traffic | Simplest correction that works on logged impressions; exploration keeps the tail learnable |
| Blending | Calibrate each head (isotonic), utility = weighted sum, weights outside the loss | The business retunes what a save is worth versus a click without retraining; Pinterest ships exactly this |
| Calibration | Downsampling correction plus post-hoc isotonic per head, ECE monitored | Negative downsampling distorts the base rate, and one consumer feeds a threshold, so probabilities must mean what they say |
| Evaluation | Per-head AUC and logloss, NDCG@k on the rendered k, ECE per head, time-based split; online A/B gates launch | Each metric earns a distinct place (section 5); a time split keeps future engagement out of training |
| Serving | One batched forward pass, feature store reads, calibrate, blend, sort | 20 ms divided by hundreds of candidates permits exactly one model call |

**Funnel counts.** Retrieval hands over up to ~1,000 candidates; take 500 as the
working number from [section 6](06-serving-and-scaling.md). All 500 go through
one batched forward pass, the utility blend sorts them, and the top tens render.
If retrieval ever grows to thousands, the fix is not a faster full ranker but a
lightweight distilled pre-ranker graded on recall of the full ranker's top-k
([section 4](04-model-development.md)).

**Latency.** The budget arithmetic from [section 6](06-serving-and-scaling.md):
500 candidates inside 20 ms at p99 is roughly 0.04 ms per candidate end to end.
An illustrative split that fits: ~6 ms for feature assembly (user context
fetched once, item and cross features in one batched feature-store read), ~11 ms
for the batched forward pass, ~1 ms for calibration, utility blend, and sort,
leaving ~2 ms of headroom. The number that matters is not any single row but the
shape: assembly and one model call, nothing per-candidate that fans out.

**Label volumes.** The log carries billions of impression rows with a click rate
of 1-2% ([section 3](03-data-preparation.md)). Illustrative: 2 billion
impressions per day at 1.5% is 30 million positives; downsampling negatives to
10:1 leaves roughly 330 million training rows per day, and the downsampling
correction from section 3 restores calibrated probabilities at inference. The
save and long-dwell heads see far fewer positives than the click head, which is
one reason the shared bottom helps: the rare heads borrow representation from
the abundant one.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: train/serve feature skew (log the features the
server actually used and diff them against the training pipeline's values; an
offline AUC gain with a flat online test is this seam until proven otherwise),
calibration drift on the threshold consumer (ECE per head as a live metric; a
drifting head silently moves the threshold's behavior even though ordering looks
fine), and the position-bias feedback loop starving cold items (track the
impression share of items younger than a week; if the exploration slice is too
thin, the ranker keeps re-showing what the old ranker showed and new items never
accumulate the labels that would let them win).

## The same techniques under different constraints

The question that matters in practice is not "which ranker is best" but "which
ranker is best under my constraints." Here is the same stage built three times.
Only the middle column is the build above; the other two keep the identical
stage interfaces and swap nearly every implementation choice.

| | Travel-marketplace search | Personalized feed (this chapter) | Ads ranker feeding an auction |
|---|---|---|---|
| Objective | Single: booking | Multi: click, long dwell, save | Multi: pCTR and pCVR, and the score is a price input |
| Labels | Sparse: a listing books at most ~365 times a year | Billions of impressions, clicks at 1-2% | Abundant clicks; conversions arrive delayed, needing an attribution window |
| Model | GBDT LambdaMART first, neural later; NDCG-weighted pairs optimize the list directly | DCN-v2 shared bottom, three heads, utility blend | Multi-task DNN (MMoE/PLE when heads conflict), user tower computed once per request |
| Id embeddings | Avoid: per-listing supervision is too sparse, lean on content features | Yes, with hashing for rare ids | Yes, with hourly warm-start because ad ids churn constantly |
| Calibration | Not central: ordering is the product, nothing reads magnitudes | Per-head isotonic; one consumer feeds a threshold | First-class: bid = value times predicted probability, so ECE is monitored live and drift is revenue |
| Eval | NDCG@k; reward the best listing at rank 1 | Per-head AUC, NDCG at rendered k, ECE, online A/B | Normalized entropy and ECE ahead of AUC; miscalibration means mis-priced bids |
| What would be over-engineering | Multi-task heads, calibration pipeline, per-id embeddings | Auction-grade live calibration monitoring | Pairwise LambdaMART: the auction needs probabilities, not just order |

Two lessons fall out. First, the marketplace column is mostly subtractions: one
objective removes the multi-task machinery, sparse per-id labels remove id
embeddings, and a product that only ever sorts removes calibration, which is why
LambdaMART, the objective that spends everything on order, wins there. Second,
the ads column shows what happens when the score stops being a sort key: the
moment a probability becomes a price input, calibration jumps from optional
pipeline step to the metric watched most closely, and the eval stack reorders
around it ([section 7](07-how-teams-do-it-in-production.md) shows Snap, Spotify,
and DoorDash making exactly these moves).

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any architectures.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Candidate count | Funnel stages | Few hundred: full ranker directly. Thousands: add a distilled pre-ranker graded on recall of the full ranker's top-k |
| Latency budget | Model size and serving shape | Budget divided by candidates gives per-item cost; one batched forward pass, user features fetched once, everything else precomputed |
| Score consumer | Calibration | Pure sorting: skip it. A threshold, auction, or cross-head blend: calibrate per head and monitor ECE live |
| Number of objectives | Head structure | One: single head. Several correlated: shared bottom. Negatively correlated (seesaw measured): MMoE or PLE |
| Labels per id | Embedding strategy | Dense repeated supervision: id embeddings. Sparse per id: content and context features; embeddings would memorize noise |
| Click-log bias | Label treatment | Always correct position bias (feature-then-fix or IPW); add exploration traffic or the tail is never learned |
| Class imbalance | Sampling | Click rate under ~5% at billions of rows: downsample negatives, then apply the calibration correction |
| Business retune cadence | Where utility weights live | Weights outside the loss: reordering the feed is a config change, not a retrain |

## The smallest runnable debiased ranker

The chapter's sharpest lesson is that the click log lies: items shown at the top
get clicked regardless of quality, so a ranker trained naively on raw clicks
learns to keep whatever was already on top. The file below shows the whole
mechanism with zero installs. Every production component is swapped for the
smallest thing with the same interface: the impression log becomes a simulated
session loop with a known examination model, the two "trained rankers" become
click counters (one raw, one inverse-propensity weighted), and the eval harness
becomes the chapter's own two-function NDCG from
[section 5](05-evaluation.md), scored against true relevance labels the rankers
never see.

```python
"""Position bias in miniature: a naive click ranker vs inverse-propensity
weighting, judged by NDCG against true relevance. Stdlib only."""
import math, random

random.seed(7)

# Ten items with true graded relevance (0-3); production: human-rater labels.
# The legacy ranker got it wrong: the best items sit near the bottom.
ITEMS = ["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"]
TRUE_REL = {"i0": 0, "i1": 1, "i2": 0, "i3": 2, "i4": 1,
            "i5": 3, "i6": 0, "i7": 2, "i8": 1, "i9": 3}

def examine_prob(pos):
    """P(user even looks at slot pos); production: estimated via swap experiments."""
    return 1.0 / (pos + 1) ** 2

def click_given_examined(rel):
    """P(click | examined) rises with true relevance; hidden from the ranker."""
    return 0.05 + 0.15 * rel

def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at_k(order, k=5):
    rels = [TRUE_REL[item] for item in order[:k]]
    ideal = sorted(TRUE_REL.values(), reverse=True)[:k]
    return dcg(rels) / dcg(ideal)

# --- generate a click log under the legacy order --------------------------
# Every session shows the same legacy order; clicks require examination first,
# so top slots accumulate clicks regardless of relevance.
SESSIONS = 20_000
naive, ipw = {i: 0.0 for i in ITEMS}, {i: 0.0 for i in ITEMS}
for _ in range(SESSIONS):
    for pos, item in enumerate(ITEMS):          # legacy order = i0 ... i9
        p_seen = examine_prob(pos)
        clicked = (random.random() < p_seen and
                   random.random() < click_given_examined(TRUE_REL[item]))
        if clicked:
            naive[item] += 1.0                  # raw click count
            ipw[item] += 1.0 / p_seen           # weight click by 1/propensity

# --- "train" two rankers: sort by each click estimate ---------------------
naive_order = sorted(ITEMS, key=lambda i: naive[i], reverse=True)
ipw_order = sorted(ITEMS, key=lambda i: ipw[i], reverse=True)

print(f"{'ranker':<14}{'top-5 it shows':<38}NDCG@5 vs true relevance")
for name, order in [("legacy", ITEMS), ("naive-clicks", naive_order),
                    ("ipw-debiased", ipw_order)]:
    top5 = " ".join(f"{i}(r{TRUE_REL[i]})" for i in order[:5])
    print(f"{name:<14}{top5:<38}{ndcg_at_k(order):.3f}")
```

Run it and three lines tell the whole story. The legacy order scores NDCG@5 of
0.263 against true relevance: the two relevance-3 items sit at positions 5 and
9 where almost nobody looks. The naive ranker, trained on raw clicks from that
log, scores 0.515 and its top two slots are still a relevance-1 and a
relevance-0 item: the top of the old list collected so many
examination-driven clicks that the naive counter mistakes exposure for quality,
which is the feedback loop from [section 3](03-data-preparation.md) in
miniature. The IPW ranker, given exactly the same clicks but weighting each one
by the inverse of its slot's examination propensity, scores 1.000: both
relevance-3 items rise to the top and the recovered order matches true
relevance exactly. Swap the session loop for a real impression log, the
propensity function for one estimated from swap experiments, the counters for a
model trained with the weighted loss from section 3, and the NDCG harness for
the evaluation stack of [section 5](05-evaluation.md), and you have rebuilt the
debiasing half of this chapter.
