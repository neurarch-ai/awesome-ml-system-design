# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it shows
how the same decisions flip when the constraints change. It closes with the
smallest runnable sequence recommender, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing encoders before producing a single user
vector. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Models
change yearly, but the interface of each stage (define the sequence, featurize
it, encode it, train against negatives, evaluate on a time split, serve it
fresh) does not, so pick per stage by interface and treat any specific encoder
as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Sequence construction | Sort by timestamp, dedup adjacent repeats, cap at recent 100 events, causal sliding-window pairs | Sessions are naturally short and bounded: cap lower and save encode cost | [3](03-data-preparation.md) |
| Features | Item ID embedding + action-type embedding (4-8 dims) + time-gap encoding | Catalog churns fast or is long-tail: add content features, hash the ID space | [3](03-data-preparation.md) |
| Encoder | SASRec-style causal self-attention, 1-2 layers | Sessions under ~20 events and serving must be dirt cheap: GRU4Rec; one model must serve many surfaces and data is plentiful: BERT4Rec | [4](04-model-development.md) |
| Loss and negatives | Sampled softmax with in-batch negatives + logQ popularity correction | A daily batch embedding must stay fresh without streaming: all-action window loss | [4](04-model-development.md) |
| Evaluation | Time-based split, last interaction held out, Recall@k and NDCG@k at the k you pass downstream, full-catalog ranking | Full-catalog scoring is unaffordable: fixed large negative sample, bias-corrected, never compared across samples | [5](05-evaluation.md) |
| Serving | Streaming ingest into a per-user KV store, re-encode on request, cache by (user, sequence-hash) | Same-session reaction is not the product requirement: daily batch embedding with all-action loss | [6](06-serving-and-scaling.md) |
| Cold start | Degradation ladder inside one model: short sequence, then content features, then popularity + context | Never a separate cold-start model | [8](08-interview-qa.md) |

The row beginners get wrong is the first one: the sequence-construction logic
must be one shared piece of code called by both the batch training pipeline and
the streaming serving pipeline. Every other row can be tuned later; a
construction mismatch silently poisons everything downstream and is the
headline operational risk of the whole system
([section 6](06-serving-and-scaling.md)).

## The complete build

Return to the scenario from
[section 1](01-clarifying-requirements.md): a main-feed ranking feature that
turns the user's recent ordered behavior (clicks, long-dwell views, purchases)
into a user-intent vector, for tens of millions of daily active users at
several feed loads per day, reacting within the same session, degrading
gracefully for new users. Here is the whole system with every choice committed
and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Funnel position | Ranking feature (user-intent vector into the ranker) | The funnel already retrieves candidates; a feature slots in without new retrieval infrastructure |
| Events in the sequence | Clicks + long-dwell views + purchases, ordered by timestamp, action type embedded | Behavior mix is heterogeneous; a purchase and a skim are different intent signals |
| Sequence construction | Dedup adjacent repeats, cap at recent 100, shared batch/streaming code | 100 captures most of the recall-vs-length curve at bounded O(L^2) encode cost |
| Time signal | Time-gap encoding, not a bare positional index | One second apart and one month apart must look different to the attention |
| Encoder | SASRec-style causal self-attention, 2 layers | Parallel training, causal serving with no mask protocol, handles 100-event histories a GRU compresses away |
| Loss | Sampled softmax, in-batch negatives with logQ correction | Full-catalog softmax is infeasible; uncorrected in-batch punishes head items |
| Evaluation | Recall@k and NDCG@k at the downstream k (10-50), time-based split, full-catalog ranking | Sampled metrics can reorder models (Krichene and Rendle); the split must match how it serves |
| Serving | Kafka-style ingest into a fast per-user KV store; re-encode per request; cache by (user, sequence-hash); GPU for the encoder | The next feed load after three cooking videos must reflect them; that is the stated product value |
| Cold start | Degradation ladder: encode whatever exists, content features carry cold items, popularity + context for empty | One model, no discontinuity at the cold-warm boundary |
| Launch gate | Online A/B on session engagement + diversity guardrail | Offline NDCG can rise while a filter bubble tightens |

**Traffic and state.** Call it 30M DAU at 5 feed loads per day (Illustrative;
the interviewer pinned "tens of millions, several loads per day"): 150M
encode requests per day, about 1,700 QPS average and roughly 5,000 QPS at peak
with a 3x diurnal swing. The per-user state is small: 100 events at ~20 bytes
each (item ID, timestamp, action type) is 2 KB per user, about 60 GB across
30M users (Illustrative), which fits a replicated in-memory KV store without
heroics. Streaming write volume is a few tens of events per user per day,
roughly 10k events per second at peak: routine for a Kafka-class pipeline.

**Sequence length and encode cost.** Self-attention is O(L^2) in sequence
length ([section 6](06-serving-and-scaling.md)), so the cap is the single
biggest latency lever: 100 events costs 100x less attention compute than
1,000, and the recall-vs-length curve has already flattened by then. The model
itself is small; the item embedding table is not. At 10M catalog items and 128
dimensions in float32, the table is about 5.1 GB (Illustrative) against an
encoder of a couple million parameters, so memory planning is table planning,
and the table must be versioned and rolled together with the encoder to avoid
the embedding-churn hazard. The (user, sequence-hash) cache absorbs the
common case of a user reloading the feed without acting: only requests that
follow a new event pay for a fresh encode.

**Quality target.** No absolute Recall@10 floor is honest at this level of
abstraction; the chapter's own evidence sets the relative bar instead. The
model must beat a bag-of-interactions baseline by a margin consistent with
what order is worth: Instacart measured a 10-45% recall drop from shuffling
sequence order alone ([section 8](08-interview-qa.md)), so an
order-aware model that only matches the bag baseline is broken, not modest.
Measure Recall@k at the k actually passed downstream (10-50 for this ranking
feature), on a time-based split, then gate the launch on the online A/B.

**What breaks in month one.** Three failure modes dominate early operations,
so wire their signals before launch: online-below-offline gap (online
engagement consistently under what offline Recall@k predicted means the
streaming pipeline builds sequences differently than the batch pipeline did;
diff the two constructions for the same users, daily), power-user tail latency
(p99 encode time climbing with sequence length means the cap is not enforced
where you think it is), and staleness of the same-session promise
(event-to-store ingest lag exceeding the typical gap between a user's actions
means the next request encodes a sequence missing the action that just
happened, and the product value quietly reverts to yesterday's batch model).

## The same techniques under different constraints

The review question that matters in practice is not "which encoder is best"
but "which encoder is best under my constraints." Here is the same system
built three times. Only the middle column is the build above; the other two
keep the identical stage interfaces and swap nearly every implementation
choice.

| | Boutique marketplace | Consumer feed (this chapter) | Lifelong-history ads CTR |
|---|---|---|---|
| Scale / traffic | 200k DAU, a few loads per day (Illustrative) | 30M DAU, ~5,000 QPS peak | Hundreds of millions of users; every impression scored |
| Freshness | Daily batch embedding; all-action loss keeps it from going stale | Streaming ingest, same-session reaction | Offline clustering of lifelong history + online inference |
| Encoder | GRU4Rec-class, or 1-layer attention; sessions are short | SASRec-style, 2 layers, last 100 events | Two-stage attention (retrieve-then-score, TWIN-style) over ~10^6 events |
| History | Recent 20 events | Recent 100 events | Lifelong; a recent-N cap would throw away the signal that pays |
| Loss | In-batch negatives, no correction needed at small scale | In-batch + logQ correction | CTR objective per candidate, not next-item softmax |
| Serving | Nightly job writes user vectors to the feature store; no streaming infrastructure at all | KV store + per-request encode + cache + GPU | Two-stage keeps online cost tractable; single-stage attention over 10^6 events is infeasible |
| Eval | Recall@k on a time split, hand-run A/B | Full-catalog Recall@k + online A/B + diversity guardrail | CTR AUC offline, revenue and CTR online |
| What would be over-engineering | Streaming pipeline, GPU serving, BERT4Rec | Lifelong two-stage attention, generative foundation model | A plain transformer over the full history; O(L^2) at 10^6 events |

Two lessons fall out. First, the boutique column is mostly deletions, and the
one addition is a loss function standing in for infrastructure: the all-action
window loss ([section 4](04-model-development.md)) trains a daily batch
embedding to predict a window of future actions, closing most of the freshness
gap that would otherwise require the streaming pipeline the team cannot staff.
Second, the ads column shows history length flipping the architecture rather
than the hardware: at 10^6 events, O(L^2) attention is not a latency problem
to optimize but an impossibility to design around, so retrieval moves inside
the model (a cheap first stage picks the relevant slice of history, an
expensive second stage attends over only that), which is the same
retrieve-then-score shape the recommendation funnel itself has.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any models.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Same-session reaction required | Streaming vs batch | Required: streaming ingest + per-request encode. Not required: daily batch with all-action loss closes most of the gap for a fraction of the cost |
| Typical session length | Encoder family | Under ~20 events: GRU4Rec is enough. Longer: causal self-attention; cap at recent 50-200 where the recall curve flattens |
| Power-user history depth | Truncation vs two-stage | Cap to recent N first. Only when deep history demonstrably pays (ads CTR): two-stage retrieve-then-score attention |
| Catalog size | Softmax strategy | Beyond ~100k items full softmax is off the table: sampled softmax or in-batch negatives, with logQ correction under popularity skew |
| Training data volume | Causal vs bidirectional | SASRec by default; BERT4Rec only with the data to feed it and the masking discipline to serve it |
| Latency budget | Depth, cap, cache, hardware | Cache by (user, sequence-hash); shallow encoder on CPU, multi-layer on GPU; the sequence cap is the biggest single lever |
| Cold-start share of traffic | Feature mix | High share of new users or items: content features and the degradation ladder, never a second model |
| Number of surfaces served | Model scope | Several surfaces: one shared model (Instacart, Netflix pattern) amortizes training and keeps representations consistent |
| Popularity skew | Negative sampling | Always present: logQ-correct in-batch negatives; mix in uniform negatives so head items are not systematically punished |
| Offline metrics distrusted | Eval protocol | Full-catalog ranking on a time split; if you must sample negatives, fix the sample, correct the bias, never compare across samples |

## The smallest runnable sequence recommender

The claim under this whole chapter is that order carries signal a bag of
counts cannot see, and that longer context buys accuracy only when there is
enough data to fill it. Both claims fit in one file with zero installs. Every
production component is swapped for the smallest thing with the same
interface: the interaction log becomes seeded sessions generated from a
ground-truth genre flow with popularity noise on top, the sequence encoder
becomes transition-count tables, the encoder ladder becomes first-order versus
second-order context, and the degradation ladder becomes count backoff. The
shape is the lesson; every section of this chapter upgrades one function of
this file.

```python
"""Popularity vs first-order vs second-order next-item prediction, no installs."""
import random
from collections import Counter, defaultdict

GENRES = ["cook", "travel", "gear", "grocery", "luggage"]
ITEMS = {g: [f"{g}{i}" for i in range(5)] for g in GENRES}
CATALOG = [i for g in GENRES for i in ITEMS[g]]

def next_genre(prev2, prev1):
    """Ground truth: 'gear' is reachable from both cooking and travel, and what
    follows it depends on how the user got there. One item of context cannot
    distinguish the two flows; two can."""
    if prev1 == "gear":
        return "grocery" if prev2 == "cook" else "luggage"
    return {"cook": "gear", "travel": "gear", "grocery": "cook", "luggage": "travel"}[prev1]

def make_session(rng, length=10, noise=0.25):
    genres = [rng.choice(["cook", "travel"]), "gear"]
    while len(genres) < length:
        genres.append(next_genre(genres[-2], genres[-1]))
    session = []
    for g in genres:                       # popularity noise rides on top of intent
        if rng.random() < noise:
            session.append(CATALOG[min(int(rng.expovariate(0.4)), len(CATALOG) - 1)])
        else:
            session.append(rng.choice(ITEMS[g]))
    return session

def fit(sessions):
    pop, m1, m2 = Counter(), defaultdict(Counter), defaultdict(Counter)
    for s in sessions:
        pop.update(s)
        for i in range(1, len(s)):
            m1[s[i - 1]][s[i]] += 1
            if i >= 2:
                m2[(s[i - 2], s[i - 1])][s[i]] += 1
    return pop, m1, m2

def top5(counter, *backoffs):
    """Ranked list from this order's counts, padded from lower orders."""
    ranked = [item for item, _ in counter.most_common(5)]
    for fallback in backoffs:
        for item in fallback:
            if item not in ranked and len(ranked) < 5:
                ranked.append(item)
    return ranked

def hit_at_5(model, train, test):
    pop, m1, m2 = fit(train)
    pop5 = top5(pop)
    hits = 0
    for s in test:
        context, target = s[:-1], s[-1]
        if model == "popularity":
            preds = pop5
        elif model == "markov1":
            preds = top5(m1[context[-1]], pop5)
        else:  # markov2 backs off through markov1 to popularity when its context is unseen
            preds = top5(m2[(context[-2], context[-1])], top5(m1[context[-1]], pop5))
        hits += target in preds[:5]
    return hits / len(test)

rng = random.Random(7)
test = [make_session(rng) for _ in range(2000)]
for label, n_train in [("dense (5000 sessions)", 5000), ("sparse (60 sessions)", 60)]:
    train = [make_session(rng) for _ in range(n_train)]
    print(label)
    for model in ["popularity", "markov1", "markov2"]:
        print(f"  hit@5 {model:<10} {hit_at_5(model, train, test):.3f}")
```

Run it and the six numbers demonstrate the chapter's core claims in about
seventy lines. Dense regime: popularity 0.493, first-order 0.696, second-order
0.769. Order alone is worth 20 points of hit@5 over popularity, the toy
version of Instacart's shuffle result, and the second item of context is worth
another 7 because the ground truth is genuinely second-order: what follows
"gear" depends on whether the user came from cooking or travel, and a
one-item context cannot see the difference. Sparse regime (60 training
sessions): first-order holds at 0.653 while second-order regresses to 0.644,
because the (prev2, prev1) table now has one noisy observation per context
where it has any, and single-count noise events outrank the better first-order
estimate that backoff would have supplied. That regression is the whole
SASRec-versus-BERT4Rec data argument in miniature: richer context is not free,
it is bought with data. The mapping to production: the seeded genre flow is
the interaction log's latent intent structure, the popularity noise is
head-item traffic, the held-out last item per session is the time-based split
of [section 5](05-evaluation.md), hit@5 is Recall@5, the count tables are the
smallest possible sequence encoder, and the backoff chain is the degradation
ladder of [section 8](08-interview-qa.md). Swap the count tables for a trained
causal-attention encoder, the session list for a streaming KV store, and the
hold-out loop for the online A/B, and you have rebuilt this chapter.
