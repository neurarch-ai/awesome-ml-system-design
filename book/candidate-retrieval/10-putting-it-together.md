# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it shows
how the same decisions flip when the constraints change. It closes with the
smallest runnable two-tower retriever, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing ANN libraries before retrieving a single
candidate. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Libraries
change yearly, but the interface of each stage (source candidates, build pairs,
train towers, index items, serve, evaluate) does not, so pick per stage by
interface and treat any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Candidate sources | Union of retrievers: two-tower + popularity + fresh-items | An explicit query or hard constraints exist: add a query-driven lexical source | [2](02-frame-as-ml-task.md) |
| Training pairs | Logged strong positives (click with long dwell), time-based split, some exploration traffic mixed in | Positives are sparse: relax the positive definition and re-check label quality | [3](03-data-preparation.md) |
| Features | ID embeddings for active users and items; content features for cold-start; hashing for unbounded IDs | Catalog has no content features: budget for a dedicated fresh-items source instead | [3](03-data-preparation.md) |
| Model | Two separate towers, no shared weights, joined by a dot product | Catalog is small enough to score online: a cross-feature model beats it | [4](04-model-development.md) |
| Negatives | In-batch negatives with the logQ correction | Batches are user-concentrated: add user-level masking; easy negatives plateau: mix in mined hard negatives gradually | [4](04-model-development.md) |
| ANN index | HNSW; add product quantization when memory binds | Heavy item churn or cheap filters needed: IVF | [6](06-serving-and-scaling.md) |
| Freshness | Minutes-cadence re-embed and upsert; content features carry cold items | Catalog churns slowly (daily batch is fine): save the streaming infra | [6](06-serving-and-scaling.md) |
| Evaluation | Recall@k at the downstream k, time-based split, measured through the real ANN index | Never skip it. Add coverage the moment recall starts climbing | [5](05-evaluation.md) |

The last row's qualifier is the one beginners skip and regret: recall measured by
brute force flatters the system, because the served path goes through an
approximate index that loses a few points of recall on its own. Measure through
the same index you serve, at the same k you hand to ranking.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): the
retrieval stage of a personalized feed over roughly 100 million items, returning
a few hundred candidates in tens of milliseconds, optimized for recall of items
the user would engage with, with new items retrievable within minutes. Here is
the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Framing | Learn embeddings whose dot product ranks well; retrieval = nearest-neighbor lookup | The only framing that moves 100M-item scoring offline ([2](02-frame-as-ml-task.md)) |
| Candidate sources | Two-tower ANN + popularity + fresh-items, unioned and deduped | Each source covers a failure mode of the others; cold items need a non-embedding path |
| Training pairs | Long-dwell clicks as positives, exploration traffic mixed in | Pure ranking-loop positives make the model imitate the old system |
| Model | Two separate towers, dot product join, no early feature crossing | Any early crossing kills the item-side precompute and the ANN lookup |
| Similarity | Dot product, not cosine | Magnitude carries popularity signal for free; the index metric flag is set to inner product to match ([6](06-serving-and-scaling.md)) |
| Negatives | In-batch sampled softmax + logQ correction + user-level masking | The catalog is power-law, so uncorrected in-batch negatives over-penalize head items; masking stops same-user false negatives |
| ANN index | HNSW with product quantization, sharded | Best recall per millisecond on a mostly-stable catalog; PQ fits the index in memory |
| Freshness | Minutes-cadence item re-embed and index upsert; user tower always fresh | The stated requirement; content features carry an item until its ID embedding trains |
| Evaluation | Recall@k at the downstream few hundred, time-based split, through the real index; launch gated on online engagement and coverage | Recall alone can rise while the feed collapses to popular items |

**Index sizing.** 100M items at 128 dimensions (illustrative; the chapter does
not pin a dimension) and float32 is 100M x 128 x 4 bytes, about 51 GB of raw
vectors before any HNSW graph overhead, which is why the memory column of the
ANN tradeoff in [section 6](06-serving-and-scaling.md) is not optional reading.
Etsy-style 4-bit product quantization stores each dimension in half a byte, 64
bytes per vector, about 6.4 GB, an 8x reduction that turns a sharded-cluster
problem into a couple-of-replicas problem. The recall cost of quantization is
then measured, not assumed: compare the quantized index against exact search on
a held-out query set before shipping it.

**The candidate funnel.** Per request, the two-tower source pulls a few hundred
nearest neighbors from the index, the popularity and fresh-items sources
contribute on the order of a hundred each (illustrative splits), and the union
is deduplicated down to the few hundred candidates the requirement names. From
100 million to a few hundred is a reduction of roughly five orders of magnitude,
and every bit of it is either precomputed (the item embeddings, the index) or a
single cheap lookup (the ANN query). Ranking then spends its heavy model on the
survivors, which is the funnel's division of labor from
[section 2](02-frame-as-ml-task.md).

**Latency.** The tens-of-milliseconds budget decomposes as roughly (illustrative)
5ms to fetch user features, 5ms for one user-tower forward pass, 10-15ms for the
ANN search across shards, and a few ms to union, dedup, and serialize, landing
near 30ms with headroom. Note what is absent: no per-item model calls, no
network hop per candidate. If the ANN search is the line that blows the budget,
the knobs are search depth (efSearch for HNSW, nprobe for IVF) traded against
recall, before any architectural change.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: new-item retrievability (the fraction of items
retrieved within minutes of creation; if the upsert pipeline silently stalls,
cold items vanish and creators notice before dashboards do), catalog coverage
(a shrinking fraction of the catalog ever being retrieved means popularity
collapse, usually a missing or broken logQ correction, and raw recall@k will
look fine or even improve while it happens), and embedding-space version skew
(recall craters right after a retrain because a new user tower is being served
against an index built by the old item tower; version the towers and the index
together and redeploy them atomically, per [section 4](04-model-development.md)).

## The same techniques under different constraints

The interview question that matters in practice is not "which index is best" but
"which index is best under my constraints." Here is the same retrieval stage
built three times. Only the middle column is the build above; the other two keep
the identical stage interfaces and swap nearly every implementation choice. The
outer columns are illustrative deployments assembled from the production
patterns in [section 7](07-how-teams-do-it-in-production.md).

| | Small-app feed | 100M-item feed (this chapter) | High-churn marketplace search |
|---|---|---|---|
| Catalog / traffic | 100k items, low QPS | 100M items, feed-scale QPS | 5M listings; explicit queries with geo and availability filters |
| Latency budget | Loose; tens of ms is easy at this scale | Tens of ms, every feed load | Tens of ms inside a search page |
| Candidate sources | Two-tower alone, or co-visitation if behavioral signal is thin | Two-tower + popularity + fresh, unioned | Query-driven lexical source + behavioral two-tower, merged |
| Negatives | Plain in-batch; at mild popularity skew the logQ term barely moves the model | In-batch + logQ + user-level masking | Journey seen-not-booked negatives; an impression inside a search session beats a random item |
| Index | Flat exact search; at 100k vectors brute force is a few ms and there is nothing to tune | Sharded HNSW with 4-bit PQ | IVF: inserts and deletes are posting-list edits, and a geo filter becomes cheap cluster selection |
| Freshness | Re-embed the whole catalog nightly in one batch | Minutes-cadence upsert | Daily batch re-embed; churn is in price and availability metadata, handled by the index, not the tower |
| Evaluation | Recall@k against exact search (it is the same thing here) | Recall@k through the ANN index + coverage + online A/B | Recall@k plus filter-correctness checks; a fast wrong-city listing is worthless |
| What would be over-engineering | ANN index, sharding, streaming upserts, hard-negative mining | A cross-feature model anywhere before ranking | HNSW, whose rebuild cost the churn would eat alive |

Two lessons fall out. First, the small-app column is mostly deletions: at 100k
vectors exact search is fast, correct, and removes the entire train-vs-ANN
recall gap from [section 4](04-model-development.md)'s pitfall table, and mild
popularity skew means the sampling corrections buy little. Complexity should
arrive when a symptom demands it, not before. Second, the marketplace column
shows that "which negatives" and "which index" are decided by the catalog, not
by fashion: churn and filters pick IVF over the HNSW default, and session
structure yields a smarter negative than any sampling trick, which is exactly
Airbnb's pair of choices in [section 7](07-how-teams-do-it-in-production.md).

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Catalog size | Index family and precompute | Under ~1M: flat or simple HNSW. At 100M: shard, quantize, and everything item-side moves offline |
| Latency budget | ANN search depth, not architecture | Tune efSearch / nprobe against recall first; redesign only when the knobs run out |
| Item churn | Index family and refresh cadence | Stable catalog: HNSW. Heavy churn or frequent metadata updates: IVF, batch cadence |
| Hard filters (geo, eligibility) | Index family and sourcing mode | IVF turns a filter into cluster selection; an explicit query adds a query-driven lexical source |
| Popularity skew | Training-time correction | Power-law catalog: logQ on the training logits, never at serving |
| Batch composition | Masking, not batch size | User-concentrated batches: user-level masking; a bigger batch makes false negatives worse |
| Cold-start rate | Feature mix and sources | High item birth rate: content features + a dedicated fresh-items source + minutes-cadence upserts |
| Memory budget | Quantization | Vectors outgrow RAM: product quantization (4-8x smaller) before sharding further |
| Training similarity | Index metric flag | Dot product trained: inner-product index. Cosine trained: normalize both sides, then inner product |
| Product health bar | Launch gates | Gate on engagement and coverage together; a recall win that shrinks the tail loses long-term |

## The smallest runnable two-tower

Frameworks hide the mechanism: embedding tables, a sampled softmax, and a
nearest-neighbor scan are all this stage is. So here is the entire loop in one
file with zero installs. Every production component is swapped for the smallest
thing with the same interface: the towers become plain learned vectors, the
in-batch negatives become uniform random draws (which is why no logQ term
appears; sample uniformly and there is no popularity bias to correct), and the
ANN index becomes an exact scan. A synthetic world with known structure (users
have latent tastes, items have latent topics, engagement mixes taste with a
popularity pull) lets us check the one claim that matters: the tower recovers
taste structure that popularity alone cannot.

```python
"""Two-tower retrieval in miniature: stdlib only, one file, seeded."""
import math, random
random.seed(7)

T, USERS, ITEMS, D = 4, 80, 200, 8          # latent topics, users, catalog size, embedding dim
topic = [i % T for i in range(ITEMS)]        # each item covers one latent topic
pop = [(i + 2) ** -0.8 for i in range(ITEMS)]  # power-law head: low ids are globally popular
taste = [random.randrange(T) for _ in range(USERS)]  # each user has one latent taste

def sample_item(u):
    """Engagement mixes taste match with the popularity pull of the head."""
    w = [pop[i] * (25.0 if topic[i] == taste[u] else 1.0) for i in range(ITEMS)]
    r, s = random.uniform(0, sum(w)), 0.0
    for i, wi in enumerate(w):
        s += wi
        if s >= r:
            return i
    return ITEMS - 1

logged = {u: {sample_item(u) for _ in range(60)} for u in range(USERS)}
train = {u: set(random.sample(sorted(s), len(s) // 2)) for u, s in logged.items()}
held = {u: logged[u] - train[u] for u in logged}          # held-out positives to recall

U = [[random.gauss(0, 0.1) for _ in range(D)] for _ in range(USERS)]  # user tower output
V = [[random.gauss(0, 0.1) for _ in range(D)] for _ in range(ITEMS)]  # item tower output

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

pairs = [(u, i) for u in train for i in train[u]]
LR, NEG = 0.1, 4
for epoch in range(20):
    random.shuffle(pairs)
    for u, pos in pairs:
        cands = [pos] + [random.randrange(ITEMS) for _ in range(NEG)]  # sampled softmax
        logits = [dot(U[u], V[i]) for i in cands]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        Z = sum(exps)
        for c, i in enumerate(cands):
            g = exps[c] / Z - (1.0 if c == 0 else 0.0)  # d loss / d logit
            for d in range(D):
                U[u][d], V[i][d] = U[u][d] - LR * g * V[i][d], V[i][d] - LR * g * U[u][d]

def recall_at_10(score):
    per_user = []
    for u in range(USERS):
        if not held[u]:
            continue
        unseen = [i for i in range(ITEMS) if i not in train[u]]
        top = sorted(unseen, key=lambda i: score(u, i), reverse=True)[:10]
        per_user.append(len(set(top) & held[u]) / len(held[u]))
    return sum(per_user) / len(per_user)

def taste_share(score):
    """Fraction of retrieved items matching the user's latent taste."""
    hits = total = 0
    for u in range(USERS):
        unseen = [i for i in range(ITEMS) if i not in train[u]]
        for i in sorted(unseen, key=lambda i: score(u, i), reverse=True)[:10]:
            hits += topic[i] == taste[u]
            total += 1
    return hits / total

pop_score = lambda u, i: pop[i]
tower_score = lambda u, i: dot(U[u], V[i])
print(f"recall@10  popularity baseline: {recall_at_10(pop_score):.3f}")
print(f"recall@10  two-tower:           {recall_at_10(tower_score):.3f}")
print(f"taste match in top-10  popularity: {taste_share(pop_score):.3f}  (chance = {1/T:.2f})")
print(f"taste match in top-10  two-tower:  {taste_share(tower_score):.3f}")
```

Run it and the seeded output reads: popularity recall@10 0.158, two-tower
recall@10 0.210, and the second pair of lines explains the gap. Popularity's
top-10 matches each user's taste 17% of the time, below the 25% chance floor,
because the head is taste-blind by construction; the tower's top-10 matches 78%.
The tower wins recall precisely because it learned per-user structure the
popularity baseline is incapable of expressing, which is the whole argument for
personalizing the retrieval stage. Each toy piece stands in for a production
component: `sample_item` is the interaction log, the `U` and `V` tables are the
tower outputs (production replaces the rows with MLPs over features, which is
what lets a never-seen user or a cold item get a vector at all), the
random-negative sampled softmax is the in-batch loss of
[section 4](04-model-development.md) (production negatives arrive
popularity-skewed, which is what makes the logQ correction necessary there),
and the exact scan inside `recall_at_10` is the ANN index of
[section 6](06-serving-and-scaling.md), affordable here only because the
catalog is 200 items instead of 100 million. The random hold-out stands in for
the time-based split of [section 5](05-evaluation.md); the toy has no
timestamps, a real evaluation must. Swap those four pieces for their production
versions and you have rebuilt this chapter.
