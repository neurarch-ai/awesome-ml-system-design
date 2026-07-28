# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it shows
how the same decisions flip when the constraints change. It closes with the
smallest runnable embedding trainer, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing losses and index libraries before embedding a
single entity. Skip that. The stack below is a sane default for a first
production build; each row names when to deviate and which section explains why.
Frameworks change yearly, but the interface of each stage (define related, pick
negatives, train, embed, index, serve, evaluate) does not, so pick per stage by
interface and treat any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Positive pairs | Co-engagement and session co-occurrence joined from behavioral logs | The consumer needs a different "related": query reformulation pairs, graph random walks, dropout views for text | [3](03-data-preparation.md) |
| Training objective | InfoNCE dual-encoder, dot-product similarity, temperature tuned on a validation retrieval metric | You have a triplet-mining pipeline and want an explicit margin: triplet loss | [4](04-model-development.md) |
| Negatives | In-batch with the logQ correction at training time; add a small hard fraction only after the easy loss saturates | Strong domain structure: sample from a domain-aware pool (Airbnb's same-market negatives) | [3](03-data-preparation.md), [4](04-model-development.md) |
| Encoder | Two-tower, content features on the precomputed side so the encoder is inductive | The entity set is fixed and cold start is a non-event: a transductive LightGCN-class baseline is cheaper | [2](02-frame-as-ml-task.md), [4](04-model-development.md) |
| Dimension | 128, tuned against the downstream ANN consumer, never in isolation | Memory binds: quantize before changing d; consumers need different widths: Matryoshka | [4](04-model-development.md), [6](06-serving-and-scaling.md) |
| Evaluation | Recall@k at the downstream k on a time-based split, tail recall reported separately | Never skip. Add alignment and uniformity when probes look fine but ranking sags | [5](05-evaluation.md) |
| Index | HNSW in RAM | Heavy churn or attribute filters: IVF; memory is the binding constraint: IVF-PQ or HNSW+PQ | [6](06-serving-and-scaling.md) |
| Versioning | Atomic full reindex on every retrain; never mix vectors across model versions | Never | [6](06-serving-and-scaling.md) |
| Serving | Precomputed side batch-embedded offline and upserted on the freshness cadence; other side encoded online per request | Both sides are cheap and static: batch both and skip the online encoder | [6](06-serving-and-scaling.md) |

The versioning row is the one beginners skip and regret: a retrain moves the axes
of the space, so a single upsert of new-model vectors into an old-model index
silently poisons every similarity score it touches. Treat the reindex as part of
the retrain, not as an optimization to schedule later.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): tens of
millions of items and users, behavioral logs with no curated similarity labels,
vectors consumed by ANN retrieval and reused as ranking features, items
precomputed offline and users computed online, a new item retrievable within an
hour, and retrieval latency in the tens of milliseconds. Here is the whole system
with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Framing | Contrastive metric learning, dual-encoder (two-tower) | No similarity labels, only co-occurrence; the score factorizes so the item side precomputes against an ANN index |
| Positive pairs | User-item engagements plus same-session item co-occurrence | This join is the definition of "related" that retrieval and ranking both inherit |
| Negatives | In-batch with logQ correction at training time, plus a small tuned hard fraction after the easy loss saturates | Free negatives at batch scale; logQ undoes popularity bias; hard negatives sharpen the boundary without destabilizing training |
| Loss | InfoNCE with tuned temperature and user-level masking in the softmax denominator | The canonical two-tower retrieval loss; masking removes same-user false negatives that grow with batch size |
| Encoder | Item tower consumes text title and category attributes (inductive); user tower consumes behavioral aggregates, computed online | The one-hour cold-start SLA rules out id-only: a brand-new item must embed from content alone |
| Dimension | 128, float32 | The recall gain past 128 saturates while memory and latency keep scaling linearly (section 4's dimension figure) |
| Index | HNSW in RAM, hourly upsert of new and changed items | Tens-of-ms latency at 50M vectors with top recall-per-latency; the catalog is stable enough that graph insertions absorb the hourly write load |
| Freshness | Inductive embed on item creation, upserted within the hour | The SLA is met structurally by the encoder choice, not by faster batch jobs |
| Versioning | Weekly retrain (Illustrative), full re-embed of the item set, atomic blue/green index swap | Space drift makes cross-version vectors incomparable; the swap must be atomic or the index mixes versions |
| Evaluation | Recall@k at the downstream k on a time-based split, tail vs head recall, alignment and uniformity; launch gated on online A/B engagement, coverage, and new-item retrievability | Retrieval recall is the quality ceiling, and the tail and collapse failure modes are invisible to the average |

**Index memory.** 50 million items at 128 dimensions and float32 is 128 x 4 =
512 bytes per vector, about 500 MB per million entities, so roughly 25 GB of raw
vectors ([section 6](06-serving-and-scaling.md)). That fits in RAM on one
well-provisioned box or two replicated shards, which is what makes HNSW viable.
The same catalog at 512 dimensions is 100 GB and forces sharding or quantization
immediately, which is why the dimension was tuned against the index rather than
picked off a leaderboard. Users are never indexed at all: user vectors are
computed online per request and exist only for the duration of the lookup.

**Two clocks.** Freshness runs on the upsert cadence: a new item embeds from its
title and category the moment it is created, and the hourly upsert makes it
retrievable well inside the SLA. Space drift runs on the retrain cadence: the
weekly retrain moves the axes, so all 50 million items are re-embedded and a new
index is built and swapped atomically. At an illustrative batch-embedding
throughput of 5,000 items per second, the full re-embed is under three hours,
comfortably inside a weekly window. The version-skew rule that falls out: a
vector is valid only against an index built from the same model version, and the
online user encoder must cut over in the same deploy as the item index, because a
new user tower scoring against old item vectors is exactly the cross-version dot
product that means nothing.

**What breaks in month one.** Three failure signals dominate early operations, so
wire them before launch. First, new-item retrievability against the one-hour
promise: monitor upsert queue depth and the age of the youngest retrievable item,
because a stalled write path is invisible to every quality metric. Second, tail
recall and catalog coverage diverging from average recall: a space that drifts
toward head items keeps its average recall while starving the long tail, and the
first external symptom is sellers of niche items reporting they can no longer be
found; check the logQ correction is actually applied and read the uniformity
diagnostic. Third, a mixed-version index after a retrain: hold out a canary set
of known-neighbor pairs and check their similarities across every cutover,
because a partial swap shifts the whole similarity distribution and shows up as
an unexplained recall drop the day after a deploy.

## The same techniques under different constraints

The review question that matters in practice is not "which loss is best" but
"which loss is best under my constraints." Here is the same pipeline built three
times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Startup semantic text search | Marketplace retrieval (this chapter) | Session embeddings for fraud |
|---|---|---|---|
| Entities / scale | 200k product descriptions (Illustrative); low QPS | 50M items, tens of millions of users; tens-of-ms retrieval | Tens of millions of customer sessions; no user-facing lookup |
| "Related" defined by | Text meaning; no behavioral logs, so dropout views of the same sentence form the positive pair (SimCSE) | Engagement and session co-occurrence from logs | A next-page pretext task over browsing sequences; no explicit negatives at all (Wayfair Melange) |
| Encoder | Pretrained sentence encoder, contrastively fine-tuned to fix anisotropy | Two-tower, inductive item side, online user side | Self-supervised sequence model over page-type sequences |
| Negatives | In-batch only; the corpus is too small to need mining | In-batch + logQ + small hard fraction | None; the pretext objective replaces the contrast |
| Index | Flat scan or a small HNSW; 200k vectors fit trivially and exact search is a fine ceiling | 128-d HNSW, ~25 GB, hourly upserts | No ANN index at all; vectors land in a feature store on an hourly pipeline |
| Freshness / versioning | Re-embed everything on deploy; the corpus fits in one batch | Hourly upsert, weekly atomic reindex | Hourly pipeline refresh; the fraud model retrains against the current vector version |
| Evaluation | Retrieval recall on a small labeled probe set | Recall@k time-split, tail recall, alignment/uniformity, online A/B | Downstream fraud PR-AUC only; retrieval metrics are meaningless here |
| What would be over-engineering | logQ correction, hard-negative mining, sharding, two towers | A cross-encoder at retrieval scale | Any ANN index, any recall metric, any serving latency work |

Two lessons fall out. First, the startup column is mostly deletions: with no
behavioral logs the entire negatives apparatus reduces to in-batch, and at 200k
vectors exact search removes the whole index-tuning surface. The one thing it
cannot skip is contrastive fine-tuning, because a raw pretrained encoder's space
is anisotropic and its cosines are not shaped for search
([section 5](05-evaluation.md)). Second, the fraud column shows that the ANN
index, which felt structural all chapter, is actually optional: when the only
consumer is a downstream model reading from a feature store, the embedding
pipeline keeps its training stages and drops its serving stages, and evaluation
collapses to the single downstream metric the system exists to move.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Cold-start SLA | Encoder family | Minutes-to-hours: inductive content encoder; retrain-cadence tolerable: transductive is cheaper and fine |
| Entity count x dimension | Index family and memory | ~500 MB per million entities at 128-d float32; fits RAM: HNSW; does not: quantize or IVF-PQ before touching d |
| Catalog churn | Index family | Heavy churn or geo/attribute filters: IVF absorbs updates HNSW pays graph-rebalancing cost for |
| Popularity skew | Negative strategy | In-batch plus logQ at training time, never at serving; report tail recall separately from head |
| No behavioral logs | Positive-pair source | Augmentation views (dropout, cropping) or a pretext task replace the log join entirely |
| Retrain cadence | Reindex and rollout | Every retrain is an atomic full reindex; never upsert new-model vectors into an old-model index |
| Downstream consumer | Similarity and eval k | The ANN metric must match the training similarity; measure recall@k at the k you actually pass downstream |
| Tens-of-ms retrieval | Probe depth, then dimension | Sweep efSearch or nprobe at a fixed index first; distance cost scales with d, so width is the second lever |
| Multi-task reuse | Evaluation | Judge by downstream lift (NDCG, MRR, PR-AUC) across consumers; one-task eval undersells the whole economic case |

## The smallest runnable embedding trainer

The review of every embedding tutorial is the same: the reader imports a
framework and never sees the mechanism. So here is the chapter's core loop in one
file with zero installs. Every production component is swapped for the smallest
thing with the same interface: the behavioral log becomes seeded sentences drawn
from three topic clusters, the InfoNCE batch becomes skip-gram negative sampling
updated by plain SGD, the encoder becomes a lookup table (deliberately
transductive: a token outside the vocabulary has no vector, which is the
cold-start problem in miniature), and the ANN index becomes an exhaustive cosine
scan. The shape is the lesson; every section of this chapter upgrades one
function of this file.

```python
"""Skip-gram with negative sampling in one file, runnable with no installs."""
import math, random

CLUSTERS = {
    "music":   ["guitar", "drums", "piano", "melody", "chord"],
    "cooking": ["oven", "flour", "butter", "simmer", "recipe"],
    "sports":  ["goal", "referee", "stadium", "coach", "league"],
}
VOCAB = [w for ws in CLUSTERS.values() for w in ws]

def make_corpus(rng, sentences=300, length=6):
    """Each sentence draws every token from one topic; production: behavioral logs,
    where a session or co-engagement plays the role of the sentence."""
    topics = list(CLUSTERS.values())
    return [[rng.choice(t) for t in [rng.choice(topics)] * length] for _ in range(sentences)]

CORPUS = make_corpus(random.Random(0))          # fixed: both runs train on identical data

def init_vecs(rng, dim):
    return {w: [rng.uniform(-0.5, 0.5) for _ in range(dim)] for w in VOCAB}

def dot(a, b): return sum(x * y for x, y in zip(a, b))

def cosine(a, b):
    na, nb = math.sqrt(dot(a, a)), math.sqrt(dot(b, b))
    return dot(a, b) / (na * nb) if na and nb else 0.0

def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

def train(seed, dim=16, epochs=8, lr=0.05, k_neg=4):
    """SGD on the negative-sampling objective; production: InfoNCE over a batch."""
    rng = random.Random(seed)                   # controls init + negatives, not the data
    center, context = init_vecs(rng, dim), init_vecs(rng, dim)
    for _ in range(epochs):
        for sent in CORPUS:
            for i, w in enumerate(sent):
                for j, c in enumerate(sent):
                    if i == j: continue
                    pairs = [(c, 1.0)] + [(rng.choice(VOCAB), 0.0) for _ in range(k_neg)]
                    for tok, label in pairs:    # positive pulls, negatives push
                        g = lr * (sigmoid(dot(center[w], context[tok])) - label)
                        for d in range(dim):
                            cw = center[w][d]
                            center[w][d] -= g * context[tok][d]
                            context[tok][d] -= g * cw
    return center

def neighbors(vecs, probe, n=3):
    ranked = sorted(((cosine(vecs[probe], vecs[w]), w) for w in VOCAB if w != probe),
                    reverse=True)
    return "  ".join(f"{w} {s:+.2f}" for s, w in ranked[:n])

untrained = init_vecs(random.Random(1), 16)
run_a, run_b = train(seed=1), train(seed=2)     # same corpus, independent training runs

print("nearest neighbors by cosine (before vs after training, run A)")
for probe in ("guitar", "flour", "referee"):
    print(f"  {probe:8s} before: {neighbors(untrained, probe)}")
    print(f"  {probe:8s} after : {neighbors(run_a, probe)}")

print("\nversion skew: two runs, identical data, independent training")
for probe, partner in (("guitar", "drums"), ("flour", "butter"), ("referee", "coach")):
    print(f"  cos(A.{probe}, B.{probe}) = {cosine(run_a[probe], run_b[probe]):+.2f}"
          f"   vs within-run cos(A.{probe}, A.{partner}) = {cosine(run_a[probe], run_a[partner]):+.2f}")
```

Run it and the two printouts demonstrate the chapter's two structural claims in
about sixty lines. Before training, a probe's neighbors are noise: guitar's
nearest tokens include referee, and flour's include drums, at cosines in the 0.2
to 0.4 range that carry no signal. After eight epochs, every probe's top
neighbors are its own topic cluster at cosine 0.94 and above: co-occurrence in
the corpus has become geometric neighborhood, which is the entire premise of
[section 2](02-frame-as-ml-task.md). The second printout is the version-skew
hazard from [section 6](06-serving-and-scaling.md) made concrete: run B trained
on the identical corpus, differing only in initialization and negative draws, yet
the cosine between run A's guitar and run B's guitar lands near zero (observed
-0.02, +0.34, +0.05 across the three probes) while within-run same-topic cosines
sit at +0.96. Both runs learned an equally good space; the contrastive objective
constrains only relative geometry, so each run lands in an arbitrary rotation of
it, and a dot product across runs compares coordinates that were never trained to
mean the same thing. That is why a retrain forces an atomic full reindex. Swap
the sentences for interaction logs, the lookup table for a two-tower encoder with
content features, the SGD loop for InfoNCE with logQ, and the cosine scan for an
HNSW index, and you have rebuilt this chapter.
