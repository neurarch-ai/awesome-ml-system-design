# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage of the query-document ranking funnel with
its options and tradeoffs; section 7 showed where real teams diverge. What none
of them show is a single system with every decision made. This capstone does
three things: it gives you an opinionated default stack so option paralysis
never blocks a first build, it walks the chapter's scenario end to end with
every choice committed and sized, and it shows how the same decisions flip when
the constraints change. It closes with the smallest runnable ranking funnel,
one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing retrieval libraries before ranking a single
document. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Tools
change yearly, but the interface of each stage (understand the query, retrieve
candidates, assemble features, rank, evaluate, serve) does not, so pick per
stage by interface and treat any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Query understanding | Spelling correction + intent classifier + conservative expansion, all cached for the head | Queries are structured (SKU lookups, filters): a parser beats a classifier | [6](06-serving-and-scaling.md) |
| Lexical retrieval | BM25 over an inverted index, top ~500 | Never retire it; it is unbeatable on exact and rare terms | [4](04-model-development.md) |
| Dense retrieval | Dual-encoder + ANN (HNSW), top ~500, run in parallel with lexical | Corpus is tiny or queries are all exact-term: defer the dense arm | [4](04-model-development.md) |
| Fusion | Union + dedupe to ~1,000; RRF if retrieval order matters downstream | Never sum raw scores; the scales are incomparable | [4](04-model-development.md) |
| Feature set | Match (BM25, cosine), quality (position-normalized CTR), freshness, light context | Behavioral features join post-query data: point-in-time join is mandatory | [3](03-data-preparation.md) |
| Label pipeline | Dwell-filtered clicks debiased with IPW or position-as-feature, calibrated on human judgments | Never train on raw clicks; the model learns position, not relevance | [3](03-data-preparation.md) |
| LTR model | LambdaMART over hand-crafted features, batch-scored | Many sparse categorical features and data to burn: deep ranker (DLRM, DCN V2) | [4](04-model-development.md) |
| Evaluation | NDCG@10 on a time-based split offline; interleaving or A/B on engagement + reformulation as the ship gate | Never ship on offline NDCG alone | [5](05-evaluation.md) |

The label-pipeline row is the one beginners skip and regret: without debiasing,
every offline NDCG gain is suspect, because the model may simply have learned to
reproduce the order you already shipped. Position bias is the dominant label
problem of this chapter, and it is fixed in the data, not the model.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md):
e-commerce product search over a few hundred million products, tens of
thousands of queries per second at peak, low hundreds of milliseconds
end-to-end with tens of milliseconds for ranking, graded relevance judged at
the top positions, labels that are mostly biased click logs plus a thin layer
of human judgments, and new listings searchable within minutes. Here is the
whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Funnel shape | Retrieve ~1,000, rank to ~100, show ~10 | Tens of milliseconds rules out scoring 100M documents with any learned model |
| Query understanding | Cached spelling correction, intent classifier, conservative expansion | Single-digit ms; everything downstream waits on it; over-expansion drifts precision |
| Lexical arm | BM25 over a sharded inverted index, WAND early termination | Product codes and brand names are exact-term queries the dense arm blurs |
| Dense arm | Dual-encoder trained with in-batch negatives, HNSW index | Closes the "laptop" to "notebook computer" vocabulary gap lexical cannot |
| Fusion | Union + dedupe; both arms in parallel | Failure modes are complementary; series execution wastes 10-20 ms |
| Labels | Dwell-filtered clicks, IPW-debiased, calibrated on human-judged pairs | Clicks alone encode position; judgments alone cannot cover the tail |
| Features | BM25 + cosine as features, position-normalized CTR, freshness x intent, point-in-time joined | The ranker should see the retrieval signal, not rebuild it |
| Ranker | LambdaMART, batch forward pass over all candidates | Metric is position-weighted NDCG; listwise loss aligns with what is reported |
| Freshness | Streaming embed-and-upsert pipeline, minutes cadence | The minutes SLA rules out nightly index rebuilds |
| Evaluation | NDCG@10 + MRR offline on a time split; interleaving then A/B on engagement and reformulation as the gate | Offline NDCG is computed against biased labels and can lie |

**Funnel arithmetic.** Each stage prunes by roughly a factor of a hundred
([section 4](04-model-development.md)): 100M documents in the corpus, ~500 from
each retrieval arm unioned and deduped to ~1,000 candidates, ~100 survivors of
the ranker's forward pass, ~10 shown. The consequence worth stating is where
capacity goes: the LTR model only ever sees one hundred-thousandth of the
corpus per query, so a retrieval miss is unrecoverable and retrieval is tuned
for recall, never precision.

**Index scale.** At 100M products and 128-dimensional float32 embeddings
(Illustrative), the dense index holds 100M x 128 x 4 bytes, about 51 GB of raw
vectors before HNSW graph overhead, which forces sharding; the inverted index
shards the same way, by document. Catalog churn sets the write path: a
streaming pipeline on the order of thousands of embeddings per second (the
Shopify reference in [section 7](07-how-teams-do-it-in-production.md) sustains
2,500/s) keeps the minutes-level freshness promise.

**Label volumes.** At tens of thousands of QPS, one day of logs is on the
order of a billion queries with ~10 impressions each (Illustrative), so click
labels have volume and tail coverage to spare; their problem is bias, not
scarcity. The human-judgment budget is the scarce side: a WANDS-scale set
(hundreds of head queries, a couple hundred thousand graded pairs,
[section 7](07-how-teams-do-it-in-production.md)) anchors calibration and is
refreshed on the head query distribution, which is heavy and stable.

**Latency budget.** The component budget from
[section 6](06-serving-and-scaling.md): ~5-10 ms query understanding, then the
two retrieval arms in parallel at ~10-20 ms (the slower arm sets the cost),
~20-30 ms for the batched LTR forward pass over ~1,000 candidates, ~5 ms for
diversity re-rank. That lands the ranking path at roughly 40-65 ms, inside the
tens-of-milliseconds ranking budget and leaving the rest of the low-hundreds
end-to-end budget for network and rendering. The caching layers (spelling,
query embedding, full results for the top few hundred queries) exist because
the query distribution has a heavy head; they buy back the p99 that
inverted-index fan-out on terms like "shoes" would otherwise spend.

**What breaks in month one.** Three failure modes dominate early operations,
so wire their signals before launch: dense-arm staleness (age of the newest
listing present in the ANN index against the minutes promise; new products
appear in lexical results but vanish from semantic ones, and sellers notice
first), the offline-online gap (offline NDCG climbing while reformulation rate
and engagement stay flat is the signature of position bias or a point-in-time
join leak, [section 5](05-evaluation.md)), and head-query tail latency (p99
spikes on broad terms are inverted-index fan-out, fixed with WAND and result
caching, not with a smaller ranker).

## The same techniques under different constraints

The review question that matters in practice is not "which ranker is best" but
"which ranker is best under my constraints." Here is the same funnel built
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Internal wiki search | E-commerce product search (this chapter) | News search |
|---|---|---|---|
| Corpus / traffic | 200k documents; 2 QPS | Hundreds of millions of products; tens of thousands QPS | Tens of millions of articles, thousands added hourly; spiky QPS |
| Latency budget | A second is fine | Tens of ms for ranking, low hundreds end-to-end | Similar to product search, but freshness beats latency tuning |
| Query understanding | Spelling correction only | Cached correction + intent + conservative expansion | Intent classifier is load-bearing: "breaking" vs evergreen steers everything |
| Retrieval | BM25 alone; add the dense arm only when synonym misses show up in logs | Both arms in parallel, union + dedupe to ~1,000 | Both arms with a time-decayed index; recency filters inside retrieval |
| Ranking model | Hand-tuned linear over a few features, upgraded to pointwise LTR later | LambdaMART over match + quality + freshness features | LTR where freshness x intent dominates the feature importances |
| Labels | A few hundred human-judged pairs; too little traffic to debias clicks | IPW-debiased clicks calibrated on WANDS-scale judgments | Clicks decay fast; labels expire with the news cycle and retraining is frequent |
| Freshness | Reindex nightly; nobody notices | Streaming embed + upsert within minutes | The product: minutes-old articles must outrank yesterday's on the same terms |
| Eval | Golden set of judged queries, checked by hand | NDCG@10 + MRR offline; interleaving then A/B as the gate | NDCG on a sliding time window; an eval set older than a week is meaningless |
| What would be over-engineering | Dense arm, ANN index, debiasing pipeline, interleaving infra | Cross-encoder over the full candidate set; LLM rewrites of every query | Deep personalization; the story is the same for everyone |

Two lessons fall out. First, the wiki column is mostly deletions: at 200k
documents and 2 QPS there is no click volume worth debiasing, no latency
problem worth caching, and BM25 alone covers a corpus where employees search
by exact page titles; the Yelp row of
[section 7](07-how-teams-do-it-in-production.md) shows the same shape shipped
in production. Second, the news column shows the same stages with a different
binding constraint: when the corpus turns over hourly, freshness stops being
one feature among four families and becomes the axis the index, the labels,
and the eval window are all built around.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Corpus size | Funnel depth | Small enough to score every document in budget: one stage. Otherwise retrieve ~1,000 then rank; each stage prunes ~100x |
| Ranking latency budget | Candidate count and ranker family | Tens of ms: batched LambdaMART over ~1,000. Tighter: shrink candidates before shrinking the model |
| Query mix | Which retrieval arm dominates | Codes, brands, rare strings: lexical leads. Paraphrase and natural language: dense leads. Mixed (the default): both arms, always |
| Click volume | Debiasing method | Enough traffic for a small randomization experiment: estimate propensities for IPW. Otherwise position-as-a-train-time-feature |
| Label budget | Loss and model class | Judgments only, no click volume: pointwise on a small set. Graded labels at volume: pairwise or listwise (LambdaMART) |
| Freshness SLA | Write-path architecture | Minutes: streaming embed + upsert. Daily: batch reindex. The lexical arm is fresh on index write either way |
| Rendered k | Eval metric | Report NDCG at the k you render; add MRR when navigational queries need "the one right answer first" |
| Head-heavy query distribution | Caching layers | Heavy stable head: spelling, query-embedding, and full-result caches pay immediately; long-tail-only traffic: they do not |
| Personalization scope | Feature families | Location and language always; behavioral signals light and secondary; if the query stops dominating, you are building a feed, not search |

## The smallest runnable ranking funnel

The review of every search tutorial is the same: the reader stands up
Elasticsearch and a vector database and still cannot see the funnel. So here is
the entire query-document ranking path in one file with zero installs. Every
production component is swapped for the smallest thing with the same interface:
the inverted index becomes a dict over a seeded toy catalog, the LTR model
becomes three hand-weighted features, the human judgment set becomes a labeled
dict, and the metric is the real NDCG@5 from
[section 5](05-evaluation.md). The shape is the lesson; every section of this
chapter upgrades one function of this file.

```python
"""The ranking funnel in one file: BM25 retrieval, feature rerank, NDCG@5."""
import math, random
from collections import Counter

# --- seeded toy catalog: id -> (title, body, age_days) -----------------------
random.seed(0)
FILLER = ("blender kettle toaster lamp desk chair sofa curtain rug shelf vase "
          "drill hammer wrench ladder bucket paint brush tarp glove").split()
DOCS = {f"f{i:02d}": ("home item", " ".join(random.choices(FILLER, k=25)), 90)
        for i in range(40)}                       # 40 irrelevant filler products
DOCS.update({
    "p1": ("trail running shoes", "lightweight trail running shoes with grip", 3),
    "p2": ("shoes outlet", " ".join(["shoes"] * 60) + " clearance bargain", 400),
    "p3": ("road running shoes", "cushioned road running shoes for training", 8),
    "p4": ("waterproof hiking jacket", "breathable waterproof jacket for rain", 5),
    "p5": ("leather dress shoes", "classic leather dress shoes for formal wear", 30),
    "p6": ("rain jacket", "packable rain jacket with waterproof seams", 2),
    "p7": ("winter jacket", "waterproof jacket waterproof jacket, discontinued", 500),
})
REL = {  # graded labels: 3 perfect, 2 good, 1 fair, 0 bad (human-judged stand-in)
    "running shoes":     {"p1": 3, "p3": 3, "p5": 1, "p2": 0},
    "waterproof jacket": {"p4": 3, "p6": 3, "p7": 1},
}

toks = lambda s: s.lower().split()
TOKS = {d: toks(t) + toks(b) for d, (t, b, _) in DOCS.items()}
N = len(DOCS)
AVGDL = sum(len(v) for v in TOKS.values()) / N
DF = Counter(t for v in TOKS.values() for t in set(v))

def score_tf(q, d):                       # naive baseline: raw term frequency
    tf = Counter(TOKS[d])
    return sum(tf[t] for t in toks(q))

def score_bm25(q, d, k1=1.2, b=0.75):
    tf, dl = Counter(TOKS[d]), len(TOKS[d])
    s = 0.0
    for t in toks(q):
        idf = math.log((N - DF[t] + 0.5) / (DF[t] + 0.5) + 1)   # rare terms weigh more
        s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / AVGDL))
    return s

def rank(q, scorer, n=None):
    return sorted(DOCS, key=lambda d: scorer(q, d), reverse=True)[:n]

def rerank(q, cands, m=5):                # stage 2: tiny feature-based LTR stand-in
    qt, top = set(toks(q)), score_bm25(q, cands[0]) or 1.0
    def f(d):
        title, _, age = DOCS[d]
        return (1.0 * score_bm25(q, d) / top                    # retrieval score
                + 0.8 * len(qt & set(toks(title))) / len(qt)    # title match
                + 0.5 / (1 + age / 30))                         # freshness
    return sorted(cands, key=f, reverse=True)[:m]

def ndcg(q, ranked, k=5):
    dcg = lambda rs: sum(r / math.log2(i + 2) for i, r in enumerate(rs[:k]))
    got = [REL[q].get(d, 0) for d in ranked]
    return dcg(got) / dcg(sorted(REL[q].values(), reverse=True))

print('retrieval scorers on "running shoes" (p2 = stale doc stuffed with "shoes"):')
print("  raw-tf top3:", rank("running shoes", score_tf, 3), " tf rewards stuffing")
print("  bm25   top3:", rank("running shoes", score_bm25, 3), " idf + length norm fix it")
for q in REL:
    cands = rank(q, score_bm25, 20)                             # stage 1: top-20
    two = rerank(q, cands)                                      # stage 2: top-5
    print(f'query "{q}"')
    print(f"  bm25-only NDCG@5 = {ndcg(q, cands):.3f}  {cands[:5]}")
    print(f"  two-stage NDCG@5 = {ndcg(q, two):.3f}  {two}")
```

Run it and the output walks the chapter's two core claims in about seventy
lines. The scorer comparison shows raw term frequency ranking the
keyword-stuffed stale document p2 first for "running shoes" while BM25 drops it
to third: IDF discounts "shoes", which nearly every relevant product contains,
and length normalization stops sixty repetitions in a long document from
buying rank. The funnel comparison then shows why retrieval alone is not
ranking: for "waterproof jacket", BM25-only scores 0.815 NDCG@5 because the
discontinued p7 packs the query terms densely and outranks the two perfect
products, and the two-stage pass reaches 1.000 by letting title match and
freshness demote it, exactly the quality-and-freshness feature families of
[section 3](03-data-preparation.md). Swap the dict for an inverted index and an
ANN arm, the hand weights for LambdaMART trained on debiased clicks, the `REL`
dict for judged and debiased labels, and the loop for an interleaving
experiment, and you have rebuilt this chapter.
