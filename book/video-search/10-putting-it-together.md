# 10. Putting it together: the complete build

Every earlier section presented options. This one commits: one stack, decided, with
the arithmetic that justifies it, then the same system re-derived under three
constraint sets, and a runnable experiment you can execute with nothing but Python 3.

The scenario from [section 1](01-clarifying-requirements.md): 100 million videos
growing by tens of thousands an hour, a strong navigational head plus a descriptive
tail that keyword search handles badly, partial transcripts, searchable within
minutes of upload, 300 ms end to end, on a constrained index budget.

## The default stack

| Decision | Committed choice | Why, in one line |
|---|---|---|
| Retrieval | Lexical (BM25 over text fields) plus dense text, both always on | They fail in different places: vocabulary mismatch versus rare exact strings |
| Visual tower | Added only for transcript-poor languages and visually specific queries | It is the most expensive recall per dollar; earn it with a measured slice |
| Fusion | Query-aware weights from a lightweight query classifier, with RRF as the fallback when scores are uncalibrated | Equal-weight fusion regresses the head, which the experiment below reproduces |
| Document representation | Multi-field, one pooled vector per video (top-k over sampled frames) | One vector per video is a search product; per-frame is a moment-retrieval product |
| Frame sampling | Uniform every 10 to 30 seconds plus scene-change frames | Frames per video is the index multiplier |
| Transcript | ASR with language ID first, word timings stored | The strongest tail signal, and timings make moment retrieval possible later |
| Engagement | Ranking only, with a prior for new items | In retrieval it makes new uploads unreachable and closes the feedback loop |
| Reranking | Cross-encoder over the top 50, first thing dropped under load | Best accuracy per millisecond at the only depth users see |
| Labels | Duration-normalized long watch, reformulation as negative evidence, position-corrected, anchored by a human-rated set | Clicks vote for thumbnails, raw watch time votes for length |
| Index | Product-quantized dense index plus the lexical index, staged ingestion | Quantization is not optional at 100M; staging is what makes the freshness promise |
| Evaluation | Recall@k for retrieval on a pooled relevant set, NDCG@k for ranking over a fixed candidate set, sliced by query type, language, and video age | The expected failure is a tail win with a head regression |

## The arithmetic

**Index.** One pooled 768-dimension vector per video is 307 GB in fp32, 154 GB in
fp16, and 6.4 GB product-quantized to 64 bytes. Twenty frame vectors per video at
the same quantization is 128 GB. Six hundred (one per second on a ten-minute video)
is 3.8 TB, which is the arithmetic that ends the "index every frame" conversation.

**Ingestion.** Tens of thousands of uploads an hour, each needing transcode, ASR,
frame sampling, and embedding, is a continuous GPU workload whose size scales with
upload volume rather than with query volume. It is normally the largest line, which
is why "sample fewer frames" is the first cost lever and "use a smaller query
encoder" is nearly always the wrong one.

**Query.** One query embedding, two retrievals, a ranker over about a thousand
candidates, and a cross-encoder over 50, inside 300 ms. The cross-encoder is roughly
a third of that budget and is the designated thing to shed under load.

## The same system under three constraint sets

**Small corpus, no ML team yet (a few hundred thousand videos).** Lexical over
title, description, and transcript gets most of the way, and the highest-value
addition is not a retriever at all, it is better transcripts. Add an off-the-shelf
dense retriever with RRF fusion when the tail complaints start, skip the visual
tower entirely, and skip reranking. One vector per video in fp16 fits in memory
without quantization, which removes an entire tuning problem.

**Transcript-poor corpus (short-form, music, many languages).** The visual tower
stops being optional and becomes the main tail signal, so the budget shifts from ASR
to frame sampling and embedding. Expect the training data to be thinnest exactly
where the model matters most, so invest early in a human-rated set per language, and
weight the fusion by language as well as by query type.

**Moment retrieval promised in the product.** The index granularity changes from
video to segment, which multiplies the vector count by twenty or more, so it is
scoped: only long-form, popular, or explicitly opted-in videos get segment vectors,
and everything else stays whole-video. Word-level transcript timings become
load-bearing rather than a nice-to-have, and the ranker gains a segment-selection
stage on top of the video-selection one.

## The smallest runnable experiment

```python
"""Hybrid video retrieval on one page. Python 3, standard library only.

Five retrieval strategies over the same synthetic corpus and query mix, scored on
recall@10 and MRR, sliced by query type. The point is the shape, not the absolute
numbers: lexical owns navigational queries, dense owns descriptive ones, naive
equal-weight fusion regresses the head, and query-aware weighting is what makes
fusion worth shipping.

Model: for each query, each retriever draws a score N(signal, 1) for the relevant
video, while distractors score N(0, 1). Fusing with weights w gives the relevant
video a score N(w.signal, |w|), so a retriever with no signal for that query type
contributes pure noise and dilutes the ones that do.

Run: python3 video_search_sim.py
"""

import random
from math import erf, sqrt

random.seed(3)

CORPUS = 1_000_000          # videos the retriever ranks against
INDEX_CORPUS = 100_000_000  # videos in the production index, for the cost table
QUERIES = 3000              # queries per type
RERANK_DEPTH, RERANK_ACC = 50, 0.75    # cross-encoder over the top 50

RETRIEVERS = ("lexical", "dense", "visual")
# how much signal each retriever carries for the relevant video, by query type
SIGNAL = {
    "navigational": {"lexical": 5.6, "dense": 3.2, "visual": 0.0},
    "descriptive":  {"lexical": 1.6, "dense": 5.4, "visual": 0.6},
    "visual":       {"lexical": 0.8, "dense": 2.6, "visual": 5.2},
}
MIX = {"navigational": 0.45, "descriptive": 0.40, "visual": 0.15}   # traffic share


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))          # standard normal CDF


def rank_from(z):
    """How many of CORPUS distractors (unit-normal scores) beat this score."""
    p = 1 - phi(z)
    mean, var = CORPUS * p, CORPUS * p * (1 - p)
    return 1 + max(0.0, random.gauss(mean, sqrt(var + 1e-9)))


def normalized(w):
    """Scale weights so the fused distractor score stays unit-normal (a fair comparison)."""
    n = sqrt(sum(x * x for x in w.values())) or 1.0
    return {k: v / n for k, v in w.items()}


def reranked(rank):
    """A cross-encoder over the top RERANK_DEPTH promotes a true hit with prob RERANK_ACC."""
    return 1.0 if rank <= RERANK_DEPTH and random.random() < RERANK_ACC else rank


EQUAL = normalized({r: 1.0 for r in RETRIEVERS})
STRATEGIES = ("lexical only", "dense only", "fusion, equal weights",
              "fusion, query-aware", "query-aware + rerank")
results = {s: {qt: [] for qt in SIGNAL} for s in STRATEGIES}

for qt, sig in SIGNAL.items():
    aware = normalized({r: max(sig[r], 0.1) for r in RETRIEVERS})   # a query classifier's weights
    for _ in range(QUERIES):
        draw = {r: random.gauss(sig[r], 1.0) for r in RETRIEVERS}   # one score per retriever
        results["lexical only"][qt].append(rank_from(draw["lexical"]))
        results["dense only"][qt].append(rank_from(draw["dense"]))
        results["fusion, equal weights"][qt].append(
            rank_from(sum(EQUAL[r] * draw[r] for r in RETRIEVERS)))
        qa = rank_from(sum(aware[r] * draw[r] for r in RETRIEVERS))
        results["fusion, query-aware"][qt].append(qa)
        results["query-aware + rerank"][qt].append(reranked(qa))


def recall_at(ranks, k=10):
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mrr(ranks):
    return sum(1.0 / r for r in ranks) / len(ranks)


def weighted(strategy, fn):
    return sum(MIX[qt] * fn(results[strategy][qt]) for qt in SIGNAL)


print("recall@10 by query type, then traffic-weighted overall")
print(f"{'strategy':>22} {'navig.':>7} {'descr.':>7} {'visual':>7} {'overall':>8} {'MRR':>6}")
for s in STRATEGIES:
    row = [recall_at(results[s][qt]) for qt in SIGNAL]
    print(f"{s:>22} {row[0]:7.2f} {row[1]:7.2f} {row[2]:7.2f} "
          f"{weighted(s, recall_at):8.2f} {weighted(s, mrr):6.2f}")
print("note: equal-weight fusion beats each single retriever overall and still loses")
print("      on the navigational head, where one retriever is contributing pure noise.")

print()
print(f"index cost at {INDEX_CORPUS // 1_000_000}M videos: videos x vectors x bytes per vector")
for label, vecs, bytes_per_vec in [("1 vec, fp32 768d", 1, 768 * 4),
                                   ("1 vec, fp16 768d", 1, 768 * 2),
                                   ("1 vec, PQ 64B", 1, 64),
                                   ("20 frame vecs, PQ 64B", 20, 64),
                                   ("600 frame vecs, PQ 64B", 600, 64)]:
    print(f"  {label:>23}: {INDEX_CORPUS * vecs * bytes_per_vec / 1e9:9.1f} GB")
print("  frames per video multiplies everything, which is why moment retrieval is")
print("  scoped to a subset of the corpus rather than applied to all of it.")
```

Output:

```text
recall@10 by query type, then traffic-weighted overall
              strategy  navig.  descr.  visual  overall    MRR
          lexical only    0.91    0.00    0.00     0.41   0.36
            dense only    0.14    0.86    0.05     0.41   0.33
 fusion, equal weights    0.78    0.53    0.76     0.68   0.52
   fusion, query-aware    0.99    0.91    0.95     0.95   0.87
  query-aware + rerank    0.99    0.94    0.97     0.97   0.95
note: equal-weight fusion beats each single retriever overall and still loses
      on the navigational head, where one retriever is contributing pure noise.

index cost at 100M videos: videos x vectors x bytes per vector
         1 vec, fp32 768d:     307.2 GB
         1 vec, fp16 768d:     153.6 GB
            1 vec, PQ 64B:       6.4 GB
    20 frame vecs, PQ 64B:     128.0 GB
   600 frame vecs, PQ 64B:    3840.0 GB
  frames per video multiplies everything, which is why moment retrieval is
  scoped to a subset of the corpus rather than applied to all of it.
```

Three things to take from it. **Neither single retriever is acceptable**: lexical
answers the head and returns nothing useful for descriptive or visual queries, dense
is the mirror image. **Equal-weight fusion is a trap**: it beats both single
retrievers overall (0.68 against 0.41) while dropping the navigational head from 0.91
to 0.78, which in traffic terms is the worst possible trade and is exactly the
regression that gets a semantic launch rolled back. **Query-aware weighting is what
makes fusion free**: 0.99 on the head, 0.91 and 0.95 on the tail, and the
cross-encoder then buys most of its value in MRR (0.87 to 0.95) rather than in
recall, because its job is ordering the results a user actually sees.
