# 6. Serving and scaling

## Two indexes, one ingestion pipeline

The read path is cheap and the write path is where the engineering is.

**Read path budget** for a 300 ms promise: query understanding and embedding (5 to
15 ms), lexical retrieval (5 to 20 ms), ANN retrieval (5 to 20 ms), fusion
(negligible), ranking over about a thousand candidates (20 to 50 ms), cross-encoder
rerank over 50 (20 to 60 ms), plus network and page assembly. It fits, and the
cross-encoder is the first thing to drop under pressure.

**Write path** is the freshness promise: upload, transcode, language ID, ASR, frame
sampling, embedding, index write. The critical design decision is that the pipeline
must be **staged, not atomic**: publish the video to the lexical index as soon as
title and description exist (seconds), then upgrade it with the transcript and
embeddings as they land (minutes). A pipeline that waits for every modality before
anything is searchable turns a minutes promise into an hours promise.

## Index size math

For $N$ videos with $v$ vectors each, dimension $d$, at $b$ bytes per component:

$$\text{index bytes} \approx N \cdot v \cdot d \cdot b$$

| Configuration | Vectors | Size |
|---|---|---|
| 100M videos, 1 vector, 768 dims, fp32 | 100M | about 307 GB |
| Same, fp16 | 100M | about 154 GB |
| Same, product-quantized to 64 bytes | 100M | about 6.4 GB |
| 100M videos, 20 frame vectors, 768 dims, PQ to 64 bytes | 2B | about 128 GB |
| 100M videos, 600 frame vectors (per-second) | 60B | not a plan |

Two conclusions to state out loud. **Quantization is not optional at this scale**,
and it costs recall, so the recall-versus-memory curve is a design artifact you
measure rather than a default you accept. And **per-frame indexing multiplies
everything by frames per video**, which is why moment retrieval is usually scoped to
a subset of the corpus (popular, long-form, or explicitly opted-in) rather than
applied to all of it.

## The re-embedding problem

The line item nobody puts in the design and everybody hits in month six: **changing
the encoder invalidates the entire index.** Embedding 100 million videos is a batch
job measured in GPU-days, and during it the index contains two incompatible vector
spaces.

Mitigations, in increasing order of engineering:

1. **Version the index and swap.** Build the new index alongside, dual-write new
   uploads to both, cut over, delete the old. Costs double storage during the
   migration and is the simplest correct answer.
2. **Freeze the encoder, iterate on the ranker.** Most quality work does not need a
   new embedding space, and separating the two schedules is what keeps the migration
   rare.
3. **Backfill by priority.** Re-embed the head of the corpus first (what actually
   gets retrieved) and let the tail lag, accepting a temporary mixed-quality regime.

Say the migration plan in the interview when you propose the encoder. Proposing a
new embedding model without a re-embedding story is the tell of someone who has not
operated one.

## Caching and the head

Query traffic is Zipfian: a small number of queries are most of the volume. Cache the
full result page for head queries with a short TTL, which cuts both retrieval and
ranking cost, and be explicit that the cache must be invalidated by freshness rules
for time-sensitive queries (news, live). The tail, which is where the semantic
retriever earns its keep, is uncacheable by construction.

## Bottlenecks

| Bottleneck | First sign | Fix | Tradeoff |
|---|---|---|---|
| Ingestion backlog | Freshness SLO slips during upload peaks | Stage the pipeline; autoscale ASR and embedding workers; shed frame sampling first | New videos are text-only searchable for longer |
| ANN recall drop after compression | Offline recall falls, online tail quality falls | Re-tune the index (more probes, better PQ codebooks) or spend memory | Latency or cost |
| Cross-encoder latency | p99 breaches under load | Cut k, distil the reranker, or drop it under load shedding | Top-of-page quality |
| Re-embedding migration | A model upgrade is blocked for a quarter | Dual index and priority backfill | Double storage during migration |
| Duplicate results | Slots wasted on reuploads | Perceptual-hash plus transcript dedup at index time and again at rerank | Some legitimate near-duplicates collapse |
| Fresh videos never retrieved | Creator complaints; the fresh-result share falls | Content-only retrieval path plus an exploration allocation | Some low-quality new items get impressions |
| Head regression from semantic retrieval | Aggregate looks fine, navigational queries degrade | Fusion weighting that respects exact matches; slice-gated rollout | The semantic retriever contributes less on the head |

## Cost shape

The recurring costs sort into three buckets, and knowing which dominates is the
answer to "make it cheaper":

- **Ingestion** (ASR plus embedding) scales with upload volume and is usually the
  largest ongoing GPU line.
- **Index memory** scales with corpus size and vectors per video.
- **Query serving** scales with QPS and is the smallest of the three unless the
  cross-encoder is unbounded.

That ordering is why "sample fewer frames" and "one vector per video" are the first
two levers, and "use a smaller query encoder" is nearly always the wrong one.
