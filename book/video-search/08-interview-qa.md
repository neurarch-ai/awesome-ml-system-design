# 8. Interview Q&A

## Commonly asked

**Q: Walk me through the design.**
A: A funnel with two retrievers. Ingestion turns each upload into a multi-field
multimodal document: title, description, tags, channel, an ASR transcript, OCR on
sampled frames, and visual embeddings over a coarse frame sample. Retrieval runs
lexical (BM25 over the text fields) and dense (query embedding against an ANN index)
in parallel and fuses the candidates, because the two fail in different places:
lexical misses vocabulary mismatch, dense misses rare exact strings. A learned ranker
scores about a thousand candidates with relevance, engagement, freshness, and context
features, and a cross-encoder reranks the top 50 if the latency budget allows. Two
constraints shape everything: engagement features cannot be in retrieval or new
uploads are unreachable, and the document side must be precomputable or it cannot be
an index.

**Q: How do you handle the fact that most videos have a bad title?**
A: Stop treating the title as the document. The transcript is usually the strongest
tail signal, so the ingestion pipeline is where the quality comes from: ASR with
language identification first, OCR on sampled frames for tutorials and slides, and
visual embeddings for content where neither exists. Then the retrieval side needs a
representation that can match a descriptive query against text the uploader never
wrote, which is what the dense retriever is for. If transcripts are unavailable in
a language, that is where the visual tower earns its cost, and it is also where the
labels will be thinnest, so I would expect to measure that slice separately.

**Q: How much of the video do you embed?**
A: As little as the product allows, because frames per video multiplies the entire
index. Whole-video retrieval works with one to a few pooled vectors from a coarse
uniform sample plus scene-change frames. Moment retrieval (returning a timestamp)
needs per-frame or per-segment vectors, which multiplies the index by a factor of
twenty or more, so it is scoped to a subset of the corpus rather than applied
everywhere. For pooling, top-k rather than mean, because video relevance is usually
driven by the best-matching moment and mean pooling averages it away.

**Q: What is the label?**
A: Constructed, not raw. Click is a vote for the thumbnail, raw watch time is a vote
for long videos, and position is a vote for the current ranker. I would build a
graded label from duration-normalized watch, with reformulation and quick-back as
negative evidence, corrected for position with propensities from randomization or
interleaving, and anchor the whole thing with a few thousand human-rated (query,
video) pairs sliced by query type and language. The human set is what adjudicates
when offline and online disagree.

**Q: How do you evaluate it?**
A: By stage. Retrieval on recall@k against a pooled, human-anchored relevant set,
because recall against the current system's clicks punishes a retriever that finds
genuinely new results. Ranking on NDCG@k with graded labels over a fixed candidate
set, so a retrieval change does not silently move the ranking metric. Navigational
queries on MRR. Online: duration-normalized long-watch rate, reformulation rate,
abandonment, with latency and fresh-result share as guardrails. And everything
sliced by query type, language, and video age, because the expected failure of
adding semantic retrieval is a tail win with a head regression.

## Tricky

**Q: You added a dense retriever. Tail metrics improved, head queries got worse. Why,
and what do you do?**
A: The dense retriever generalizes, which is exactly wrong for a navigational query
where the user typed a title and wants that title. Semantically similar candidates
crowd out the exact match, and if the fusion weights the retrievers equally, the
exact match can lose. Fixes in order: make the fusion query-aware, so exact and
entity-heavy queries weight the lexical side much more heavily; add exact-match and
field-match features to the ranker so it can restore the right result even when
retrieval blurred it; and gate the rollout on the navigational slice rather than the
aggregate. The general lesson is that a semantic layer should be additive by
construction, not a replacement, which is why production systems keep the keyword
path.

**Q: Your index is 300 GB and the budget is 50. What do you cut, in order?**
A: Vectors per video first, since it is a multiplier: drop from a frame-level index
to a pooled one, keeping per-frame vectors only for the subset that needs moment
retrieval. Then precision: fp32 to fp16 is free-ish, product quantization to 64
bytes is roughly a 12x cut and it costs recall, so I would measure the
recall-versus-memory curve rather than pick a setting. Then dimension, via a smaller
encoder or a learned projection, which is the most quality-costly lever. What I would
not cut is the lexical index, which is small, exact, and carries the head.

**Q: A new encoder version is 4 points better offline. What does shipping it
actually involve?**
A: Re-embedding the corpus, which is the part people leave out. 100 million videos is
a GPU-days batch job, and during it the index holds two incompatible vector spaces,
so the plan is: build the new index alongside, dual-write new uploads to both, backfill
by retrieval priority (the head of the corpus first), then cut over and delete the
old, with double storage for the duration. That cost is also the argument for
freezing the encoder and iterating on the ranker instead, since most quality work does
not need a new embedding space. If the 4-point offline gain came from a benchmark
that is not sliced by query type, I would want that before committing the migration.

**Q: How do you keep a video uploaded 60 seconds ago findable?**
A: Stage the ingestion pipeline rather than making it atomic. Title and description
go into the lexical index immediately, so the video is searchable in seconds. The
transcript, OCR, and embeddings land minutes later and upgrade the document in place.
On the model side, retrieval must be content-only so that zero engagement is not
disqualifying, and the ranker's engagement features need a prior for new items rather
than a zero, plus a small exploration allocation so new uploads get the impressions
that generate their first real signal.

## Commonly answered wrong

**Q: Just embed each video into one vector and do nearest-neighbor search. Done?**
A: That throws away three things you need. **Exactness**: the head of the traffic is
navigational and wants a title match, which a smoothed semantic vector is worst at.
**Attribution**: with one vector you cannot say which field matched, which you need
for debugging, spam defense, and explaining results. **Graceful degradation**: a
missing transcript should weaken one field, not corrupt the whole representation.
The production shape everywhere is multiple fields, two retrievers, and fusion.

**Q: Use watch time as the relevance label, it is the most honest signal.**
A: Raw watch time is duration-confounded: a 40-minute video that holds someone for
four minutes outranks a two-minute video that answered the question completely.
Normalize by the video's own length or use a duration-aware long-watch threshold,
and pair it with reformulation as negative evidence, because a user who searched
again immediately did not find what they wanted no matter how long they watched.
Also worth naming: watch time is measured only on results that were *seen*, so it
carries position bias, and correcting for that is a separate step.

**Q: CLIP is zero-shot, so we can skip training data.**
A: It is a strong starting point and a weak ending point. An image-text aligned
encoder gives you a shared space without task-specific data, which is genuinely
useful in the languages and content types where you have nothing else. But it is
shallow per frame, weak on rare entities and on text rendered inside the video, and
it has no idea what *your* users mean by a query. The production path is to use it as
initialization, then train on your own (query, video) pairs with hard negatives mined
from your own lexical index, which is where most of the quality comes from.

**Q: Put engagement features in the retriever so popular videos are retrieved first.**
A: That makes new uploads structurally unreachable: no impressions means no
engagement means never retrieved means never any impressions. It also tightens the
feedback loop, because the retriever now trains on the exposure it caused. Engagement
belongs in ranking, where a cold item can still be reached and given a prior, with an
exploration allocation to break the loop.

**Q: The offline recall of the new retriever is lower, so it is worse.**
A: Check what the relevant set is first. If it was built from the current system's
clicks, then any retriever that surfaces videos the old system never showed is
penalized for finding new things, which is the opposite of what you want. Recall has
to be measured against a pooled relevant set with human judgments over candidates
from *both* systems. This is the single most common way a good retrieval change gets
killed offline.
