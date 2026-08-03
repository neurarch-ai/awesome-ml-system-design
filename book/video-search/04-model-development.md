# 4. Model development

## The dual encoder, and what it must not be

Retrieval needs a scoring function that factorizes, so the document side can be
precomputed and indexed:

$$s(q, d) = \langle f_{\text{query}}(q),\; g_{\text{doc}}(d) \rangle$$

That factorization is the entire reason the system scales: $g$ runs once per video at
ingestion, $f$ runs once per query. Anything that needs the query and the document
together (a cross-encoder) cannot be an index and therefore cannot be retrieval.

Training is contrastive, with the softmax over in-batch and mined hard negatives:

$$\mathcal{L} = -\log \frac{\exp(s(q, d^{+})/\tau)}{\exp(s(q, d^{+})/\tau) + \sum_{d^{-}} \exp(s(q, d^{-})/\tau)}$$

The temperature $\tau$ controls how hard the loss pushes on near-misses, and the
negatives decide what "near" means. This is the same machinery as
[embeddings](../embeddings/) and [candidate retrieval](../candidate-retrieval/); what
is specific here is the document side.

## The document side: fields and modalities

**Text tower.** Encode the concatenated text fields, but not naively: title and
transcript have different lengths and different reliability, and a transcript can be
tens of thousands of words. Practical pattern: encode fields separately, then combine
with learned field weights, or encode a truncated composite (title, description,
plus the highest-scoring transcript passages) and let the lexical index carry exact
long-tail terms. Long transcripts are usually chunked into passages, which turns one
video into several retrievable units.

**Visual tower.** Encode sampled frames with an image-text aligned encoder so that
queries and frames live in one space (the CLIP line of work,
[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)),
then aggregate over time:

| Aggregation | Behavior | Cost |
|---|---|---|
| Mean pool | Smooths everything; a 5-second cake shot in a 20-minute vlog disappears | Cheapest |
| Max or top-k pool | Preserves "this moment matched", which is usually what the user wants | Cheap |
| Attention pool over frames | Learns which frames matter for retrieval | Needs training data |
| No aggregation (per-frame index) | Enables moment retrieval | Multiplies the index by frames per video |

Top-k pooling is the default worth defending in an interview: video relevance is
usually driven by the best-matching moment rather than the average frame, and mean
pooling is exactly the operation that destroys it.

**Fusion.** Late fusion (separate towers, combine scores) is the pragmatic choice:
each modality can be trained, refreshed, and debugged independently, and a missing
modality (no transcript) degrades gracefully. Early fusion (one tower over all
modalities) can be stronger but couples everything, including your re-embedding
schedule.

## Retrieval fusion: combining lexical and dense

Two candidate lists have to become one. Reciprocal rank fusion is the strong
baseline because it needs no score calibration between systems:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$

with $k$ a small constant (60 is the common default). A learned fusion using the raw
scores plus query features can beat it, at the cost of needing recalibration whenever
either retriever changes. Spotify's production description of adding natural-language
search alongside keyword search is the reference shape: the dense retriever supplies
semantic candidates, and a final ranker blends them with the existing keyword
results rather than replacing them
([Introducing Natural Language Search for Podcast Episodes](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes)).

## Ranking and reranking

The ranker sees a thousand candidates and can afford features retrieval cannot:
query-document relevance scores from both retrievers, field-level match features,
engagement priors, freshness, channel authority, and user context. It is a
learning-to-rank problem, and the [search ranking](../search-ranking/) chapter owns
the objective choices.

The reranker sees the top 50 and can afford a **cross-encoder**, which reads query
and document jointly and is materially more accurate. Two ways to use it:

1. **Online**, on the top-k, when latency allows (tens of milliseconds for a small
   model over short text).
2. **Offline as a teacher**, distilling its judgments into the dual encoder, which
   is how you get some of the accuracy without the latency. This is the standard
   move when the cross-encoder does not fit the budget.

## Cold and fresh videos

A video uploaded a minute ago has no engagement, so any model that leans on
engagement cannot retrieve it. The design consequence: **retrieval is content-only**
(text and visual), engagement enters at ranking with a prior that decays to a
default for new items, and a small exploration allocation gives new uploads
impressions they would not otherwise earn (see [cold start](../cold-start/)).

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Dense text retrieval | Descriptive tail queries, transcripts exist | A visual tower, until text is exhausted |
| Image-text aligned visual tower | Poor transcript coverage or visually specific queries | Training a video encoder from scratch |
| Top-k frame pooling | Whole-video retrieval where the best moment matters | Mean pooling, which averages the signal away |
| Per-frame index | The product promises timestamps | Pooling, which cannot localize |
| RRF fusion | You need something robust today | A learned fusion that needs recalibration on every change |
| Cross-encoder online | Latency budget allows a second pass over 50 candidates | Growing the dual encoder for the same accuracy |
| Cross-encoder distillation | Latency does not allow it | Shipping the cross-encoder in retrieval, which cannot be indexed |

**Provenance.** The dual-encoder retrieval pattern with in-batch negatives is dense
passage retrieval ([DPR](https://arxiv.org/abs/2004.04906), Facebook AI, 2020); the
image-text aligned space is CLIP (OpenAI, 2021); the transcript field is produced by
a modern ASR system such as [Whisper](https://arxiv.org/abs/2212.04356) (OpenAI,
2022); ANN serving descends from [FAISS](https://arxiv.org/abs/1702.08734) (Meta,
2017) and [ScaNN](https://arxiv.org/abs/1908.10396) (Google, 2020).

**Tools.** Dual encoders are trained with sentence-transformers or a custom PyTorch
loop; ANN indexes are FAISS, ScaNN, or a managed vector store; the lexical half is
Elasticsearch, OpenSearch, or Lucene directly; ASR is Whisper or a managed speech
API; frame extraction is ffmpeg. The reranker is a small cross-encoder from the same
sentence-transformers ecosystem.
