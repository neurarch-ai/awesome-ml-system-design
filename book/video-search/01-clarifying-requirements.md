# 1. Clarifying the requirements

**Candidate:** What does the query distribution look like? Mostly short
navigational queries, or natural-language questions?

**Interviewer:** Both. A large head of title-like queries, and a long tail of
descriptive ones that our current keyword search handles badly.

**Candidate:** That split decides where the budget goes. The head is already served
well by a lexical index and I should be careful not to regress it; the tail is where
a semantic retriever earns its cost.

**Interviewer:** Assume you must improve the tail without touching the head.

**Candidate:** What is the corpus size and the upload rate? Index freshness is
usually the constraint people underestimate.

**Interviewer:** About 100 million videos, tens of thousands of new ones an hour.
New uploads should be searchable within minutes.

**Candidate:** Do we have transcripts, and for which languages? Text on the video is
almost always the strongest tail signal, and whether it exists changes the whole
design.

**Interviewer:** Auto-generated captions for the top few languages, patchy elsewhere.

**Candidate:** So the visual tower carries more weight in the languages where
transcripts are thin, which is also where our labels will be sparsest. Next: what
counts as a good result? Click, watch, or something else?

**Interviewer:** We care about whether the user found what they wanted. We log
clicks, watch time, and reformulations.

**Candidate:** Then I would build a relevance label rather than use a raw behavioral
one, because click favors thumbnails and watch time favors long videos. Last:
latency and cost budgets?

**Interviewer:** 300 ms end to end for search, and the index budget is real. We are
not embedding every frame of every video.

Let us summarize. **We are asked to add semantic retrieval to a video search system
with a strong lexical head, over 100 million videos growing by tens of thousands an
hour, with partial transcripts, searchable within minutes of upload, in 300 ms, on a
constrained index budget, judged by whether users find what they wanted rather than
by clicks.**

Two consequences fall out immediately.

**Consequence 1: a video is a multi-field multimodal document, not an item with an
embedding.** Title, description, tags, channel, transcript, on-screen text, frames,
and thumbnail are different fields with different reliability, different coverage,
and different cost to produce. Collapsing them into one vector throws away the
ability to say *why* a video matched, and it destroys the head queries that match a
title exactly. The design keeps a lexical index over text fields and a dense index
for semantics, and fuses them, which is the same conclusion Spotify reached when it
added natural-language search alongside keyword search rather than replacing it.

**Consequence 2: behavioral labels are confounded, so the label is something you
construct.** A click is a vote for the thumbnail. Raw watch time is a vote for long
videos. Position is a vote for whatever the current ranker already liked. The usable
signal is closer to "long watch relative to the video's own length, given that the
result was seen," plus reformulation and abandonment as negative evidence, plus a
small human-rated set to anchor the whole thing. Every evaluation number in this
chapter is only as good as that construction.

A third point worth stating early because it changes the cost estimate: **the
expensive part is not query-time, it is ingestion and re-embedding.** Embedding a
query is one forward pass. Keeping 100 million videos embedded, and re-embedding all
of them when the encoder version changes, is the line item that decides whether this
design is affordable, and it belongs in the first ten minutes of the answer rather
than the last.
