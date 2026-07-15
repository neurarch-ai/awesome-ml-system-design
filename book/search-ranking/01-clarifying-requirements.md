# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Every question either removes
work or changes the design.

**Candidate:** Is this web search, product search, or internal-document search?
**Interviewer:** Product search for an e-commerce platform. Users type queries
like "running shoes" or "waterproof jacket under 100 dollars."

**Candidate:** How large is the corpus, and how many queries do we serve?
**Interviewer:** A few hundred million products. Peak traffic is tens of thousands
of queries per second, with a heavy head of common queries and a long tail of
rare or never-before-seen ones.

**Candidate:** What is the end-to-end latency budget?
**Interviewer:** Low hundreds of milliseconds for the whole request, including
rendering. Ranking itself gets at most a few tens of milliseconds.

**Candidate:** What does "good ranking" mean here? Binary relevant/not-relevant,
or graded?
**Interviewer:** Graded. Think of it as perfect, good, fair, or bad. The top
slots matter far more than the bottom ones.

**Candidate:** What training signal do we have? Click logs, purchases, human
judgments?
**Interviewer:** Mostly click logs, plus a small set of human-judged query-product
pairs. Purchases exist but are sparse.

**Candidate:** Is personalization in scope? Can the same query return different
results for different users?
**Interviewer:** Some. Location and language always matter. Light user-history
signals are fine. But the query dominates; this is not a feed.

**Candidate:** How fresh must new products be?
**Interviewer:** Searchable within minutes of being listed. The catalog turns over
continuously.

---

Let us summarize. **We are asked to design the system that, given a user query,
returns a ranked list of products from a corpus of hundreds of millions, under a
latency budget of tens of milliseconds for ranking, optimized for graded relevance
at the top positions.** Labels are mostly biased click logs plus a thin layer of
human judgments. New products must be searchable within minutes.

Three consequences fall out of this immediately, and stating them early is most of
the signal in the interview:

- **Two stages are forced by scale.** Tens of milliseconds rules out running any
  learned model over hundreds of millions of products per query. The design must
  first narrow the field cheaply (retrieval: lexical plus semantic), then score
  the survivors with a richer model (ranking). Those two stages have different
  objectives, different models, and different latency budgets.

- **Retrieval needs two arms, not one.** Lexical matching (BM25, a fast keyword-overlap scoring formula, over an inverted
  index, a lookup table from each word to the documents that contain it) is fast
  and unbeatable on exact-term and rare-term queries, but it cannot
  match "laptop" to "notebook computer." Semantic matching (a dual-encoder, a model
  that maps query and document into the same vector space, plus ANN search,
  approximate nearest-neighbor lookup that finds close vectors without scanning all
  of them) closes that vocabulary gap but drifts on rare terms and exact
  strings. Neither arm is optional; the design unions them.

- **The dominant label problem is position bias, not multi-task engagement.**
  Unlike recommendation, where the main label challenge is deciding which
  engagement signal to trust, search labels are clicks that reflect where a result
  was shown as much as whether it was relevant. Naive training teaches the model
  to predict position, not relevance. Fixing that is the hardest part of the label
  pipeline and shapes every modeling decision downstream.
