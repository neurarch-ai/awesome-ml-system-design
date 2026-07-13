# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Every question below
either removes work or changes the architecture. Notice how much the answers matter
before a single model is discussed.

---

**Candidate:** What kinds of entities are we embedding? Users, items, queries,
or nodes in a graph?
**Interviewer:** Users and items. Items have text titles, categories, and a few
numeric attributes. Users have interaction history.

**Candidate:** How large are the entity sets?
**Interviewer:** Tens of millions of items and tens of millions of users. That
sizes both the index and the dimension budget.

**Candidate:** What signal do we have that two entities are related? Explicit
labels are rare.
**Interviewer:** Behavioral logs: which users engaged with which items, sessions,
and co-occurrence counts. No curated similarity labels.

**Candidate:** What consumes the embeddings downstream? That determines whether
we optimize a dot product, a cosine, or just a feature vector.
**Interviewer:** Primarily approximate nearest-neighbor retrieval, but the same
vectors also feed the ranking model as features.

**Candidate:** Do items have content features, or only IDs? A content-based
encoder can embed a brand-new item with zero history; an ID-only encoder cannot.
**Interviewer:** Yes, text titles and category attributes. Cold start is a real
concern.

**Candidate:** What latency and freshness are needed? Are vectors precomputed
offline or computed on demand?
**Interviewer:** Precomputed offline for items; online for users. A new item must
become retrievable within an hour. Retrieval latency is tens of milliseconds.

---

Let us summarize the problem statement. **We are asked to design a representation
learning system that produces fixed-length vectors for users and items, such that
geometric closeness reflects behavioral relatedness, vectors are served through an
ANN index, and new items can be embedded from content features without waiting for
a retrain.**

Three consequences fall out immediately, and stating them early is most of the
signal:

- **The training problem is contrastive, not supervised.** We have no similarity
  labels; we have co-occurrence and engagement. "Related" is defined by behavior,
  and the loss must exploit that implicit signal.
- **The hardest design choice is the negatives.** Positives come almost for free
  from the logs. Choosing what counts as "not related" is where the embedding space
  is actually shaped. Name that before discussing any architecture.
- **The cold-start requirement forces an inductive encoder.** An encoder that
  consumes content features embeds a brand-new entity from its attributes with zero
  history. An id-only encoder has nothing until the next retrain and needs a
  fallback.
