# 1. Clarifying the requirements

Before drawing anything, pin down what the system must do. Here is a typical
exchange. Notice that each question either removes work or changes the design.

**Candidate:** Which cold-start problem (serving useful recommendations for
brand-new users or items that have no interaction history yet) dominates:
item-side (lots of new supply, as in a UGC platform or marketplace) or
user-side (lots of new signups, as after a viral campaign)?

**Interviewer:** Both matter. Item side is heavier; we ingest thousands of
new listings per day. User signups spike occasionally.

**Candidate:** How fast do items churn? Is cold start an edge case, or the
steady state?

**Interviewer:** Most items live for weeks, but about 20% of impressions go
to items less than 24 hours old. So cold start is common, not occasional.

**Candidate:** How rich is the metadata on a brand-new item?

**Interviewer:** We have category, creator, free-text description, and a
thumbnail. No interaction history on day zero.

**Candidate:** What is the reward signal, and how long until it arrives?

**Interviewer:** Click is available immediately. Session completion and
retention arrive hours or days later. You decide which to optimize.

**Candidate:** How large is the action space? Are we picking among a handful
of content variants, or from a catalog of millions?

**Interviewer:** Millions of items. A two-stage retrieval-then-rank funnel
cuts it to a few hundred before any bandit or ranker runs.

**Candidate:** Is there an experimentation platform I should integrate with,
or does this ship as a standalone service?

**Interviewer:** We have an experiment platform. Prefer to reuse it where
possible.

**Candidate:** Are there guardrails? Surfaces like checkout or trust-andsafety
where a bad exploratory impression is unacceptable?

**Interviewer:** Yes. The main feed can accept some exploration cost. The
checkout and notification surfaces cannot.

---

Let us summarize the problem statement. **We are asked to design a system
that gives reasonable recommendations on day zero for both cold users and cold
items, while preventing the feed from ossifying around what it already knows.**
The input is a user request with context; the output is a ranked feed that
balances exploitation (what we know is good) with exploration (what we are
uncertain about but should learn). A new item must be retrievable within
minutes of upload. A new user must see a plausible feed on their first
request.

Three consequences fall out immediately, and stating them early is most of
the signal in this question:

- **Cold start is a representation problem.** A model that keys off learned
  ID embeddings (compact learned vectors that place each entity as a point in
  a shared space, so similar entities sit close together) has nothing to say
  about a brand-new entity. The fix is to
  build item and user representations from content and context, so a fresh
  entity inherits a location in embedding space from things it resembles
  rather than starting at a random vector.
- **Pure exploitation ossifies the corpus.** A greedy ranker only collects
  labels for what it already ranks highly. Items it ranks low never surface,
  so their estimates go stale and the corpus narrows. Fixing this requires
  deliberately spending some impressions on uncertain items.
- **Exploration is only rational under a long-horizon objective.** It costs
  short-term reward. You must justify it against the value of the information
  it buys: a better model, a broader corpus, less staleness.
