# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question
either removes work or changes the design.

**Candidate:** What kinds of features are we talking about: batch aggregates
updated daily, or real-time signals updated per event?
**Interviewer:** Both. Think user engagement counts refreshed daily and session
signals like "items clicked in this session" that need to be fresh within seconds.

**Candidate:** How many teams and models will consume features?
**Interviewer:** A dozen models across five teams right now, and growing. Every
team recomputes the same things today.

**Candidate:** What is the online latency budget for feature retrieval?
**Interviewer:** Feature fetch sits on the critical path of each ranking request.
It must complete in single-digit milliseconds.

**Candidate:** How large is the entity space? Users, items, or both?
**Interviewer:** About 50 million active users and 10 million items. Writes arrive
at up to 500k events per second during peak.

**Candidate:** Does training need features at the exact value they had when the
event happened, or is "latest value" good enough?
**Interviewer:** Labels can arrive hours after the events. We have had label
leakage before. Point-in-time correctness (pairing each label with the feature value as it existed at the moment of the event, not a later value) is required.

**Candidate:** Is this a greenfield build, or are we migrating from ad-hoc pipelines?
**Interviewer:** Migration. Teams have 200-plus existing feature pipelines. Keep
the disruption minimal.

Let us summarize the problem statement. **We are asked to design a feature
platform that serves features online with single-digit-millisecond latency,
produces point-in-time-correct training datasets, supports both batch and streaming
computation, and is reusable across a dozen models without each team recomputing
the same features from scratch.**

Three consequences fall out of this immediately, and stating them early is most
of the signal in this question:

- **Online and offline must derive from one definition.** If a team writes
  separate SQL for training and a separate service for scoring, any divergence in
  logic creates skew. The fix is a single shared definition that compiles to both
  paths. This is the architectural invariant the whole system protects.
- **A low-latency (answers in a few milliseconds) lookup store is mandatory.** Single-digit milliseconds rules out
  computing features at request time for anything non-trivial. The system must
  precompute and materialize the latest value into a key-value store. That
  requirement is what forces the dual-store split in section 4.
- **Point-in-time correctness forces full feature history.** "Latest value" is not
  enough. The offline store must keep timestamped history so that training can
  reconstruct the feature value as it existed just before each label event. This
  is what the as-of join in section 3 provides.
