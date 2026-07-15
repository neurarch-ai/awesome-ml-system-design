# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Every question either removes
work or changes the design.

**Candidate:** Where in the funnel does this model sit? Is it powering a
retrieval user tower, a ranking feature, or both?
**Interviewer:** Let us say the main feed ranking stage. The system should turn
the user's recent behavior into a feature that the ranker can use.

**Candidate:** What counts as a "behavior event" in the sequence? Clicks,
purchases, watch time, skips?
**Interviewer:** Primarily clicks and long-dwell views on the main feed. You can
also include purchases if relevant. Treat them as ordered by timestamp.

**Candidate:** How long can the sequence be? Are we modeling the last 10 events
or the last 10,000?
**Interviewer:** Keep it practical. Recent history is what matters most. Assume
tens to a few hundred events per user in the recent window.

**Candidate:** How fresh must the user state be? Does the feed need to react to
the user's last action in the same session (one continuous visit, before the user leaves and comes back)?
**Interviewer:** Yes. That is actually the whole point. If someone watches three
cooking videos, the very next refresh should reflect that, not tomorrow's batch.

**Candidate:** What scale are we at? How many active users and requests per
second?
**Interviewer:** Tens of millions of daily active users, each generating several
feed loads per day.

**Candidate:** How do we handle a brand-new user with no history?
**Interviewer:** The system should degrade gracefully, not break. Use whatever
context is available.

Let us summarize the problem statement. **We are asked to model a user's recent
ordered behavior so the ranking stage can react to current intent within the same
session.** The input is an ordered sequence of recent interactions plus optional
context; the output is a user-intent representation fed to the ranker as a
feature. The system must handle tens of millions of users, react within the same
session, and degrade sensibly for new users.

Two consequences fall out immediately, and stating them early is most of the
signal in this question:

- **The whole value of this system is freshness.** A sequence model that is only
  updated in a nightly batch is barely better than a lifetime interest aggregate.
  If the user state is not updated within the session, the requirement is not met.
  That single constraint forces a streaming pipeline; the model design is almost
  secondary.
- **Order and recency carry intent in a way that aggregates do not.** A user who
  just watched three cooking videos looks identical to a lifetime cooking fan under
  a bag-of-categories feature. The sequence model sees the recent shift. Every
  design choice in this chapter is about preserving that signal cheaply enough to
  serve at low latency.
