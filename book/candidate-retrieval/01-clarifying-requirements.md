# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question
either removes work or changes the design.

**Candidate:** Are we retrieving for a feed (no query) or for search (with a
query)?
**Interviewer:** A personalized feed. There is no search query; the input is the
user and their context.

**Candidate:** How large is the catalog, and how many candidates should
retrieval return?
**Interviewer:** About 100 million items. Retrieval should hand roughly a few
hundred candidates to the ranking stage.

**Candidate:** What is the latency budget for retrieval?
**Interviewer:** Tens of milliseconds. It runs on every feed load.

**Candidate:** What signal do we have that a user "liked" an item? Clicks, watch
time, purchases?
**Interviewer:** Assume logged positive interactions (a click that led to a long
dwell). You choose how to use them.

**Candidate:** Do we need the single best item, or high recall of the good ones?
**Interviewer:** High recall. Retrieval only has to get the good items into the
few hundred; ranking sorts them.

**Candidate:** How fresh must new items and new users be? Is a several-hour-old
index acceptable?
**Interviewer:** New items should be retrievable within minutes. Users can be a
bit staler.

Let us summarize the problem statement. **We are asked to design the retrieval
stage of a personalized feed.** The input is a user and their context; the output
is a few hundred candidate items out of 100 million, returned in tens of
milliseconds, optimized for recall of items the user would engage with. New items
must enter the index within minutes.

Two consequences fall out of this immediately, and stating them early is most of
the signal in this question:

- **Retrieval is a recall problem, not a precision problem.** Its job is to not
  miss good items, cheaply. Precision is the ranking stage's job. This is why we
  can use a coarse, fast model here and a heavy one later.
- **We cannot score 100 million items per request.** Tens of milliseconds rules
  out running any model over the whole catalog online. The design must precompute
  as much as possible and reduce the online cost to a lookup. That single
  constraint is what forces the two-tower architecture in the next section.
