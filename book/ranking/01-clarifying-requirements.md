# 1. Clarifying the requirements

Before choosing an architecture, pin down what the ranker must do. Here is a
typical exchange between a candidate and an interviewer. Every question either
removes work or changes the design.

**Candidate:** What is the primary business objective: click-through rate,
conversion, watch time, or a blend of several signals?
**Interviewer:** A blend. We care about clicks, long dwell, and saves. You decide
how to handle multiple objectives.

**Candidate:** How many candidates come in from retrieval, and how many items go
out in the final ranked list?
**Interviewer:** A few hundred to around a thousand in from retrieval. The ranked
list shows the top tens. You score all candidates and sort.

**Candidate:** What is the end-to-end latency budget, and how much of it can
ranking use?
**Interviewer:** The whole request is in the low tens of milliseconds. Ranking
can use roughly 20 ms of that at p99.

**Candidate:** What features are available? Specifically: do we have user-item
cross features, and can we compute them online within budget?
**Interviewer:** User features, item features, and cross features are all on the
table. Assume a feature store can serve pre-computed signals.

**Candidate:** How large is the training corpus? Impressions with engagement
labels, or something else?
**Interviewer:** Billions of impression rows, each with an engagement outcome.
Negative examples vastly outnumber positives.

**Candidate:** Does the ranking score need to be a calibrated probability, or is
ordering the list enough?
**Interviewer:** At least one use case feeds a threshold, so calibration matters.

**Candidate:** Are there fresh items or users with little history?
**Interviewer:** Yes. Cold-start is a real concern for new items.

Let us summarize the problem statement. **We are asked to design the ranking
stage of a multi-objective personalized feed.** The input is a batch of a few
hundred candidates plus the user and their context. The output is a calibrated,
ordered list within roughly 20 ms. The model must blend signals from clicks,
dwell, and saves into one utility score that respects a per-candidate latency
budget of well under a millisecond.

Two consequences fall out of this immediately, and stating them early is most of
the signal in this question:

- **Ranking is a precision problem inside a latency budget.** Retrieval already
  filtered for recall; ranking's job is to get the order right on the survivors.
  But scoring hundreds of candidates per request means the per-candidate model
  cost must stay flat and cheap, which constrains the architecture as much as
  accuracy does.
- **Cross features between user and item are the biggest accuracy lever.** A
  model that only sees user features and item features in isolation cannot recover
  the signal in "how many times has this user engaged with this item's category
  in the past seven days." That cross feature is often the single largest accuracy
  move, and forgetting to mention it is the most common gap in ranking answers.
