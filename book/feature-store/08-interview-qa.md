# 8. Interview Q&A

The questions an interviewer actually asks about feature stores and training-serving
skew, grouped by how they are used. The commonly-missed ones are where interviews
are won or lost.

## Commonly asked

**Q: What is training-serving skew, and what causes it?**

A: It is the gap between the feature values the model saw during training and the
values it sees at serving time. Three causes. Code skew: separate code for
training (SQL) and serving (service function) that drifts apart. Time skew:
training joins on the latest feature value instead of the value at the time of
the event, leaking future information. Data skew: offline and online pipelines pull
from sources that have diverged in freshness or logic. Name all three. Interviewers
are listening for the distinction.

**Q: Why do you need two stores (offline and online)? Why not one?**

A: Different access patterns. The offline store must support bulk scans over billions
of timestamped rows for an as-of join; columnar warehouses do this well. The online
store must return a single entity's features in milliseconds; column-oriented storage
does this poorly. Trying to serve ranking requests from BigQuery will blow the
latency budget. Trying to do point-in-time historical joins from Redis without
historical rows is impossible. Two stores with different technologies solve two
genuinely different problems.

**Q: What is an as-of join and why does training require it?**

A: For each labeled event with timestamp $T_i$, the as-of join retrieves the feature
value valid at $\max\{t : t \leq T_i\}$: the most recent write no later than $T_i$. It prevents time leakage by ensuring the model is trained only on information
that would have been available at scoring time. "Join on latest" is not the same
thing; it picks up feature writes that happened after the event.

**Q: What freshness tier do you assign a feature?**

A: The least-strict tier that the model accuracy can tolerate. A 30-day purchase
count changes slowly; daily batch is correct and cheap. Session click count changes
per event; streaming is required. Assigning streaming freshness to a slow feature
wastes infrastructure with no accuracy benefit.

**Q: How do you detect that skew has entered the system?**

A: Served-vs-computed parity: for a sample of entities, compare the value in the
online store to what a fresh offline computation produces. Target parity above 0.999
per feature. Also monitor PSI between the training feature distribution and the
serving feature distribution; PSI above 0.1 is a warning. Both metrics should run
on a schedule and alert before model quality degrades.

## Tricky (the follow-ups that separate people)

**Q: Your model looked great offline, but engagement dropped the day after launch.
What do you look for first?**

A: Training-serving skew. Check served-vs-computed parity for each feature the model
uses. If parity is fine, check PSI of the training distribution versus what the
online store is returning. Then look at whether the offline evaluation used a
random split instead of a time-based split (which leaks future data and flatters
offline metrics). The pattern "great offline, bad online immediately after launch"
is almost always skew or an evaluation bug, not a model bug.

**Q: When is logging features at serving time better than recomputing them offline?**

A: Always, when you have it. Recomputing features offline to match serving requires
exact reproducibility of the computation: same null handling, same aggregation, same
time-zone assumptions. Logging the exact values at serving time bypasses that
fragility entirely. The model then trains on literally what it saw in production.
Google's Rules of ML prioritizes this approach precisely because recomputation is
where code skew silently enters. The cost is storage; the benefit is perfect parity
by construction.

**Q: How do you handle a backfill that covers a period when the feature definition
was different?**

A: You have two options. Apply the current definition uniformly and accept that the
backfilled values do not match what the old pipeline would have produced; train only
on data after the definition change. Or version the feature definition and apply
the version that was active at each point in time. Most teams choose the first
option; it is simpler and avoids versioned backfill complexity. The key constraint
is: do not mix training data from before and after a definition change without
explicitly flagging the discontinuity.

**Q: Uber logged 10,000 features and governance became the critical challenge. What would
you do differently?**

A: Governance from the start: every feature must have a declared owner, a description,
a freshness SLA, and at least one consumer in the registry. Enforce deprecation:
a feature with no consumer for 90 days is automatically sunset. Enforce naming
conventions and feature groups so related features are discovered together. Without
governance, a feature store is a write-only system: features accumulate, ownership
erodes, and nobody dares delete them.

## Commonly answered wrong (the traps)

**Q: Can you compute heavy features (neural embeddings, expensive aggregates) on
the serving path to keep the online store small?**

A: Only if the latency budget allows it, which it usually does not. Serving budgets
are 5-10ms. A neural embedding forward pass is 10-100ms on CPU. The online store
exists precisely to move that computation offline. The right answer is: precompute
offline, materialize to the online store, serve a lookup. Reserve request-time
computation for features that literally depend on the request (device, query text).

**Q: Should training always use the latest feature value from the online store?**

A: No. The online store holds only the latest value; it has no historical rows. For
training, you need the value as of the event time, which requires the offline store's
timestamped history and an as-of join. Using the online store for training is the
join-on-latest bug: it will pass offline evaluation (the latest value is often
close to correct) while introducing subtle time leakage that inflates offline metrics.

**Q: Is PSI measured between training and serving distributions, or between training
and holdout?**

A: Between training and serving (live traffic). PSI between training and a holdout
split tells you about train-test distribution shift, which is a different problem.
PSI between training and serving tells you whether the feature values arriving at
the model in production match what it was trained on. That is the skew signal. Both
are useful; they measure different things.

**Q: Can you just retrain more frequently to fix skew?**

A: No. Frequent retraining reduces distributional shift between training data and
the current world, but it does not fix code skew, time skew, or materialization
bugs. A model retrained daily on features computed with the wrong logic will still
have skew; it just trains on a fresher version of the wrong distribution. The fix
is correct infrastructure, not retraining cadence.
