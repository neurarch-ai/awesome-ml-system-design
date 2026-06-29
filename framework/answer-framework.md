# The answer framework

ML system design interviews reward the same thing every system design interview
rewards: a structured answer that starts broad, commits to a design, and then
stress-tests it. The content is domain-specific; the shape is not.

Use these six steps. They map onto a 45-minute loop with time to spare.

## 1. Clarify and scope (5 min)

Do not start designing. The fastest way to fail is to build the wrong system
confidently. Pin down:

- **The task and the objective.** "Recommend items" is not a spec. Are you
  optimizing clicks, watch time, purchases, or retention? The objective decides
  your labels, your loss, and your metric. A surprising amount of recsys design
  is just choosing what to predict.
- **Scale.** Number of users, number of items (the catalog), requests per second
  at peak. A catalog of ten thousand items and one of a billion are different
  systems. These set every downstream number.
- **Latency budget.** An interactive feed tolerates tens of milliseconds for the
  whole recommendation call. A batch email recommender tolerates hours. This
  decides whether you precompute, retrieve online, or both.
- **Quality bar and how it is measured.** Offline (AUC, NDCG, recall@k) and
  online (engagement, conversion) often disagree. Name both and say which one
  ships the model.
- **Freshness.** How fast must a new item become recommendable, and how fast must
  the system react to a user's latest action? Minutes, hours, or next-day are
  three different architectures.

State your assumptions out loud and write them down. The interviewer will correct
the ones that matter.

## 2. Define functional and non-functional requirements (3 min)

Turn the clarifications into a short list.

- **Functional:** what the system must do (retrieve candidates, score them, apply
  business rules, rank, log impressions and outcomes for training).
- **Non-functional:** end-to-end latency p50/p99, throughput, freshness of items
  and of user state, training cadence, and the offline metric the model is gated
  on.

Name the one or two non-functional requirements that will dominate the design.
For most large recommenders it is the **latency budget split across a multi-stage
funnel**: you cannot score a billion items, so the architecture is shaped by how
cheaply you can narrow them down.

## 3. Sketch the high-level data flow (8 min)

Draw the boxes and the arrows. For a classic ML system there are almost always
two distinct paths, and conflating them is the most common mistake in the room:

- **The offline path:** logging, label joining, feature computation, training,
  evaluation, and pushing artifacts (a trained model, refreshed embeddings, a
  rebuilt index) to production. Runs on a schedule (hourly, daily) or on a
  trigger, not per request.
- **The online path:** what happens per request. Fetch features, retrieve
  candidates, score, rank, apply policy, return, and log the impression so it can
  become a training label later.

The two paths meet at two seams, and both are where bugs live:

- **The feature seam.** A feature must be computed the same way offline (for
  training) and online (for serving). When they diverge you get
  **training-serving skew**: the model was trained on a feature distribution it
  never sees in production. This is the single most common silent failure in
  applied ML. Call it out by name.
- **The label seam.** Today's impressions become tomorrow's training labels. The
  system generates its own training data, which means it also generates its own
  feedback loops and biases (you only get labels for what you showed).

Separating offline and online lets you reason about freshness and skew (offline
and seam concerns) independently from latency (an online concern).

## 4. Go deep on the components the interviewer cares about (15 min)

This is where most of the signal is. You cannot go deep on everything, so read
the room and pick. The high-value deep dives in classic ML systems are usually:

- **The two-stage funnel: retrieval then ranking.** Almost every large
  recommender narrows a huge catalog to a small ranked list in stages. Cheap
  retrieval (often a [two-tower](../topics/01-candidate-retrieval.md) model over
  an approximate-nearest-neighbor index) pulls a few hundred to a few thousand
  candidates from millions. An expensive [ranker](../topics/02-ranking-model.md)
  then scores that short list with rich features. Knowing why the split exists,
  and what each stage optimizes, is the backbone of the answer.
- **Features.** Where the real accuracy lives. User features, item features, and
  **cross features** (user-times-item interactions) that a model cannot recover
  on its own. Sparse categorical features become embeddings; dense features feed
  the network directly. How you compute, store, and serve them is the feature
  store discussion.
- **Training-serving skew and the feature store.** How you guarantee the feature
  the model trained on is the feature it serves on, including point-in-time
  correctness when joining labels to features (no peeking at the future).
- **The eval loop.** Offline metrics to gate a candidate model, then an online
  A/B test to decide if it ships. Strong candidates bring this up unprompted and
  are honest that offline gains often do not survive online.

When you discuss the model, be concrete about the architecture. "We use a DLRM,
so sparse features go through embedding tables and then we take explicit pairwise
feature interactions before the MLP" is a stronger answer than "we use a deep
model," and it invites the follow-up you want.

## 5. Identify bottlenecks and scale (8 min)

Walk the request path and ask where it breaks first as load grows:

- Scoring too many candidates per request: that is what the retrieval stage is
  for. Narrow harder upstream.
- ANN index latency at catalog scale: shard the index, tune the recall-versus-
  latency knob, replicate for QPS.
- Feature fetch fanning out to many stores: co-locate features, batch the lookup,
  cache user features for the session.
- Embedding tables too large for memory: hashing, lower dimensions, or
  feature pruning.
- Model retraining falling behind a shifting distribution: more frequent
  retrains, or incremental/online updates for the fast-moving parts.

For each bottleneck, name the fix and its tradeoff. There is no free lunch and
interviewers want to see that you know it.

## 6. Address failure modes and quality (5 min)

The thing that separates senior answers:

- **Training-serving skew.** The quiet killer. Detect it by comparing logged
  serving features against the training distribution.
- **Cold start.** New users and new items have no interaction history. Fall back
  to content features, popularity, or exploration.
- **Feedback loops and bias.** The model only sees labels for items it chose to
  show, so it reinforces its own past behavior. Position bias, popularity bias,
  and the need for some exploration all live here.
- **Distribution drift.** Tastes and catalogs move. Stale models decay; monitor
  online metrics, not just offline ones.
- **Graceful degradation.** What happens when the feature store is slow, the
  index is stale, or the ranker times out. A popularity fallback that always
  returns something beats an error.

---

## The estimation you should be able to do in your head

Almost every recommender question reduces to a **funnel** and a **latency
budget**. Be ready to estimate both. The numbers below are illustrative orders of
magnitude to anchor the reasoning, not benchmarks.

The candidate funnel, stage by stage:

```
catalog            ~10,000,000 items
  └─ retrieval (ANN, cheap)        ──▶  ~1,000 candidates
       └─ ranking (rich model)     ──▶  ~100 scored
            └─ policy / re-rank     ──▶  ~10 shown
```

Each stage trades cost for precision. Retrieval touches everything but scores
almost nothing per item (a dot product over precomputed embeddings). Ranking
touches little but spends a lot per item (hundreds of features, a deep network).
The whole point of the funnel is to spend your expensive compute only where it
changes the answer.

The latency budget, working backwards from the request:

- The total recommendation call often has a budget on the order of tens of
  milliseconds at p99.
- Out of that, feature fetch, retrieval, and ranking each get a slice. If
  ranking gets, say, 20 ms and you must score 500 candidates, that is well under
  a tenth of a millisecond per candidate including feature assembly. That
  constraint, not accuracy, is often what picks your ranking architecture.

If you can reason from "I have N items, a p99 budget of T milliseconds, and an
expensive scorer" to "therefore a two-stage funnel with these candidate counts,"
you are ahead of most candidates. Topics 01 and 02 build this out with real
architectures.

---

## Common mistakes

- Jumping to a solution before scoping the objective and the scale.
- Designing one path and forgetting the other (online versus offline).
- Trying to score the whole catalog instead of retrieving a candidate set first.
- Ignoring training-serving skew, the most common real-world failure.
- No eval story, or only an offline one. "AUC went up" does not mean it ships.
- Forgetting cold start and the feedback loop the system creates about itself.
