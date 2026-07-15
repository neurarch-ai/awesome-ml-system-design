# 8. Interview Q&A

The questions interviewers actually ask about real-time ML serving and
deployment, grouped by how they are used. The commonly-answered-wrong ones are
where interviews are won or lost.

## Commonly asked

**Q: Why a dedicated model server instead of loading the model inside the
application?**
A: Three reasons. First, you can redeploy the model without redeploying the
application: a daily checkpoint swap is a registry pointer change, not an app
release. Second, the server owns batching, warm-up, multi-version routing, and
metrics in one place so every team gets them for free. Third, you can hot-swap a
version while traffic is live, which no application code pattern handles cleanly.
Loading the model in the app couples these two timelines and produces bespoke,
unmonitored serving code per team.

**Q: What is dynamic batching and what does it cost?**
A: Dynamic batching collects requests arriving in a short window and sends them
through the model as one batch, amortizing fixed-cost overhead and filling GPU
or CPU lanes. The cost is added tail latency: a request must wait up to the full
window before dispatch. Size the window against the p99 budget, not for peak
throughput on idle hardware.

**Q: Shadow vs canary: when do you use each?**
A: Shadow proves the new version does not break anything, at zero user risk, by
running it on a copy of traffic and throwing the output away. Canary proves the
new version actually helps, on a small real slice of users. Shadow cannot measure
user impact because no one sees its output; canary cannot eliminate user risk,
only bound it. A mature pipeline does both in sequence: shadow, then canary, then
ramp.

**Q: How do you roll back quickly?**
A: Keep the prior version in the registry and wire an automated trigger off a
health or metric regression. The rollout controller shifts traffic to the prior
version; the rollback takes seconds. A rollback that requires a rebuild under
pressure takes minutes to hours and is not a real rollback.

**Q: Why p99, not average latency?**
A: Average latency hides the fat tail. If 7 percent of requests take 10 times
longer than the median, the average looks fine and the p99 is in breach of the
SLA. Every user whose request lands in the tail waits. Alert on p99 and p999.

## Tricky (the follow-ups that separate people)

**Q: Your new version looked fine in shadow but tanked in canary. Why?**
A: Shadow throws its output away, so it cannot capture the effect of the new
model's predictions reaching users. Feedback loops, changes in diversity or
coverage, and any user-behavior effect are invisible in shadow. "Great in shadow,
tanked in canary" is expected, not a contradiction: shadow proves no mechanical
breakage, canary proves actual user impact. The two are measuring different things
and both are necessary.

**Q: You scaled out by adding replicas but latency stayed high. What do you
check?**
A: First, confirm the scale-out signal. If autoscaling fired on CPU but the
bottleneck is GPU memory bandwidth or request queue depth, new replicas with
idle GPUs do not help. Second, check whether new replicas are receiving traffic
before they are warm: a readiness probe failure or a missing warm-up pass means
new replicas blow the latency budget for the first several hundred requests.
Third, check batch fill efficiency: at low load the batch is mostly empty and
the window wait dominates.

**Q: When would you choose offline batch scoring instead of online serving?**
A: When the prediction does not depend on real-time context and freshness can be
hours old. A daily user affinity score, a precomputed item quality rank, or an
entity-level propensity that changes on the timescale of days are all candidates
for batch scoring written to a fast lookup store. Online serving should be
reserved for what genuinely depends on the request (query text, real-time
inventory, session context). Pushing work to batch cuts critical-path cost and
eliminates the serving infrastructure for that prediction.

**Q: What is training-serving skew and does it make serving unsafe?**
A: Training-serving skew is the divergence between features at training time and
features at serving time: a different preprocessing step, a stale feature, a
missing join. A model can serve at p99 under budget and still be producing wrong
predictions if the features it receives do not match the distribution it trained
on. The fix is to log served features and compare their distribution against the
training set. Serving safely is a necessary but not sufficient condition for
quality.

**Q: Each hop in your inference chain has a p99 of 20 ms, so the end-to-end p99
is about 20 ms, right?**
A: No, tail latency compounds. If a request passes through N roughly independent
hops in sequence, the chance it dodges every hop's slow tail is about
(0.99)^N, so the end-to-end p99 drifts well past any single hop's p99. Fan-out is
worse than a chain: a request that must wait on the slowest of K parallel feature
shards sees the maximum of K tails, so scattering wider makes it more likely one
shard lands in a tail. The mechanisms that fight this are hedged (backup) requests
(fire a duplicate after a short delay and take whichever returns first) and
tail-tolerant fan-out (proceed once a deadline passes with the shards you have).
This is why the tail is budgeted per hop, not just measured end to end.

## Commonly answered wrong

**Q: Autoscale on CPU utilization for an inference service, right?**
A: Wrong. The bottleneck for an inference service is almost never CPU
utilization; it is GPU memory bandwidth, request queue depth, or batch latency.
Scaling on CPU lets the queue grow and latency breach the SLA while the
autoscaler waits for CPU to spike. Scale on queue depth, GPU utilization, or
batch latency percentile instead.

**Q: A bigger shadow phase means more confidence before canary, so run it as
long as possible.**
A: Wrong in two ways. Shadow doubles inference cost for every second it runs;
at high QPS on GPU hardware, this is real money. More importantly, shadow cannot
tell you anything about user impact no matter how long you run it, because its
output never reaches users. Run shadow for the minimum time needed to confirm no
latency regression and no prediction distribution collapse, then proceed to
canary to measure actual user impact.

**Q: Keep two full blue-green fleets for every daily checkpoint update.**
A: Too costly for a daily cadence. Blue-green is the right tool for high-stakes
deploys (a major architecture change, a new model architecture, a rollout you
need to reverse instantly) where the instant cutover and instant rollback justify
two full fleets. For daily checkpoint updates, canary plus gradual ramp with an
automated rollback trigger gives the same safety at half the cost.

**Q: A model that passes shadow is safe to ramp to 100 percent.**
A: No. Shadow proves mechanical correctness and no latency regression. It cannot
prove that the model helps users (or does not hurt them). A version that passes
shadow has earned a canary, not a full rollout. The canary result is what earns
the ramp to 100 percent.

**Q: Prediction caching is free latency, so cache everything.**
A: Wrong for personalized serving, because hit rate is the whole game. A cache
only pays off when inputs repeat. For a per-user personalized ranker the key space
is effectively unbounded, so the hit rate is near zero and the cache becomes pure
overhead: a lookup, a miss, then the full model call anyway, plus the memory and
invalidation cost. Caching earns its place on low-cardinality repeating inputs
(globally popular items, static context) and is dead weight on high-cardinality
real-time personalized inputs. Measure the projected hit rate before adding a
cache, not after.
