# 1. Clarifying the requirements

Before drawing a single box, pin down what the system must do. Here is a typical
exchange. Notice that every question either removes work or changes the design.

**Candidate:** What are we serving: a small dense net, a large ranker with
embedding tables, or an LLM?
**Interviewer:** A ranking model, roughly the kind used to score a few hundred
candidates in a recommendation pipeline. Large embedding tables, a few dense
layers on top.

**Candidate:** What is the p99 latency target (the latency that 99 percent of
requests come in under, i.e. the slow-tail experience) for the whole prediction
call, including feature fetch?
**Interviewer:** About 50 ms end-to-end at p99. The feature store adds some of
that; you decide how to split it.

**Candidate:** Is this online serving on the user's critical path, or is it
near-line scoring that writes to a cache?
**Interviewer:** Online, synchronous. The caller blocks on this response.

**Candidate:** What is the expected QPS (queries per second), and is traffic
steady or spiky?
**Interviewer:** Peak around 50 000 QPS, with a strong diurnal curve and
occasional launch spikes up to 3x the normal peak.

**Candidate:** How often does a new model version ship?
**Interviewer:** About once a day. The team retrains on fresh data nightly.

**Candidate:** What does a safe deploy mean here? Is there tolerance for some
users getting the new model early?
**Interviewer:** We need gradual rollout with an automatic rollback trigger. A
bad version that reaches 100 percent of traffic is an incident.

Let us fix the problem statement. **We are designing the serving layer for a
large ranking model that answers online requests at p99 under 50 ms, up to
50 000 QPS at peak, with daily model updates that roll out gradually and roll
back automatically if a regression appears.**

Two consequences fall out immediately, and naming them early is most of the
interview signal:

- **Tail latency, not average latency, is the contract.** A healthy average with
  a fat p99 still misses the SLA for the requests that matter. Every design
  decision from here bends around the slow requests, not the fast ones.
- **Daily deploys make safe rollout load-bearing, not a nice-to-have.** At
  once-a-day cadence, the deploy tooling is as critical as the serving
  infrastructure. A system with no shadow, no canary, and no rollback path is
  one bad model checkpoint away from a production incident every night.
