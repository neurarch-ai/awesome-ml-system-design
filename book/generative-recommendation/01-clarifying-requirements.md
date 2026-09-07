# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Every question either removes work or
changes the design.

---

**Candidate:** "Generative recommendation" means at least three different systems.
Generative retrieval, which replaces the ANN index. Generative ranking, which replaces
the ranker with a scaled sequence model. And an LLM in the loop, which turns the
problem into text. Which one are we talking about?

**Interviewer:** Start with retrieval. Assume ranking is a follow-up conversation.

---

**Candidate:** How large is the catalogue, and how fast does it change? The item
representation has to be recomputed when the catalogue moves, so this is a cost
question, not a modeling one.

**Interviewer:** About 50 million items, roughly 100,000 new per day, and a long tail
that almost never gets interactions.

---

**Candidate:** What is the latency budget for the retrieval stage? An ANN probe is a
few milliseconds; decoding is several sequential forward passes.

**Interviewer:** 30 ms p99 for retrieval, and no increase in ranking latency.

---

**Candidate:** Why are we doing this? Cold start and long-tail generalization,
engineering velocity, or a scaling-law bet? Those are three different projects.

**Interviewer:** Officially cold start. Unofficially, every new surface costs us a
quarter of feature engineering and we would like that to stop.

---

**Candidate:** Good, because those point at different first steps. What has to survive
the change? Business rules, availability, locale, diversity?

**Interviewer:** All of them. If it cannot be filtered, it cannot ship.

---

**Candidate:** And how will we decide it worked? I ask because offline recall against
a sampled candidate set is exactly the metric that flatters this architecture.

**Interviewer:** Say more about that in the evaluation section. Assume an online test
with a guardrail set.

---

## What we are building

A generative retrieval stage for a 50M-item catalogue with 100k new items a day,
p99 under 30 ms, feeding the existing ranker, keeping every existing filter, judged
online with slice-level reporting. The stated goal is cold start; the real goal is the
engineering cost of onboarding a surface, and those two lead to different first
projects, which section 4 comes back to.

**Requirements**

- Functional: retrieve a few hundred candidates from history, cover new items within
  hours of ingestion, support hard filters and a diversity constraint, and leave a path
  to reuse the representation in ranking.
- Non-functional, with numbers: p99 under 30 ms, serving cost within a small multiple
  of the ANN path, item representation refreshed daily and stable enough that
  yesterday's model still works with today's codes.
- Explicitly out of scope: replacing the whole cascade in one step.

**The one thing to carry into the next section.** Nothing above is a modeling
constraint. They are all consequences of how you decide to name an item.
