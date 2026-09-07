# 8. Interview Q&A

## Commonly asked

**Q: What is a semantic ID, and why does it help a cold item?**

An item's content embedding quantized into a short tuple of discrete codes, coarse to
fine, by a residual quantizer. It helps a cold item because the codes come from
content, not interactions, so a brand-new item lands next to items with similar content
and inherits their neighbourhood. It adds no interaction information; it is a
content-similarity prior expressed in the model's vocabulary.

**Q: Why can a model decode over a 50 million item catalogue at all?**

Because the vocabulary collapsed. Predicting an item directly needs an output layer
with one logit per item. Predicting four codes from a 256-entry codebook needs an
output layer of width 256, applied four times, and $256^4$ is about 4.3 billion
representable items. You never score 50 million things.

**Q: Is generative retrieval cheaper to serve than ANN?**

No, and this is the most common wrong answer. A beam decode is several sequential
forward passes with a batch of hypotheses; ANN is a single index probe. It buys
generalization and a simpler data pipeline, not latency.

**Q: What happens when you retrain the quantizer?**

Every item's semantic ID changes, so the sequence model trained on the old codes is
reading a vocabulary that no longer exists. The refresh cycles are coupled. Version the
codebooks, retrain both together, and keep the old mapping through a deprecation
window.

**Q: How do business rules and filters work here?**

Exactly where they did before, after candidate generation. The difference is that you
have to over-generate, because a beam of 200 can survive filtering as a dozen items for
some users, and you have to measure how often the fallback path fires.

## Tricky (the follow-ups that separate candidates)

**Q: Your offline recall went up 12 points and online is flat. What did you do wrong?**

Almost certainly sampled-candidate evaluation: ranking the true item against a handful
of sampled negatives is the task the vocabulary collapse makes easy, and it is not the
production task. Re-run full-catalogue, report per slice, then shadow the decode against
the live ANN path and compare candidate sets before testing online.

**Q: Where exactly does "no index to maintain" stop being true?**

At the valid-prefix structure, the item-to-code map, the collision groups, and the
codebooks themselves, all of which are indexes with refresh cycles. The ANN index is
replaced by a smaller set of structures plus a second model, not by nothing.

**Q: You removed the boundary between retrieval and ranking. What did you lose?**

The place you used to intervene. Stage boundaries are where filters, business rules,
diversity and calibration live, and a unified model has to reintroduce all of them
deliberately. You also lost a graceful failure: a regression in a unified model is a
whole-surface regression, so shadow mode and a one-flag rollback stop being optional.

**Q: The interviewer says "our catalogue is 500k stable items with dense interaction
data". Do you still build this?**

Probably not, and saying so is the strongest answer available. The gains concentrate
where interaction data is scarce. With a stable catalogue and dense interactions, an
atomic-ID two-tower model with a tuned ranker is hard to beat, and the honest
recommendation is semantic IDs as ranking features at most.

## Commonly answered wrong (the traps)

**Q: How would you evaluate a generative retriever?**

**The tempting answer:** recall@k and NDCG against the held-out next item.

**Why it is wrong:** it omits the three things that are specific to this architecture
and that fail silently. The validity rate (tuples that map to nothing, or to items the
user cannot see), the slice breakdown (the gain and the risk both live in the tail), and
diversity (beam search is mode-seeking and concentrates output).

**What to say instead:** full-catalogue metrics per slice, plus validity and coverage
as first-class numbers, then shadow, then an online test with guardrails.

---

**Q: Is this the end of the multi-stage cascade?**

**The tempting answer:** yes, the frontier has moved and the funnel is legacy.

**Why it is wrong:** the published production systems still feed an existing ranker, and
the ones that unified stages did it at a scale that comes with a compute budget most
teams do not have. The cascade is also where filters, calibration and business rules
live.

**What to say instead:** the cascade is what nearly everyone runs today, the generative
foundation model is where the frontier is, and the interesting question is which
surface you would pilot on and what would make you stop.

---

**Q: The candidate proposes an LLM ranker for the whole system.**

**The tempting answer:** LLMs have world knowledge, so let one rank everything.

**Why it is wrong:** token cost and latency per request at recommendation QPS, plus
hallucinated items and position bias in a ranking prompt. It is the most expensive
option and it is being applied where a cheap model already works.

**What to say instead:** scope it to where features and history do not exist yet (a new
content type, a new surface), name engineering velocity as the benefit being bought,
and keep the cheap path for everything else.
