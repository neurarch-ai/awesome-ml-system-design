# 7. Summary and talking points

## One-page recap

- **Retrieval is a recall problem.** Get the good items into a few hundred
  candidates, cheaply; ranking handles precision.
- **The latency budget forces a two-tower model.** Item embeddings are
  user-independent, so precompute all 100 million offline and index them; run only
  the user tower online, then do an ANN lookup.
- **Train with in-batch negatives plus the logQ correction.** The other items in
  the batch are the negatives; the correction removes popularity bias. Forgetting
  the correction is the most common mistake.
- **Serve with ANN (HNSW or IVF-PQ).** Trade a little recall for the speedup that
  makes 100M-item retrieval fit in tens of milliseconds.
- **Freshness is a minutes-cadence upsert**, and cold items ride on content
  features until their ID embedding trains.
- **Evaluate with Recall@k at the downstream k**, then gate the launch on online
  engagement and coverage, not offline recall alone.

## Likely follow-ups

- **"Why not one model over user and item features?"** More accurate, but it
  cannot precompute item embeddings, so it cannot meet the latency budget at 100M
  items. Use it for re-ranking a few hundred, not retrieval.
- **"How do you handle a brand-new item?"** Its ID embedding is untrained, so
  retrieve it via content features and a fresh-items source until it gathers
  interactions.
- **"Your recall improved but engagement dropped. What happened?"** Likely
  popularity collapse or reduced coverage. Check diversity and the tail; a recall
  gain that resurfaces popular items can lose the product.
- **"How do negatives work if you only logged positives?"** In-batch negatives:
  the other items in the batch. Add hard-negative mining once random negatives
  stop teaching.
- **"What breaks when you retrain the towers?"** The user and item towers must be
  versioned and re-indexed together, or the two embedding spaces drift apart and
  recall collapses.

## Questions to test yourself

1. Why does the two-tower structure make item embeddings cacheable, and a
   cross-network not?
2. What exactly does the logQ correction subtract, and what bias does it remove?
3. At what k should you measure Recall, and why does the choice of k matter?
4. When would you pick IVF-PQ over HNSW?
5. Why can an offline recall win still fail an online A/B?

## Further reading

- The dense reference version of this topic (comparison table, math, production
  case studies) lives in [topics/01-candidate-retrieval.md](../../topics/01-candidate-retrieval.md).
- Trace a two-tower graph live in the [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo).
