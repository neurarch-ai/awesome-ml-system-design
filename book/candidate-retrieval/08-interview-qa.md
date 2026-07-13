# 8. Interview Q&A

The questions an interviewer actually asks about candidate retrieval, grouped by
how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Why two towers instead of one model over user and item features?**
A: Latency at scale. A single cross-feature model is more accurate, but its score
depends on the user, so you would have to run it over all 100M items per request,
which no latency budget allows. Two towers factor the problem: item embeddings are
user-independent, so you precompute all of them offline and index them, and only
run the user tower online. The single model is the right choice for re-ranking a
few hundred items, not for retrieval over the whole catalog.

**Q: Where do negatives come from if you only logged positives?**
A: In-batch negatives. For a batch of B positive (user, item) pairs, each user
treats the other B-1 items in the batch as negatives; one matrix multiply gives
all B-by-B scores. Add hard-negative mining once random in-batch negatives get too
easy to keep teaching.

**Q: How do you evaluate retrieval?**
A: Recall@k against a time-based (future) hold-out, measured at the k you actually
pass to ranking. Then gate the launch on an online A/B for engagement and coverage.
Do not headline precision; that is ranking's metric.

**Q: How does a brand-new item get retrieved?**
A: Its ID embedding is untrained, so it rides on content features (title,
thumbnail, category) and a dedicated fresh-items retrieval source until it gathers
enough interactions to train its ID embedding. Freshness is a minutes-cadence
re-embed and index upsert.

## Tricky (the follow-ups that separate people)

**Q: Your recall@k improved offline but engagement dropped online. What happened?**
A: Most likely popularity collapse. The model started resurfacing head items,
which lifts raw recall (they are easy to hit) while shrinking coverage and
diversity, so the feed feels generic. Diagnose with coverage and tail metrics, and
verify the logQ correction is actually applied. A recall gain that costs diversity
usually loses long-term.

**Q: Bigger batches give more free negatives, so should you always scale the batch?**
A: No. If batches are request-sorted or user-concentrated, a user's own other
engaged items land in the same batch and get scored as negatives. Pinterest saw
the in-batch false-negative rate rise from near 0% to about 30%. The fix is
user-level masking (drop same-user items from the softmax denominator), not simply
a larger batch.

**Q: Everyone uses HNSW. When would you not?**
A: When the catalog churns hard or needs filters. Airbnb chose IVF because HNSW's
rebuild cost could not absorb price and availability updates, and geo filters ran
poorly over graph traversal; IVF turns a filter into cheap cluster selection. Match
the index to update rate, filtering, and memory, not to a default. Use HNSW with
product quantization (Etsy) when the index must fit memory at large N.

**Q: Dot product or cosine similarity, and does it matter?**
A: It matters. Dot product lets embedding magnitude carry signal (popular or
high-quality items can score higher for free), because `u . v = |u| |v| cos(theta)`.
Cosine normalizes magnitude away for pure semantic match. Pick based on whether you
want popularity baked into the geometry (Airbnb reasons about exactly this).

**Q: What breaks when you retrain the towers?**
A: The user and item towers must be versioned and re-indexed together. If you ship
a new user tower against an index built by the old item tower, the two embedding
spaces drift apart and recall collapses. Coordinate the redeploy.

## Commonly answered wrong (the traps)

**Q: Do the two towers share weights to save parameters?**
A: No. Users and items have different feature distributions, so the towers stay
separate; the only thing they share is the output embedding space, enforced by the
dot-product loss. (Uber deliberately shares a UUID embedding layer to collapse many
per-city models into one, but that is the exception, not the rule.)

**Q: Is the logQ correction applied at serving time, by subtracting log-popularity
from the ANN results?**
A: No. logQ is subtracted from the logits during **training**, so the embedding
space itself comes out unbiased. Serving stays a plain dot-product or cosine lookup
with no correction term. Applying it at serving is a common and wrong answer.

**Q: Can you add a few cross features between user and item for accuracy?**
A: Not in retrieval. Any early crossing makes the score depend on the user, which
kills offline precompute and the ANN lookup. Early crossing is the ranking stage's
job (cross networks, NCF); retrieval must keep the join at a final dot product.

**Q: Should retrieval optimize precision so ranking has less to do?**
A: No. Retrieval is a recall stage: get the good items into the candidate pool
cheaply. Precision is ranking's job on the few hundred survivors. Optimizing
precision in retrieval wastes the funnel's division of labor and usually hurts
recall.
