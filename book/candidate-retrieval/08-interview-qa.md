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
**Why:** the offline metric is complicit. The future hold-out is itself
popularity-skewed (head items account for a disproportionate share of future
interactions), so a model that leans into the head hits more hold-out labels and
recall@k rises even as the lived experience narrows. The metric and the failure
share a cause, which is why recall alone cannot catch it.

**Q: Bigger batches give more free negatives, so should you always scale the batch?**
A: No. If batches are request-sorted or user-concentrated, a user's own other
engaged items land in the same batch and get scored as negatives. Pinterest saw
the in-batch false-negative rate rise from near 0% to about 30%. The fix is
user-level masking (drop same-user items from the softmax denominator), not simply
a larger batch.
**Why:** every false negative is a gradient with the wrong sign: the loss
actively pushes an item the user engaged with away from that user's embedding.
Scaling the batch raises the chance that any given user's other positives
co-occur in it, so the corruption grows with exactly the knob you turned for
more signal.

**Q: Everyone uses HNSW. When would you not?**
A: When the catalog churns hard or needs filters. Airbnb chose IVF because HNSW's
rebuild cost could not absorb price and availability updates, and geo filters ran
poorly over graph traversal; IVF turns a filter into cheap cluster selection. Match
the index to update rate, filtering, and memory, not to a default. Use HNSW with
product quantization (Etsy) when the index must fit memory at large N.
**Why:** the structural reason is what an update touches. An HNSW insert or
delete mutates a navigable graph whose search quality depends on carefully
maintained neighbor lists, so heavy churn degrades it and deletions are often
tombstoned rather than removed. An IVF update is appending or dropping a vector
in a cluster's posting list, which is trivially cheap, and a filter becomes
"only scan the eligible lists" instead of repeatedly hitting filtered-out nodes
mid-traversal.

**Q: Dot product or cosine similarity, and does it matter?**
A: It matters. Dot product lets embedding magnitude carry signal (popular or
high-quality items can score higher for free), because `u . v = |u| |v| cos(theta)`.
Cosine normalizes magnitude away for pure semantic match. Pick based on whether you
want popularity baked into the geometry (Airbnb reasons about exactly this).

**Q: What breaks when you retrain the towers?**
A: The user and item towers must be versioned and re-indexed together. If you ship
a new user tower against an index built by the old item tower, the two embedding
spaces drift apart and recall collapses. Coordinate the redeploy.
**Why:** the training loss only constrains relative geometry, the dot products
among vectors trained together; nothing pins the space to absolute coordinates,
so each run lands in an arbitrarily rotated and scaled version of "the" space.
A dot product between a vector from run N and a vector from run N-1 is therefore
meaningless, even if both runs individually reached identical recall.

**Q: You switched the loss from dot product to cosine, but the ANN index still
scores with inner product. What breaks?**
A: Recall silently collapses unless the index geometry matches the training
geometry. Cosine similarity is inner product on L2-normalized vectors (each scaled to length
1, dividing by its L2 norm, the square root of the sum of its squared entries), so if the
towers were trained with cosine you must normalize both the stored item vectors
and the query vector before indexing, then an inner-product index gives the right
ranking. If you normalize on one side only (or not at all), the index is now
ranking by `|u| |v| cos(theta)` while the model learned `cos(theta)`, so long
vectors win for free and the retrieved set no longer matches what the loss
optimized. The rule: the index distance metric and the training similarity must be
the same function, normalization included.

**Q: In-batch negatives and uniformly sampled negatives look similar; when does
the difference actually matter?**
A: Both plug the same hole (no logged negatives) and both feed the same sampled
softmax, but they draw from different distributions. In-batch negatives are
other users' positives, so they arrive in proportion to item popularity;
uniform negatives arrive in proportion to nothing. The difference matters in two
places. First, bias: popularity-proportional sampling over-penalizes head items,
which is survivable only if you apply the logQ correction; uniform sampling
needs no correction but mostly serves up tail items the model already scores
low, so its gradients go soft early. Second, cost: in-batch negatives are free
(one matrix multiply reuses the batch), while uniform negatives require a
separate sampling and embedding path. On a small catalog with mild popularity
skew the two train nearly identical models; at web scale with a heavy head, the
choice plus its correction visibly moves which items the space favors.

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
**Why:** the bias being corrected exists only during training. In-batch
negatives are sampled in proportion to item popularity (they are other users'
positives), so the sampled softmax over-penalizes popular items relative to the
full softmax; subtracting log of the sampling probability from the logit undoes
exactly that over-sampling. At serving there is no sampling, so there is no bias
left to correct, and subtracting log-popularity there would instead inject a new
anti-popularity distortion the model never learned.

**Q: Can you add a few cross features between user and item for accuracy?**
A: Not in retrieval. Any early crossing makes the score depend on the user, which
kills offline precompute and the ANN lookup. Early crossing is the ranking stage's
job (cross networks, NCF); retrieval must keep the join at a final dot product.

**Q: Should retrieval optimize precision so ranking has less to do?**
A: No. Retrieval is a recall stage: get the good items into the candidate pool
cheaply. Precision is ranking's job on the few hundred survivors. Optimizing
precision in retrieval wastes the funnel's division of labor and usually hurts
recall.
**Why:** the error costs are asymmetric by construction. An item retrieval drops
can never be ranked, so a false negative here is unrecoverable anywhere
downstream; a false positive just gets demoted by a model that is better at
precision than retrieval could ever be. Spending retrieval's capacity on
precision buys accuracy the funnel already has while selling recall the funnel
cannot get back.

**Q: Harder negatives always teach more, so should you mine hard negatives from
step one?**
A: No, and the failure mode is specific. Retrieval labels are implicit: a
non-interaction is not a confirmed negative, just an unobserved one. Mining the
"hardest" items (nearest in embedding space to the anchor) disproportionately
surfaces items the user would actually like but never saw, so you train the model
to push away true positives, which caps recall and can destabilize the embedding
geometry. The standard recipe is to warm up on in-batch (easy) negatives so the
space is roughly organized, then introduce mined hard negatives gradually, often
mixed with in-batch negatives rather than replacing them. "Hardest possible from
the start" is the wrong answer.
