# 8. Interview Q&A

The questions an interviewer actually asks about embeddings and representation
learning, grouped by how they are used. The "commonly answered wrong" group is
where interviews are won or lost.

## Commonly asked

**Q: Why learn embeddings instead of using hand-built features?**
A: Three reasons. One-hot id vectors are huge, sparse, and orthogonal: every item
is equally far from every other, so the representation carries no similarity
information at all. Manually engineered features (category, price bucket, tags)
capture only the axes a human thought to encode. A learned embedding discovers
the axes that actually predict the behavior you trained on, compresses millions of
sparse ids into a few hundred dense dimensions, and crucially makes "similar" a
geometric fact you can query with nearest-neighbor search. That last property is
what turns representation learning into a serving primitive.

**Q: How do you train an embedding model without explicit similarity labels?**
A: Contrastively. Positives come from co-occurrence or interaction logs (the user
engaged with this item, these two items co-occur in sessions, this node is a
neighbor of that one via random walk). The negatives are sampled explicitly. The
labels are implicit in behavior; you never need a human to say "these two items
are 0.7 similar."

**Q: Where do negatives come from?**
A: In-batch for free (every batch of B positive pairs yields B times B minus B
negative pairs at no extra cost), plus mined hard negatives for sharpness once
in-batch negatives get too easy. Add the logQ correction when the catalog is
popularity-skewed to undo the fact that popular items appear as negatives far more
often.

**Q: How do you handle a brand-new item with no interaction history?**
A: Structurally, not as a patch. Use an inductive encoder that consumes content
features (text, category, image, graph neighbors). A new entity maps to a sensible
point in the space from its attributes alone, with zero history. Id-only embeddings
(LightGCN, classic matrix factorization) have no vector for an unseen entity at
all and need a content-based fallback or a fresh-entity source until interactions
accumulate. State which side you are on.

**Q: How do you pick the embedding dimension?**
A: Name all three parts of the tradeoff: recall (more dimensions give more capacity
but gains diminish), memory (index memory scales with dimension), and latency
(distance computations scale with dimension). Pick a modest starting point (64 to
256), tune it against the downstream consumer rather than in isolation, and reach
for quantization before chasing dimension if memory is the binding constraint.

## Tricky (the follow-ups that separate people)

**Q: You added hard negatives and training loss dropped, but downstream recall
stayed flat. What happened?**
A: The most likely cause is false negatives. Some of the items you mined as "hard
negatives" are actually items the user would have engaged with if shown (unlabeled
positives). Training on them teaches the model the wrong thing and can actually
degrade the decision boundary precisely where it was about to improve. Cap the hard
fraction, filter obvious near-duplicates of the anchor, and cross-check that the
hard-negative source excludes known positives.

**Q: Bigger batches give more free negatives. Should you always scale the batch?**
A: Not without checking batch composition. If batches are user-concentrated (many
rows from the same user in the same batch), a user's own other engaged items appear
as negatives and the model learns to push them down. This is the false-negative
problem at scale. The fix is user-level masking (drop same-user items from the
softmax denominator), not a larger batch. Pinterest measured the in-batch
false-negative rate rising to roughly 30% without masking.

**Q: Your embedding space looks healthy on cosine probes but production engagement
dropped. What could explain it?**
A: Representation collapse is a candidate. A weak loss or too-easy negatives can
map most entities into a narrow region of the space where all cosine similarities
are high (above 0.9) and ranking is meaningless. Cosine probes on a probe set
still look fine because all pairs are "similar." Check the uniformity diagnostic:
if $\ell_{\text{unif}}$ is high (near zero), the space is not spread and collapse
is happening. Also check tail recall and catalog coverage; popularity collapse
means head items dominate, which can look fine on average recall while crushing
the long tail.

## Commonly answered wrong (the traps)

**Q: The logQ correction: apply it at training time or at serving time?**
A: Training time only. Subtracting $\log Q(y)$ from the logit during training
reshapes the embedding space itself so it comes out unbiased. At serving time you
run a plain dot-product or cosine lookup with no correction term. Applying logQ at
serving is a classic and incorrect answer; the embedding is already unbiased after
the trained correction, and double-correcting at serve would re-introduce bias in
the opposite direction.

**Q: Should the two towers share weights to save parameters?**
A: No, in the general case. Users and items have different feature distributions
and different feature types; sharing weights would force the same transformation
on inputs that are not comparable. The two towers share only the output embedding
space, enforced by the contrastive loss. (Uber deliberately shares a UUID
embedding layer to collapse many per-city models into one global model, but that
is a domain-specific trick enabled by the fact that drivers and riders share the
same ID namespace, not a general principle.)

**Q: What happens to the index when you retrain the encoder?**
A: The axes of the embedding space move. A vector produced by the new encoder is
not comparable to a vector produced by the old one, so you cannot upsert new
vectors into an index full of old vectors. You must full-reindex the entire entity
set against the new encoder, atomically, and coordinate the cutover. The mistake
is to upsert incrementally and end up with a mixed-version index where similarity
scores are meaningless between old and new vectors.

**Q: Can you add a cross feature between the user and item towers for accuracy?**
A: Not in the embedding step. Any early crossing makes the score depend on both
sides simultaneously, which means you can no longer precompute the item side and
serve it from an ANN index. Early crossing is the ranking stage's job; the
retrieval embedding must keep the two sides separate until the final dot product.
Adding cross features at the retrieval stage defeats the whole reason the two
towers exist.
