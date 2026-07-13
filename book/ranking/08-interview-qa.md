# 8. Interview Q&A

The questions an interviewer actually asks about ranking models, grouped by how
they are used. The "commonly answered wrong" group is where interviews are won or
lost.

## Commonly asked

**Q: Why have a separate ranking stage at all? Why not just use the retrieval
model to pick the final order?**
A: Because retrieval and ranking operate under different constraints. Retrieval
runs over a hundred million items under a tens-of-millisecond budget, so its
model must be cheap and factored (a two-tower dot product). Ranking runs over a
few hundred survivors and can afford rich cross features, explicit interaction
layers, and multi-task heads, all of which would be impossible at catalog scale.
Retrieval maximizes recall cheaply; ranking maximizes precision expensively.
Folding them together means giving up one or the other.

**Q: What are cross features and why do they matter?**
A: Cross features are user-times-item signals that cannot be recovered from user
features and item features separately. For example: "how many times has this user
engaged with this item's category in the last 7 days," or "the semantic overlap
between this user's recent queries and this item's title." These are typically
the single biggest accuracy lever in ranking because they encode the specific
relationship between this user and this item right now. A model that only sees
the user's global history and the item's global stats is leaving the most
informative signal on the table.

**Q: How do you handle multiple objectives (clicks, saves, long dwell)?**
A: Use a multi-task ranker: a shared lower body with per-objective output heads,
each trained with its own binary cross-entropy loss. Calibrate each head
separately so the per-objective probabilities are meaningful. Then combine them
into a single utility score as a weighted sum, with the weights kept outside the
model so the business can retune them without retraining. Pinterest built exactly
this: changing the weight on saves versus clicks reorders the feed within hours
without touching model weights.

**Q: How do you evaluate a ranker offline?**
A: AUC for binary discrimination quality; NDCG@k when order and position matter
(as in search or when you want to reward the best item at rank 1); logloss when
you need to track probability sharpness alongside AUC; calibration error (ECE)
when scores feed an auction or a utility blend. Use a time-based hold-out, not a
random split, to avoid leaking future engagement into training. Then gate the
ship decision on an online A/B test on the business metric, because offline
gains routinely fail to survive the training-serving seam.

**Q: How do you keep a ranker scoring hundreds of candidates in under 20 ms?**
A: State the budget out loud and design backward. With 500 candidates and 20 ms,
that is roughly 0.04 ms per candidate. Batch the forward pass so all candidates
go through one model call. Fetch user and context features once and broadcast
across all candidates; only item and cross features vary. Precompute as much as
possible in the offline feature pipeline so online work is assembly plus one
model call. Keep the per-candidate MLP small enough that the bottleneck is
batched linear algebra, not a fan-out of individual calls.

## Tricky (the follow-ups that separate people)

**Q: Your offline AUC improved but online engagement fell. What went wrong?**
A: Suspect the training-serving seam first. The most common cause is
training-serving skew: a feature computed one way in the training pipeline and a
different way in the serving code means the model operates on a distribution it
never trained on. Other candidates: label leakage from a non-point-in-time join
(the feature secretly encodes the outcome), position bias uncorrected in the
offline labels (the ranker learned to predict position, not relevance), or an
offline metric that does not match the online objective. Run a feature
distribution comparison between training and serving before suspecting the model.

**Q: Where exactly in DLRM do the feature interactions happen? Why does the
placement matter?**
A: After the embedding tables and the bottom MLP, and before the top MLP. The
bottom MLP projects dense features to the same width as the embeddings. Explicit
pairwise dot products are then taken between all pairs of these vectors to form
second-order interaction terms. Those terms concatenate with the dense vector and
feed the top MLP. The placement matters because the dot products are undefined
unless the dense and embedding vectors share the same width, and because putting
the interaction after the top MLP loses the structured second-order signal
entirely. Diagrams that draw the interaction before the embeddings, or that feed
concatenated embeddings straight into a top MLP, are describing a different and
less powerful model.

**Q: When should you calibrate, and when is ordering enough?**
A: Calibration earns its place the moment a score leaves pure sorting. If you
are only ordering a list and never using the scores as probabilities, raw order
is enough and calibration adds pipeline cost for nothing (Airbnb, Yelp). The
moment a score feeds an auction bid (bid = value times predicted probability), a
threshold, or a weighted utility blend across tasks, the probabilities must mean
what they say. Negative downsampling and stratified training both distort the
base rate, so apply a post-hoc Platt or isotonic step and monitor ECE
continuously. Spotify monitors ECE as a live metric because miscalibration
directly means over- or under-bidding in the ad auction.

**Q: A colleague wants to add listing-id embeddings to the Airbnb ranker for
personalization. Why is that dangerous?**
A: Because each Airbnb listing books at most roughly 365 times a year, so the
per-listing id embedding sees very few training examples and overfits. The
embedding carries noise, not signal. For sparse-per-id settings, lean on content
features (location, price, amenities, listing type) and contextual features that
generalize across listings. Reserve id embeddings for entities with dense,
repeated supervision (product ids in a marketplace with billions of transactions,
not individual travel listings).

## Commonly answered wrong (the traps)

**Q: Do the two towers in a ranker share weights to save parameters?**
A: The question confuses retrieval two-towers with the user and ad towers in
rankers like Snap. In retrieval, the user and item towers stay separate (different
feature distributions). In rankers like Snap's ad system, the user-feature tower
and ad-feature tower are separate so the user side can be computed once and
shared across all candidate ads. Neither case involves sharing weights between
the user side and the item side. Sharing would destroy the ability to precompute.

**Q: Can you add a few user-item cross features for accuracy without rethinking
the architecture?**
A: In retrieval, no: any user-item crossing kills the offline precompute. In
ranking, yes: cross features are the biggest accuracy lever and they belong
exactly here, where you already have both the user and the item at hand. The
constraint is only that the cross feature must be computable at serving latency.
Pre-compute slow cross signals offline and serve from a feature store.

**Q: Is the logQ correction applied at ranking?**
A: No. logQ (the sampling-bias correction) is a retrieval training trick that
de-biases the embedding space by subtracting the log-sampling-probability from
in-batch logits. Ranking does not use in-batch negatives and does not have a
sampling-bias problem in the same form. The corrections relevant to ranking are
position bias correction in the labels and negative-downsampling calibration
correction in the output probabilities.

**Q: Does adding more objectives in a multi-task ranker always help?**
A: No. Objectives that are positively correlated (clicks and saves) help each
other through shared representation. Objectives that are negatively correlated
(for example: "close-up" and "repin" at Pinterest) can interfere under a single
shared body, letting one task's gradient hurt the other. The fix is MMoE or PLE
gating (each task picks its own mixture of experts) rather than a flat shared
body, and monitoring per-task metrics so you can see when one task is degrading
another.
