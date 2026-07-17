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
**Why:** the model's weights encode assumptions about the exact feature
distribution it trained on, so a feature computed differently at serving time
puts the model in silent extrapolation, where its errors are unbounded. And
offline evaluation cannot catch this by construction: the eval set is built by
the same training pipeline, so the seam between the two codepaths is invisible
until real serving traffic crosses it.

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
**Why downsampling distorts the base rate:** if you keep 1 in k negatives, the
model sees k times more positives per negative than reality contains, so the
probabilities it learns are honest about the training data but the training data
lied about the world. A post-hoc monotone map (Platt, isotonic) undoes exactly
that distortion without touching the ordering.

**Q: A colleague wants to add listing-id embeddings to the Airbnb ranker for
personalization. Why is that dangerous?**
A: Because each Airbnb listing books at most roughly 365 times a year, so the
per-listing id embedding sees very few training examples and overfits. The
embedding carries noise, not signal. For sparse-per-id settings, lean on content
features (location, price, amenities, listing type) and contextual features that
generalize across listings. Reserve id embeddings for entities with dense,
repeated supervision (product ids in a marketplace with billions of transactions,
not individual travel listings).
**Why:** an id embedding is a block of free parameters updated only by that id's
own training examples, and no other listing's data can correct it because no
other listing shares those parameters. With a few hundred noisy labels driving a
multi-dimensional vector, gradient descent memorizes which particular bookings
happened rather than any transferable property of the listing. Content features
do generalize because every listing with the same price band or location keeps
updating the same shared weights.

**Q: LambdaMART optimizes NDCG, but NDCG is flat almost everywhere and
discontinuous in the model scores. How does gradient boosting train on it at
all?**
A: It never differentiates NDCG. LambdaMART (Microsoft, 2010) specifies a
per-pair gradient (a "lambda") directly, without any closed-form loss whose
derivative it is. For each (winner, loser) pair it takes the RankNet (Microsoft,
2005) pairwise-logistic gradient and scales it by the magnitude of the NDCG
change that swapping just those two items would produce. Summing those lambdas
over every pair an item appears in gives that item's gradient, and the regression
trees fit that gradient. So the "loss" is implicit: you write down the gradient
you wish existed, weighted so that mis-ordering a high-value pair near the top
pulls hardest, which is exactly why the model moves NDCG without NDCG ever being
differentiated.

**Q: AUC and NDCG both measure ranking quality and usually move together; when
does the difference actually matter?**
A: AUC is position-blind: every (positive, negative) inversion counts the same,
so fixing a mis-ordering at rank 400 moves AUC exactly as much as fixing one at
rank 2. NDCG@k discounts gain by the log of the rank and truncates at k, so it
only rewards fixes a user can see. The difference bites whenever the display
surface is short: a model change that cleans up mid-list order can lift AUC
while leaving the rendered top ten untouched, which is one honest reason an
offline AUC win ships flat online. The reverse also happens: swapping ranks 1
and 3 barely dents AUC over millions of pairs but visibly changes the product.
Use AUC to confirm the model discriminates at all, and gate on NDCG@k at the k
you actually render.

## Commonly answered wrong (the traps)

**Q: Do the two towers in a ranker share weights to save parameters?**
A: The question confuses retrieval two-towers with the user and ad towers in
rankers like Snap. In retrieval, the user and item towers stay separate (different
feature distributions). In rankers like Snap's ad system, the user-feature tower
and ad-feature tower are separate so the user side can be computed once and
shared across all candidate ads. Neither case involves sharing weights between
the user side and the item side. Sharing would destroy the ability to precompute.
**Why:** the compute saving comes from the split itself, not from the parameter
count: because the user tower's output depends only on user features, it can be
computed once per request (or cached) and reused across every candidate. Tying
weights would also force one function to embed two unrelated feature spaces
(user behavior history versus ad content) with different vocabularies and
semantics, so nothing is gained and both representations get worse.

**Q: Can you add a few user-item cross features for accuracy without rethinking
the architecture?**
A: In retrieval, no: any user-item crossing kills the offline precompute. In
ranking, yes: cross features are the biggest accuracy lever and they belong
exactly here, where you already have both the user and the item at hand. The
constraint is only that the cross feature must be computable at serving latency.
Pre-compute slow cross signals offline and serve from a feature store.
**Why the retrieval "no" is absolute:** the two-tower factorization works because
item vectors are user-independent, so the whole catalog can be embedded once and
indexed in the ANN. A single user-item cross feature makes every item's
representation a function of the current user, which means re-embedding the
catalog per request, which is exactly the hundred-million-item cost the
factorization exists to avoid. Ranking never precomputed item scores in the
first place, so it has nothing to lose.

**Q: Is the logQ correction applied at ranking?**
A: No. logQ (the sampling-bias correction) is a retrieval training trick that
de-biases the embedding space by subtracting the log-sampling-probability from
in-batch logits. Ranking does not use in-batch negatives and does not have a
sampling-bias problem in the same form. The corrections relevant to ranking are
position bias correction in the labels and negative-downsampling calibration
correction in the output probabilities.
**Why retrieval needs it and ranking does not:** in-batch negatives are drawn in
proportion to item popularity (popular items appear in more training rows), so
popular items get hammered as negatives far more often than a uniform sample
would dictate, and the embedding space learns to underscore exactly the items
users like most; subtracting the log-sampling-probability cancels that
distortion. Ranking's negatives are logged impressions that were actually shown
and not clicked, arriving at their natural serving-time rate, so there is no
artificial sampling distribution to undo.

**Q: Does adding more objectives in a multi-task ranker always help?**
A: No. Objectives that are positively correlated (clicks and saves) help each
other through shared representation. Objectives that are negatively correlated
(for example: "close-up" and "repin" at Pinterest) can interfere under a single
shared body, letting one task's gradient hurt the other. The fix is MMoE or PLE
gating (each task picks its own mixture of experts) rather than a flat shared
body, and monitoring per-task metrics so you can see when one task is degrading
another.
**Why:** shared layers receive the sum of every task's gradient; when two tasks
want the shared representation to move in opposite directions, the updates cancel
or seesaw, so features one task depends on get overwritten by the other (negative
transfer). Gating routes each task's gradient mostly into its own preferred
experts, so the conflicting updates land on different parameters instead of
fighting over the same ones.

**Q: Does applying Platt or isotonic calibration change the ranking order within
a single model?**
A: No, and stopping there is the trap. Platt scaling (Platt, 1999) is a monotone
increasing logistic map and isotonic regression fits a monotone non-decreasing
step function, so both are order-preserving on one model's scores: AUC and NDCG
are unchanged by calibration. But that does not mean calibration is optional. The
moment two calibrated quantities are combined (a multi-task utility blend, or an
auction bid = value times predicted probability), a monotone remap of one head
relative to another absolutely reorders the blended result. Calibration is
order-neutral within a head and order-changing across heads, which is precisely
why it earns its keep only once scores leave pure sorting.
