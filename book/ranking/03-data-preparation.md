# 3. Data preparation

## Labels from logs: impressions and outcomes

The raw material is the impression log. Each row is one candidate shown to one
user, with an engagement outcome.

| user | item | context | cross: cat-affinity-7d | position | label (click) |
|---|---|---|---|---|---|
| u_8421 | v_1120 | evening, mobile | 0.72 | 3 | 1 |
| u_8421 | v_3304 | evening, mobile | 0.08 | 7 | 0 |
| u_9011 | v_0882 | morning, web | 0.44 | 1 | 1 |

The join must be **point-in-time correct**: when building the training row, use
feature values as they were at impression time, not their current values. Joining
on the latest feature snapshot instead leaks the future into training and inflates
offline metrics without helping online.

## Position bias

Items shown at higher positions get more clicks regardless of their true quality.
A naive ranker trained on these labels learns to predict position, not relevance.
Two standard corrections:

- **Position as a training feature, held fixed at serving.** Include position in
  the training features so the model can learn its effect, then set position to a
  neutral constant (say 1) at serving time. This is the simplest approach and
  widely used.
- **Inverse-propensity weighting (IPW).** Estimate the probability of an item
  being examined at each position, then upweight observations at lower positions
  so the ranker does not over-credit top-position clicks.

## Feature engineering: three families

Three families of features, and naming all three signals experience.

**User features.** User id embedding (learned), demographic attributes if
available, aggregated interaction history (category affinities, engagement rates
over multiple time windows), and context (time of day, device, location).

**Item features.** Item id embedding (learned when the item has enough
interactions), category, popularity, content embeddings (text, image), age of
the item, and historical engagement rates computed from past data.

**Cross features.** User-times-item signals: "how many times has this user
engaged with this item's category in the last 7 days," "this user's affinity for
this author," "the query-item semantic match." These **cannot** be recovered from
user and item features separately. They are typically the single biggest accuracy
lever in ranking. A model that sees only separate user and item features is
leaving signal on the table.

Sparse categorical features (user id, item id, category) feed into **embedding
tables**: each id indexes a row of learned weights. The embedding tables, not the
MLP layers, hold most of the model's parameters (millions of ids times an
embedding dimension). Dense numeric features feed the network directly after
normalization.

## Sampling

**Negative downsampling.** With billions of impressions per day and a click rate
of 1-2%, training on the full set is wasteful. Downsample non-engagement rows to
a manageable ratio. When you do, the model's raw output is no longer a calibrated
probability. Apply a correction at inference:

$$\hat{p}_{\text{corrected}} = \frac{\hat{p}}{(\hat{p} + (1 - \hat{p}) / w)}$$

where $w$ is the downsampling rate for negatives. Pinterest applies this
correction as a standard step before combining head outputs into a utility score.

**When to use which data treatment.**

| Reach for | When | Instead of |
|---|---|---|
| Point-in-time feature join | Any ranking training pipeline | Joining on current features, which leaks the future |
| Position as training feature (fixed at serve) | Impression logs where position is recorded | Ignoring position bias, which teaches the ranker to predict position |
| Negative downsampling with calibration correction | Click rate under 5%, billions of rows | Training on the full imbalanced set when compute is limited |
| Cross features (user x item) | Always; they are the biggest accuracy lever | Relying on separate user and item features hoping the MLP learns interactions |
| Content and context features for cold items | New items with few or no interactions | Per-item id embeddings, which are uninformative for new ids |
