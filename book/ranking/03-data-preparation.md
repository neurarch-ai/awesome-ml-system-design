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
  so the ranker does not over-credit top-position clicks. Concretely, each row is
  weighted by the inverse of its examination propensity and that weight scales its
  loss term:

$$w_i = \frac{1}{p(\text{examined} \mid \text{position}_i)}, \qquad \mathcal{L}_{\text{IPW}} = \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \ell(\hat{y}_i,\, y_i)$$

  A click at a rarely-examined low position (small propensity) thus counts for
  more than a click at the top, correcting the position bias in the labels.

## Feature engineering: three families

Three families of features, and naming all three signals experience.

**User features.** User id embedding (a short learned vector that stands in for
each id, so the model can represent millions of users in a compact numeric form),
demographic attributes if
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

where $w$ is the downsampling rate for negatives. Apply this correction as a
standard step so that head outputs remain calibrated probabilities that can be
combined into a utility score.

**When to use which data treatment.**

| Reach for | When | Instead of |
|---|---|---|
| Point-in-time feature join | Any ranking training pipeline | Joining on current features, which leaks the future |
| Position as training feature (fixed at serve) | Impression logs where position is recorded | Ignoring position bias, which teaches the ranker to predict position |
| Negative downsampling with calibration correction | Click rate under 5%, billions of rows | Training on the full imbalanced set when compute is limited |
| Cross features (user x item) | Always; they are the biggest accuracy lever | Relying on separate user and item features hoping the MLP learns interactions |
| Content and context features for cold items | New items with few or no interactions | Per-item id embeddings, which are uninformative for new ids |

## Where the labels come from

Ranking labels come from three sources, and knowing which one produced a row tells you
how much to trust it and how to correct it.

| Label source | What it gives you | The bias, and the fix |
|---|---|---|
| Implicit feedback (interaction logs) | Abundant and cheap: clicks, dwell, and downstream conversions on every impression | Position, exposure, and selection bias: top slots win clicks regardless of relevance, and you only log outcomes for items the current ranker surfaced. Correct with position-as-feature (fixed at serve), inverse-propensity weighting, and a slice of exploration data. |
| Human raters | High-quality relevance judgments that are unbiased by the product's own exposure | Slow, costly, low volume. Use for the golden eval set and to calibrate a cheaper learned model. |
| Targeted collection / synthesis | Coverage for cold-start and rare queries or items where logs are thin | Not organic, so treat as a supplement. Use open relevance datasets, augmentation, or a stronger model as a teacher to seed sparse regions. |

Because interaction logs are time-ordered, split train and test by time (train on
earlier sessions, evaluate on later ones), never by a random shuffle, so future
outcomes cannot leak backward into training.
