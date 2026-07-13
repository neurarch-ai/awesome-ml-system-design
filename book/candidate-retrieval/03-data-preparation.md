# 3. Data preparation

## Data engineering: building training pairs

The raw material is the interaction log: which users engaged with which items.
Each **positive pair** is a (user, item) the user genuinely engaged with (a click
with long dwell, per our requirements). A row in the training set looks like this.

| user | item | context | label |
|---|---|---|---|
| u_8123 | v_5501 | evening, mobile | positive |
| u_8123 | v_9020 | evening, mobile | positive |
| u_4471 | v_1200 | morning, web | positive |

The subtle part is **negatives**: items the user did *not* engage with. We do not
store explicit negatives. Instead, the model will treat other items in the same
training batch as negatives for a given user (in-batch negatives), which the next
chapter covers. For now, the data job is to produce clean positive pairs and to
avoid two traps:

- **Popularity bias.** Popular items appear in almost everyone's positives, so the
  model learns "recommend popular things." We correct for this at training time
  with a sampling correction (the logQ correction), not by throwing away data.
- **Leakage from the ranking loop.** If positives come only from what the old
  system already showed, the model just imitates it. Mix in some exploration
  traffic so the data is not fully self-fulfilling.

## Feature engineering

Almost all ML models accept only numeric input, so every feature becomes a number
or an embedding.

- **User features.** ID embedding (a learned vector per user), plus aggregates of
  recent behavior (categories watched, average session length) and context (time
  of day, device).
- **Item features.** ID embedding, category, language, age of the item, and
  content features (a title embedding, thumbnail features).
- **Sparse IDs to vectors.** High-cardinality IDs (user, item) map to dense
  vectors through an embedding table, the same lookup-table idea used throughout
  ML: an ID indexes a row of learned weights.

**When to use which feature treatment.**

| Reach for | When | Instead of |
|---|---|---|
| Learned ID embedding | the ID recurs enough to learn a stable vector (active users, catalog items) | hashing, which you use only when the ID space is unbounded or long-tail |
| Feature hashing | unbounded or rarely-seen IDs (raw search terms, new device models) | a lookup table, which would grow without bound |
| Content features (title, thumbnail) | cold-start items with little interaction history | relying on the ID embedding, which is untrained for new items |
| Behavioral aggregates | you have enough history per user | ID-only, which underfits for users with rich histories |

The cold-start point matters: a brand-new item has an untrained ID embedding, so
its **content features** are what let it be retrieved at all in its first minutes.
With data and features in hand, we can build the model.
