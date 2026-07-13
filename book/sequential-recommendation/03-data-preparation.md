# 3. Data preparation

## Building the sequence dataset

The raw material is the interaction log: timestamped events keyed by user. For
each user, sort events by timestamp, optionally deduplicate adjacent identical
items (a user who refreshed the same item three times in a row is not expressing
three separate intents), and cap the sequence at a recent window of N events
(50-200 is typical). From each sequence of length L, we generate L-1 training
pairs by sliding a window forward one step at a time:

| context (sequence so far) | target (next item) |
|---|---|
| [item_1] | item_2 |
| [item_1, item_2] | item_3 |
| [item_1, ..., item_{L-1}] | item_L |

Each pair is **causal**: the context contains only events that happened before the
target. This is the key constraint that prevents leakage. A sequence model trained
on non-causal pairs learns to "see the future" during training and performs
randomly at serving time where the future is genuinely unknown.

## Two leakage traps

**1. Time-based split is mandatory.** If you split pairs randomly, a user's early
events land in the training set while their later events land in the test set, but
those later events were still used to build the training-set contexts (since you
built L-1 pairs per sequence before splitting). That leaks the future. The correct
split: sort all users' events by time, hold out each user's last interaction as
the test target and the second-to-last as the validation target, and train on
everything earlier.

**2. Online and offline sequence construction must be identical.** The batch
pipeline that builds training pairs and the streaming pipeline that updates user
state online will drift unless they share the same deduplication rules, filtering
logic, and tie-breaking for simultaneous events. That drift is the classic
training-serving skew for sequence models: the encoder trains on one sequence
distribution and serves on another. Share the construction code, not just the
algorithm.

## Feature engineering

Each element in the sequence becomes a vector: the model needs a numeric
representation of every position.

- **Item ID embedding.** A learned dense vector per item, looked up from an
  embedding table. The same table can be shared with a retrieval system to keep
  item representations consistent across the funnel.
- **Action type.** Whether the event was a click, a long dwell, a purchase, a
  skip. Encode as a small learned embedding (4-8 dimensions) and add it to the
  item embedding.
- **Time signal.** Positional index (1st, 2nd, 3rd action) is a start. The
  refinement that most candidates skip: encode the **actual time gap** between
  consecutive events, not just their order. Two events one second apart differ
  from two events one month apart; a plain positional index does not see that
  difference.
- **Item content features.** Category, price tier, popularity bucket. These let
  the model generalize across items that share attributes, and they carry
  cold-start items (whose ID embeddings are untrained) into the sequence.

**When to use which feature treatment.**

| Reach for | When | Instead of |
|---|---|---|
| Learned ID embedding | the item recurs enough to train a stable vector | hashing, which you use only for unbounded or very-long-tail item spaces |
| Time-gap encoding (not just positional index) | recency matters and session gaps vary widely | a plain 1..N position index that treats actions equally spaced |
| Action-type embedding | behavior mix is heterogeneous (clicks, purchases, skips all in one sequence) | a single interaction type, which discards intent-signal variation |
| Item content features | cold-start items or sparse catalogs with many new items | relying purely on ID embeddings, which are untrained for new items |

The cold-start note: a new item's ID embedding is a random initialization, so
the model's first exposure is almost noise. Content features (category, price
tier, visual embedding) carry cold items until they accumulate enough interactions
to train a stable ID vector.
