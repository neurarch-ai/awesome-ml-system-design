# 3. Data preparation

## Building positive pairs

The raw material is the interaction log: a record of which users engaged with
which items, which items co-occurred in sessions, or which graph nodes share an
edge. A **positive pair** is any two entities whose co-occurrence carries the
"related" signal you want to encode.

Common positive pair constructions, and what each teaches the space:

| Signal source | Positive pair | What the space encodes |
|---|---|---|
| User clicked item then dwelled | (user, item) | user taste matched to item content |
| Two items bought in the same session | (item A, item B) | item-item co-purchase relatedness |
| Random walk on a social graph | (node A, node B nearby in walk) | graph neighborhood proximity |
| Reformulated search query | (failed query, successful query) | query semantic equivalence |
| Same sentence, two dropout masks | (sentence view A, sentence view B) | sentence-level semantic meaning |

The subtle point: the choice of positive pair is the most consequential design
decision in the whole system. It defines what "related" means in the embedding
space, and every downstream consumer inherits that definition. Choose it to match
what retrieval or ranking actually needs.

## Negatives

You will not have explicit "unrelated" labels. You must sample or construct
negatives. There are three strategies, each with different costs and risks:

**In-batch negatives.** In a batch of $B$ positive pairs, treat every anchor's
non-partner items in the same batch as its negatives. Negatives come for free with
the batch; larger batches give more negatives. The cost: negatives are drawn from
the data distribution, so popular items appear as negatives far more often and the
model learns to push them down unfairly. The fix (the **logQ correction**) is
covered in section 4.

**Hard negatives.** Mine items that are close to the anchor in the current
embedding space but are not actually related (e.g., a different laptop when the
query is "laptop"). These sharpen the decision boundary where the model is
currently wrong. The risk: some "hard negatives" are false negatives, meaning
items the user would have engaged with if shown. Too many hard negatives destabilize
training. The defensible recipe is mostly in-batch with a small, carefully tuned
hard fraction.

**Domain-aware negatives.** Sample negatives from a restricted pool that is
harder than random but still provably unrelated. Airbnb samples negatives from
the same geographic market so the model learns fine within-market distinctions
rather than trivially separating New York from Tokyo listings.

## Data augmentation

For text encoders (SimCSE-style), data augmentation creates a second view of the
input to form a positive pair without any behavioral signal:

- **Dropout noise**: pass the same sentence through the encoder twice with
  independent dropout masks; the two output vectors are the positive pair. This is
  SimCSE's unsupervised trick and costs nothing beyond a second forward pass.
- **Synonym replacement or cropping**: replace words or truncate sentences to
  create surface variants that should map to the same representation.
- **Back-translation**: translate to another language and back to produce a
  semantically equivalent paraphrase.

For behavioral systems, augmentation usually means enriching features (session
context, device, time of day) rather than perturbing the entity itself.

## Feature engineering

**When to use which feature treatment.**

| Reach for | When | Instead of |
|---|---|---|
| Learned ID embedding | the ID recurs enough to train a stable vector (active users, popular items) | hashing, which is reserved for unbounded or sparse ID spaces |
| Content features (text, category, image) | cold-start entities need a vector before any interaction history exists | id-only embedding, which has nothing for an unseen entity |
| Behavioral aggregates (recent categories, session length) | users have rich interaction history | id-only, which underfits users with long histories |
| Graph neighbor features | the entity sits in a graph and neighbor features carry extra signal | flat content features alone, when graph structure is available |

The cold-start point is structural: if the encoder consumes content features, a
brand-new entity maps to a sensible point in the space from its attributes alone.
That is exactly the inductive property discussed in section 2.
