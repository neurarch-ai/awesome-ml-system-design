# 3. Data preparation

Two pipelines, and the second one is the one that surprises people: the interaction
sequences you already build, and the item representation that did not exist before.

## The content embedding comes first

Semantic IDs are only as good as the embedding they quantize. Whatever produces it
(a text encoder over title and description, an image encoder, or a multimodal model)
is now part of your recommender, with the consequences that implies:

- **Coverage is a data-quality metric.** Items with thin metadata get weak embeddings
  and therefore meaningless codes. Measure the share of the catalogue with usable
  content before you promise cold-start gains.
- **The encoder's domain matters.** A general-purpose text encoder puts "cooking
  video" near "recipe blog", which is usually what you want, and it also puts every
  sequel near its original, which sometimes is not.
- **It has to be reproducible.** The exact encoder version is part of the item's
  identity from now on.

## The quantizer, and the refresh that couples everything

A residual quantizer encodes the embedding coarse to fine: quantize, subtract, repeat.
Each level's codebook is trained on the residual left by the previous one.

```mermaid
flowchart LR
  E["content embedding e"] --> L1["level 1 codebook<br/>nearest entry -> c1"]
  L1 --> R1["residual e - q1"]
  R1 --> L2["level 2 codebook -> c2"]
  L2 --> R2["residual"]
  R2 --> L3["level 3 codebook -> c3"]
  L3 --> DIS["collision? add a<br/>disambiguating position c4"]
  DIS --> SID["semantic ID"]
```

Three operational facts about this pipeline, in the order teams hit them:

1. **Collisions are normal.** Distinct items quantize to the same tuple. The standard
   fix is an extra position that simply counts within the collision group, which grows
   the sequence and therefore the decode cost. Track the collision rate as a metric.
2. **Retraining the quantizer changes every item's ID.** The sequence model was
   trained on the old codes and is now reading a different language. The two refresh
   cycles are coupled whether or not anyone planned for it. Version the codebook,
   retrain both together, and keep the old mapping for a deprecation window.
3. **New items are cheap, until the codebook is stale.** A new item is encoded and
   quantized against the existing codebooks at ingestion time, which takes
   milliseconds and is the actual cold-start mechanism. But as the catalogue's content
   distribution shifts, new items quantize into increasingly wrong regions and nothing
   raises an alarm. Monitor **quantization error** and **code usage entropy** on new
   items, not on the training snapshot.

## Building the sequences

Everything from [Sequential Recommendation](../sequential-recommendation/03-data-preparation.md)
applies unchanged: one ordered sequence per user, deduplication and filtering decided
once, a recent-N cap, and identical build logic offline and online. Two additions
specific to this design:

- **The sequence is now a sequence of tuples**, so a history of 50 items at $m = 4$
  is 200 positions. Sequence length, and therefore training cost, multiplies by $m$.
- **Targets are code tuples**, so a leaked future item leaks its codes too. The
  temporal boundary audit matters exactly as much as before, and looks different
  enough that it is worth re-doing rather than assuming the old check covers it.

## What to store

| Artifact | Refresh | Why it exists |
|---|---|---|
| Content embeddings | On item change | The input to quantization |
| Codebooks (versioned) | Weeks to months | Retraining changes every ID |
| Item to semantic ID map | On ingestion | The serving lookup after decoding |
| Valid-prefix structure | With the map | Constrained decoding needs it |
| Collision groups | With the map | Disambiguation at decode time |
| Sequences of tuples | Daily or streaming | Training and serving input |

The row people forget is the fourth. "No index to maintain" is the usual pitch for
generative retrieval, and this table is the honest answer: the ANN index is replaced by
a smaller set of structures, not by nothing.
