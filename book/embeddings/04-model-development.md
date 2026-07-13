# 4. Model development

The goal of training is a space where entities that co-occur in behavior land near
each other and unrelated entities land far apart. When the encoder is trained well,
this structure emerges geometrically: items from the same category form clusters,
and the distance between clusters reflects behavioral dissimilarity.

![Learned embedding clusters in 2D projection](assets/fig-embedding-clusters.png)

*PCA projection of a learned item embedding space. Items from the same behavioral
category pull into a cluster; the cluster centroids (stars) are roughly separated
by category. Schematic illustration; a real space has hundreds of dimensions and
finer-grained clusters by brand, style, and price.*

## Contrastive losses

The two workhorses are InfoNCE and triplet/margin loss. Both pull related entities
together and push unrelated ones apart, but they differ in how they define the
contrast and how sensitive they are to negative quality.

### InfoNCE (softmax contrastive loss)

For an anchor $x_i$ and its positive $x_i^{+}$, treat the positive as the correct
class among a set of $B$ candidates (the positive plus $B - 1$ negatives) and
minimize the cross-entropy:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp\!\bigl(\text{sim}(z_i,\, z_i^{+}) / \tau\bigr)}{\exp\!\bigl(\text{sim}(z_i,\, z_i^{+}) / \tau\bigr) + \sum_{j \neq i} \exp\!\bigl(\text{sim}(z_i,\, z_j) / \tau\bigr)}$$

where $z_i = f(x_i)$ is the embedding, $\text{sim}$ is dot product or cosine, and
$\tau$ is a temperature hyperparameter. Small $\tau$ makes the softmax peaky,
concentrating the loss on the hardest negatives; large $\tau$ spreads the gradient
over all negatives. This is the workhorse loss for dual-encoder retrieval, and it
is exactly the loss two-tower models train under.

### Triplet / margin loss

For an anchor $a$, a positive $p$, and a negative $n$, require the anchor to be
closer to the positive than to the negative by a margin $m$:

$$\mathcal{L}_{\text{triplet}} = \max\!\bigl(0,\ d(a, p) - d(a, n) + m\bigr)$$

where $d(\cdot, \cdot)$ is a distance (e.g., Euclidean or $1 - \text{cosine}$).
The loss is zero when the margin is already satisfied, so only triplets near the
boundary contribute gradients. This makes triplet loss very sensitive to how
triplets are mined; a poor mining strategy means most gradients are zero and
training stalls.

### The logQ popularity correction

In-batch negatives are sampled from the data distribution, so popular items appear
as negatives far more often and the model learns to push them down unfairly. The
fix is to subtract the log sampling probability from each item's logit before the
softmax:

$$s'(x, y) = s(x, y) - \log Q(y)$$

where $Q(y)$ is the frequency-based sampling probability of item $y$. This turns
the in-batch softmax into an unbiased estimate of the full-corpus softmax. Mention
this unprompted; it is the single most cited fix in production retrieval writeups
(YouTube, Expedia), and applying it at serving rather than training is one of the
most common wrong answers in interviews.

## Negative sampling in detail: in-batch vs hard negatives

![In-batch vs hard negatives training loss curves](assets/fig-inbatch-vs-hard.png)

*Training loss under three negative strategies. In-batch negatives alone plateau
early because the negatives are too easy once the space has basic structure. A
mix of in-batch plus a small hard fraction reaches the lowest floor with stable
training. A majority of hard negatives drives the loss lower but introduces
instability from false negatives. Schematic; relative shapes are the point.*

The figure captures the practical recipe: **mostly in-batch negatives, with a
small, carefully tuned fraction of mined hard negatives added once the easy loss
has saturated.** The hard fraction is a dial, not a binary switch, and should be
increased slowly.

## Encoder architectures

**Dual-encoder (two-tower).** Two separate encoders map the two sides (user and
item, query and document) into the same space; the score is their dot product or
cosine. Because the score factorizes into one vector per side, you precompute one
side offline and serve the other against an ANN index. The towers share an output
space but not weights, because the two sides have different features. This is the
canonical architecture for query-item and user-item retrieval.

> **Open the validated graph.** Trace a two-tower model at real dimensions in the
> live [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). Follow
> each tower down to the similarity layer and note that the features never mix
> before it, which is exactly what lets you precompute one side.

**GraphSAGE-style (inductive graph encoder).** Instead of learning a fixed vector
per node, learn aggregation functions that build a node's embedding from its own
features plus a sample of its neighbors' features, stacked over $K$ hops. Because
the embedding is computed from features, a brand-new node gets a vector with zero
history. This is the cold-start-safe approach.

> **Open the validated graph.** Trace a GraphSAGE-style recommender in the live
> [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). Notice how
> the neighbor aggregation and the self-feature are concatenated at each hop, and
> how the same aggregation function runs on every new node at serving time.

**LightGCN (transductive graph encoder).** Simplified graph convolution for
collaborative filtering: drop feature transforms and nonlinearities, keep only
linear neighborhood aggregation over the user-item interaction graph, and take a
weighted sum across all propagation layers. Fast and a strong collaborative
filtering baseline, but strictly transductive: a new user or item has no vector
until the next retrain.

## Dimensionality

The dimension $d$ is a three-way tradeoff. Name all three rather than picking a
number:

- **Recall / quality.** More dimensions give the space more capacity to separate
  entities; gains are real but diminish, and very high dimensions can overfit and
  hurt ANN recall.
- **Memory.** Index memory scales roughly as (number of entities) times $d$ times
  bytes per component. At tens of millions of entities, doubling the dimension
  doubles a large bill.
- **Search latency.** Distance computations scale with $d$, so wider vectors make
  every ANN probe slower.

![Recall@100 vs embedding dimension](assets/fig-recall-vs-dim.png)

*Recall improves with dimension but saturates; the gains from 128 to 1024 are
real but modest compared to better negatives (blue vs gray). Memory and latency
double with each step. Pick a modest $d$, tune it against the downstream consumer,
and reach for quantization before chasing dimension if memory is the constraint.
Schematic; qualitative trend is the point.*

## When to use which

**Losses and negative strategies.**

| Reach for | When | Instead of |
|---|---|---|
| InfoNCE (softmax contrastive) | default; many candidates per anchor, same loss as two-tower retrieval | triplet, unless you can mine high-quality triplets and want an explicit margin |
| Triplet / margin loss | you have a triplet mining pipeline and want an explicit similarity margin in the geometry | InfoNCE with in-batch negatives, which is simpler when you do not have a miner |
| In-batch negatives | negatives for free; scale with batch size | small batches that starve the loss or a setting where false negatives are prevalent |
| Hard negatives | recall has plateaued and the boundary is fuzzy on look-alike items | more model capacity; the fix is almost always better negatives, not a bigger encoder |
| logQ / sampled-softmax correction | in-batch negatives and the catalog is popularity-skewed | skipping it and accepting head-item bias in the learned space |

**Encoder families.**

| Reach for | When | Instead of |
|---|---|---|
| Dual-encoder (two-tower) | query-vs-item or user-vs-item relatedness; one side precomputable | cross-encoder at retrieval scale, where precompute is impossible |
| GraphSAGE-style (inductive) | rich node features and new entities arrive constantly | id-bound transductive methods when cold start matters |
| LightGCN (transductive) | a fixed user-item interaction graph and you want a cheap, strong collaborative baseline | inductive graph encoder when the graph is sparse or entities arrive frequently |
| SimCSE / text contrastive | the entity is a sentence or passage; semantic text similarity from dropout or NLI pairs | behavioral dual-encoder, when all you have is text with no engagement signal |
