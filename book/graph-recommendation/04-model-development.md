# 4. Model development

## The ladder: heuristics, then embeddings, then GNNs

Link prediction has a strong cheap baseline and a strong expensive ceiling. Climb
the ladder only as far as the bar requires.

**1. Graph heuristics.** Score a pair from shared structure alone, no training:

$$\text{CommonNeighbors}(u,v) = |N(u) \cap N(v)|, \qquad \text{Jaccard}(u,v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$

$$\text{AdamicAdar}(u,v) = \sum_{w \in N(u) \cap N(v)} \frac{1}{\log |N(w)|}$$

```python
import math
def common_neighbors(N, u, v):                     # N: node -> set of neighbor ids
    return len(N[u] & N[v])                         # count of shared neighbors
def jaccard(N, u, v):
    return len(N[u] & N[v]) / len(N[u] | N[v])      # shared / total distinct neighbors
def adamic_adar(N, u, v):
    # each shared neighbor w contributes 1/log(degree of w), so hubs count for less
    return sum(1.0 / math.log(len(N[w])) for w in (N[u] & N[v]))
# N = {"u": {"a","b","c"}, "v": {"b","c","d"}, "b": {"x","y"}, "c": {"x","y","z"}}
# shared neighbors of u, v are {b, c}
# common_neighbors(N, "u", "v") -> 2
# jaccard(N, "u", "v")          -> 2 shared / 4 distinct = 0.5
# adamic_adar(N, "u", "v")      -> 1/log(2) + 1/log(3) = 2.352934267515801
```

Adamic-Adar is the workhorse: it counts shared connections but down-weights hubs (a
shared celebrity connection means little; a shared niche colleague means a lot).
These are hard to beat cheaply and are always the baseline to state first.

**2. Shallow embeddings.** node2vec / DeepWalk run random walks (random hops from
node to neighbor that produce node sequences, treated like sentences of words) and
learn a vector per node with a skip-gram objective, so proximity in the walk implies closeness in
the embedding. Transductive: they only embed nodes seen at training time, so a
brand-new member has no vector.

**3. Inductive GNNs.** The production ceiling. A GNN like **GraphSAGE** (the basis
of Pinterest's PinSage) embeds a node by sampling a fixed fan-out of neighbors and
aggregating their features inward, layer by layer:

$$h_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \text{concat}\left(h_v^{(k-1)},\ \text{AGG}\left(\lbrace h_u^{(k-1)} : u \in \mathcal{N}(v) \rbrace\right)\right)\right)$$

Concretely, one such layer (with mean aggregation) is just: average the neighbor
vectors, glue that to the node's own vector, then apply a linear map and a
nonlinearity.

```python
import numpy as np
def sage_layer(h, neighbors, W):
    # h: dict node -> feature vector from the previous layer
    # neighbors: dict node -> list of neighbor ids;  W: weight matrix
    out = {}
    for v in h:
        agg = np.mean([h[u] for u in neighbors[v]], axis=0)  # average neighbor vectors
        combined = np.concatenate([h[v], agg])               # self features + neighbor summary
        out[v] = np.maximum(W @ combined, 0.0)               # linear map, then ReLU
    return out
# stacking k such layers lets a node's vector see k hops out into the graph
```

Because it aggregates **features**, it is **inductive**: it embeds a node it never
saw in training (a new member) from that node's features and neighbors. That is why
GraphSAGE-style models, not node2vec, run in production.

![Inductive GNN neighbor sampling](assets/fig-neighbor-sampling.png)

*The GNN embeds target node v by sampling a fixed fan-out of 1-hop and 2-hop
neighbors and aggregating their features inward. Sampling a fixed fan-out is what
keeps cost bounded on a billion-edge graph.*

## Compare and contrast: GraphSAGE vs GCN

Both are message-passing GNNs, and at whiteboard distance their layers look
interchangeable: aggregate neighbor vectors, combine with the node's own vector,
apply a learned transform and a nonlinearity. The confusion is natural because
GraphSAGE grew out of the GCN line; the split is in which neighbors each layer
touches and what that implies for training and serving at scale.

| Dimension | GCN | GraphSAGE |
|---|---|---|
| Core layer | message passing: aggregate, transform, nonlinearity (same) | message passing: aggregate, transform, nonlinearity (same) |
| Over-smoothing with depth | yes, collapses past 2-3 layers (same) | yes, collapses past 2-3 layers (same) |
| Neighborhood per layer | the full neighbor set, via the degree-normalized adjacency matrix | a fixed-size sampled fan-out (e.g. 10-25 neighbors per hop) |
| Cost on a hub node | proportional to the hub's full degree; one celebrity node pulls in every follower | capped by the fan-out, regardless of true degree |
| Training regime | classically full-batch over the whole graph's adjacency | minibatch over sampled subgraphs, which is what fits a billion-edge graph in memory |
| New node at serving time | the formulation is tied to the training graph's adjacency (transductive as originally stated) | explicitly inductive: any node with features and neighbors can be embedded |
| Aggregator | fixed degree-normalized mean | a learnable choice (mean, pooling, LSTM) concatenated with the self vector |

The difference changes the design the moment the graph is power-law and growing:
bounded sampling is what makes minibatch training tractable and per-node inference
cost predictable, so on a social graph with hubs and a constant stream of new
members, GraphSAGE-style sampling is a feasibility requirement, not a tuning choice.

## The link-prediction head and loss

Given node embeddings, score a pair with a dot product or a small MLP, and train
with a binary cross-entropy over positive edges and sampled negatives:

$$\hat{y}_{uv} = \sigma\!\left(z_u^{\top} z_v\right), \qquad \mathcal{L} = -\sum_{(u,v) \in E^{+}} \log \hat{y}_{uv} - \sum_{(u,v) \in E^{-}} \log\left(1 - \hat{y}_{uv}\right)$$

Two subtleties an interviewer probes:

- **Heterogeneous graphs.** Real networks have many node and edge types (member,
  company, school, group; connected, viewed, messaged). Production systems (LinkedIn
  LiGNN, Twitter TwHIN) model these as typed edges, which carries far more signal
  than a single homogeneous friend graph.
- **GNNs have a known blind spot for common-neighbor count.** Standard message
  passing (the step where each node collects and aggregates vectors from its
  neighbors) with set-pooling cannot directly count shared neighbors, exactly the
  signal Adamic-Adar captures. Recent work (subgraph methods like SEAL, and adding
  the heuristic as an explicit feature) fixes this. The practical answer: **feed the
  heuristics in as features**, do not expect the GNN to rediscover them.

![Link-prediction AUC by method](assets/fig-linkpred-auc.png)

*Heuristics are strong and cheap; inductive GNNs win when node features and graph
structure both matter, especially for cold-start members. Illustrative.*

## Why deeper is not better: over-smoothing

A natural instinct is to stack more GNN layers so each node sees further into the
graph. It backfires, and the reason is mechanical. One message-passing layer
replaces a node's vector with a blend of its own and its neighbors' vectors, which is
a smoothing (low-pass) operation on the graph. Stacking $k$ layers applies that
smoothing $k$ times, and repeated averaging over a connected neighborhood is a
diffusion that converges toward a state where every node in a connected component
holds essentially the same vector (up to a degree-dependent scale). Li et al. (2018)
made this precise by showing a GCN layer is a form of Laplacian smoothing. Past two
or three hops the representations wash out and become indistinguishable, so a
link-prediction head can no longer separate pairs. This **over-smoothing** is why
production GNNs stay shallow (two to three layers) and why the useful escape hatches
are residual or jumping-knowledge connections (which re-inject earlier, sharper
representations) rather than more depth.

![Over-smoothing: embeddings collapse as layers stack](assets/fig-over-smoothing.png)

*Simulated on a random 60-node graph with pure mean-aggregation propagation: the mean
pairwise cosine similarity between node embeddings climbs from about 0 toward 1.0 as
layers stack, meaning every node converges to the same vector. Learned transforms and
residual connections slow this collapse but do not remove it, which is why production
GNNs stay at two or three hops.* It is also why long-range graph signal is
usually better captured by a heuristic feature or a subgraph method than by simply
adding layers.

**When to use which model.**

| Reach for | When | Instead of |
|---|---|---|
| Adamic-Adar / common neighbors | a baseline, or a warm member with a rich neighborhood | a GNN, when a heuristic already clears the bar |
| node2vec / DeepWalk | a static graph and you want cheap embeddings | a GNN, when new nodes never need embedding |
| Inductive GNN (GraphSAGE / PinSage) | cold-start matters, features are rich, scale is large | transductive embeddings that cannot embed a new member |
| Heterogeneous GNN (LiGNN / TwHIN) | many node and edge types carry the signal | a homogeneous friend graph that throws away edge types |
| Heuristic as an explicit GNN feature | you need common-neighbor counts the GNN cannot learn | expecting message passing to rediscover Adamic-Adar |

**Provenance.** node2vec (Stanford, 2016) produces embeddings via biased random walks
on a static graph. The inductive GNN line runs from GraphSAGE (Stanford, 2017), which
learns aggregator functions so an unseen node can be embedded from its features and
neighborhood, through PinSage (Pinterest, 2018), which scaled GraphSAGE to a
web-scale recommendation graph. The heterogeneous, many-edge-type case is TwHIN
(Twitter, 2022); the underlying message-passing formulations are GCN (Kipf and
Welling, 2017) and GAT (2018).

> **Open the validated graph.** Trace a GraphSAGE-style recommendation graph at
> real dimensions in the live
> [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo): see where
> neighbor sampling, aggregation, and the pairwise scoring head attach.

## Implementation and training pitfalls

Graph link-prediction models fail in ways specific to the graph: the split leaks
edges through message passing, depth collapses embeddings, and lazy negative
sampling makes an offline AUC look nearly perfect while live recall stays poor.

| Problem | Symptom | Fix |
|---|---|---|
| Edge leakage in the split | AUC near perfect offline, live recall poor | remove test edges before message passing, split edges by time rather than randomly |
| Neighbor explosion | memory and latency blow up on high-degree nodes | fixed-size neighbor sampling (GraphSAGE), cap the fan-out per hop |
| Over-smoothing | deeper GNN makes node embeddings collapse to indistinguishable | keep depth to 2 or 3 layers, add residual or jumping-knowledge connections |
| Hard-negative collapse | training loss stalls, model cannot separate close pairs | mix random and mined hard negatives, curriculum from easy to hard |
| Degree and popularity bias | recommendations dominated by hub nodes | degree-normalized aggregation, down-weight hubs the way Adamic-Adar does |
| Transductive cold start | a new node has no embedding at serving time | use an inductive model (GraphSAGE) that embeds from features and neighborhood |
| Trivial-negative sampling | offline metric high, ranks obvious non-edges | sample negatives from plausible candidates, not the whole node set |
| Stale neighborhood at serving | a precomputed embedding misses recent edges | refresh embeddings on a cadence matched to graph churn, or recompute for active nodes |

The through-line: any link-prediction AUC near the ceiling almost always means test
edges leaked into message passing or the negatives were trivial, so distrust it until
the split and the sampler are both audited.

```mermaid
flowchart TD
  A["offline AUC suspiciously high"] --> B{test edges removed<br/>before message passing?}
  B -- no --> C["edge leakage:<br/>rebuild split, mask test edges"]
  B -- yes --> D{negatives sampled from<br/>plausible candidates?}
  D -- no --> E["trivial negatives:<br/>mine hard negatives"]
  D -- yes --> F{new nodes need<br/>embeddings at serve time?}
  F -- yes --> G["use inductive GNN,<br/>not node2vec"]
  F -- no --> H["metric likely trustworthy"]
```
