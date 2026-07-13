# 7. How teams do it in production

People You May Know and its cousins (accounts to follow, related pins) converge on
the same shape: build a graph, learn node embeddings with an inductive GNN,
precompute them offline, and use embedding nearest neighbors plus graph structure to
generate candidates that a pairwise model ranks. What differs is graph
heterogeneity, the embedding method, and the serving-freshness cadence. The clear
trend from 2022 to 2026 is a move from heuristics and shallow embeddings to
**large-scale inductive, heterogeneous GNNs**.

## Where the real designs diverge

| System | Graph | Embedding method | What is notable |
|---|---|---|---|
| LinkedIn LiGNN | heterogeneous, ~100B nodes (members, companies, posts, notifications) | inductive GNN framework in production | powers People You May Know, feed, jobs, notifications; temporal and multi-hop encoding at scale (KDD 2024) |
| Pinterest PinSage | pins and boards, ~3B nodes, ~18B edges | random-walk GraphSAGE (GCN) | the reference web-scale inductive GNN; on-the-fly convolutions, importance-pooled neighborhoods (KDD 2018) |
| Twitter TwHIN | heterogeneous information network (users, tweets, ads) | knowledge-graph embeddings (TransE-style) | one embedding set reused across follow-rec, ads, search, and safety; large candidate-generation recall gains (KDD 2022) |
| Snapchat GiGL | large-scale friend and content graph | open library for billion-scale GNN training and inference | tackles the systems problem: partitioning, sampling, and distributed inference (2025) |

The dividing line is graph heterogeneity and the inductive requirement: a
homogeneous friend graph with node2vec is a fine baseline, but production systems
model many node and edge types with an inductive GNN so a brand-new member can be
embedded from features on day one, and so one embedding set serves many downstream
tasks.

## The systems (first-party writeups)

- **LinkedIn** [LiGNN: Graph Neural Networks at LinkedIn](https://arxiv.org/abs/2402.11139): a production GNN framework over a roughly 100-billion-node heterogeneous graph, powering People You May Know and other surfaces (KDD 2024).
- **Pinterest** [PinSage: A new graph convolutional neural network for web-scale recommender systems](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48): random-walk GraphSAGE learning embeddings on a 3-billion-node graph, the reference for web-scale inductive GNNs.
- **Twitter** [TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation](https://arxiv.org/abs/2202.05387): heterogeneous knowledge-graph embeddings reused across follow recommendation, ads, and search (KDD 2022).
- **Snapchat** [GiGL: Large-Scale Graph Neural Networks at Snapchat](https://arxiv.org/abs/2502.15054): an open library for training and serving GNNs at billion scale, focused on the partitioning and distributed-inference systems problem (2025).

These are first-party engineering and research writeups; read them for the graph
construction, the sampling and partitioning systems work, and the eval bar that a
whiteboard answer skips.
