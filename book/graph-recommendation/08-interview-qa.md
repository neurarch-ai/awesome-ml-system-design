# 8. Interview Q&A

## Commonly asked

**Q: Why is this link prediction and not classifying a member?**
A: The label lives on a pair of members, and the signal is the graph structure
(shared connections, communities) plus node features. You score whether an edge
would form, so the model must consume the neighborhood, not a member in isolation.

**Q: Why not just use the Adamic-Adar heuristic?**
A: You often start there; it is a strong, cheap baseline that down-weights hub
connections. But it uses only structure, so it fails for cold-start members with an
empty neighborhood, and it cannot use profile features. A GNN adds features and
generalizes, which is why production climbs past heuristics.

**Q: How do you generate candidates without scoring all pairs?**
A: Two stage. Precompute a node embedding per member offline, index them in an ANN
store, and online fetch nearest neighbors plus cheap graph-structure candidates
(two-hop neighbors, shared connections), then rank the merged pool with a heavier
pairwise model.

**Q: How do you evaluate it?**
A: Hits@k and MRR (and AP) on a time-based split, where you train on the graph as
of T and predict edges that form after T. Then gate the launch on online
invitation-acceptance rate, not invitations sent.

## Tricky

**Q: Why an inductive GNN (GraphSAGE) instead of node2vec?**
A: node2vec is transductive, so it only embeds nodes present at training time; a
brand-new member has no vector. GraphSAGE aggregates neighbor features, so it is
inductive: it embeds a never-seen member from that member's features and neighbors.
Cold start is the whole reason production uses inductive GNNs.

**Q: Your GNN underperforms Adamic-Adar on shared-connection-heavy pairs. Why?**
A: Standard message passing with set-pooling provably cannot count common
neighbors, exactly the signal Adamic-Adar captures. The fix is to feed the
heuristic in as an explicit pairwise feature (or use a subgraph method like SEAL),
not to expect the GNN to rediscover it. Mechanism: a message-passing GNN embeds each
node independently from its own rooted neighborhood, and its expressive power is
bounded by the 1-dimensional Weisfeiler-Leman test. It never sees the two candidate
nodes jointly, so it has no way to represent "how many neighbors do u and v share,"
which is a property of the pair, not of either node alone. Adamic-Adar computes
exactly that pairwise intersection, weighted by inverse-log degree, which is why it
stays competitive on shared-connection-heavy pairs.

**Q: Hard negatives lifted offline Hits@k but online acceptance rate fell. What went
wrong?**
A: Hard negatives (two-hop, same-community non-edges) sharpen the decision boundary,
but "structurally close yet not connected" is not the same as "a good suggestion."
Two people in the same community who have deliberately not connected can be exactly
the pair a user does not want surfaced. Pushing the model to rank those pairs highly
improves offline ranking against sampled negatives while degrading the real product
objective. The fix is to keep hard negatives proportionate (mix them with random and
degree-corrected negatives rather than training on hard negatives alone) and to gate
on the online acceptance rate, since the offline metric and the product goal have
quietly diverged.

**Q: Uniform negative sampling hurt the model. What happened?**
A: The graph is power-law, so uniform non-edges over-sample hub nodes, and the model
learns to avoid popular members. Correct for degree (the graph analogue of the logQ
correction) and add hard negatives (two-hop, same community) so the boundary is
where reject-cost actually lives.

**Q: How do you keep it fresh when two members connect right now?**
A: Re-infer embeddings incrementally for the affected neighborhoods and upsert them
into the ANN index on a minutes-to-hours cadence, plus cheap graph-structure
candidates that reflect the new edge immediately. A nightly rebuild is too slow.

## Commonly answered wrong

**Q: Can you evaluate with a random train/test split of the edges?**
A: No. A random split leaks the future: an edge's own endpoints already look
connected in the training graph. Always split by time.

**Q: Should you optimize invitations sent?**
A: No. Optimize invitations **accepted**. A suggestion that gets invited and
rejected annoys the recipient and hurts the network, so sent-count rewards spammy
over-suggesting.

**Q: Do you run the GNN per request to score a member?**
A: No. You batch-infer embeddings for the whole graph offline and index them; online
is an embedding lookup plus an ANN query plus ranking. Running a multi-hop GNN per
request blows the latency budget. Mechanism: an L-hop GNN needs the target node's
entire L-hop neighborhood at inference time, and that neighborhood grows roughly like
the average degree raised to the power L. On a power-law social graph a single hop
through one hub can pull in millions of nodes, so a per-request forward pass would
have to fetch and aggregate an unbounded subgraph within the request budget.
Precomputing embeddings amortizes that fan-out into an offline batch job and leaves
online serving as a bounded lookup plus ANN query.

**Q: Is a homogeneous friend graph enough?**
A: Usually not at scale. Real signal is heterogeneous (members, companies, schools,
groups; connected, viewed, messaged). Production systems (LiGNN, TwHIN) model typed
nodes and edges, which carries far more signal than a single friend graph.
