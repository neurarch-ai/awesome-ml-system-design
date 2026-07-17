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
**Why:** node2vec's embedding is a free lookup-table row per node ID, trained so
nodes that co-occur on random walks land nearby; the vector itself is the learned
parameter, so a node that did not exist at training time simply has no row.
GraphSAGE learns aggregation functions (how to pool neighbor features into an
embedding), and a function applies to any node you can describe by features and
neighbors, which is exactly what makes it inductive.

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
**Why:** a hub has millions of non-neighbors, so it keeps appearing in uniformly
drawn negative pairs, and every appearance pushes its embedding away from the rest
of the graph; the learned score becomes entangled with degree instead of affinity.
The degree correction subtracts the log of each node's sampling probability from the
score during training, so the model learns "more connected than chance given this
popularity" rather than raw popularity avoidance.

**Q: How do you keep it fresh when two members connect right now?**
A: Re-infer embeddings incrementally for the affected neighborhoods and upsert them
into the ANN index on a minutes-to-hours cadence, plus cheap graph-structure
candidates that reflect the new edge immediately. A nightly rebuild is too slow.
**Why:** in an L-layer model, a new edge only changes the inputs of nodes within L
hops of its endpoints, so incremental re-inference of that bounded region captures
the full effect at a tiny fraction of full-graph cost. The cadence matters because
the minutes after a connection forms are when friend-of-friend suggestions are most
actionable; a nightly rebuild surfaces them a day after the social context that
motivated them has passed.

**Q: A GNN embedding behind an ANN index and a two-tower retrieval model look
similar; when does the difference actually matter?**
A: Operationally they are identical: precompute vectors offline, index them in an
ANN store, serve nearest-neighbor lookups in milliseconds. The difference is what
information the vector encodes. A two-tower model embeds each entity from its own
features and its own interaction history; information moves between entities only
through co-occurrence in training pairs. A GNN embedding additionally propagates
features along multi-hop structure, so an entity's vector inherits signal from its
neighborhood even when its own history is thin. The difference matters when the
interaction data is sparse or the signal is inherently structural: a low-activity
member's two-tower embedding collapses toward a feature-only prior, while graph
propagation fills it in from neighbors, and transitive patterns like
friend-of-friend simply are not expressible as independent tower outputs. When
interactions are dense and structure adds little, the two-tower model wins on
training cost and simplicity, and the GNN's extra machinery buys nothing.

## Commonly answered wrong

**Q: Can you evaluate with a random train/test split of the edges?**
A: No. A random split leaks the future: an edge's own endpoints already look
connected in the training graph. Always split by time.
**Why:** with a random edge split, the training graph contains edges that formed
after the held-out edge, including triangles that only closed because the held-out
edge existed. The model then reads a neighborhood that is causally downstream of the
very edge it is asked to predict, so offline metrics inflate relative to deployment,
where the future graph genuinely does not exist yet.

**Q: Should you optimize invitations sent?**
A: No. Optimize invitations **accepted**. A suggestion that gets invited and
rejected annoys the recipient and hurts the network, so sent-count rewards spammy
over-suggesting.
**Why:** sent-count is controlled entirely by the suggester's willingness to click,
which UI placement can inflate arbitrarily without any real match existing.
Acceptance requires the recipient's independent judgment, so it is the only signal
in the loop that verifies the predicted edge was real rather than merely clickable.

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
**Why:** each edge type is evidence of a different relationship strength, and typed
message passing lets the model weight and transform each relation separately.
Collapsing them into one generic "connected" relation forces a single aggregation to
average semantically different signals, so a coworker edge and a
viewed-profile-once edge become indistinguishable, which destroys exactly the
contrast the ranking needs.
