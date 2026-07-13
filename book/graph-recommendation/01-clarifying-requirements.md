# 1. Clarifying the requirements

Pin down the problem before designing. A typical exchange, where each question
removes work or changes the design.

**Candidate:** Are we recommending new connections (an undirected friend graph) or
accounts to follow (a directed graph)?
**Interviewer:** People You May Know: an undirected connection graph. A suggestion
is good if both sides would accept.

**Candidate:** How large is the graph?
**Interviewer:** Hundreds of millions of members, tens of billions of edges. Assume
it grows continuously.

**Candidate:** What is the success signal, an invitation sent or an invitation
accepted?
**Interviewer:** Accepted. A sent-but-rejected invite is a cost, not a win, so
optimize for accepted connections.

**Candidate:** How many suggestions do we return, and what is the latency budget?
**Interviewer:** A ranked list of tens of suggestions per visit, in tens of
milliseconds.

**Candidate:** Do brand-new members with almost no connections need suggestions
too?
**Interviewer:** Yes. Cold-start members are exactly where the feature matters
most, so you cannot rely only on their (empty) neighborhood.

**Candidate:** How fresh must it be? If two members connect now, should that change
suggestions within minutes?
**Interviewer:** New edges should influence suggestions quickly, within minutes to
hours, not a nightly rebuild.

Let us summarize. **We are asked to design People You May Know: link prediction on
a member graph of hundreds of millions of nodes and tens of billions of edges.**
The input is a member (plus their neighborhood and profile); the output is a ranked
list of not-yet-connected members likely to accept an invitation, returned in tens
of milliseconds, with new edges reflected within minutes to hours.

Two consequences drive the whole design, and stating them early is most of the
signal:

- **This is link prediction, not classification of a member in isolation.** The
  signal lives in the graph structure (shared connections, communities) and in
  node features, so the model has to consume the neighborhood, not just the member.
- **We cannot score all pairs.** Hundreds of millions squared is astronomical, so
  the system must be two stage: cheaply generate a few hundred candidate members
  per member (via graph structure and embedding nearest neighbors), then rank those
  with a heavier pairwise model. That constraint is why node embeddings plus an ANN
  index sit at the center of the design.
