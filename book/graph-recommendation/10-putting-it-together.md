# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable link predictor, one file, no installs.

## The default stack: start here, deviate with reason

Every stage of link prediction has several credible options, and a first-time
builder can burn a week comparing GNN libraries before scoring a single pair.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Frameworks change
yearly, but the interface of each stage (build the graph, sample negatives,
embed nodes, generate candidates, rank pairs, evaluate) does not, so pick per
stage by interface and treat any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Graph construction | Homogeneous accepted-connection graph plus node features (profile, activity) | Two or more meaningful edge types exist (viewed, messaged, same-group): type them | [3](03-data-preparation.md), [7](07-how-teams-do-it-in-production.md) |
| Train/test split | Time-based: train on the graph as of T, predict edges after T, test edges masked before message passing | Never. A random edge split leaks the future | [3](03-data-preparation.md), [4](04-model-development.md) |
| Negative sampling | Degree-corrected random non-edges plus a minority of mined hard negatives | First baseline only: uniform non-edges to get moving | [3](03-data-preparation.md) |
| Baseline | Adamic-Adar, stated and measured first | It clears the product bar: ship it and skip the GNN entirely | [4](04-model-development.md) |
| Embedding model | 2-layer inductive GraphSAGE, heuristics fed in as explicit pairwise features | Graph is static and no node is ever new: node2vec is cheaper | [4](04-model-development.md) |
| Depth and fan-out | 2 layers, fan-out ~25 then ~10 | You need longer-range signal: residual / jumping-knowledge or a heuristic feature, not more depth | [4](04-model-development.md) |
| Candidate generation | ANN over precomputed embeddings, unioned with 2-hop graph candidates | Graph is small enough to score 2-hop candidates exactly: skip the ANN | [6](06-serving-and-scaling.md), [2](02-frame-as-ml-task.md) |
| Ranking | Pairwise head (dot product, then a small MLP) over the merged few-hundred-candidate pool | The candidate pool is tiny: rank with the heuristic scores directly | [4](04-model-development.md) |
| Freshness | Incremental re-embedding of affected neighborhoods, minutes-to-hours upserts | The product is not socially immediate: nightly batch is simpler and fine | [6](06-serving-and-scaling.md) |
| Evaluation | Hits@k / MRR and AP on the time split offline; acceptance rate and coverage online, built before tuning anything | Never. Build the eval first | [5](05-evaluation.md) |

The second and last rows are the ones beginners skip and regret: with a leaked
split or trivial negatives, every offline number is fiction, and you cannot
tell whether any model change helped. Audit the split and the sampler before
trusting a single AUC point.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): People
You May Know on an undirected member graph of hundreds of millions of nodes and
tens of billions of edges, a ranked list of tens of suggestions per visit in
tens of milliseconds, optimized for accepted invitations, with cold-start
members served on day one and new edges reflected within minutes to hours. For
the arithmetic below, fix the scale at 300 million members and 30 billion
edges (Illustrative, but inside the stated range). Here is the whole system
with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Framing | Link prediction: score pairs, factored into per-node embeddings | The label lives on a pair; scoring all pairs is astronomically infeasible, so the model must factor |
| Graph | Heterogeneous: typed nodes (member, company, school) and typed edges (connected, viewed, messaged) | Typed edges carry different relationship strengths; collapsing them destroys the contrast ranking needs |
| Split | Time-based, test edges removed before message passing | A random edge split leaks the future through closed triangles and inflates every metric |
| Negatives | Degree-corrected random plus a proportionate mix of 2-hop hard negatives | The graph is power-law; uniform sampling teaches hub avoidance, and hard-only training diverges from acceptance |
| Model | 2-layer GraphSAGE-style inductive GNN, Adamic-Adar and common-neighbor counts as explicit features | Inductive embeds a day-one member from features; message passing provably cannot count shared neighbors, so feed them in |
| Candidate generation | ANN over precomputed embeddings, unioned with 2-hop graph candidates | Two-stage is forced by scale; the union covers what either source alone misses |
| Ranking | Pairwise link-prediction head over the merged few-hundred pool | Precision where it is affordable: hundreds of pairs, not hundreds of millions |
| Freshness | Incremental re-embedding of affected L-hop neighborhoods, minutes-to-hours ANN upserts | A new edge changes the whole neighborhood's inputs, and friend-of-friend suggestions decay fast |
| Evaluation | Hits@k / MRR and AP on the time split; launch gated on acceptance rate and coverage | The product is a short ranked list, and sent-count rewards spam; acceptance is the only verified signal |

**Graph scale and embedding storage.** 300M members and 30B undirected edges
give an average degree of 2 x 30B / 300M = 200. At 128 dimensions and float32
(Illustrative), the embedding table is 300M x 128 x 4 bytes, about 154 GB; int8
quantization brings it near 38 GB plus ANN graph overhead, which shards
comfortably. The number to notice is that the edges dominate the nodes by two
orders of magnitude: the graph itself, not the embedding table, is the storage
and partitioning problem, which is why the production frameworks in
[section 7](07-how-teams-do-it-in-production.md) are mostly systems papers.

**Neighborhood-sampling fan-out.** The unbounded 2-hop neighborhood at average
degree 200 is 1 + 200 + 200 x 200, about 40,000 nodes per target, and through
one celebrity hub it is millions. With GraphSAGE fan-outs of 25 and 10, every
target costs exactly 1 + 25 + 25 x 10 = 276 sampled nodes regardless of degree
([section 6](06-serving-and-scaling.md)). That is a ~150x reduction on the
average node and unbounded savings on hubs, and it is the single decision that
makes both minibatch training and predictable inference possible on this graph.

**Training and inference cost.** Embedding all 300M nodes is 300M bounded
forward passes of 276 sampled nodes each. At an illustrative 20,000 nodes per
second per GPU, that is 15,000 GPU-seconds, call it 4 GPU-hours per full
refresh, so a handful of GPUs turns the full pass into about an hour.
Freshness then does not come from running full passes faster: an L-layer model
means a new edge only changes inputs within L hops of its endpoints, and with
the fan-out cap that affected region is a few hundred nodes, so incremental
re-embedding plus ANN upsert meets the minutes-to-hours promise at a tiny
fraction of full-graph cost ([section 6](06-serving-and-scaling.md)).

**Serving latency.** The request budget is tens of milliseconds, and the GNN
never runs inside it. Illustrative component budget: ~1ms embedding lookup for
z_u, ~10ms ANN query on the sharded index, ~5ms fetching 2-hop graph
candidates, ~10ms for the pairwise ranker over a few hundred merged candidates,
totaling under ~30ms. The counterfactual justifies the whole offline path: a
per-request 2-hop GNN forward pass would fetch an unbounded subgraph (millions
of nodes through a hub) inside a 30ms budget, which is why "do you run the GNN
per request" is a [commonly-answered-wrong interview question](08-interview-qa.md).

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: embedding staleness (the age of the oldest
un-upserted neighborhood against the minutes-to-hours promise; users notice
"I just connected with her, why no friend-of-friend suggestions" before any
dashboard does), coverage collapse (suggestion coverage across the degree
distribution from [section 5](05-evaluation.md); a falling tail means the
system is re-suggesting hubs, who get flooded with invites while acceptance
sinks), and offline-online divergence (offline Hits@k climbing while
acceptance is flat almost always means edge leakage in the split or negatives
gone trivial, so re-audit both before touching the model).

## The same techniques under different constraints

The review question that matters in practice is not "which GNN is best" but
"which rung of the ladder is best under my constraints." Here is the same
system built three times. Only the middle column is the build above; the other
two keep the identical stage interfaces and swap nearly every implementation
choice.

| | Team-directory suggestions (startup) | People You May Know (this chapter) | Related-content marketplace |
|---|---|---|---|
| Graph / traffic | 200k members, ~2M edges (Illustrative) | 300M members, 30B edges | ~3B nodes, ~18B edges, bipartite items and collections (PinSage scale, [section 7](07-how-teams-do-it-in-production.md)) |
| Latency budget | Tens of ms, trivially met at this scale | Tens of ms, met only by precompute + ANN | Tens of ms, same precompute shape |
| Model | Adamic-Adar plus a logistic ranker over pairwise features; no GNN | 2-layer inductive GraphSAGE, heuristics as features | Random-walk GraphSAGE with importance-pooled neighborhoods |
| Candidate generation | Score all 2-hop candidates exactly; no embeddings, no ANN | ANN over embeddings, unioned with 2-hop candidates | ANN over item embeddings |
| Negative sampling | Uniform is nearly fine at this scale; degree-correct if metrics skew | Degree-corrected plus proportionate hard negatives | In-batch plus curriculum hard negatives |
| Freshness | Nightly rebuild of everything; the whole graph re-scores in minutes | Incremental re-embedding, minutes-to-hours upserts | Daily batch; catalog churn is slower than social churn |
| Eval | Hits@k on a time split, maintained by hand; acceptance online | Time-split Hits@k / MRR / AP; acceptance and coverage gates | Time-split Hits@k; save / engagement rate online |
| What would be over-engineering | Any GNN, any ANN index, incremental freshness | Per-request GNN inference, 4+ layers of depth | Minutes-level freshness, member-style cold-start machinery for items with content features |

Two lessons fall out. First, the startup column is mostly deletions: at 2M
edges the entire 2-hop candidate set can be scored exactly with heuristics,
which removes the embedding pipeline, the ANN index, and the freshness
machinery in one stroke, and [section 4](04-model-development.md)'s ladder says
to state that baseline first precisely because it often clears the bar. Second,
the marketplace column shows that scale alone does not dictate the design;
churn does. It is at PinSage scale, yet its freshness bar is a daily batch,
because an item catalog does not have the "we connected five minutes ago"
immediacy that forces incremental re-embedding on a social graph.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Graph size | Model rung and candidate generation | Up to a few million edges: heuristics and exact 2-hop scoring. Beyond: embeddings + ANN. Billions of edges: partitioned, sampled GNN |
| New-node stream | Inductive vs transductive | Any steady arrival of new members rules out node2vec; features carry the day-one signal, so GraphSAGE-style models win |
| Freshness bar | Embedding refresh strategy | Nightly acceptable: full batch pass. Minutes-to-hours: incremental re-embed of affected L-hop neighborhoods, upsert into ANN |
| Power-law skew | Fan-out cap and negative sampler | Always cap fan-out per hop; always degree-correct negatives at scale, or the model learns hub avoidance |
| Latency budget | Where the GNN runs | Tens of ms means never per request: precompute embeddings, serve lookup + ANN + rank |
| Reject cost | Negative mix and the gate metric | Mix hard negatives proportionately, never train on them alone; gate launches on acceptance, never on invitations sent |
| Edge-type diversity | Homogeneous vs heterogeneous graph | Two or more meaningful relation types: model them as typed edges (the LiGNN / TwHIN move) |
| Coverage floor | Ranking diversity and monitoring | Track suggestion coverage across the degree distribution; inject exploration before the filter bubble closes |
| Long-range signal | Depth vs features | Never chase it with depth (over-smoothing); use a heuristic feature, residual connections, or a subgraph method |

## The smallest runnable link predictor

The review of every GNN tutorial is the same: the reader assembles a graph
library, a sampler, and a trainer and still cannot see why graph structure
beats popularity. So here is the chapter's core mechanism in one file with zero
installs: personalized PageRank by a restarting random walk, the training-free
ancestor of the embedding pipeline (PinSage literally uses such walks for
neighborhood importance). Every production component is swapped for the
smallest thing with the same interface: the member graph becomes twenty nodes,
the GNN-plus-ANN candidate generator becomes the visit counter of a restarting
walk, and the popularity shelf becomes a degree sort. The shape is the lesson.

```python
"""Link prediction by random walks with restart, one file, no installs."""
import random
from collections import Counter

random.seed(7)

# --- a tiny member graph: two communities and one celebrity hub -------------
EDGES = []
acme = [f"acme{i}" for i in range(8)]        # a dense workplace community
univ = [f"univ{i}" for i in range(8)]        # a dense school community
for group in (acme, univ):                   # ring + chords: ~4 in-group edges each
    for i in range(8):
        EDGES.append((group[i], group[(i + 1) % 8]))
        EDGES.append((group[i], group[(i + 2) % 8]))
EDGES.append(("acme0", "univ0"))             # one weak tie bridges the communities
for m in acme[:6] + univ[:6]:
    EDGES.append(("hub", m))                 # the celebrity: highest degree by far
EDGES.append(("newbie", "hub"))              # a cold member: one edge, to the hub

N = {}                                       # adjacency: node -> set of neighbors
for u, v in EDGES:
    N.setdefault(u, set()).add(v)
    N.setdefault(v, set()).add(u)

# --- two candidate generators -----------------------------------------------
def by_popularity(seed, k=4):
    """Global degree ranking; production: the most-followed / trending shelf."""
    cands = [v for v in N if v != seed and v not in N[seed]]
    return sorted(cands, key=lambda v: (len(N[v]), v), reverse=True)[:k]

def by_walks(seed, k=4, steps=20000, restart=0.15):
    """Personalized PageRank by a restarting random walk;
    production: GNN embeddings + an ANN index playing the same role."""
    visits, cur = Counter(), seed
    for _ in range(steps):
        if random.random() < restart:
            cur = seed                       # teleport home: keeps the walk local
        else:
            cur = random.choice(sorted(N[cur]))
        if cur != seed and cur not in N[seed]:
            visits[cur] += 1                 # only score not-yet-connected members
    top = [v for v, _ in visits.most_common(k)]
    share = visits[top[0]] / sum(visits.values())  # mass on the top suggestion
    return top, share

def community_frac(seed, recs):
    group = acme if seed in acme else univ if seed in univ else None
    return f"{sum(r in group for r in recs) / len(recs):.0%}" if group else "n/a"

# --- demo: a warm in-community member, then a cold one ----------------------
for label, seed in [("acme3  (warm: 5 edges in a dense community)", "acme3"),
                    ("newbie (cold: 1 edge, to the hub)", "newbie")]:
    pop = by_popularity(seed)
    ppr, share = by_walks(seed)
    print(label)
    print(f"  popularity baseline: {pop}  in-community {community_frac(seed, pop)}")
    print(f"  personalized walks : {ppr}  in-community {community_frac(seed, ppr)}"
          f"  top-pick share {share:.0%}")
```

Run it and the two probes demonstrate the chapter's two central claims in
under sixty lines. For the warm member, the popularity baseline recommends the
bridge nodes and the other community's members (25% in-community), because
degree is blind to where the member actually lives in the graph; the
restarting walk surfaces acme6 and acme7, ordinary-degree colleagues the
popularity shelf never ranks, at 75% in-community. For the cold member with a
single edge to the hub, the walk has nothing local to exploit: its top picks
collapse onto the same nodes as the popularity baseline and its visit mass
goes diffuse (top-pick share 8% versus 17% for the warm member). That
degradation is precisely the cold-start argument of
[section 4](04-model-development.md): structure-only methods fail exactly
where the feature-driven, inductive GNN is needed, embedding a day-one member
from profile features instead of an empty neighborhood. Swap the visit counter
for GraphSAGE embeddings in an ANN index, the degree sort for a coverage-aware
ranker, and add a pairwise head over the merged pool, and you have rebuilt
this chapter.
