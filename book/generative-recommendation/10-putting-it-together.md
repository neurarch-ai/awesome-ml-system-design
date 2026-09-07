# 10. Putting it together: the complete build

## The default stack: start here, deviate with reason

For a team with a real cascade and real cold-start pain, this is the answer, and the
reason each piece is in it.

| Layer | Default choice | Why | When to deviate |
|---|---|---|---|
| Item representation | Semantic IDs **alongside** atomic IDs | Head items still want memorization | Text only where language is the product |
| Content encoder | An off-the-shelf text or multimodal encoder, pinned | It is the component everything else depends on | Train your own only after the pinned one proves the idea |
| First project | Semantic IDs as ranking features | Feature-rollout risk, and it tests the encoder | Skip only if the encoder is already proven |
| Retrieval | Keep ANN, add generative for a traffic slice | Bounded blast radius and cost | Full replacement only after a slice wins online |
| Codes | 3 to 4 levels, K = 256, plus disambiguation | Covers a large catalogue at a small output width | Fewer levels when decode latency binds |
| Decoding | Constrained beam search over a valid-prefix trie | The model cannot invent items | Never unconstrained |
| After the decode | Dedup, over-generate, filter, enforce diversity | Beam search is mode-seeking and filters are hungry | Nothing here is optional |
| Fallback | The ANN path, one flag | Failures here are product regressions, not errors | Nothing |
| Evaluation | Full-catalogue, per slice, plus validity and coverage | The standard metrics flatter this design | Nothing |
| Refresh | Codebooks and sequence model deployed together | Retraining one invalidates the other | Nothing |

## The complete build

The scenario from section 1: 50M items, 100k new per day, p99 under 30 ms, feeding the
existing ranker, keeping every filter.

**Phase 1, weeks 1 to 4: the representation.** Pin a content encoder, embed the
catalogue, train a residual quantizer at 3 levels of 256 with a disambiguating fourth
position. Publish the item-to-code map and the valid-prefix structure. Measure
coverage (what share of the catalogue has usable content), collision rate, and
quantization error on newly ingested items. **Nothing is served yet.**

**Phase 2, weeks 5 to 8: the cheap win.** Add the codes as categorical features to the
existing ranker. Report per slice. If cold and tail do not move here, stop: the content
encoder is the problem, and generative retrieval will not fix it. This phase costs a
feature rollout and answers the question the rest of the project depends on.

**Phase 3, weeks 9 to 16: generative retrieval in shadow.** Train the sequence model
over code tuples, decode in parallel with the live ANN path, and compare. The numbers
that decide are candidate-set overlap, validity rate, how often filters empty the
generated set, and the real p99 under production load.

**Phase 4: one surface, a traffic slice, online.** With the ANN path as a one-flag
fallback and slice-level guardrails. Expect the aggregate to move little and the tail
to move a lot; if the aggregate moves a lot, check for a leak before celebrating.

**The arithmetic that decides the rollout.** At 4 code positions and a beam of 128,
retrieval is 4 sequential model steps on a batch of 128 per request, against one ANN
probe. That is the budget line, and it is why phase 4 is a slice rather than the whole
surface. The cost of the project is not the model: it is running two retrieval systems
in parallel for a quarter.

## The same techniques under different constraints

| | Large catalogue, cold-start pain (default) | Small stable catalogue, dense data | Velocity-bound, many new surfaces |
|---|---|---|---|
| Objective | Tail and cold-item recall | Do no harm | Weeks to onboard a surface |
| Representation | Semantic IDs plus atomic | Atomic IDs | Verbalized text |
| First project | Codes as ranking features | Probably none of this | An LLM ranker on the new surface only |
| Retrieval | ANN, plus generative on a slice | ANN | Whatever the surface already has |
| The metric that decides | Tail recall, online | Guardrails only | Engineering weeks, before and after |
| Main risk | Two systems for a quarter | Rebuilding something that works | Token cost and latency per request |

## What each constraint decides

**Cold-start-bound.** The constraint points at the representation, not the model. Every
phase above exists to get a content prior into the system with the least risk, and the
sequence model is the last thing you change.

**Stable and dense.** The honest answer is to not do this. An atomic-ID two-tower model
with a tuned ranker is hard to beat when interactions are plentiful, and saying so is a
stronger interview answer than proposing the fashionable architecture.

**Velocity-bound.** The constraint points at text, because verbalization is what makes
a surface with no features and no history work on day one. Scope it to that surface,
measure onboarding weeks, and keep the cheap path everywhere else.

## The smallest runnable version

Standard library only, deterministic. It quantizes a synthetic catalogue, measures what
the codes actually carry (including for items the quantizer never saw), and decodes with
and without a validity constraint.

```python
"""Semantic IDs end to end: quantize a catalogue, measure what the codes buy, and
decode with and without a validity constraint.

Standard library only, one fixed seed, so two runs give the same numbers. It answers
the three questions the chapter keeps returning to: do content codes actually carry a
neighbourhood, does a brand-new item inherit one, and what does constrained decoding
buy you.
"""

import random
from collections import Counter, defaultdict

RNG = random.Random(0)
DIM, GENRES, CATALOGUE, HELD_OUT = 8, 24, 2000, 200
LEVELS, CODEBOOK = 2, 16


def make_catalogue():
    """Items are content embeddings drawn around a latent genre centroid."""
    centroids = [[RNG.gauss(0, 1) for _ in range(DIM)] for _ in range(GENRES)]
    items = []
    for i in range(CATALOGUE):
        g = i % GENRES
        vec = [centroids[g][d] + RNG.gauss(0, 0.35) for d in range(DIM)]
        items.append({"id": i, "genre": g, "vec": vec})
    return items


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(vectors, k, iters=12):
    """Deterministic k-means: initialise from the first k distinct vectors."""
    centers = [list(v) for v in vectors[:k]]
    for _ in range(iters):
        groups = defaultdict(list)
        for v in vectors:
            groups[min(range(k), key=lambda c: dist2(v, centers[c]))].append(v)
        for c in range(k):
            if groups[c]:
                centers[c] = [sum(v[d] for v in groups[c]) / len(groups[c]) for d in range(DIM)]
    return centers


def train_quantizer(vectors):
    """Residual quantization: quantize, subtract, repeat."""
    codebooks, residuals = [], [list(v) for v in vectors]
    for _ in range(LEVELS):
        centers = kmeans(residuals, CODEBOOK)
        codebooks.append(centers)
        residuals = [
            [r[d] - centers[min(range(CODEBOOK), key=lambda c: dist2(r, centers[c]))][d]
             for d in range(DIM)]
            for r in residuals
        ]
    return codebooks


def encode(vec, codebooks):
    codes, residual = [], list(vec)
    for centers in codebooks:
        c = min(range(len(centers)), key=lambda i: dist2(residual, centers[i]))
        codes.append(c)
        residual = [residual[d] - centers[c][d] for d in range(DIM)]
    return tuple(codes)


def purity(items, codes):
    """Share of items whose code group is dominated by their own genre."""
    groups = defaultdict(list)
    for it, code in zip(items, codes):
        groups[code].append(it["genre"])
    hits = sum(Counter(g).most_common(1)[0][1] for g in groups.values())
    return hits / len(items)


def main() -> None:
    items = make_catalogue()
    known, cold = items[:-HELD_OUT], items[-HELD_OUT:]

    codebooks = train_quantizer([it["vec"] for it in known])
    known_codes = [encode(it["vec"], codebooks) for it in known]
    cold_codes = [encode(it["vec"], codebooks) for it in cold]

    print("catalogue")
    print(f"  items                    {len(items)} ({len(known)} known, {len(cold)} never quantized before)")
    print(f"  semantic ID scheme       {LEVELS} levels x {CODEBOOK} codes "
          f"= {CODEBOOK ** LEVELS} representable tuples")
    print(f"  distinct tuples used     {len(set(known_codes))}")
    collisions = len(known) - len(set(known_codes))
    print(f"  collisions               {collisions} items share a tuple with another "
          f"({100 * collisions / len(known):.1f}%), so a disambiguating position is needed")

    # What the codes buy: a group of items that share a tuple is mostly one genre.
    print("\nwhat the codes carry (a random grouping scores about 1/24 = 0.04)")
    coarse_known = [c[:1] for c in known_codes]
    coarse_cold = [c[:1] for c in cold_codes]
    print(f"  known items, coarse code only  {purity(known, coarse_known):.2f}"
          f"   ({CODEBOOK} groups for {GENRES} genres, so this one is hard)")
    print(f"  cold items, coarse code only   {purity(cold, coarse_cold):.2f}"
          "   <- never seen by the quantizer")
    print(f"  known items, full tuple        {purity(known, known_codes):.2f}"
          f"   ({len(set(known_codes))} groups, finer than the genres, so expect a high number)")
    print(f"  cold items, full tuple         {purity(cold, cold_codes):.2f}")

    # Decoding, with and without a validity constraint.
    valid = set(known_codes)
    prefixes = {c[:1] for c in valid}

    def beams_from(history, constrained, width=8):
        """Score each code position from the user's mean history vector."""
        mean = [sum(v[d] for v in history) / len(history) for d in range(DIM)]
        level0 = sorted(range(CODEBOOK), key=lambda c: dist2(mean, codebooks[0][c]))
        out = []
        for c0 in level0:
            if constrained and (c0,) not in prefixes:
                continue
            residual = [mean[d] - codebooks[0][c0][d] for d in range(DIM)]
            level1 = sorted(range(CODEBOOK), key=lambda c: dist2(residual, codebooks[1][c]))
            for c1 in level1:
                if constrained and (c0, c1) not in valid:
                    continue
                out.append((c0, c1))
                if len(out) == width:
                    return out
        return out

    users = [[known[RNG.randrange(len(known))]["vec"] for _ in range(5)] for _ in range(200)]
    for label, constrained in (("unconstrained decode", False), ("constrained decode", True)):
        beams = [b for u in users for b in beams_from(u, constrained)]
        real = sum(1 for b in beams if b in valid)
        print(f"\n{label}")
        print(f"  candidates returned      {len(beams)}")
        print(f"  map to a real item       {real} ({100 * real / len(beams):.1f}%)")

    m, b, = LEVELS, 8
    print(f"\ncost per request: {m} sequential decoder steps x {b} hypotheses, "
          f"against 1 ANN probe.")
    print("Generative retrieval buys generalization, not latency.")


if __name__ == "__main__":
    main()
```

Three things to read off the output. The **collision rate** is high at two levels of 16
codes, which is why production schemes use more levels and a disambiguating position.
The **coarse-code purity** is the cold-start mechanism measured: items the quantizer
never saw land in groups dominated by their own genre at the same rate as items it
trained on, far above the chance rate of one in twenty-four. And the **constrained
decode** returns only tuples that map to real items, while the unconstrained one does
not, which is the difference between a demo and a service.
