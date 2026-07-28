# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable CTR model, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing architectures before pricing a single auction.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Architectures change
yearly, but the interface of each stage (define labels, assemble features, score,
calibrate, price, evaluate, serve) does not, so pick per stage by interface and
treat any specific model family as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Label definition | Click label finalized in seconds; conversions held unresolved inside a bounded attribution window; time-based train/test split | Continuous training on fast-moving campaigns: fake-negative weighted loss or the two-model delay approach | [3](03-data-preparation.md) |
| Features | Feature hashing for unbounded id spaces (user, ad); row-per-id embeddings for small stable fields (placement, device); log the serving-time feature vector | Every id space is small and bounded: row-per-id everywhere, hashing is needless collision | [3](03-data-preparation.md) |
| Model family | Logistic regression baseline, then Wide and Deep once interactions justify it | Billions of ids where pairwise dot products carry the signal: DLRM; explicit high-degree crosses matter: DCN V2 | [4](04-model-development.md) |
| Calibration | Platt scaling layer refit hourly on fresh held-out data, decoupled from the daily retrain | Reliability curve is regionally warped and the hold-back slice is large: isotonic; small unbiased hold-back set: transfer-learning fine-tune | [4](04-model-development.md) |
| Auction integration | eCPM = bid x pCTR, second-price; apply the downsampling correction before any score reaches the auction | Never. The correction is arithmetic, not a preference | [2](02-frame-as-ml-task.md), [5](05-evaluation.md) |
| Bias control | Small exploration slice, logged propensities with IPW, position as a train-time feature neutralized at serving | Traffic is too thin to spare an exploration slice: lean harder on IPW and accept slower tail learning | [3](03-data-preparation.md) |
| Evaluation | AUC and NE for ranking quality, sliced ECE for pricing quality, offline; A/B on RPM and advertiser ROI online | Never AUC alone. Build the calibration monitoring first | [5](05-evaluation.md) |
| Serving | Fetch shared features once per request, batch the forward pass, cache hot embedding rows | Tables exceed one host's memory: shard model-parallel and co-locate co-accessed tables | [6](06-serving-and-scaling.md) |

The evaluation row is the one beginners skip and regret: a launch gate of AUC
alone passes exactly the failure mode this system is most exposed to, a monotone
scale shift that reorders nothing and mis-prices everything. Wire the sliced ECE
dashboard before the first model ships, not after the first revenue incident.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a pCTR
system feeding a real-time second-price auction, tens to a few hundred candidate
ads per request, low tens of milliseconds to score them, a sparse feature space
of billions of categorical ids, and continuous retraining to track campaign
churn. Here is the whole system with every choice committed and the reason it
won.

| Decision | Choice | Why it won |
|---|---|---|
| Label pipeline | Click labels finalized in seconds; pCVR labels held unresolved inside the attribution window with a fake-negative weighted loss; time-based split | Treating a pending conversion as 0 biases pCVR down and under-bids real value; a random split leaks future labels |
| Feature treatment | Hashing for user, ad, advertiser, creative ids; row-per-id for the dozen placements and devices; serving-time feature vector logged | Open-ended id spaces cannot bound a row-per-id table; logged vectors kill train/serve skew at the source |
| Model | DLRM-style: sharded embedding tables, pairwise dot-product interactions, small top MLP | At billions of sparse ids the trees of GBDT+LR blow past memory and the wide branch of Wide and Deep needs hand-built crosses |
| Training cadence | Continuous embedding updates on streamed impressions; full dense retrain daily | New campaigns launch hourly; their embeddings are noise until fresh gradients arrive |
| Sampling correction | Negatives downsampled at rate w for class balance; closed-form odds correction p = q / (q + (1 - q) / w) applied before calibration | Downsampling shifts the prior odds by exactly 1/w; AUC hides it, every price shows it |
| Calibration | Platt scaling refit hourly on a fresh held-out slice fed by exploration traffic | Calibration drifts with campaign mix, far faster than ranking; two parameters refit in minutes while the DNN retrains daily |
| Bias control | Exploration slice with logged propensities, IPW at training, position feature neutralized at serving | The model only sees outcomes for ads its predecessor served; without correction it entrenches those biases |
| Evaluation | AUC and NE offline, ECE sliced by placement, device, and ad type; launch gated on A/B over RPM and advertiser ROI | A model calibrated on average can still mis-price the high-value slices the auction cares about most |
| Serving | Shared features fetched once per request, all candidates scored in one batched forward pass, hot-id cache in front of sharded tables | Embedding lookup, not the MLP, is the latency driver; the cold-lookup fraction bounds p99 |

**Traffic and label volume.** Illustrative: 20,000 ad requests per second at
peak, 150 candidates scored per request, is 3 million model scores per second.
One served impression per request is about 1.3 billion impressions per day; at a
0.5 percent CTR that is roughly 6.5 million clicks per day. Downsampling
negatives at w = 0.05 keeps all 6.5M positives plus about 65M negatives, near
70M training rows per day, an 18x smaller stream that the continuous trainer can
actually keep up with. The price of that convenience is the odds shift the
correction row exists to undo.

**Embedding memory.** [Section 4](04-model-development.md)'s own arithmetic: 500
million user ids at embedding dimension 64 is 32 billion floats, 128 GB at
float32 for one table, before ad, advertiser, and creative tables add tens more.
The hash table for each open-ended id space is sized to keep n/H near the 5
percent collision line from [section 3](03-data-preparation.md). Total sparse
parameters land in the low hundreds of GB against a top MLP under one million
parameters, five orders of magnitude apart, which is why the tables shard
model-parallel across hosts while the MLP replicates data-parallel.

**Latency.** Illustrative decomposition of a 20 ms scoring budget: shared user
and context features fetched once, about 5 ms; embedding lookups for 150
candidates times 10 sparse fields, 1,500 rows against the sharded tables, the
dominant and most variable cost, budgeted near 10 ms with a hot-id cache
carrying most reads; one batched forward pass through interactions and top MLP,
2 to 3 ms; the two-parameter calibration map and the eCPM sort, microseconds.
The p99 is set by the cache miss rate on cold ids, not by the arithmetic, which
is why the cache tier and table co-location are serving decisions, not
afterthoughts.

**Calibration and auction math.** At w = 0.05 the raw head's odds are inflated
20x: a raw score of 0.40 corrects to p = 0.40 / (0.40 + 0.60 / 0.05) = 0.032.
An advertiser bidding \$2.00 on that impression carries eCPM = 1000 x 2.00 x
0.032 = \$64. If the runner-up's eCPM is \$48, the second-price charge is 48 /
(1000 x 0.032) = \$1.50 per click. Skip the correction and the same auction
books eCPM = \$800, charges a nonsense price, and reports revenue that will
never arrive. The correction is one line of arithmetic standing between the
model and every dollar the platform invoices.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: sliced calibration rot (global ECE flat while
a high-value placement drifts as its campaign mix shifts; the hourly Platt refit
must be checked per slice, not just in aggregate), train/serve skew (offline NE
improves while online CTR for the same inputs does not; diff the logged
serving-time feature vector against the training-time recomputation the day it
happens, not after the revenue review), and new-campaign starvation (fresh ad
ids hash into noisy embeddings and never win auctions, so they never earn
gradients; a shrinking exploration-slice coverage of new ids is the early
warning, and advertiser complaints about under-delivery are the late one).

## The same techniques under different constraints

The review question that matters in practice is not "which interaction model is
best" but "which is best under my constraints." Here is the same system built
three times. Only the marketplace column is the build above; the other two keep
the identical stage interfaces and swap nearly every implementation choice.

| | Niche vertical ad network | Large marketplace (this chapter) | Real-time bidder on external exchanges |
|---|---|---|---|
| Scale | ~5M ids, ~200 requests/s | Billions of ids, ~20k requests/s | Millions of bid requests/s across exchanges, but you answer only what you can win |
| Latency budget | Comfortable double digits, self-imposed | Low tens of ms, page-load bound | ~10 ms hard, imposed by the exchange, network included |
| Model | Logistic regression with FTRL, or GBDT+LR; the id space still fits in trees | DLRM-style sharded embeddings, dot-product interactions | Compact model, aggressively small embedding dims; the deadline is not yours to negotiate |
| Calibration | The LR head is naturally calibrated; verify with a weekly reliability curve | Platt hourly, decoupled from daily retrain, sliced ECE | Even more load-bearing: every over-prediction is cash paid out on a losing bid, so calibrate per exchange and per inventory type |
| Labels and delay | Daily batch retrain; bounded attribution window is fine | Continuous training, fake-negative weighted loss | Win notices and clicks arrive on exchange timelines you do not control; label joins are their own pipeline |
| Bias control | Position feature; light exploration | Exploration slice + IPW + position feature | Severe: you observe outcomes only on won auctions, so IPW over the win probability is mandatory, not optional |
| Eval | AUC + reliability curve; A/B when traffic allows | AUC, NE, sliced ECE; A/B on RPM and advertiser ROI | Offline replay on logged bid landscapes; spend pacing and win-rate curves join the metric set |
| What would be over-engineering | Sharded embeddings, hourly recalibration, continuous training | A single global ECE with no slicing | Deep interaction stacks that miss the deadline; a model that arrives late loses every auction |

Two lessons fall out. First, the niche column is mostly deletions: at 5M ids the
trees of GBDT+LR hold the whole feature space, the linear head is naturally
calibrated for free, and continuous training is dead weight against a daily
batch. The chapter's machinery exists for a scale problem this column does not
have. Second, the bidder column shows the calibration thread tightening rather
than loosening: when the pCTR prices bids against other people's money on other
people's clocks, miscalibration stops being a silent revenue leak and becomes a
direct cash outflow, and the selection bias (you only see what you win) is the
feedback loop of [section 3](03-data-preparation.md) at its most vicious.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any architectures.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Id-space size | Feature treatment and model family | Fits in tree memory: GBDT+LR keeps a calibrated head for free. Hundreds of millions and growing: hash into fixed tables and go embedding-based |
| Latency budget | Embedding dimension, candidate count, cache tier | The lookup, not the MLP, is the cost; shrink dims and cache hot ids before touching the network |
| Base CTR / class imbalance | Downsampling rate w, then the correction | Whatever w you pick, the odds shift is exactly 1/w; apply the closed-form correction before Platt, never instead of it |
| Conversion delay | Label pipeline | Short and predictable: bounded window. Continuous training: fake-negative weighted loss. Long and variable: two-model delay approach |
| Campaign churn rate | Training and calibration cadence | Fast churn moves the base rate hourly; recalibrate on the fast clock, retrain on the slow one, never couple them |
| Per-slice volume | Platt vs isotonic | Millions of impressions per slice: isotonic can flatten a warped curve. Thin hourly slices: Platt, or isotonic overfits to noise |
| Who owns the inventory | Bias-control severity | Own placements: exploration slice + IPW. External exchanges: you see only won auctions, so propensity weighting is mandatory |
| Where the money moves | The launch gate | The auction reads the number, not the order; gate on sliced ECE and an online A/B over RPM, with AUC as a diagnostic only |

## The smallest runnable CTR model

The chapter's central claim is that a model can rank perfectly and still lose
money, and the fastest way to believe it is to watch it happen. The file below
is the whole loop with zero installs: hashed sparse features, logistic
regression by SGD on a seeded synthetic click stream, negatives downsampled for
class balance (the standard move, and the calibration wrecker), then the
closed-form correction from [section 4](04-model-development.md). Every
production component is swapped for the smallest thing with the same interface:
the hash trick stands in for TorchRec tables, the SGD loop for FTRL or the
continuous trainer, the correction for the Platt layer, and the final two lines
for the auction. The shape is the lesson; every section of this chapter
upgrades one function of this file.

```python
"""Hashed-feature CTR model, one file, no installs: train, calibrate, price."""
import math, random, zlib

random.seed(7)
H, DIM_FIELDS, LR, W_NEG = 2 ** 12, ("user", "ad", "place"), 0.15, 0.1

def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))

def bucket(field, value):
    """Stable hash of 'field=value' into H slots; production: TorchRec hashing."""
    return zlib.crc32(f"{field}={value}".encode()) % H

# --- synthetic ad platform: the true click process the model must recover ----
ads = [{"q": random.gauss(0, 1.0), "bid": random.uniform(0.5, 2.5)} for _ in range(200)]
users = [random.gauss(0, 0.6) for _ in range(1000)]
places = {"top": 0.4, "mid": 0.0, "side": -0.4}

def impression():
    u, a = random.randrange(1000), random.randrange(200)
    pl = random.choice(list(places))
    true_p = sigmoid(-3.5 + ads[a]["q"] + users[u] + places[pl])   # avg CTR ~ 4%
    return (u, a, pl), int(random.random() < true_p)

# --- training: SGD on hashed features, negatives downsampled at rate W_NEG ---
w = [0.0] * H
bias = 0.0
for _ in range(60000):
    x, y = impression()
    if y == 0 and random.random() > W_NEG:      # class rebalancing: keep 10% of negatives
        continue                                 # ...this is what wrecks calibration
    idx = [bucket(f, v) for f, v in zip(DIM_FIELDS, x)]
    p = sigmoid(bias + sum(w[i] for i in idx))
    g = p - y
    bias -= LR * g
    for i in idx:
        w[i] -= LR * g

def raw_score(x):
    return sigmoid(bias + sum(w[bucket(f, v)] for f, v in zip(DIM_FIELDS, x)))

def recalibrate(q, wneg=W_NEG):
    """Closed-form prior correction for downsampling; production: Platt/isotonic."""
    return q / (q + (1.0 - q) / wneg)

# --- evaluation on a fresh stream: AUC, calibration table, auction revenue ---
test = [impression() for _ in range(20000)]

def auc(scored):
    ranked = sorted(scored, key=lambda t: t[0])
    pos_ranks = [r for r, (_, y) in enumerate(ranked, 1) if y == 1]
    n_pos = len(pos_ranks)
    return (sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * (len(ranked) - n_pos))

def calibration_table(name, preds):
    rows = sorted(zip(preds, (y for _, y in test)))
    print(f"{name:>12}  bucket  mean_pred  empirical_ctr")
    k = len(rows) // 4
    for b in range(4):
        chunk = rows[b * k:(b + 1) * k]
        mp = sum(p for p, _ in chunk) / len(chunk)
        ec = sum(y for _, y in chunk) / len(chunk)
        print(f"{'':>12}  {b + 1:>6}  {mp:>9.4f}  {ec:>13.4f}")

raw = [raw_score(x) for x, _ in test]
cal = [recalibrate(q) for q in raw]
print(f"AUC raw {auc(list(zip(raw, (y for _, y in test)))):.3f}  "
      f"AUC calibrated {auc(list(zip(cal, (y for _, y in test)))):.3f}  (identical order)")
calibration_table("RAW", raw)
calibration_table("CALIBRATED", cal)

# --- why the auction cares: eCPM = bid x pCTR prices every impression -------
for name, preds in (("raw", raw), ("calibrated", cal)):
    booked = sum(ads[a]["bid"] * p for ((_, a, _), _), p in zip(test, preds))
    real = sum(ads[a]["bid"] * y for (_, a, _), y in test)
    print(f"{name:>10} model: booked ${booked / 20:.2f} vs realized ${real / 20:.2f} per 1000 impressions")
```

Run it and the output tells the chapter's story in four acts. AUC is 0.728
before and after recalibration, identical to the third decimal, because the
correction is monotone and AUC cannot see it. The raw calibration table shows a
model predicting mean CTRs of 0.17 to 0.63 against empirical rates of 0.016 to
0.12, roughly the 10x odds inflation that downsampling at 10 percent
guarantees, yet the empirical column still rises monotonically with the
predicted one: the model is genuinely discriminative and genuinely wrong about
every price. After the one-line correction the buckets read 0.021 against
0.016, 0.042 against 0.028, 0.071 against 0.053, 0.160 against 0.119, tracking
the empirical rates within the noise of a toy. The closing lines are the money:
the raw model books \$578.85 of expected revenue per 1000 impressions against
\$80.46 realized, a 7x overstatement, while the calibrated model books \$110.20
against the same \$80.46. The residual gap is real too: the closed-form formula
undoes the sampling shift exactly, but hashing collisions and model
misspecification leave a drift it cannot see, which is precisely why production
teams fit a learned Platt or isotonic layer on top of the correction rather
than trusting either alone. Swap the hash for TorchRec tables, the loop for a
continuous trainer, the correction plus a Platt fit for the hourly calibration
job, and the last two lines for a second-price auction, and you have rebuilt
this chapter.
