# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable moderation loop, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing model families before enforcing a single
policy. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Models
change yearly, but the interface of each stage (taxonomy, hash gate, classify,
threshold, enforce, review, appeal, evaluate, retrain) does not, so pick per
stage by interface and treat any specific model as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Policy taxonomy | One policy per harm class, each with its own head, threshold, calibration, and owner | Never merge policies; share only the encoder backbone | [2](02-frame-as-ml-task.md) |
| Hash gate | Perceptual hash (PDQ class) at conservative Hamming radius, before any classifier | Text-only harm: cluster membership or feature lookup instead of image hashing | [4](04-model-development.md), [6](06-serving-and-scaling.md) |
| Model tiers | Cheap unimodal models at ingest (fine-tuned encoder for text, EfficientNet-B0 class for images); joint fusion gated behind ambiguity | Single-modality platform: drop the fusion tier entirely | [4](04-model-development.md) |
| Labels | Reviewer decisions as gold, user reports as triggers only, plus a random audit stream; three-grader consensus on borderline items | Never skip the audit stream; it is the only unbiased recall estimate | [3](03-data-preparation.md) |
| Thresholds | Per-policy precision floor, threshold from a PR-curve scan on a calibration holdout, Platt recalibration after every retrain | Never set 0.5 or max-F1; costs are asymmetric on every policy | [4](04-model-development.md), [5](05-evaluation.md) |
| Enforcement | Policy engine separate from the model; graduated actions (remove, age-gate, interstitial, downrank, nudge), CSAM and terrorism never auto-actioned on a classifier score | Cheap-FP policies (spam invites): auto-block at a high floor is fine | [1](01-clarifying-requirements.md), [2](02-frame-as-ml-task.md) |
| Human review loop | Priority-ranked queue, priority = severity x reach; every decision persists as a training label | Very high-confidence proxy labels exist (invite acceptance): queue shrinks to a periodic audit | [6](06-serving-and-scaling.md), [7](07-how-teams-do-it-in-production.md) |
| Appeals | Fast and cheap; overturn rate tracked per policy as the live false-positive signal | Never. An appeal is a free confirmed label | [5](05-evaluation.md) |
| Retraining | Per-policy cadence: near-weekly for spam and scams, monthly for stable classes; automated label-to-training pipeline | A class shows no adversarial drift: relax the cadence, keep the monitoring | [3](03-data-preparation.md), [6](06-serving-and-scaling.md) |
| Evaluation | Recall at the precision floor, time-based split, reported per policy, per modality, per language | Never use accuracy or a random split; both flatter the model | [5](05-evaluation.md) |

The last row is the one that fails interviews and launches alike: a random
split leaks future content into training and flatters recall by 5 to 15
percent, and a single global number hides the weakest language and the
weakest policy, which is exactly where the adversary goes next.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md):
hundreds of millions of users posting text, images, video, and live voice
across many languages; eight harm classes of unequal severity; a cheap fast
path handling billions of items per day; synchronous scoring for text,
async for video, near-real-time rolling windows for voice; a policy engine
that turns scores into actions; and an existing reviewer team whose decisions
feed back as labels. Here is the whole system with every choice committed and
the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Taxonomy | Eight per-policy heads on shared per-modality encoders | Operating points, drift rates, and legal duties differ by orders of magnitude; one "badness" score cannot express that |
| First gate | PDQ perceptual hash, conservative radius, bank grown from confirmed items | Near-zero cost and near-zero false positive at the same time; removes re-upload mass before any model spends compute |
| Text | Fine-tuned ModernBERT behind a Unicode-normalization front-end | Posts exceed 512 tokens and arrive obfuscated; unnormalized input loses to zero-width and homoglyph tricks in hours |
| Images | EfficientNet-B0 at ingest; larger model only for review-queue re-ranking | B0 survives the per-hour compute budget at billions of items; the heavy model runs where volume is already small |
| Video | Keyframe sampling plus audio track, async queue, escalate suspicious segments | Full-fidelity frame-by-frame is unaffordable; async is acceptable because video has no post-time gate |
| Voice | Distilled WavLM student (~48M params) on 10-to-15-second rolling windows | Live voice has no pre-publish gate; only a distilled, quantized student meets sub-100ms at 2,000+ requests per second |
| Cross-modal | CLIP-style joint model, gated behind unimodal ambiguity | OR-ing unimodal scores passes hateful memes by construction; ungated fusion blows the budget |
| Loss | Class-weighted BCE per policy, weight tuned on a calibration holdout | Unweighted loss learns to output near-zero for everything at sub-percent prevalence |
| Thresholds | Per-policy PR-curve scan against each precision floor; Platt after every retrain | The floor is the durable policy decision; the threshold is a disposable artifact re-derived from it |
| CSAM | Hash match auto-actions known material; classifier only prioritizes the queue | A false accusation is legally unacceptable; humans always make the call on novel material |
| Queue | Priority = severity x reach, decisions persisted as labels | Finite reviewer capacity concentrates where a miss costs most; the queue doubles as the label pipeline |
| Retraining | Spam near-weekly, nudity monthly, calibration after every retrain, Monday-evasion-to-Wednesday-model pipeline | A frozen model decays against an active adversary; cadence follows drift rate, not the calendar |
| Evaluation | Random audit stream, time-based split, recall at the floor per policy/modality/language, appeal-overturn rate live | Flagged-only measurement cannot estimate recall's denominator; the audit stream is the only unbiased view |

**Content volume.** The fast path sees on the order of 2 billion items per day
(Illustrative; [section 1](01-clarifying-requirements.md) pins "billions").
The funnel is what makes that affordable: the hash gate and cheap unimodal
classifiers resolve nearly everything, and the joint fusion model runs on well
under 1 percent of traffic, only where both modalities are present and the
unimodal signals conflict. Cost per item falls three to four orders of
magnitude from the fusion tier to the hash tier, which is why the funnel
ordering, not any single model choice, is the scaling decision.

**Auto-action versus review split.** Take one representative policy (hate
speech) on one surface at 1 million items per day and 0.5 percent prevalence
(Illustrative). The base-rate arithmetic from
[section 5](05-evaluation.md) governs everything: a classifier with 99 percent
recall and 99 percent specificity still delivers roughly 9 percent precision at
0.1 percent prevalence, so the auto-remove threshold must sit far out on the
score tail to meet the floor, and the borderline band between auto-remove and
auto-allow is where the residual errors concentrate. At thresholds tuned for a
high precision floor, the band holds a few percent of traffic, tens of
thousands of items per day, and it is mostly benign: the negative pool is two
hundred times larger than the positive pool, so even a small false-positive
rate fills the queue with edgy-but-fine content. The runnable at the end of
this section reproduces exactly this split.

**Reviewer headcount.** At roughly 500 decisions per reviewer per day
(Illustrative), a 75,000-item daily band on that one surface is about 150
reviewers before appeals, golden-set curation, and adjudication overhead. This
is why [section 6](06-serving-and-scaling.md)'s coupling matters operationally:
losing one percentage point of classifier precision can double queue volume,
which at this scale is another 150 heads or a blown severe-item SLA. Threshold
tuning is headcount planning.

**Prevalence measurement.** The audit stream is sized by the statistics, not
by convenience. Labeling 100,000 random items per day at 0.1 percent
prevalence yields about 100 true positives, giving a 95 percent confidence
interval of roughly plus or minus 20 percent relative on the prevalence
estimate (Illustrative). That is adequate for a global trend line and useless
per language: slicing the same budget across 30 languages leaves a handful of
positives each, which is why low-resource languages get targeted audit
oversampling rather than a proportional share.

**What breaks in month one.** Three failure modes dominate early operations,
so wire their signals before launch: appeal-overturn rate rising after the
first retrain (the threshold was carried over on a shifted score scale;
recalibrate and re-derive it from the floor, per
[section 5](05-evaluation.md)), review-queue volume spiking on one policy (the
classifier lost precision in the dense borderline band, and severe-item SLA
degrades first, per [section 6](06-serving-and-scaling.md)), and a step drop
in a policy's flag rate with confirmed violations holding steady (a new
evasion pattern is live; red-team it, augment, fast-retrain, per
[section 3](03-data-preparation.md)).

## The same techniques under different constraints

The review question that matters in practice is not "which classifier is best"
but "which classifier is best under my constraints." Here is the same system
built three times. Only the middle column is the build above; the other two
keep the identical stage interfaces and swap nearly every implementation
choice.

| | Startup UGC app | Multi-modal platform (this chapter) | B2B invite spam |
|---|---|---|---|
| Volume / modality | ~50k text and image posts/day | Billions of items/day; text, image, video, live voice | Invites with text and metadata; one harm class |
| Harm mix | Spam, nudity, harassment; CSAM hash-matching still legally mandatory | Eight policies, unequal severity, per-policy floors | Spam only; a false block is cheap |
| Models | One fine-tuned text encoder, one small image CNN; no fusion tier | Full funnel: hash gate, unimodal tiers, gated joint fusion, distilled voice student | Sparse logistic regression over engineered features (Slack class) |
| Hash gate | Industry hash lists (CSAM, terror) via vendor API; no home-grown bank | PDQ bank grown from own confirmed items, multiple algorithms | Feature-pattern lookup; no perceptual hashing |
| Thresholds | High auto-action floors; wide band to humans because volume permits it | Per-policy floors, PR-scan thresholds, recalibration after every retrain | Single high floor; auto-block at send time |
| Human loop | Two moderators reviewing every borderline item; consensus via escalation to founder | Priority-ranked queue, severity x reach, thousands of reviewers, decisions become labels | Nearly none: proxy label (invite acceptance) replaces review; periodic audit of blocks |
| Retraining | Monthly, manual; drift is slow at low visibility | Per-policy cadence, automated Monday-to-Wednesday pipeline | Retrain when the proxy metric degrades |
| Eval | Appeal overturns plus a small weekly audit sample | Audit stream, recall at floor per policy/modality/language, reach-before-action | Blocked-invite acceptance rate as the false-block proxy |
| What would be over-engineering | Joint fusion, distilled voice models, priority-ranked queue infrastructure | A single global "toxicity" score; it cannot carry eight floors | Deep encoders, a review platform, multi-grader consensus |

Two lessons fall out. First, the startup column is mostly deletions: at 50k
items per day two humans can see every borderline case, so the elaborate queue
machinery deletes, but the legal minimum does not: hash matching for CSAM is
mandatory at any size, which is why the vendor hash list survives when
everything else shrinks. Second, the invite-spam column shows what a
high-quality proxy label buys: when the recipient's acceptance decision labels
every block for free, the human loop and the consensus pipeline, the most
expensive parts of the chapter's build, collapse to a periodic audit.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any models.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Prevalence | Precision floor feasibility, audit-sample size | Below ~1 percent, precision collapses by Bayes even for a "99 percent" model; size the audit stream to yield ~100+ positives per slice you report |
| Cost of a false positive | Auto-action stance | Legally catastrophic (CSAM): classifier never auto-actions, hash only. Cheap (spam invite): auto-block at a high floor |
| Reviewer capacity | Band width between auto-allow and auto-remove | Headcount = band volume / decisions-per-reviewer-day; a 1pp precision loss can double the band, so tune thresholds and staffing together |
| Adversary speed | Retraining cadence, augmentation | Fast-mutating class (spam): near-weekly retrain, adversarial augmentation, flag-rate alerts in both directions. Stable class: monthly |
| Latency budget at ingest | Model tier on the hot path | Hundreds of ms: fine-tuned encoder inline. Sub-100ms streaming: distill and quantize a student. No budget (video): go async |
| Modality mix | Encoder stack, fusion tier | Single modality: no fusion. Image plus caption harm: gated joint model; OR-ing unimodal scores passes it by construction |
| Language spread | Eval breakdown, labeling budget | Report recall at the floor per language; oversample audits in low-resource languages instead of proportional sampling |
| Own review platform or not | Label source | With reviewers: decisions are gold labels, build the flywheel. Without: proxy labels or vendor labels, and accept the quality ceiling |

## The smallest runnable moderation loop

Every framework demo hides the decision that actually runs a moderation
system: where the two thresholds sit. So here is the whole enforcement split
in one file with zero installs. Every production component is swapped for the
smallest thing with the same interface: the calibrated per-policy classifier
becomes a seeded score distribution with an ambiguous middle, the policy
engine becomes two comparisons, and the reviewer team becomes an assumption of
perfect decisions plus a headcount bill. The shape is the lesson: one
threshold has two outcomes and this problem has three.

```python
"""Two thresholds, three outcomes: the tradeoff one threshold cannot satisfy."""
import random

random.seed(7)

DAILY_VOLUME = 1_000_000        # items/day on one surface, one policy
PREVALENCE = 0.005              # 0.5% of items truly violate
REVIEWS_PER_DAY = 500           # decisions per reviewer per day
N = 200_000                     # simulated sample, scaled up to DAILY_VOLUME
SCALE = DAILY_VOLUME / N

def clamp(x):
    return min(1.0, max(0.0, x))

def score(violating):
    """Calibrated-classifier stand-in: overlapping score distributions.
    Most content is easy; a minority sits in a genuinely ambiguous middle."""
    if violating:   # 75% blatant, 25% subtle (satire-adjacent, obfuscated)
        mu, sd = (0.82, 0.09) if random.random() < 0.75 else (0.55, 0.13)
    else:           # 92% clearly benign, 8% edgy-but-fine (news, counter-speech)
        mu, sd = (0.10, 0.07) if random.random() < 0.92 else (0.45, 0.14)
    return clamp(random.gauss(mu, sd))

items = [(v, score(v)) for v in
         (random.random() < PREVALENCE for _ in range(N))]
n_violating = sum(1 for v, _ in items if v)

def evaluate(t_low, t_high):
    """Auto-remove >= t_high; human review in [t_low, t_high); allow < t_low.
    Toy assumption: reviewers are perfect, so band errors become queue cost."""
    auto_benign = sum(1 for v, s in items if s >= t_high and not v)
    auto_total  = sum(1 for v, s in items if s >= t_high)
    queued      = sum(1 for _, s in items if t_low <= s < t_high)
    missed      = sum(1 for v, s in items if s < t_low and v)
    false_removal = auto_benign / auto_total if auto_total else 0.0
    miss_rate = missed / n_violating
    queue_day = queued * SCALE
    return false_removal, miss_rate, queue_day, queue_day / REVIEWS_PER_DAY

print(f"{DAILY_VOLUME:,} items/day, prevalence {PREVALENCE:.1%}, "
      f"reviewer throughput {REVIEWS_PER_DAY}/day\n")
print(f"{'T_low':>6} {'T_high':>7} {'false-removal':>14} "
      f"{'missed-violation':>17} {'queue/day':>10} {'reviewers':>10}")
configs = [
    (0.30, 0.30),   # one threshold, permissive
    (0.50, 0.50),   # one threshold, "balanced"
    (0.85, 0.85),   # one threshold, cautious
    (0.30, 0.85),   # review band
    (0.20, 0.92),   # wider band
]
for t_low, t_high in configs:
    fr, miss, queue, heads = evaluate(t_low, t_high)
    kind = "band" if t_low != t_high else "single"
    print(f"{t_low:>6.2f} {t_high:>7.2f} {fr:>13.1%} {miss:>16.1%} "
          f"{queue:>10,.0f} {heads:>10.0f}  {kind}")
```

Run it and the five rows demonstrate the chapter's central coupling in about
sixty lines. Every single-threshold row fails in one direction: at 0.30,
93 percent of removals are benign content; at 0.85, false removals fall to
11.6 percent (still high, because the benign pool is 200 times larger, the
base-rate arithmetic of [section 5](05-evaluation.md)) but 73.3 percent of
violations sail through. No point on that dial delivers both low false
removals and low misses, because the ambiguous middle is on both sides of any
single cut. The band rows buy both at once: 0.30-to-0.85 hits 0.6 percent
missed violations with the same 11.6 percent auto-remove impurity, at the
price of 74,115 queued items per day, 148 reviewers. Widening the band buys
further error reduction at double the headcount, which is
[section 6](06-serving-and-scaling.md)'s threshold-staffing coupling made
concrete. Swap `score` for a calibrated per-policy classifier, the two
comparisons for the policy engine with graduated actions, the perfect-reviewer
assumption for consensus grading whose decisions feed retraining, and put a
hash gate in front of all of it, and you have rebuilt this chapter.
