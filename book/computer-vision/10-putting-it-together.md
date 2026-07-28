# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable detection post-processor, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a month comparing backbones before labeling a single image.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Model checkpoints
change yearly, but the interface of each stage (frame the task, label, augment,
pick a trunk, pick a head, evaluate, serve) does not, so pick per stage by
interface and treat any specific architecture as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Task framing | Map each product ask to an output shape before touching a model; classification unless position matters | The target can occupy a small corner of the image: pay for detection | [2](02-frame-as-ml-task.md) |
| Labeling | Mine free labels first (upload context, review decisions, CLIP zero-shot), then active learning for paid budget | Labels are abundant and cheap already (rare): random sampling is simpler | [3](03-data-preparation.md) |
| Augmentation | Flip, crop, color jitter, with the decode-resize-normalize path pinned identical to serving | The transform changes the label (flips for OCR, color jitter for blur scoring): drop it | [3](03-data-preparation.md) |
| Backbone | Pretrained ResNet-50, partial fine-tune with a small LR | Cost per million images binds: EfficientNet; multi-scale detection: Swin; no labels at all: frozen DINOv2 or CLIP | [4](04-model-development.md), [3](03-data-preparation.md) |
| Task head | One shared trunk, one lightweight head per task; sigmoid per class for multi-label | Output shapes conflict at the trunk too (rare): split models and eat the serving cost | [4](04-model-development.md) |
| Evaluation | Per-class recall at a fixed precision floor for gates; mAP at IoU for detection; recall at the serving k for retrieval | Never accuracy on imbalanced data. There is no deviation that makes accuracy right | [5](05-evaluation.md) |
| Serving | Async batch by default; only the publish gate runs sync, distilled and quantized | A task must gate a user action in real time: it earns the sync path and its latency budget | [6](06-serving-and-scaling.md) |

The framing row is the one beginners skip and regret: reaching for
classification when the product needs localization, or a fixed class list when
the catalog is open, is unrecoverable by tuning. State the output shape per
product ask first; everything downstream follows from it.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
marketplace with 80 million catalog photos and 5 million uploads per day at
peak, needing a real-time moderation gate on the publish path (three harm
classes, p99 under a few hundred milliseconds, fail closed on high harm), async
multi-label room-type tagging from a small labeled set, and visual search whose
index is fresh within minutes. Here is the whole system with every choice
committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Task split | Three heads on one shared trunk: binary-per-class gate, multi-label classifier, embedding | Output shapes are incompatible; one head cannot serve three metrics, but one trunk serves three heads |
| Ingest | One decode / EXIF-fix / resize / normalize stage feeding every model | Preprocessing skew is the most common silent killer; one pinned path, byte-identical at train and serve |
| Backbone | ResNet-50 fine-tuned on domain data; EfficientNet-B0 distillation for the gate | Proven trunk for the label counts we have; the gate needs the 50-60% FLOP cut to hit p99 |
| Moderation gate | Per-class sigmoid classification, calibrated thresholds, detection deferred | Classification labels cost 1x; boxes cost 5-10x. Ship the cheap gate, add detection only if small-region harms prove to be the binding miss |
| Cold-start labels | CLIP zero-shot scoring plus perceptual hashing to seed, human review decisions wired back as gold labels | Moderation has zero labels on day one; review decisions are the highest-quality labels in the pipeline |
| Tagging head | Multi-label sigmoid, per-class thresholds calibrated on validation | Airbnb-style fixed taxonomy; one global threshold is wrong when the tail classes have different operating points |
| Search | CLIP-style embedding head, ANN index (HNSW with PQ), no class list | The catalog is open and growing; classification would need retraining per taxonomy change |
| Index freshness | Embed and upsert new uploads within minutes; full re-embed only on backbone swap | Users search their own upload immediately; a stale index is a visible product bug |
| Fail policy | High-harm classes fail closed to human review; low-harm fail open; async paths just delay | Stated per class before launch, because "what happens on timeout" is a product decision, not an ops accident |
| Evaluation | Recall at 90% precision per harm class; macro P/R for tagging; recall at the served k for search; overturn rate online | Each metric is the one the product actually gates on; accuracy appears nowhere |

**Labeling budget.** The relative-cost table in
[section 3](03-data-preparation.md) is the arithmetic that decided the
moderation head. At image-level tags as the 1x unit, the same annotation budget
buys 5-10x fewer bounding boxes and 20-30x fewer masks. Illustrative: a budget
that funds 50,000 moderation tags funds only 5,000-10,000 boxes, and a
three-class gate cannot reach a per-class precision floor on a few thousand
examples of anything. So the gate ships as classification, active learning
spends the budget on the model's least-confident uploads, and detection waits
until measurement shows small-region harms are the failure mode worth 5-10x per
label.

**Throughput and the sync path.** 5 million uploads per day is about 58 images
per second on average; assume a peak factor of 3x, roughly 175 per second
(Illustrative). Every upload passes the shared trunk once, and the trunk
amortizes across all three heads, which is the structural cost win from
[section 4](04-model-development.md). Only the moderation gate is sync, so only
the gate pays for distillation: the distilled student runs on every image at
5-10x lower latency, and the heavy teacher runs solely on the ambiguous
escalate band, so average cost tracks the small escalate rate rather than the
teacher's per-image price. Tagging and embedding drain a queue on
throughput-optimized capacity where a batch is always full and spot instances
are viable.

**Index sizing.** The search side is a memory problem before it is a model
problem. 80 million catalog embeddings at 512 dimensions (a CLIP ViT-B/32
output) in float32 is 80M x 512 x 4 bytes, about 164 GB of raw vectors, which
is why [section 6](06-serving-and-scaling.md)'s index table sends an 80M-item
catalog to HNSW with product quantization: at 8-bit codes the same vectors fit
in roughly a quarter of the memory or less, trading a small recall loss
(Illustrative sizing). The number to internalize is the coupling: embedding
dimension is an index-memory decision, not a leaderboard decision, and a
backbone retrain forces a full catalog re-embed and a coordinated index
rebuild.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: train-serve preprocessing skew (run a fixed
probe set through both pipelines and alert on logit divergence; offline metrics
are structurally blind to this bug, per [section 5](05-evaluation.md)),
per-class overturn rate in the human review queue (aggregate moderation
precision will look fine while one rare harm class sits at near-zero recall;
overturn rate is the live per-class signal), and search relevance decay after
the first backbone retrain (embeddings from the new trunk land in an index
built by the old one; recall at k against a frozen query set catches the
version mismatch before click rate does).

## The same techniques under different constraints

The review question that matters in practice is not "which backbone is best"
but "which backbone is best under my constraints." Here is the same pipeline
built three times. Only the marketplace column is the build above; the other
two keep the identical stage interfaces and swap nearly every implementation
choice.

| | Marketplace platform (this chapter) | On-device document check | Continental satellite mapping |
|---|---|---|---|
| Task / output shape | Classify (gate + tags) and embed; open catalog | Classify image quality, detect document fields | Semantic segmentation, per-pixel building masks |
| Traffic / latency | 5M uploads/day; gate p99 under a few hundred ms | One image at a time, real-time on a mid-range phone | Offline batch over continental imagery; no latency budget at all |
| Backbone | ResNet-50 trunk shared by three heads; distilled EfficientNet gate | Quantized MobileNet-class TFLite model, multi-task | U-Net, fully fine-tuned; satellite is far from natural-image pretraining |
| Labels | Free sources + active learning; boxes deferred on cost | Field boxes are unavoidable; the product is localization | Mask labels at 20-30x cost; accepted because pixels are the product |
| Serving | Sync gate + async queue + ANN index | Entirely on-device; server sees only uploads that pass | Throughput-optimized batch; spot capacity, rebuilt on a cadence |
| Eval | Recall at a precision floor; recall at k; overturn rate | Auto-capture rate and per-field accuracy | Precision-recall at 0.5 IoU, sliced by terrain type |
| Fail policy | Fail closed per harm class | Fail open to server-side checks; the gate is a cost saver, not a legal gate | None needed; a failed tile re-runs in the next batch |
| What would be over-engineering | Segmentation masks, a ViT trunk at current label counts | Any server round-trip on the capture path, any trunk over a few MB | Distillation, quantization, dynamic batching, anything latency-shaped |

Two lessons fall out. First, the on-device column inverts the budget: the
binding constraint is model size and phone-CPU latency, so quantization is
mandatory rather than an optimization, and the expensive box labels are
accepted because localization is the product, not a nice-to-have. Second, the
satellite column shows what happens when latency disappears entirely: every
serving lever from [section 6](06-serving-and-scaling.md) goes unused, the
label budget flows to the most expensive annotation type in the book, and the
transfer-learning row flips to full fine-tune because the domain is too far
from natural images for frozen features to survive.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any models.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Labeling budget | Task head | Tags 1x, boxes 5-10x, masks 20-30x; pick the cheapest output shape that answers the product question |
| Where the target sits in frame | Classification vs detection | Fills the frame: classify. Can hide in a corner: detect, and pay the box cost |
| Sync on a user action | Model size on that path | Distill and quantize the gate model; run the heavy model only on the escalate band |
| Upload volume | Backbone FLOPs, trunk sharing | Cost per million images, not per request; share one trunk across heads and the cost divides by the head count |
| Catalog open or fixed | Classifier vs embedding | Growing or unbounded class list: embedding + ANN; stable taxonomy: classifier with per-class thresholds |
| Label scarcity per class | Head vs retrieval | Under a few dozen examples: ANN over embeddings, hashing, zero-shot; a trained head cannot generalize from 40 samples |
| Domain distance from natural images | Transfer strategy | Close: linear probe or partial fine-tune. Far (satellite, medical): full fine-tune; scratch almost never |
| Crowded scenes | NMS threshold | Tune to scene density; low merges adjacent objects, high keeps duplicates, and mAP will not warn you either way |
| Asymmetric error cost | Metric and threshold | Fix the precision floor, maximize recall there, and report the review-queue rate the operating point implies |

## The smallest runnable NMS

The review of every detection tutorial is the same: the reader watches a
framework draw boxes and still cannot see the post-processing decision that
production teams actually tune. So here is that decision in one file with zero
installs. The detection head is replaced by a seeded jitter function that emits
overlapping candidates around known objects, exactly the raw output shape a
real head produces, and everything downstream (IoU, greedy NMS, matching
against ground truth at a fixed evaluation IoU) is the real algorithm from
[section 4](04-model-development.md), not a stand-in.

```python
"""IoU and NMS from scratch: the detection post-processing loop, no installs."""
import random

def iou(a, b):                        # boxes as (x1, y1, x2, y2)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0

def nms(dets, thr):
    """Greedy NMS: keep the best-scoring box, drop overlaps above thr, repeat."""
    dets = sorted(dets, key=lambda d: d[1], reverse=True)   # (box, score)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best[0], d[0]) <= thr]
    return keep

def score_against_gt(kept, gt, match_iou=0.5):
    """Match kept boxes to ground truth greedily; the eval IoU is fixed at 0.5
    and is a different knob from the NMS threshold."""
    unmatched = list(gt)
    tp = fp = 0
    for box, _ in sorted(kept, key=lambda d: d[1], reverse=True):
        hit = next((g for g in unmatched if iou(box, g) >= match_iou), None)
        if hit is not None:
            unmatched.remove(hit); tp += 1
        else:
            fp += 1
    return tp, fp, len(unmatched)     # missed = GT boxes nothing matched

# --- seeded toy scene: what a detection head emits before post-processing ----
random.seed(7)
GT = [(10, 10, 50, 50),               # object A
      (25, 10, 65, 50),               # object B, overlapping A (crowded pair)
      (150, 120, 190, 170)]           # object C, isolated

def jitter(box, d):                   # a near-duplicate raw detection
    return tuple(v + random.randint(-d, d) for v in box)

raw = []
for g in GT:                          # 4 overlapping candidates per true object
    for _ in range(4):
        raw.append((jitter(g, 3), round(random.uniform(0.55, 0.95), 2)))
raw.append(((220, 20, 250, 60), 0.58))   # spurious background detection
raw.append(((222, 24, 254, 62), 0.52))   # its near-duplicate

gt_pair_iou = iou(GT[0], GT[1])
print(f"raw detections: {len(raw)} candidates for {len(GT)} true objects "
      f"(A-B overlap IoU {gt_pair_iou:.2f})")
for thr in (0.30, 0.50, 0.80):
    kept = nms(raw, thr)
    tp, fp, missed = score_against_gt(kept, GT)
    print(f"NMS IoU threshold {thr:.2f}: kept {len(kept):2d} boxes -> "
          f"TP {tp}, FP {fp}, missed {missed}")
```

Run it and the three thresholds print the whole tuning problem in four lines:
at 0.30 the two overlapping objects A and B (ground-truth IoU 0.45, a crowded
pair) suppress each other and one true object goes missing (TP 2, missed 1); at
0.80 the near-duplicate candidates survive suppression and false positives
quadruple (FP 4); only the middle threshold 0.50 recovers all three objects
with the single false positive being the spurious background detection the
model itself invented, which no NMS setting can remove. The jittered candidates
stand in for a dense head's raw anchor outputs, the seeded background pair
stands in for a genuine model false positive, and the fixed 0.5 matching IoU
inside `score_against_gt` is the evaluation threshold that
[section 8](08-interview-qa.md) warns you not to confuse with the NMS knob.
Swap the jitter function for a real detection head and this file is the
production post-processing path.
