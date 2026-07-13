# 8. Interview Q&A

The questions interviewers actually ask about CV systems, grouped by how they
are used. The "commonly answered wrong" section is where interviews are won or
lost.

## Commonly asked

**Q: When do you use detection instead of classification?**
A: When position or the presence of a small region drives the product decision.
If the harmful content can be a small corner of the image (a weapon, a symbol),
a whole-image classifier will miss it. Detection adds a region proposal stage
and regresses bounding boxes, so the head can flag that region. The metric
shifts from accuracy to mAP at IoU thresholds. The label cost also jumps 5-10x:
an annotator must draw and verify a box, not just tag the image.

**Q: Why almost never train from scratch?**
A: A backbone pretrained on ImageNet-scale data already encodes edges, textures,
shapes, and object parts. Fine-tuning it with a few thousand domain labels almost
always beats training from scratch with tens of thousands, because the backbone
carries all that prior learning. Training from scratch is only justified when the
domain is so far from natural images (radar, X-ray, satellite) that pretrained
features are actively misleading, and when you have hundreds of thousands of
domain labels to compensate.

**Q: What is the right metric for a moderation classifier?**
A: Recall at a fixed precision floor per harm class, not accuracy. Missing a
violation (false negative) is worse than flagging a legitimate photo (false
positive), and the costs are asymmetric and class-specific. Define the minimum
acceptable precision (e.g., 90% for nudity) and report the recall achievable at
that operating point. Also report the human-review queue rate the operating point
implies, since that tells the ops team how many reviewers they need.

**Q: How do you handle a new harm class with only 40 labeled examples?**
A: Do not train a dedicated classification head; 40 examples cannot generalize.
Use retrieval instead: compute embeddings for the 40 known violations and run
ANN search over all uploads. Also apply perceptual hashing against known-bad
content and zero-shot scoring from a CLIP-style model against text prompts for
that harm type. Route all high-similarity matches to human review. Once labels
accumulate past a few hundred, revisit a dedicated head.

**Q: How do you share one backbone across multiple tasks?**
A: Add one lightweight head per task that branches from the same feature
extraction network. The backbone runs once per image; each head adds a small
forward pass (linear layer or a few conv layers). Train jointly or in sequence,
depending on whether the tasks have aligned or conflicting objectives. This is
the Pinterest unified embedding approach: one SE-ResNeXt trunk feeds three
task-specific heads. Any backbone improvement lifts all tasks at once.

## Tricky (the follow-ups that separate people)

**Q: Your mAP improved offline but precision at the operating confidence dropped
online. What happened?**
A: The operating confidence threshold changed implicitly. mAP averages over all
thresholds; if the model learned to be less confident (lower raw scores) while
still ranking items correctly, mAP is unchanged but the chosen threshold now
hits a different precision-recall point. Always re-calibrate the threshold on
the validation set after retraining, and track precision and recall at the fixed
operating point, not just mAP.

**Q: The visual search returns visually similar but irrelevant items. How do you fix it?**
A: The embedding is optimizing visual feature similarity (texture, color, shape)
without purchase-intent signal. Two fixes: (1) add a supervised fine-tuning stage
with engagement labels (what users clicked or bought after a search), so the
embedding learns intent-aligned similarity; (2) add a re-ranking step over the
ANN candidates with a supervised model that sees both the query and the candidate.
Pinterest addressed this by training with proxy metric learning on engagement-
derived datasets, not just visual similarity.

**Q: You added a ViT backbone and mAP improved 4 points. Should you ship it?**
A: Not automatically. Measure: (1) inference latency (ViT is slower than ResNet
at the same accuracy level; does the moderation gate still hit p99?); (2) cost
per million images on the tagging and embedding paths (a 2x FLOP increase at 5M
uploads/day is a significant budget line); (3) whether the improvement is uniform
or concentrated on head classes. A 4-point mAP gain that costs 2x serving budget
and shows no improvement on the rare, high-risk classes may not be the right
trade.

**Q: How do you know if train-serve preprocessing skew is the problem?**
A: Run inference on a fixed set of known-good validation images in both the
training pipeline and the serving pipeline, and compare the output distributions.
A byte-identical preprocessing path should produce the same logits. If they
diverge, instrument each step (decode, resize, normalization constants, channel
order) until you find the mismatch. This bug is silent in offline metrics and
shows up only as a production accuracy regression.

## Commonly answered wrong (the traps)

**Q: Report accuracy as the moderation metric. Is that right?**
A: No. Accuracy treats false positives and false negatives as equal-cost errors.
In moderation, a missed violation (false negative) is far worse than a wrongly
flagged photo (false positive). The correct metric is recall at a fixed precision
floor, so you can explicitly reason about the tradeoff. Accuracy on an imbalanced
harm dataset can be 99% while recall on the rare harm class is near zero.

**Q: Train one model for all three tasks (tagging, moderation, search) to save
time. Is that a good idea?**
A: It is a good idea to share a backbone, but not to share heads. The output
shapes are incompatible (multi-label sigmoid vs. binary gate vs. unit-sphere
vector) and the loss functions conflict. Multi-task learning on the heads is
possible when tasks are related (Shopify's multi-level taxonomy), but forcing
a single head to serve three product requirements with different metric targets
and operating points will degrade all three. Share the trunk; keep the heads
separate.

**Q: Should the EXIF orientation fix happen in the model (augment with random
rotations) rather than in ingest?**
A: No. EXIF orientation is a deterministic, correctable metadata error, not
random variation you want the model to learn invariance to. Apply the fix in
the ingest stage before any model sees the image. Relying on rotation
augmentation to absorb EXIF errors means the model wastes capacity learning
invariance to a bug that should not exist in the serving pipeline, and the
augmentation does not guarantee the model actually learns the right rotation
for each image.

**Q: A higher confidence threshold on the moderation gate makes the system safer.
Is that right?**
A: It depends on what "safer" means. A higher threshold raises precision (fewer
false positives) but lowers recall (more violations slip through). For a high-harm
class, the safe direction is a lower threshold (higher recall, accepting more false
positives that the human review queue will clear). A higher threshold reduces the
review queue burden but increases missed violations. The correct framing is to fix
the minimum acceptable precision and maximize recall there, not to "raise
confidence."
