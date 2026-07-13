# 5. Evaluation

Picking the wrong metric is the most common CV interview mistake. Accuracy is
almost never the right headline number. The correct metric depends on what the
output shape is and what the product actually gates on.

## Metric by task

### Classification: per-class precision and recall

Accuracy collapses all classes into one number. A model that predicts the
majority class every time can score 95% on an imbalanced dataset while
completely failing on every rare class. Use per-class precision and recall, and
macro-average to expose the tail.

For a binary gate (moderation), the operating point matters more than any single
number. Fix a minimum precision and take the recall achievable there:

$$R_{\text{op}} = \max\!\left\{ R \;:\; P(t) \geq P_{\min} \right\}$$

For example: "at 90% precision, what is our recall on the nudity class?" That
question maps directly to the product tradeoff (how many violations do we catch
vs. how many legitimate photos do we wrongly flag).

For medical or screening use cases (diabetic retinopathy), F1 is used because
it punishes a collapse in either precision or recall equally:

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

### Detection: mAP at IoU thresholds

Average precision for one class is the area under its precision-recall curve
(the shaded area in the figure below). Mean average precision (mAP) averages
this over all classes. COCO-style mAP further averages over IoU thresholds from
0.5 to 0.95 in steps of 0.05, written mAP@[.5:.95].

![Precision-recall curves and mAP](assets/fig-pr-map.png)

*Each colored curve is one class. The shaded area under each curve is that
class's average precision (AP). mAP is the mean of the AP values across all
classes. A class with a curve that drops steeply (orange) has low AP even if
its peak precision is high, because recall does not extend far.*

Also report precision and recall at the **chosen confidence operating point**,
since the product must pick one threshold to run at.

### Segmentation: mean IoU (mIoU)

mIoU averages intersection-over-union over all classes, so a dominant class
cannot hide weak ones:

$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{\lvert A_c \cap B_c \rvert}{\lvert A_c \cup B_c \rvert}$$

Boundary IoU is a stricter variant that evaluates only the pixels near the
boundary contour, useful when edge quality matters (garment cutout, where a
rough boundary is visible to the user).

### Embedding retrieval: recall at k

Recall at k over a labeled query set is the fraction of queries where the
relevant item appears in the top-k results of the ANN lookup:

$$R@k = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \mathbf{1}\!\left[\text{rel}(q) \in \text{top-}k(q)\right]$$

Measure at the k you actually serve (the ANN shortlist before any re-ranking).
Tuning to look good at k=5 when you serve k=100 optimizes the wrong point.

Also track mean average precision over the ranked list (MAP, not mAP) for cases
where multiple relevant items exist per query.

### OCR: text accuracy and field coverage

Report character-error rate (CER) and word-error rate (WER) on a held-out
transcription set. For structured documents (ID cards, receipts), report
per-field extraction accuracy (all characters in the target field correct).

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| Per-class recall at a fixed precision floor | a harm gate (moderation) where missing a violation has an asymmetric cost | accuracy, which hides recall on rare harm classes |
| Macro precision and recall per class | multi-label tagging over a long-tailed taxonomy | overall accuracy, which is dominated by the head classes |
| F1 | screening with balanced harm cost on FP and FN (medical grading) | accuracy on imbalanced data |
| mAP at IoU (COCO style) | detection quality across all classes and thresholds | accuracy or per-class threshold, which do not measure localization |
| mIoU | semantic segmentation where per-class coverage matters | pixel accuracy, which is dominated by background |
| Recall at k at the serving k | embedding retrieval quality | classification metrics, which do not measure ranked retrieval |

## Evaluation discipline

Several practices separate a real evaluation from a flattering one:

- **Frozen per-task test set.** Never tune thresholds on the test set. Hold out a
  fixed labeled test split per task and refresh it periodically as the data
  distribution changes.
- **Time-based split for retrieval.** Hold out future queries and evaluate whether
  today's index retrieves them. A random split leaks future items into the training
  embeddings.
- **Sliced metrics.** Report metrics sliced by category, geography, skin tone, and
  photo quality, not just aggregate. A gap that passes the aggregate bar can still
  be a launch blocker on a specific slice.
- **Calibration.** Raw softmax logits are not probabilities. Temperature-scale
  (or Platt-scale) the outputs so that a threshold of 0.9 actually means 90%
  precision. Especially important for the moderation escalate band.
- **Online proxy.** Offline metrics are necessary but not sufficient. Pair each
  offline metric with an online proxy: human-review overturn rate for moderation,
  click or conversion for visual search. A regression in overturn rate is live
  precision signal.

The next section translates these metrics into serving requirements and the
cost-per-million-images number that actually drives infrastructure decisions.
