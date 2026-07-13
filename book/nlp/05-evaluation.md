# 5. Evaluation

## Per-class F1, not accuracy

Reporting aggregate accuracy on an NLP classification task is almost always wrong,
and on a safety or spam task it is actively dangerous. Consider: a model that
predicts "not spam" for every message achieves 99.5% accuracy when spam is 0.5%
of traffic. It catches zero spam. Aggregate accuracy optimizes toward "predict the
majority class," which is exactly the failure mode that makes safety systems
worthless.

The correct metric family for classification is **precision, recall, and F1 per
class**, especially the rare class:

$$P^{(c)} = \frac{TP^{(c)}}{TP^{(c)} + FP^{(c)}}, \qquad R^{(c)} = \frac{TP^{(c)}}{TP^{(c)} + FN^{(c)}}$$

$$F_1^{(c)} = \frac{2 \cdot P^{(c)} \cdot R^{(c)}}{P^{(c)} + R^{(c)}}$$

Across $C$ classes, macro-averaged F1 weights every class equally regardless of
frequency, which is exactly right for the minority class that accuracy ignores:

$$F_1^{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} F_1^{(c)}$$

![Per-class F1 under class imbalance](assets/fig-class-imbalance-f1.png)

*Left panel: accuracy looks nearly identical across both classes but F1 on the
minority class (spam) collapses to 14%. Right panel: resampling plus a cost-aware
threshold recovers minority-class F1 to 72% with negligible regression on the
majority. Accuracy alone would have hidden the failure. Illustrative.*

Report the confusion matrix in addition to per-class F1. It shows which classes
bleed into which, and that pattern usually tells you whether the problem is
labeling noise, class overlap, or a feature gap.

## Task-appropriate metrics

Different NLP tasks call for different metrics. Using the wrong one is a classic
interview mistake.

**NER and extraction:** span-level F1, not token-level accuracy. A span is correct
only if both the boundary and the entity type are correct (strict match). Partial-
match F1 (boundary correct, type wrong) is reported separately to diagnose
taxonomy confusion vs boundary errors. Token accuracy on a BIO-tagged sequence
can look high simply because O tags dominate.

**Translation:** BLEU or COMET for automatic tracking, plus human adequacy and
fluency ratings. BLEU measures n-gram overlap between hypothesis and reference and
is fast and cheap to compute, but it misses meaning: a correct paraphrase with
different word choice scores poorly. COMET (a learned metric trained on human
judgments) correlates better with human quality ratings. The release gate for
production translation systems is human ratings, not BLEU alone. Google GNMT used
a 0-6 human rater scale as the final quality bar.

**Grammatical error correction:** $F_{0.5}$ rather than $F_1$. False corrections
(editing good text) annoy users more than misses (leaving an error in), so
precision counts twice as much as recall in the harmonic mean. Grammarly's GECToR
reports $F_{0.5}$ on CoNLL-2014 and BEA-2019 benchmarks for this reason.

**Entity resolution:** pairwise precision and recall on matched pairs, measured
against a held-out set of known-synonym or known-distinct pairs.

## Calibration: turning a score into a probability

A raw model logit is not a probability you can threshold on. After fine-tuning, a
classifier is often over-confident on high scores and under-confident near the
boundary, which means the raw score does not behave like a true probability.
**Temperature scaling** fits a single scalar $T \gt 0$ to the logits on a held-out
calibration set:

$$p_{\theta}(c \mid x) = \text{softmax}\!\left(\frac{z_c}{T}\right)$$

When $T \gt 1$, the distribution softens (less confident); when $T \lt 1$, it
sharpens. Temperature scaling is cheap (one hyperparameter) and rarely hurts
accuracy. Platt scaling (logistic regression on the logit) and isotonic regression
are heavier alternatives when the calibration error is large.

Why does calibration matter operationally? It makes "0.9 means roughly 90% of
such cases are positive" hold, which is what makes the confidence-gated routing
decision principled: auto-act above 0.95, route to review between 0.5 and 0.95,
auto-allow below 0.3. Without calibration those thresholds are magic constants
that drift every time the model retrains.

![Calibration curve](assets/fig-calibration-curve.png)

*A reliability diagram for a fine-tuned classifier before and after temperature
scaling. The red line (before) overshoots near perfect confidence; the green line
(after) tracks the diagonal closely. Set the confidence threshold only after
calibrating. Illustrative.*

**Recalibrate after every retrain.** A new model can shift the score distribution,
making old thresholds over- or under-act. Recalibration is a one-step fix; missing
it is one of the most common silent production failures.

## Slicing by language and segment

A global metric number hides broken subgroups. For a multilingual system:

- **Slice eval per language.** A model that is 91% F1 in English and 52% F1 in
  Turkish shows 89% F1 globally and looks fine. Per-language reporting is not
  optional for multilingual deployments; it is the primary quality control.
- **Slice by cohort.** New users, short messages, messages in a particular product
  area. A model that fails silently on one cohort can cause outsized harm.
- **Track false block rate as a first-class metric for safety tasks.** Over-
  aggressive thresholds silence innocent users. The false block rate on a random
  sample of auto-blocked messages is a first-class release gate, tracked as
  carefully as the miss rate.

## When to use which evaluation metric

| Reach for | When | Instead of |
|---|---|---|
| Per-class F1 and PR curves, sliced per language | any classification with imbalanced classes or a safety task | aggregate accuracy, which a majority-class predictor maximizes |
| Span-level strict F1 for NER | extraction where both boundary and type must match | token-level accuracy dominated by O-tag majority |
| $F_{0.5}$ | correction tasks where false edits annoy users more than misses | plain F1 that weights precision and recall equally |
| BLEU or COMET plus human adequacy ratings | translation quality | BLEU alone (misses meaning) or human ratings alone (too slow and expensive for CI) |
| Pairwise precision/recall on matched pairs | entity resolution against a known-synonym test set | classifier accuracy on a fixed-class label set |
| Temperature or isotonic calibration before thresholding | any model whose raw score feeds a confidence gate | acting on raw logits with a fixed threshold that was set at training time |
| Time-based eval split | any offline classification or extraction eval | random split, which may leak future labels into training |
| Human review sample for safety | toxicity and abuse where offline metrics are necessary but not sufficient | purely automated eval for a task where false blocks harm real users |
