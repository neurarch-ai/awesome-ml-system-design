# 8. Interview Q&A

The questions an interviewer actually asks about production NLP systems, grouped by
how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Why not just use an LLM for all of this?**

A: Three reasons: latency, cost, and calibration. A distilled encoder classifies
in single-digit milliseconds on commodity hardware; a large decoder LLM is 100 to
300 times slower and orders of magnitude more expensive per call. At millions of
calls per day the inline path cannot afford a decoder. Second, an LLM returns text
you must parse; it does not emit a calibrated probability you can threshold.
Third, with even a few thousand labeled examples, a fine-tuned encoder matches or
beats a zero-shot LLM on a fixed label set, because it is specialized to your
exact distribution and taxonomy. The LLM's real role is offline: generating weak
labels, handling the hard-tail abstentions, and serving as a zero-shot baseline
before labels exist.

**Q: You said "NLP system." What are the actual tasks, and do they use the same
model?**

A: NLP is a family, not a task. Routing is text classification (one label per
document, encoder plus softmax head). Field extraction is NER or span extraction
(one label per token, encoder plus token-tagging head). Abuse detection is
classification under brutal class imbalance. Translation is seq2seq
(encoder-decoder). Entity resolution is bi-encoder plus ANN match. Saying "these
are five different problems with five different models, labels, and metrics" is
most of the interview signal. A single model that handles everything is usually the
wrong answer.

**Q: The positive class in your abuse detector is 0.3% of traffic. How do you train
and evaluate it?**

A: Train with class-weighted cross-entropy (weight proportional to inverse class
frequency), or oversample the positive class. Mine hard negatives once the model
handles easy ones. Evaluate with precision-recall curves and per-class F1 on the
positive class, never with accuracy (a model that predicts "not abuse" always
is 99.7% accurate and catches nothing). Set a per-class cost-aware threshold, not
a single global cutoff, and track the false block rate on auto-rejected messages
as a release gate alongside the recall number.

**Q: How do you get labeled data when the positive class is rare and expensive to
annotate?**

A: Three tools. First, weak supervision: write labeling functions (keyword lists,
regex patterns for known patterns, an existing simpler model, an LLM prompt), and
combine their noisy votes into soft labels. This bootstraps a training set without
hand-labeling millions of messages. Second, LLM annotation: prompt a large model
to label a sample, treat its output as a noisy label source, and distill it into
the small encoder. You pay the LLM once per training example, not once per
inference. Third, active learning: route uncertain predictions from the inline
model to a human review queue, and fold every review decision back into training.
The review queue is itself a labeling pipeline.

**Q: How do you handle forty languages without a separate model per language?**

A: A multilingual encoder (mBERT, XLM-R) shares a single subword vocabulary and
model weights across languages, giving cross-lingual transfer: training in English
provides some signal for Spanish or Portuguese. Tradeoffs to name: shared capacity
dilutes per-language representation, so a dedicated monolingual model will
outperform it on any single high-resource language. Morphologically rich or
non-Latin-script languages fragment into more subword tokens (more latency, more
cost). Run language ID up front, and slice eval per language so a broken language
does not hide in the global number.

## Tricky (the follow-ups that separate people)

**Q: Your per-class F1 on the abuse class improved in offline eval, but your
false block rate rose in the online A/B. What happened?**

A: The model learned a proxy for abuse that fires on legitimate messages from a
particular demographic, writing style, or topic. A classic adversarial-fairness
failure: aggregate F1 improves while a segment of innocent users gets over-blocked.
Diagnose with segment-level false block rate and per-slice PR curves, not global
metrics. The fix is adding a representative sample of the over-blocked segment to
the negative class and retraining, and adding the false block rate as a first-class
gate for every promotion.

**Q: You retrained the encoder overnight and promoted it to production. The next
morning, abuse recall dropped sharply. What went wrong?**

A: Likely a calibration or threshold issue, not a model quality issue. The new
model can shift the score distribution even if its F1 is higher on the validation
set. If the old threshold was 0.8 and the new model's scores are uniformly lower,
the same threshold now cuts off too much. Recalibrate the model on a held-out
calibration split after every promotion, then re-tune the threshold band. Missing
recalibration is one of the most common silent production failures.

**Q: You are asked to add a new abuse category (for example, coordinated
inauthentic behavior) with no labeled examples. How do you start?**

A: Same weak-supervision bootstrap as before, but with extra care. Write a handful
of labeling functions based on behavioral heuristics (coordinated posting patterns,
account age, post timing). Use an isolation-forest outlier detector on the activity
sequence to surface anomalies as a noisy positive signal (this is how LinkedIn
bootstrapped abuse detection). Prompt an LLM to label a random sample. Combine
these signals into soft labels. Fine-tune the encoder on the soft-labeled set, then
launch the review queue to collect gold labels and retrain. The goal is a usable
model in days, not a perfect one from scratch.

**Q: Classification, NER, and entity resolution all use an encoder backbone. Can
they share one?**

A: Sometimes. Sharing the encoder saves memory and can improve generalization when
the tasks are on the same domain text. The risk is task interference: a
classification fine-tuning objective can push the encoder's representations away
from what the NER head needs, especially if the tasks have different token-level vs
sequence-level supervision signals. Test shared-backbone vs separate fine-tuning
on a held-out validation set for each task. In practice, many teams fine-tune a
shared backbone jointly (multi-task learning), then add task-specific heads, which
often outperforms single-task fine-tuning while sharing compute.

## Commonly answered wrong (the traps)

**Q: The spam class is 0.5% of traffic. What accuracy should the model hit?**

A: Accuracy is the wrong metric here entirely, and naming it without caveats is
the mistake. A classifier that predicts "not spam" for everything achieves 99.5%
accuracy. The useful metric is F1 (or precision and recall separately) on the spam
class, the area under the precision-recall curve, and the operating point on that
curve that the product's cost structure implies. Stating this clearly and then
giving a target F1 on the positive class is the correct answer.

**Q: Should the calibration temperature be applied at serving time to each
inference?**

A: Yes, temperature is applied at serving time, but it is fitted (a single scalar
or a simple function) on a held-out calibration set during model development,
not refit at serving time. The common wrong answer is "we just pick a threshold on
the raw logits." A raw logit is not a probability; the decision to route to review
or auto-act should rest on a calibrated probability, not a threshold tuned against
an uncalibrated score that will shift with every retrain.

**Q: We have very little labeled data, so we should use a general-purpose LLM
instead of training a specialized model.**

A: Partially right but ultimately backward. When you have almost no labels, a
zero-shot LLM is a correct starting point: it gives you a baseline and lets you
identify the hard cases. But it is not the end state. The LLM generates labels for
your domain; those labels train a small encoder; the encoder runs inline at
millions/day scale; the LLM handles the tail. Treating the LLM as the production
system because the encoder has no labels yet is confusing the bootstrapping phase
with the final design.

**Q: For abuse detection, should we build a single large general model or many
narrow specialized classifiers?**

A: The mature answer is both, in a pipeline: a fast general classifier on the
inline path (catches the bulk of obvious cases cheaply) plus specialized narrow
models for high-recall abuse categories that the general model misses (coordinated
inauthentic behavior, hate speech targeting a specific demographic). Narrower
models with better-curated labels reliably outperform general models on the
specific category they target. The wrong answer is picking one side exclusively.
