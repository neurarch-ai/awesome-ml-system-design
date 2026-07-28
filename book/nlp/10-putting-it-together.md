# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable text classifier, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing encoders before classifying a single message.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Model checkpoints
change yearly, but the interface of each stage (normalize, tokenize, label,
baseline, fine-tune, calibrate, serve, close the loop) does not, so pick per
stage by interface and treat any specific checkpoint as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Language ID + normalization | fastText language ID (under 1 ms) first, then Unicode, casing, and homoglyph normalization before anything touches a tokenizer | Single-language corpus with no adversarial users: language ID can wait, normalization cannot | [3](03-data-preparation.md) |
| Tokenization | Subword (WordPiece / SentencePiece), pinned byte-identical between training and serving | Multilingual with high-fertility scripts: measure tokens per word per language and budget latency from day one | [3](03-data-preparation.md) |
| Labeling | Product logs for the head, weak supervision (labeling functions + LLM annotator) for the rare classes, active learning from the review queue thereafter | Labels are already abundant for every task: fine-tune directly and skip the bootstrap | [3](03-data-preparation.md) |
| Baseline | Majority-class predictor, then TF-IDF + logistic regression, then LLM zero-shot, all scored before any fine-tuning | Never. The baseline ladder is what tells you the encoder earned its complexity | [2](02-frame-as-ml-task.md), [4](04-model-development.md) |
| Inline model | Distilled encoder (DistilBERT / MiniLM class) with thin task heads on one shared backbone | Distillation's accuracy loss is measurable on your task: full BERT-base inside a 50 ms budget | [4](04-model-development.md) |
| Imbalance handling | Class-weighted cross-entropy, per-class cost-aware thresholds, hard-negative mining once easy negatives stop teaching | Classes are roughly balanced: uniform loss and a single threshold are fine | [3](03-data-preparation.md), [4](04-model-development.md) |
| Evaluation | Per-class F1 and PR curves sliced per language, span F1 for extraction, time-based splits | Never report aggregate accuracy on an imbalanced task; a majority-class predictor maximizes it | [5](05-evaluation.md) |
| Calibration | Temperature scaling on a held-out calibration split, refit on every promotion | Calibration error stays large after temperature scaling: Platt or isotonic | [5](05-evaluation.md) |
| Serving | Inline/offline split with a confidence-gated review queue whose verdicts flow back as labels | No safety task and no label scarcity: the review band can start narrow | [6](06-serving-and-scaling.md) |

The baseline row is the one beginners skip and regret: without the majority-class
number and a linear model score, every encoder result is uninterpretable, and you
cannot tell whether fine-tuning helped or the task was easy. One afternoon of
baselining pays for itself the first time a "94% accurate" model turns out to be
two points above predicting the majority class.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
support-message pipeline handling a few million messages per day. Route each
ticket to the right queue, extract product name, date, and contact reason, and
hold back abusive messages before a human reads them. Routing and abuse
detection run inline in tens of milliseconds; extraction may lag up to a second.
English first with Spanish and Portuguese next quarter, a few thousand routing
labels, fewer than five hundred confirmed abuse positives, no extraction labels
at all, and asymmetric safety errors in both directions. Here is the whole
system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Task split | Three heads, not one model: classification (routing), token tagging (extraction), imbalance-aware classification (abuse) | "NLP system" is separate problems; one model over-designs the easy tasks and under-designs the safety task |
| Backbone | One multilingual distilled encoder, shared, tokenized once per message | The Spanish/Portuguese roadmap rules out a per-language stack; heads are thin linear layers, so the encoder dominates cost and sharing it amortizes everything |
| Inline model size | Distilled encoder, ~6 ms per forward pass | Tens-of-milliseconds budget at millions per day; BERT-base (~10 ms) is the fallback only if distillation's accuracy loss is measurable |
| Routing labels | The few thousand historical tickets, then active learning from the review queue | Every review decision is a fresh label at zero marginal cost |
| Abuse labels | Weak supervision: keyword and regex labeling functions plus an LLM annotator, combined into soft labels and distilled into the encoder | Fewer than 500 positives cannot fine-tune anything alone; labeling functions vote over the unlabeled firehose for free |
| Extraction | Regex and heuristic weak labels (dates, order numbers) bootstrap the tagging head; runs async up to a second | No labels exist, and the looser latency budget tolerates longer sequences |
| Loss | Class-weighted cross-entropy on abuse; threshold tuned with beta above 1 while tracking false blocks | The positive class is well under one percent; a miss reaches staff and customers, but a false block silences an innocent user, so both directions are gated |
| Metrics | Per-class F1 and PR curves sliced per language; strict span F1 for extraction | Accuracy is maximized by predicting the majority class; a global number hides a broken language |
| Calibration + gate | Temperature scaling, then the band: auto-act above the high threshold, review the uncertain middle, auto-allow the confident bottom; recalibrate on every promotion | A threshold is a promise about the score scale, and the score scale is exactly what a retrain does not preserve |
| LLM role | Offline only: label factory, hard-tail fallback for abstentions, zero-shot baseline while labels accumulate | 100 to 300 times slower and orders of magnitude more expensive per call than the encoder; at millions of calls per day the math is not close |

**Label arithmetic.** Abuse prevalence is about 0.3% of traffic, so at 3 million
messages per day roughly 9,000 abusive messages pass through daily
(illustrative). Finding positives by random sampling means screening ~330
messages per hit, so the ~500-positive launch set already represents about 165k
screened messages; scaling that by hand does not work, which is why the
labeling functions vote over the firehose for free and the LLM annotator labels
a bootstrap sample once. After launch, a review queue receiving 0.5% of traffic
sees ~15,000 items per day (illustrative); even a modest reviewed fraction
returns more confirmed positives per week than the entire launch set, which is
why [section 6](06-serving-and-scaling.md) treats the review loop as the label
factory, not a fallback.

**Class balance.** At 0.3% prevalence, a model that predicts "not abuse" for
everything is 99.7% accurate and catches nothing, which is why accuracy never
appears on the abuse dashboard. Inverse-frequency class weighting puts roughly
330x more gradient on each positive example (illustrative), and the numbers
watched instead are per-class F1 on the positive class and the false block rate
on a sampled slice of auto-blocked messages, the release gate from
[section 5](05-evaluation.md).

**Latency.** The inline path spends under 1 ms on language ID, ~1 ms on
normalization and tokenization, ~6 ms on the shared distilled-encoder forward
pass, and sub-millisecond on the two thin heads plus temperature scaling: call
it 8 to 10 ms per message, inside the tens-of-milliseconds budget with headroom
(illustrative breakdown, model figures from
[section 4](04-model-development.md)). Three million messages per day averages
~35 QPS, call peak ~100 QPS (illustrative), which a handful of commodity
replicas absorb. The async extraction path can afford BERT-base and longer
sequence lengths because nothing user-facing waits on it.

**Cost, baseline vs transformer.** The distilled encoder serves for a fraction
of a cent per thousand calls; at an illustrative \$0.002 per thousand, the whole
3M/day inline portfolio costs about \$6 per day, because the shared backbone
means routing and abuse ride one forward pass. A decoder LLM on the same path
at an illustrative \$0.50 per thousand calls would run \$1,500 per day per task,
250x the price, while blowing the latency budget by two orders of magnitude.
The same LLM used offline as an annotator labels a 50k-example bootstrap set
for about \$25 once (illustrative). That asymmetry is the chapter's core
economic argument: pay the LLM per training example, never per inference.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: abuse recall measured immediately after each
promotion (the first retrain shifts the score distribution, and a stale
threshold silently under-acts; the recalibrate-on-promotion rule from
[section 5](05-evaluation.md) exists because this fails quietly), the false
block rate on sampled auto-blocked messages (a cohort with a distinct writing
style gets over-blocked and files complaints while global F1 looks fine), and
review-queue depth (weak-supervision noise plus adversarial drift push
uncertain volume up until reviewers saturate, and the queue is the label supply
for every retrain, so saturation starves the loop).

## The same techniques under different constraints

The review question that matters in practice is not "which encoder is best" but
"which encoder is best under my constraints." Here is the same pipeline built
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Startup helpdesk bot | Support pipeline (this chapter) | Batch listing enrichment |
|---|---|---|---|
| Traffic / labels | Hundreds of tickets/day; ~200 hand labels | A few million/day; thousands routing, <500 abuse positives | Whole corpus reprocessed on a batch cadence; ~30K taxonomy-mapped spans |
| Latency budget | Seconds are fine | Tens of ms inline; ~1 s async extraction | None; throughput and coverage only |
| Model | TF-IDF + logistic regression, or LLM zero-shot inline: at ~0.005 QPS the per-call price is irrelevant | Shared distilled encoder + thin heads | Full BERT-class NER and scorer; batch buys back the model size |
| Labels | Hand-label a few hundred; active learning once volume justifies it | Weak supervision + LLM annotator + review-queue loop | Taxonomy-mapped labels; entity resolution via bi-encoder + ANN against the catalog |
| Imbalance | Mostly moot; classes are product queues | Class-weighted loss, per-class thresholds, hard negatives | The problem is coverage, not imbalance: entities outside the taxonomy |
| Eval | A golden set of a few hundred lines, maintained by hand | Per-class F1 per language; false-block rate as release gate | Strict span F1 + taxonomy coverage; human audit sample |
| Serving | Single process; no review queue yet | Inline/offline split, confidence gate, nightly-to-weekly retrain | Batch pipeline; results land in a store, no confidence gate on a hot path |
| What would be over-engineering | Distillation, calibration bands, weak supervision | An LLM inline, or a separate model stack per language | A tens-of-ms serving stack, distillation, streaming anything |

Two lessons fall out. First, the startup column inverts the chapter's loudest
rule: the prohibition on inline LLMs is a volume argument, so at hundreds of
calls per day the zero-shot LLM is the correct production system while labels
accumulate, exactly the bootstrapping role [section 2](02-frame-as-ml-task.md)
assigns it, and the fine-tuned encoder is the thing it would be premature to
build. Second, the batch column shows what removing the latency constraint
buys: model size stops mattering, so capacity goes to accuracy and coverage
(full BERT-class scorers, longer contexts), and the operational risk moves from
threshold drift to taxonomy staleness.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any checkpoints.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Inline latency budget | Model size on the hot path | Tens of ms: distilled encoder. Up to 50 ms: BERT-base. Seconds or batch: size stops being the constraint |
| Daily volume | Where the LLM sits | Millions/day: LLM offline only (labels, tail). Hundreds/day: zero-shot LLM inline is fine while labels accumulate |
| Label supply | Supervision strategy | Thousands: fine-tune directly. Hundreds: weak supervision + LLM annotator, distill. Zero: LLM zero-shot baseline first |
| Positive-class prevalence | Loss, threshold, metric | Under ~1%: class-weighted loss, per-class cost-aware threshold, per-class F1 and PR curves; accuracy never |
| Error asymmetry | Beta and review-band width | Misses costlier: beta > 1 and a wide review band. False corrections costlier: beta < 1 (the F0.5 case) |
| Language count | Encoder family and eval slicing | Two or more languages: multilingual encoder, slice every metric per language, budget tokenizer fertility per script |
| Taxonomy churn | Fine-tune vs prompt | Label set changes weekly: prompted LLM until it stabilizes. Stable set: encoder + retrain cadence |
| Output structure | Architecture class | Fixed label: encoder + classification head. Spans: token-tagging head. Taxonomy match: bi-encoder + ANN. Generated text: seq2seq |

## The smallest runnable text classifier

The review of every framework tutorial is the same: the reader downloads a
checkpoint and still cannot see the pipeline. So here is the entire
train-and-classify path in one file with zero installs. Every production
component is swapped for the smallest thing with the same interface: the ticket
stream becomes a seeded toy corpus, the pretrained encoder becomes raw word
counts, fine-tuning becomes counting, the softmax head becomes Bayes' rule with
add-one smoothing, and the eval harness becomes a held-out slice scored against
a majority-class baseline. The shape is the lesson; every section of this
chapter upgrades one function of this file.

```python
"""The whole train-and-classify path in one file, runnable with no installs."""
import math, random
from collections import Counter

random.seed(7)

# --- a seeded toy corpus of two support intents -----------------------------

BILLING  = ["refund", "charge", "invoice", "payment", "card", "subscription", "price"]
SHIPPING = ["package", "tracking", "delivery", "shipment", "address", "carrier", "customs"]
FILLER   = ["my", "order", "please", "help", "need", "about", "the", "status", "update", "when"]

def make_line(topic_vocab):
    """Two topic words plus three filler words; production: a real ticket."""
    words = random.sample(topic_vocab, 2) + random.sample(FILLER, 3)
    random.shuffle(words)
    return " ".join(words)

data = [(make_line(BILLING), "billing") for _ in range(40)] \
     + [(make_line(SHIPPING), "shipping") for _ in range(40)]
random.shuffle(data)
train, test = data[:60], data[60:]                 # held-out split, same vocabulary

# --- training: count words per class, add-one smoothing ---------------------

counts = {c: Counter() for c in ("billing", "shipping")}   # word counts per class
prior  = Counter()                                          # documents per class
for text, label in train:
    prior[label] += 1
    counts[label].update(text.split())
vocab = {w for c in counts for w in counts[c]}

def predict(text):
    """argmax_c log P(c) + sum_w log P(w|c); production: a fine-tuned encoder."""
    scores = {}
    for c in counts:
        total = sum(counts[c].values())
        score = math.log(prior[c] / sum(prior.values()))
        for w in text.split():
            score += math.log((counts[c][w] + 1) / (total + len(vocab)))  # add-one
        scores[c] = score
    return max(scores, key=scores.get)

def accuracy(pairs):
    return sum(1 for text, label in pairs if predict(text) == label) / len(pairs)

# --- lesson 1: the strong baseline ------------------------------------------

majority = prior.most_common(1)[0][0]
majority_acc = sum(1 for _, l in test if l == majority) / len(test)
print(f"held-out accuracy : {accuracy(test):.2f}  ({len(train)} train / {len(test)} test lines)")
print(f"majority baseline : {majority_acc:.2f}  (always predict '{majority}')")

# --- lesson 2: the out-of-vocabulary cliff ----------------------------------

BILLING_SYN  = ["reimburse", "overcharged", "fee", "receipt", "billed"]
SHIPPING_SYN = ["parcel", "courier", "dispatch", "postage", "consignment"]
synonym_test = [(make_line(BILLING_SYN), "billing") for _ in range(20)] \
             + [(make_line(SHIPPING_SYN), "shipping") for _ in range(20)]
oov = {w for t, _ in synonym_test for w in t.split()} - vocab
print(f"synonym accuracy  : {accuracy(synonym_test):.2f}  "
      f"({len(oov)} topic words never seen in training)")
```

Run it and the three printed lines demonstrate the chapter's opening and
closing arguments in about sixty lines. Held-out accuracy is 1.00 against a
majority baseline of 0.50: on in-vocabulary text, counting words is a strong
baseline, which is exactly why [section 4](04-model-development.md) demands the
baseline ladder before any fine-tuning; an encoder that cannot beat this
number has earned nothing. Then the synonym set, where customers write "parcel"
and "reimburse" instead of "package" and "refund," collapses accuracy to 0.65,
barely above the coin flip, with all ten topic words out of vocabulary:
add-one smoothing keeps the math from crashing on unseen words but carries no
notion that a parcel is a package. That cliff is the precise motivation for the
climb this chapter teaches: subword tokenization so unseen words still split
into known pieces ([section 3](03-data-preparation.md)), then a pretrained
encoder whose representations already place synonyms near each other before
your first label arrives ([section 4](04-model-development.md)). Swap the
counting for that encoder, the held-out slice for per-class F1 sliced per
language, and the print statements for a calibrated confidence gate, and you
have rebuilt this chapter.
