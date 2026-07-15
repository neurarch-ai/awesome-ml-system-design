# 6. Serving and scaling

## The inline/offline split

The architecture divides into two paths based on latency requirements:

- **The inline path** runs synchronously on every incoming message, within a
  tens-of-milliseconds budget. The model here must be small and fast. For
  classification and NER, this is a distilled encoder (DistilBERT, MiniLM, or
  BERT-base at most) with a linear head, served on commodity hardware.
- **The offline path** runs asynchronously, in batch, at any latency. This is
  where the LLM lives: generating training labels, handling hard-tail abstentions,
  retraining on fresh data, and auditing production decisions.

```mermaid
flowchart LR
  subgraph Inline["inline (tens of ms, per message)"]
    MSG["incoming message"] --> NORM2["normalize + language ID"]
    NORM2 --> ENC2["distilled encoder"]
    ENC2 --> HEAD2["task head (classify / tag)"]
    HEAD2 --> CAL2["calibrate + threshold"]
    CAL2 --> ACT2["auto-route / auto-block"]
    CAL2 --> QUEUE["review queue<br/>(uncertain / high-risk)"]
  end
  subgraph Offline["offline (batch, any latency)"]
    QUEUE --> LLM["LLM fallback<br/>(hard cases)"]
    QUEUE --> HUM2["human review"]
    HUM2 --> LABELS["new labels"]
    LLM --> LABELS
    LABELS --> RETRAIN["retrain encoder<br/>(nightly or weekly)"]
    RETRAIN --> ENC2
  end
```

**How it works.** The diagram splits into two subgraphs by latency budget. In the inline path, an incoming message is normalized and language-identified, passed through a distilled encoder and its task head to produce a raw score, then calibrated and thresholded. That calibration step is the fork: confident predictions are auto-routed or auto-blocked immediately, while uncertain or high-risk ones are diverted to a review queue instead of being acted on. In the offline path, the queue drains at any latency into an LLM fallback for hard cases and into human review, and both sources emit new labels. Those labels retrain the encoder on a nightly or weekly cadence, and the refreshed encoder feeds back into the inline path, so the fast synchronous path stays cheap while the slow asynchronous path supplies the training signal that keeps it accurate.

The loop closes on the right: every human review decision and every LLM fallback
verdict returns as a labeled training example. The encoder retrains periodically
(nightly or weekly depending on drift rate), and the calibration step reruns each
time.

## Shared encoder, multiple heads

A single fine-tuned encoder backbone can serve multiple task heads, which matters
for cost and latency. Text comes in, is tokenized once, and the same forward pass
feeds a classification head for routing, a token-tagging head for extraction, and
a calibrated output for abuse scoring. The heads are thin linear layers; the cost
is dominated by the encoder stack.

One caveat: a shared encoder fine-tuned for one task can drift from the
representation another task needs. When task-specific accuracy diverges, decouple
the heads with separate fine-tuning passes, or use task-adaptive pretraining
(continued pretraining on each task's domain text before head fine-tuning).

## The human review loop

The confidence gate is not a fallback; it is a designed component. The band
between auto-act and auto-allow is the leverage point: widen it to route more
items to review, narrow it to increase automation. For safety tasks, the band
should be wide (high recall of uncertain cases) even if it increases review volume.
Review capacity is then a staffing and tooling constraint, not a modeling one.

Every message the review queue receives is a labeling opportunity. A well-designed
review interface captures the reviewer's decision and feeds it directly back to
training data. At steady state, the inline model should push the confident band
outward over time as it sees more labels, narrowing the review queue and reducing
cost.

## Bottlenecks

| Bottleneck | First sign | Fix | Tradeoff |
|---|---|---|---|
| LLM on the inline path | latency and cost blow the budget | distilled encoder for inline tasks; LLM offline for labels and tail | more models to train and maintain |
| Label scarcity at launch | model stalls; tail classes are weak; F1 on rare class near zero | weak supervision, LLM annotation, active learning from review queue | noisy labels require cleanup and a gold-set audit |
| Class imbalance | high accuracy, near-zero recall on the positive class | class-weighted loss, resampling, hard-negative mining, per-class threshold | precision/recall must be traded via the threshold |
| Adversarial drift in abuse | abuse recall decays weekly; new evasion patterns appear | continuous fresh labels from the review queue, frequent retrain cadence | ongoing labeling and retrain infrastructure cost |
| Multilingual coverage | one language is much worse; global metric hides it | multilingual encoder (mBERT, XLM-R) with per-language eval and annotator pools | capacity diluted per language vs monolingual model |
| Calibration drift after retrain | thresholds over-act or under-act after model update | recalibrate on a held-out calibration split every time a new model promotes | one extra eval step per retrain |
| Review queue saturation | reviewers cannot keep up; latency rises | tighten the auto-act band (higher confidence threshold), prioritize queue by risk score | fewer items handled automatically |
| False blocks on safety tasks | innocent-user complaint rate rises | widen the review band, audit false-block rate on sampled auto-blocked messages | more review volume |

Two details worth pinning down. First, the multilingual-coverage fix leans on
multilingual encoders (mBERT is BERT (Google, 2018) pretrained over many languages
at once): a single shared subword vocabulary lets a low-resource language borrow
representation from high-resource ones, but that same shared capacity is why a
dominant language can crowd out a rare one, so per-language eval is not optional. The
mechanism to watch is that the global F1 is a traffic-weighted average, so a language
that is 2 percent of volume can collapse to near-zero recall while the headline number
barely moves. Second, calibration drift after retrain is not a bug in the new model but
a property of thresholding: a fresh model produces a different score distribution, so a
confidence gate tuned to the old distribution now acts at a different effective
operating point, which is why the recalibration step must run on every promotion, not
only when accuracy visibly regresses.
