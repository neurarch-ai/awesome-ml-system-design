# 1. Clarifying the requirements

Before drawing any architecture, pin down what the system must do. Every question
here either removes work, changes the model, or changes the failure mode the
system needs to defend against.

## The dialogue

**Candidate:** The prompt mentions tagging, moderation, and visual search. Are
all three in scope, and do they share infrastructure?
**Interviewer:** Yes to all three. Assume they share an ingest pipeline and
ideally a shared backbone, but each has its own product latency and quality bar.

**Candidate:** Real-time or batch for each? A moderation miss that lets illegal
content publish is a legal event, which is very different from a tagging error.
**Interviewer:** Moderation must gate the publish action in real time, so it is
on the critical path. Tagging can be async. Visual search index-build is offline;
the query is online.

**Candidate:** What is the harm taxonomy for moderation? Each harm class
typically has its own precision and recall target and possibly a mandatory
reporting obligation.
**Interviewer:** Start with nudity, weapons, and off-marketplace items. Assume
each class has an independent operating point.

**Candidate:** Do we have labeled data, or are we cold-starting?
**Interviewer:** There is a small labeled set for room types. Nothing for moderation
yet. The search index can start with zero labels if we use a pretrained embedding.

**Candidate:** How large is the catalog, and what is the upload volume?
**Interviewer:** About 80 million photos in the catalog today. Around 5 million
new photos per day at peak.

**Candidate:** For moderation, what happens if the model is slow or unavailable?
**Interviewer:** Fail closed for the highest-harm classes (hold for human review).
Lower-harm classes can fail open if the queue is backed up.

**Candidate:** For visual search, how fresh must the index be? Can a newly
uploaded photo be searchable after a few minutes?
**Interviewer:** Yes, within minutes for new items. The user expects to search
their own upload immediately.

**Candidate:** Any on-device or edge constraints? Or is everything server-side?
**Interviewer:** All server-side for now.

## Scope and consequences

**Scope.** Real-time moderation gate (sync on the publish path), async multi-label
room-type tagging, offline-indexed visual search (query is online).

Two consequences fall out immediately, and stating them early is most of the signal
in this question:

**Consequence 1: task taxonomy, not one model.** "Tag, moderate, search" maps to
at least three distinct ML task types (multi-label classification, binary/multiclass
classification with high-recall operating points, and embedding retrieval). A single
model cannot serve all three without hurting each. State the task split before
proposing architecture.

**Consequence 2: labeling cost and GPU cost are the real budget lines.** You do not
know whether fine-tuning a detection head is warranted until you know the labeling
cost difference between an image-level tag and a bounding box. At 5 million uploads
per day, serving cost per million images is the number that determines whether a
ViT or a distilled EfficientNet sits on the moderation path.

## Summary

| Task | Latency | Metric | Label type | Cold start? |
|---|---|---|---|---|
| Multi-label room tagging | async batch | per-class precision and recall | image-level tags | small set available |
| Moderation (3 harm classes) | sync, p99 under a few hundred ms | recall at a fixed precision floor per class | binary per class | cold start, need labeling pipeline |
| Visual search (image query) | online query tens of ms, index offline | recall at k over a relevance set | no fixed classes needed | zero shot via CLIP-style backbone |

This table is the deliverable from requirements. Everything downstream follows
from these three rows.
