# 1. Clarifying the requirements

Before designing anything, pin down which tasks actually need to be solved. Here
is a typical exchange. Notice that every question either eliminates work, splits
the design into separate pieces, or reveals a constraint that changes the choice
of model.

---

**Candidate:** The prompt says "NLP system." Can you name the specific tasks?
Routing, extraction, abuse detection, and translation are all NLP, but they need
different models, labels, and metrics.

**Interviewer:** Good catch. The immediate priorities are: route incoming support
tickets to the right queue, extract key fields (product name, date, contact
reason) from ticket text, and hold back abusive or policy-violating messages
before a human reads them.

**Candidate:** What is the latency budget for the inline path, and what is the
daily volume?

**Interviewer:** Routing and abuse detection run inline on every message as it
arrives: the budget is tens of milliseconds. Extraction can run slightly async,
up to a second. Volume is a few million messages per day.

**Candidate:** Are we in one language or many?

**Interviewer:** English first, but there is a roadmap to add Spanish and
Portuguese in the next quarter. Assume you cannot maintain a separate model stack
per language.

**Candidate:** How much labeled data do we have for each task?

**Interviewer:** For routing, a few thousand labeled tickets from the last six
months. For abuse detection, fewer than five hundred confirmed positives, because
the positive class is rare and expensive to label. Extraction has no labels yet.

**Candidate:** What is the cost of each error type? A misrouted ticket seems
recoverable; a missed abusive message or a false block on a legitimate one seems
more serious.

**Interviewer:** Correct. A misroute adds latency to resolution but a human fixes
it. A missed abusive message is visible to staff and customers. A false block on a
legitimate message silences the user and shows up as a complaint. Safety errors
are asymmetric and expensive in both directions.

---

Let us summarize the problem statement. **We are asked to design a portfolio of
NLP models for a support-message pipeline.** The tasks are text classification
(routing), named-entity extraction (field extraction), and toxicity/abuse
classification (content moderation). Latency for inline tasks is tens of
milliseconds, volume is millions per day, the starting language is English with
multilingual expansion coming, labeled data ranges from thin to nearly absent, and
error costs are asymmetric for safety tasks.

Two consequences follow from this immediately, and stating them at the start is
most of the interview signal here.

**Volume and latency rule out a large LLM on the hot path.** A large decoder model
takes hundreds of milliseconds per call and costs orders of magnitude more per
inference than a fine-tuned encoder. At millions of calls per day the math is not
close. The inline path needs a small, fast, calibratable model. The LLM's role, if
it has one, is offline: generating weak labels to bootstrap the encoder,
processing the long tail of hard cases the cheap model abstains on, and serving as
a zero-shot baseline while labeled data accumulates.

**"NLP system" is five different problems, and each needs its own model, label,
and metric.** A single model that does everything would be over-designed for most
tasks and under-designed for the safety task. The design separates them: a
classification head for routing and abuse detection, a token-tagging head for
field extraction, and a weak-supervision pipeline to bootstrap the abuse detector
where labels are scarce. Naming the separation before picking a model is the
second large signal.
