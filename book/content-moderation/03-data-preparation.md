# 3. Data preparation

## Where labels come from

Content moderation labels are expensive and contested. Unlike recommendation data
where a click is a ground-truth signal, harm is a judgment call that reasonable people
sometimes disagree about. The label pipeline has three sources, and mixing them
deliberately is part of the design:

**Human reviewer decisions.** The highest-quality labels. Reviewers see the uncertain
middle the model routes to them, so these labels are drawn from exactly the
distribution the model finds hardest. That is the flywheel: routing the hard cases to
humans and feeding their decisions back into training makes the model improve precisely
where it was weak.

**User reports.** Community-sourced signals about content users found harmful. Reports
are noisier (motivated by convenience or malice as well as genuine harm) and skewed
toward high-reach content. They are valuable as a detection trigger but should not be
used as raw gold labels without adjudication.

**Proactive audit samples.** A random sample of content pulled independently of whether
a model or user flagged it. This is the only way to measure true precision and recall
on the full distribution. Without an independent audit stream, you can only measure
quality on the hard tail you routed to humans, which flatters the model's apparent
recall on easy cases.

## Multi-grader consensus and label quality

Borderline content is genuinely ambiguous. A single reviewer label is often wrong.
The industry standard is to route difficult items to multiple graders and require
consensus or majority vote. Roblox's rule is 80 percent inter-rater agreement across
at least three reviewers before an item becomes training data. Below that threshold,
send to adjudication (a senior reviewer or policy owner resolves the disagreement),
not to the training set.

Track reviewer agreement rates over time. Falling agreement is a signal either that
the policy boundary has become unclear (needs a policy update) or that new attack
patterns have arrived that reviewers themselves have not seen before.

## Class imbalance

Every content moderation dataset is severely imbalanced. CSAM may be one in ten
million items at a large platform. Spam may be a few percent. Nudity violating policy
may be a fraction of a percent.

| Imbalance handling | When to use | Watch out for |
|---|---|---|
| Oversample positives (SMOTE or simple repeat) | moderate imbalance (1:100 to 1:1000) | introduces duplicates that can overfit the classifier to specific positive examples |
| Undersample negatives | when the training loop cannot handle very large negative sets | loses information from the majority class; never undersample your eval set |
| Class-weighted loss | imbalance up to roughly 1:10000, especially for transformer fine-tuning | weight must be tuned; too high and the model becomes over-sensitive |
| Hard-negative mining | you have many easy negatives and boundary cases decide quality | mining can introduce selection bias toward adversarial boundary examples |
| Separate positive mining from audit negatives | when you have a natural positive set from reports/review plus a large random negative pool | keep the two sources identified so you can rebalance at eval time |

The metric consequence: on a 0.1 percent positive rate (Bumble's base rate for lewd
images), a model that flags nothing reaches 99.9 percent accuracy. Accuracy is useless
here. Always report recall at a fixed precision floor, read off the precision-recall
curve at the operating threshold.

## Labeling for multimodal data

Text is the easiest to label: show the reviewer the text, ask if it violates policy.
Image labeling adds reviewer-safety concerns: reviewers must see harmful imagery to
label it. Pinterest's Pinqueue platform introduced "Kitty Mode," swapping sensitive
images for pictures of cats in the review UI while still persisting the original for
enforcement, so a reviewer can label a decision audit trail without a screenshot
re-exposing the content.

For joint image-text items (memes, posts with captions), reviewers must see both and
be told to assess the combination, not each piece individually. A reviewer who judges
image and caption separately will mislabel most cross-modal violations.

## Adversarial drift in training data

The threat model is non-stationary: adversaries probe the model boundary and mutate
content the moment a pattern gets blocked. This means the training distribution shifts
continuously.

**Adversarial augmentation during training.** For text, augment positives with known
obfuscation patterns: leetspeak (3 for e, 0 for o), homoglyphs (Cyrillic a for Latin a),
deliberate misspellings, zero-width characters inserted between letters, and code-switching
between languages within a sentence. For images, augment with border additions, crops,
color jitter, and re-encoding artifacts. The goal is to make the model robust to
perturbations the model has never seen but that follow the patterns of known attacks.

**Continuous retraining.** The most important process defense. Fresh human labels from
the review queue, delivered to retraining within days rather than months, keep the
decision boundary tracking the attack. A model frozen at training time decays
continuously. Cadence depends on harm class: spam requires near-weekly retraining;
nudity can tolerate monthly.

**Distribution shift monitoring.** A sudden drop in a policy's flag rate is as likely
to be a successful new evasion as a genuine drop in harm. Alert on both directions.
A good monitor is a holdout of labeled items with known ground truth; if the model's
recall on that holdout drops without a corresponding drop in human-reviewed confirmed
harms, an evasion pattern has arrived.

**When to use which data strategy.**

| Reach for | When | Instead of |
|---|---|---|
| Multi-grader consensus with adjudication | borderline content where a single reviewer label is unreliable | single-reviewer labels for hard cases |
| Random audit sample as a separate eval set | you need true recall and precision on the full distribution | evaluating only on the uncertain middle routed to humans |
| Adversarial augmentation in training | the attack surface includes obfuscation and re-encoding | training only on clean originals and hoping the model generalizes |
| Near-weekly retraining on fresh labels | a harm class that mutates fast (spam, coordinated campaigns) | batch retraining once a quarter, which lets the boundary drift |
| Proxy labels bootstrapped from existing signals | you need labels fast and hand-labeling at scale would take months | waiting for hand-labels before shipping any version of the model |
