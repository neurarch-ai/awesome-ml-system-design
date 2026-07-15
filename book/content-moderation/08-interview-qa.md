# 8. Interview Q&A

The questions an interviewer actually asks about content moderation, grouped by how
they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: What metric do you optimize for a content moderation classifier?**
A: Recall at a fixed precision floor per policy. You do not optimize accuracy (a
flag-nothing model scores 99.9 percent at 0.1 percent base rate) and you do not
optimize F1 blindly (it treats false positives and false negatives as equally costly,
which they are not). The precision floor is driven by the cost asymmetry: how severe
is a miss versus a false block on this policy? Set the floor for each policy, then
find the threshold that maximizes recall subject to meeting that floor. Report it per
policy, per modality, and per language.

**Q: Why one model per policy rather than one big classifier for all harm?**
A: Three reasons. First, the operating points differ by orders of magnitude: CSAM
requires near-perfect precision before any auto-action, spam can act at a much lower
floor. You cannot express that with one threshold on one output. Second, drift rates
differ: spam mutates weekly, nudity is relatively stable, so retraining cadences
differ per class. Third, accountability is per policy: each policy usually has an
owner who tunes its threshold against real appeal and miss data. A shared encoder
with per-policy heads is fine; the calibration and thresholds must stay separate.

**Q: What is the role of human reviewers in this system?**
A: Two roles, not one. The primary role is safety: humans review the borderline items
the model is uncertain about, high-severity items the system will not auto-action,
and appeals. The second role, equally important, is data production: every reviewer
decision is a gold label drawn from exactly the distribution the model finds hardest.
Feed those decisions back into retraining and the model improves where it was weak.
This is the flywheel. Reviewer capacity is therefore the real ceiling on how much
borderline content the system can handle; model precision and queue load are tightly
coupled.

**Q: How do you handle CSAM specifically?**
A: Hash matching for known material, classifier to prioritize the queue for novel
material, but never auto-action on a classifier score alone. For known material,
perceptual hashing (PhotoDNA, Google CSAI Match) catches re-uploads robustly and
produces a near-zero false-positive signal that is legally actionable. For novel
content, AI classifiers assign priority scores so human reviewers can work the
worst cases first. Confirmed novel items are added to the shared hash database so
the next occurrence is caught upstream for free. The false-positive cost of wrong
auto-removal is legally unacceptable, which is why humans always review.

**Q: How do you moderate live voice chat?**
A: Rolling-window streaming inference. You score a rolling 10-to-15-second window
of audio at sub-100ms latency using a distilled, quantized audio model. There is no
pre-publish gate for live voice, so you act during the conversation rather than before.
A consequence model (warning, muting, session termination) fires on a flagged window.
The Roblox approach: bootstrap machine labels by running Whisper transcriptions through
the existing text-filter ensemble, then distill a small WavLM-based audio student
(approximately 48M parameters) that infers directly from audio at serve time, serving
over 2,000 requests per second at 50ms latency.

## Tricky

**Q: A classifier's flag rate drops 30 percent overnight. What happened?**
A: Two possibilities: a genuine drop in harm, or a successful new evasion pattern.
Do not assume either without investigating. Check whether confirmed violations (from
human review) also dropped proportionally. If classifier flags dropped but confirmed
violations held steady or are unknown, suspect evasion. Red-team the drop: ask what
text or image perturbation an adversary would use to defeat the current model, and
see if that matches the new traffic. If it is evasion, augment training data with
the new pattern and fast-retrain. Meanwhile, add hash or rule coverage for known
instances of the evasion technique.

**Q: How do you handle the "hateful memes" problem where neither the image nor the
caption is harmful alone?**
A: You need a joint vision-language model that reasons over image and text together.
Running a text classifier and an image classifier independently and ORing the results
will pass cross-modal violations. A CLIP-style dual encoder maps both into a shared
space; an early-fusion model (ViLBERT, VisualBERT) adds cross-attention between
image patches and text tokens for stronger joint reasoning. The joint model is too
expensive to run on everything, so gate it behind cheap unimodal pre-filters and
invoke it only when both modalities are present and the unimodal signals are
ambiguous or conflicting.

**Q: How do you handle borderline content like satire, counter-speech, or news
reporting of violence?**
A: These cases are where context decides meaning and naive classifiers fail hardest.
Do not auto-action them. The options are: soft enforcement (an interstitial warning
rather than removal, an age-gate, downranking in distribution), pre-post nudge
(ask the user to reconsider before posting, as Nextdoor does), or routing to a
human reviewer with context flags. A restored appeal on borderline content is a
confirmed false positive and a free label for retraining. Fast appeals on borderline
content are both a fairness mechanism and a data pipeline.

**Q: How do you prevent over-censorship from becoming its own harm?**
A: Several levers. Keep auto-action thresholds high so only the confident tail is
auto-actioned. Treat borderline content with soft enforcement instead of removal.
Make appeals fast and cheap (each appeal is also a label source). Monitor the
appeal-overturn rate as a live false-positive signal. Report false-block rates per
demographic, per language, and per content type to catch disparate impact. Suppressing
counter-speech, news coverage, or marginalized voices at scale is a real harm, not
a rounding error.

**Q: Why does a perceptual hash catch a re-uploaded violating image that a
cryptographic hash misses, and what is it actually comparing?**
A: A cryptographic hash (SHA-256) is avalanche by design: flip one bit of the file
and the entire digest changes, so re-saving at a different JPEG quality, cropping a
pixel, or adding a watermark produces a totally different hash and the match is
lost. A perceptual hash like PDQ (Meta) instead hashes the image's visual content:
it downscales to a fixed grid, takes a frequency-domain (DCT) transform of the
luminance, and thresholds the low-frequency coefficients into a 256-bit string, so
two visually similar images produce hashes that differ in only a few bits. Matching
is then a Hamming-distance threshold, not exact equality, which is why PDQ survives
re-encoding and mild edits while remaining near-zero false positive and needing no
per-image training. That robustness-to-edits property is exactly what a
cryptographic hash lacks.

## Commonly answered wrong

**Q: You have a new harm policy. You want a threshold. How do you set it?**
Common wrong answer: "pick 0.5" or "pick the threshold that maximizes F1."
Correct answer: label a representative sample of content covering the harm class plus
realistic negatives. Plot the precision-recall curve on a held-out calibration set.
Set the precision floor from the cost of a false positive on this policy. Take
the operating threshold as the rightmost point on the PR curve that meets that floor.
Reserve auto-action for the confident tail (high-precision region); route everything
else to humans. 0.5 is not the right answer unless the class is balanced and costs
are symmetric, which they never are in content moderation.

**Q: Should you train one model with all harm classes as labels, or separate models?**
Common wrong answer: "one big multi-label model for efficiency."
Correct answer: a shared encoder with per-policy heads is efficient and acceptable,
but the per-policy heads must have independent calibration and independent thresholds.
The reason is not architectural taste; it is that different policies have different
drift rates, different retraining cadences, and wildly different operating points.
A single threshold on a single output cannot serve them. If one policy needs
weekly retraining and another only monthly, a monolithic model forces the worst
cadence on every policy or lets the faster-drifting policy fall behind.

**Q: Can you just run a CSAM classifier and auto-action on a high score?**
Common wrong answer: "yes, if the classifier is good enough."
Correct answer: no. For CSAM specifically, you do not auto-action on a classifier
score alone. The false-positive cost is legally unacceptable: you would be accusing
a real user of the most serious crime. The correct split is: hash matching catches
known material at near-zero false positive and is legally actionable. For novel
content, the classifier assigns priority to the human review queue. Human reviewers
make the call. Only confirmed material gets added to the hash set and auto-actioned
on future re-uploads.

**Q: How do you measure whether the system is getting better over time?**
Common wrong answer: "watch the total number of policy violations removed."
Correct answer: you need a random audit stream, labeled independently of whether
the model flagged anything, to estimate true recall and precision on the full
distribution. If you only measure quality on the hard cases you routed to humans,
you lose visibility into how the model performs on the easy majority. Also track
the appeal-overturn rate as a live false-positive signal, the flag rate per policy
over time for evasion detection, and time-to-action on severe harms as a latency
bound on harm spread.

**Q: The classifier is calibrated, so a 0.9 score means about 90 percent of those
items are violations. Safe to auto-action the whole 0.9-plus band, right?**
A: Not safe, because calibration is a property of the distribution it was fit on,
not a per-item guarantee that travels. Platt or isotonic calibration makes
$\hat{p}$ match the empirical violation rate on the calibration holdout; the moment
the incoming traffic shifts, and in moderation it shifts constantly because there
is an adversary steering it, the base rate and the score-to-rate mapping both move,
so the 0.9 bucket may now be 60 percent violations. Two mechanisms drive this: base
-rate shift (calibration is sensitive to prior prevalence) and adversarial drift
(new evasions cluster at specific scores). That is why moderation recalibrates on a
recent holdout after every retrain and monitors appeal-overturn rate as a live
check, rather than trusting a once-measured calibration curve. Calibration justifies
a threshold; it does not license "set and forget" auto-action.
