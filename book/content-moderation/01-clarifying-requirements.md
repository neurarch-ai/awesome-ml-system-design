# 1. Clarifying the requirements

Before sketching any model, pin down what the system must actually do. Every question
below either removes work, changes the design, or shifts the operating point.

**Candidate:** Which harms are in scope, and are they equally severe?
**Interviewer:** CSAM, terrorism, graphic violence, hate speech, harassment, self-harm,
spam, and scams. They are not equally severe. CSAM and terrorism are legally mandatory
to remove and often to report to authorities. Spam removal has a much lower bar.

**Candidate:** So each harm class gets its own operating point?
**Interviewer:** Correct. Don't propose one "badness" score that tries to cover all of
them.

**Candidate:** Which content modalities are in scope?
**Interviewer:** Text posts and comments, images, video (frames plus audio), and live
voice chat. Treat them all, and assume the mix shifts over time.

**Candidate:** Should we block before content reaches an audience, or act after it
spreads?
**Interviewer:** Both. Synchronous scoring at post-time for most content. Async re-scoring
triggered by virality or user reports. Live voice chat gets its own near-real-time path.

**Candidate:** What latency budget applies where?
**Interviewer:** Text at post time can tolerate a few hundred milliseconds. Uploaded video
goes async. Live voice needs sub-second on a rolling window with no pre-publish gate.

**Candidate:** What is the enforcement surface? Does the model's score directly delete
things?
**Interviewer:** No. The model outputs a risk score. A policy engine turns that score
into an action: auto-remove, auto-allow, age-gate, add a warning interstitial, downrank,
or route to a human review queue. The model and the policy engine are separate.

**Candidate:** Do we have existing human review infrastructure?
**Interviewer:** Yes. Assume a team of trained reviewers and a review platform that can
record decisions. Reviewer decisions feed back as training labels.

**Candidate:** What scale are we designing for?
**Interviewer:** Hundreds of millions of users posting across many languages. The cheap
fast path must handle billions of items per day; heavy models fire only on what the
cheap path passes through.

---

Let us summarize the problem statement. **We are designing a multi-policy, multi-modal
content moderation system.** It accepts text, image, video, and voice content, runs
per-policy risk classifiers, and routes each item to the right enforcement action via
a policy engine. Human reviewers handle the uncertain middle and their decisions flow
back as labels to retrain the classifiers. The system must operate against adversaries
who continuously adapt their evasion tactics.

Three consequences fall out of this immediately:

**1. One model per policy, not one model for "bad."** CSAM, spam, and hate speech
have different precision floors, different retraining cadences, and different legal
obligations. They must stay separate, even if they share a backbone encoder.

**2. The objective is recall at a fixed precision floor, not accuracy.** A miss that
allows illegal content to spread is categorically different from a false block that
annoys one user. The precision floor per policy is what separates them. Stating this
early is the clearest signal you have done this before.

**3. The human loop is the core of the system, not a fallback.** Reviewers produce
the gold labels that keep classifiers current against adversarial drift. Human review
capacity is the real ceiling on how much borderline content the system can handle, so
model precision and reviewer queue load are tightly coupled.
