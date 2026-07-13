# 7. How teams do it in production

Every large platform converges on the same skeleton: a hash-match gate for known-bad
material, per-policy classifiers for novel content, a policy engine that turns scores
into actions, a priority-ranked human review queue, and a label flywheel that feeds
reviewer decisions back into retraining. What actually differs is four decisions:
**which modality** the harm lives in, **whether enforcement is proactive or reactive**,
**how much the auto-action threshold trusts the model**, and **how the human loop is
staffed and tooled**. The architecture everyone shares; the leverage is in those four
choices.

## Where the real designs diverge

| System | Modality / harm | Proactive vs reactive | Auto-action stance | Human loop design | Why this shape |
|---|---|---|---|---|---|
| Roblox voice safety | Audio / voice: profanity, bullying, discrimination | Proactive on live voice rolling windows | Consequence model (warning, timeout, escalation) rather than hard block | Machine-labeled training; human evaluation only | No pre-publish gate for live voice; distilled model must hit sub-100ms |
| Roblox multimodal (scale) | Text, voice, image: all policies | Proactive across all surfaces | AI auto-enforces only where it beats humans on both precision and recall | Thousands of human experts for complex cases, appeals, red-team, golden-set curation | Ship AI only where measurably better; experts handle the rest |
| Pinterest hybrid scoring | Image + graph: six violation classes | Proactive (online model on fresh Pins) + batch daily (full corpus re-score) | Image-signature grouping for uniform enforcement across matching images | Pinqueue3.0: label platform where every reviewer decision is a training label | Need freshness for new Pins plus precision from graph features that require batch |
| LinkedIn fake accounts | Account metadata and behavior | Proactive at registration, reactive via cluster and activity models | Registration scoring blocks high risk immediately; medium risk challenged | Reports and manual investigation feed back to models | Lifecycle funnel: stop bulk fakes at signup, catch late-revealing fakes with cluster and activity signals |
| LinkedIn viral spam | Text + engagement signals | Proactive DNN at feed surfacing; reactive boosted-tree model on engagement cascade | Proactive can filter or escalate; reactive throttles before wide reach | Human review for escalated items | Two classifiers serve different failure modes: ingest-time misses and viral spread signals are independent |
| Bumble Private Detector | Image: lewd images in chat | Proactive at image send time | Blur-and-warn rather than hard block; recipient keeps control | No human queue for the model; iterative dataset expansion from production misclassifications | 0.1 percent base rate; blur-and-warn avoids appeal volume by preserving user agency |
| Meta hateful memes | Image + text jointly | (Research benchmark, not a described production system) | Benchmark, no stated auto-action threshold | Hard cases require joint reasoning; production would gate joint model behind cheap unimodal pre-filters | Neither modality alone detects cross-modal hate; joint early fusion wins |
| Google CSAM tools | Image and video: CSAM | Proactive for classifier prioritization; hash-match for known material | Classifier only prioritizes the queue; never auto-actions on a classifier score alone | Always human-reviewed; confirmed items grow the shared hash set | Legal: auto-action on a CSAM classifier is too high a false-positive cost; hash is legally actionable |
| Nextdoor kindness | Text: incivility in comments | Pre-post nudge (before the user submits) | No removal; a nudge asks the author to reconsider | Model fires the nudge; author decides; 1 in 5 edits or deletes the comment | Nudge avoids the appeal entirely when it works; preserves user agency |
| Slack invite spam | Text + metadata: spam invitations | Proactive at invite-send time | Auto-block; blocked invites log to a review channel rarely checked | Human review rarely needed; periodic audit of flagged invites | High-confidence spam at a channel with a clear proxy label; a false block is cheap |

## The systems (first-party write-ups)

- **Roblox** [Deploying ML for Voice Safety](https://about.roblox.com/newsroom/2024/07/deploying-ml-for-voice-safety): distilled WavLM audio classifier on rolling 15-second windows; machine-labeled training data bootstrapped from an existing text-filter ensemble; 50ms latency at 2,000-plus requests per second; 15.3 percent reduction in severe voice-abuse reports.

- **Roblox** [How Roblox Uses AI to Moderate Content on a Massive Scale](https://about.roblox.com/newsroom/2025/07/roblox-ai-moderation-massive-scale): 6.1 billion chat messages per day; 750,000-plus requests per second across all moderation surfaces (text, image, and voice combined, not chat messages alone); distilled per-policy text models plus GPU-based PII detection; deploy AI only where it beats humans on both precision and recall; thousands of human experts handle complex cases and appeals.

- **Pinterest** [Fighting misinformation, hate speech, and self-harm content with ML](https://medium.com/pinterest-engineering/how-pinterest-fights-misinformation-hate-speech-and-self-harm-content-with-machine-learning-1806b73b40ef): hybrid batch-online scoring for six violation classes; PinSage graph embeddings in the batch model; image-signature grouping for uniform enforcement; policy-violating reports per impression fell 52 percent.

- **Pinterest** [Pinqueue3.0, Pinterest's next-gen content moderation platform](https://medium.com/pinterest-engineering/introducing-pinqueue3-0-pinterests-next-gen-content-moderation-platform-fcfa972bf39c): object-abstraction review platform where every reviewer decision persists as a training label; JSON queue configs and template engine for self-service per-queue UI; Kitty Mode for reviewer safety; auditable who-decided-what history.

- **LinkedIn** [Automated Fake Account Detection at LinkedIn](https://www.linkedin.com/blog/engineering/trust-and-safety/automated-fake-account-detection-at-linkedin): registration-scoring plus cluster-level detection plus activity anomaly models; blocked five million accounts in under a day during one attack via the registration funnel.

- **LinkedIn** [Viral spam content detection at LinkedIn](https://www.linkedin.com/blog/engineering/trust-and-safety/viral-spam-content-detection-at-linkedin): proactive DNN at feed surface plus reactive boosted-tree model on engagement cascade; cut spam-content views 7.3 percent and policy-violating views 12 percent.

- **Bumble** [Open-sourcing Private Detector](https://medium.com/bumble-tech/bumble-inc-open-sources-private-detector-and-makes-another-step-towards-a-safer-internet-for-women-8e6cdb111d81): EfficientNetV2 binary classifier; hard-negative mining for arms and legs to hold down false positives; above 98 percent balanced precision and recall at 0.1 percent base rate; open-sourced under Apache 2.0.

- **Meta AI** [Hateful Memes Challenge and dataset](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/): 10,000-plus example multimodal benchmark; benign confounders defeat unimodal shortcutting; early-fusion ViLBERT wins but all models trail human performance; benchmark released to force joint image-text reasoning.

- **Google** [Child safety toolkit: Content Safety API and CSAI Match](https://protectingchildren.google/tools-for-partners/): AI classifier for novel CSAM prioritizes the human review queue; CSAI Match perceptual hashing for known material; local fingerprinting so only hashes, not content, leave the partner; confirmed items grow the shared hash database.

- **Nextdoor** [A feature to promote kindness in neighborhoods](https://blog.nextdoor.com/2019/09/18/announcing-our-new-feature-to-promote-kindness-in-neighborhoods): ML pre-post nudge developed with bias researcher Dr. Jennifer Eberhardt; 1 in 5 prompted users edited their comment; 20 percent fewer negative comments; deliberate engagement trade for healthier interactions.

- **Slack** [Blocking Slack Invite Spam With Machine Learning](https://slack.engineering/blocking-slack-invite-spam-with-machine-learning/): sparse logistic regression over roughly 60 million features; proxy label from team-level invite acceptance; proactive blocking at send time; false-block proxy dropped from 70 percent to 3 percent versus hand-tuned rules.
