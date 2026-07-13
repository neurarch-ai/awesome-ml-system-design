# 8. Interview Q&A

The questions an interviewer actually asks about ads CTR prediction, grouped by
how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Why is calibration non-negotiable here but optional for ranking?**
A: Because the auction multiplies pCTR into a bid and derives a price from the
result. eCPM = bid times pCTR. A model with perfect AUC but 20% upward
calibration drift will rank ads correctly but will over-value every auction it
touches, mis-pricing every slot at every QPS. In pure ranking only the order
matters; advertisers are charged based on the number. AUC is blind to scale
shifts; the auction is not.

**Q: What does log loss buy you that AUC does not?**
A: Log loss is a proper scoring rule: it is minimized by the true probability, so
optimizing it simultaneously rewards calibration and ranking quality. AUC
measures rank order only and is invariant to any monotone scaling of the
scores. A model with inflated but consistent probabilities will score the same
AUC as a well-calibrated one while pricing the auction incorrectly. For a system
where the absolute probability sets money, you need log loss as the training
signal and calibration metrics at eval time.

**Q: Where do the model parameters actually live?**
A: In the embedding tables, not the top MLP. The MLP is small. Each sparse
categorical id maps to a row in a learned embedding matrix, and with hundreds of
millions of user ids and ad ids at even modest embedding dimension, the tables
reach billions of parameters and many gigabytes. This forces model parallelism
on the tables (sharded across hosts) while the dense MLP uses data parallelism.
Feature hashing into a fixed-size table bounds the memory but accepts controlled
collisions.

**Q: DeepFM, DCN, DLRM, Wide and Deep: how do they differ?**
A: All embed sparse features; they differ in how interactions are modeled.
Wide and Deep: a linear memorization branch over hand-crafted crossed features
in parallel with a deep MLP. DeepFM: replaces the hand-crafted wide side with an
FM component (pairwise dot products via latent vectors) sharing one embedding
layer with a deep MLP. DCN V2: stacks explicit bounded-degree cross layers beside
an MLP, with a mixture-of-low-rank variant for efficiency. DLRM: explicit
pairwise dot products between all embedding vectors and the dense bottom MLP
output, feeding a top MLP.

**Q: How do you handle clicks that may still convert later?**
A: Treat them as unresolved, not confirmed negatives. Concrete options: (1) a
bounded attribution window of $W$ hours before finalizing the label; (2) a
fake-negative weighted loss (Twitter) that assigns lower weight to samples that
are still plausibly converting; (3) a two-model approach (Criteo) that models
conversion probability and delay distribution separately and combines them.
Labeling a not-yet-converted click as 0 and training immediately biases pCVR
downward and under-bids real value.

## Tricky (the follow-ups that separate people)

**Q: AUC went up but revenue dropped. What happened?**
A: Suspect calibration first. The new model ranks ads better (higher AUC) but
its predicted probabilities shifted in scale, so eCPM moved even though the
ranking improved. The auction is now pricing slots incorrectly. Check a
reliability curve and sliced ECE before concluding the model is better. Also
check for training-serving skew, which can shift probabilities without changing
the rank order, and for delayed-feedback bias, which depresses pCVR.

**Q: Can you retrain more often to fix calibration drift without a separate
calibration layer?**
A: Rarely practical at scale. Full DNN retrains are expensive (billions of
parameters, terabytes of data) and take hours. Calibration can drift hourly as
campaign mixes and demand change. The decoupled approach used by Pinterest
(Platt scaling refitted hourly) and LinkedIn (isotonic regression plus a shallow
tower) is the right architecture: retrain the heavy network daily, recalibrate
the cheap layer hourly. Coupling them forces you to choose between expensive
retrains and stale calibration.

**Q: Your model only sees clicks on ads it chose to show. Isn't that circular?**
A: Yes. The feedback loop is real and tight: the model scores ads, high scores
win auctions, only served ads produce click labels, so the model trains on data
shaped by its predecessor's biases. Break it with: (1) exploration traffic that
shows ads off-policy (epsilon-greedy or a randomized slice); (2)
inverse-propensity weighting that upweights rarely-served ads; (3) a position
feature at train time, neutralized at serving, so the model learns relevance and
not slot position.

**Q: You said feature hashing accepts collisions. Doesn't that hurt quality?**
A: Controlled collision is the accepted tradeoff for bounded memory and graceful
handling of unseen ids. The collision probability follows a birthday-paradox curve:
for $n$ ids hashed into $H$ buckets, the expected collision rate is roughly
$1 - e^{-n(n-1)/(2H)}$. Increase $H$ to reduce collisions; shrink it to save
memory. In practice, collision rates below roughly 5% introduce negligible quality
loss, and for rare ids (which collide most) the embedding was undertrained anyway.
Name the tradeoff explicitly; it is the senior detail.

## Commonly answered wrong (the traps)

**Q: Can you just use a high AUC as the launch criterion?**
A: No. AUC is invariant to any monotone rescaling of the score. A model can
have identical AUC before and after a 20% upward shift in its probability scale.
The auction reads the absolute pCTR, not the rank order, so AUC alone cannot
detect the shift that would mis-price every slot. The launch criterion must
include log loss, calibration (reliability curve and ECE, sliced), and an online
A/B on revenue. AUC is a useful diagnostic but not the gate.

**Q: Do all four interaction models (DLRM, DCN, DeepFM, Wide and Deep) perform
about the same? Should I just pick the simplest one?**
A: No. They differ in what kind of interaction they can represent efficiently.
Wide and Deep requires hand-crafted wide-side cross features, which DeepFM
removes via shared embeddings and an FM component. DCN V2 gives explicit
bounded-degree crosses without a deep stack. DLRM's explicit pairwise dot
products are parameter-efficient for pairwise interactions but do not easily
express higher-degree ones without the MLP. The right choice depends on the
sparsity of the feature space, the degree of interaction that matters, and the
team's tolerance for feature engineering. Start with Wide and Deep as a
baseline; move to DLRM or DCN V2 when offline eval (log loss and ECE, not just
AUC) justifies the complexity.

**Q: Logistic regression has naturally calibrated output; why not use it for
production CTR?**
A: It is a strong baseline and was the production workhorse for years (it
remains so in moderate-scale settings). Its ceiling is feature interactions:
it cannot model user-ad cross signals without hand-crafted feature crosses, which
do not scale to billions of sparse ids. In large-scale systems with hundreds of
millions of ids, embedding-based deep models handle the full feature space
automatically. That said, deep models' raw heads still need post-hoc calibration
under negative sampling and distribution shift, whereas logistic regression's
sigmoid is naturally calibrated. The trade is interaction capacity for calibration
ease.
