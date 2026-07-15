# 4. Model development

## Gradient-boosted trees: the tabular workhorse

For a tabular fraud problem with hundreds of engineered features, gradient-
boosted trees (XGBoost, LightGBM) are the first model to reach for. They handle
mixed feature types naturally, give feature importance for free, produce
well-calibrated probability estimates with isotonic regression or Platt scaling
(post-hoc methods that remap raw model scores into true probabilities) applied
post-training, and train fast enough for daily or even hourly retraining
cycles. Capital One's AML program uses a random forest to triage suspicious-activity
alerts, prioritizing cases for investigation across hundreds of features under
regulatory scrutiny where explainability is a hard requirement. The message: trees
are not a fallback; they are the right starting point for tabular fraud signals
and a strong baseline to beat before reaching for a DNN.

## Wide-and-Deep DNN

When the feature set includes high-cardinality sparse categoricals (device ID,
merchant ID, card BIN, geo cell), a deep model can learn richer interaction
patterns. The **wide-and-deep** architecture (Cheng et al., 2016) handles this
directly.

```mermaid
flowchart TD
  SPARSE["sparse categoricals<br/>(device ID, merchant, geo, card BIN)"] --> CROSS["cross-product transforms"]
  CROSS --> WIDE["wide linear model<br/>(memorization: exact co-occurrences)"]
  SPARSE --> EMB["embedding tables<br/>(low-dim dense representations)"]
  DENSE["dense features<br/>(velocity, amount, account age)"] --> CONCAT["concatenate"]
  EMB --> CONCAT
  CONCAT --> MLP["deep MLP<br/>(generalization: unseen combos)"]
  WIDE --> SUM["weighted sum of logits"]
  MLP --> SUM
  SUM --> OUT["sigmoid → fraud probability p"]
```

The wide side memorizes specific co-occurrences (device X plus merchant Y plus
time-of-day Z was always fraud last month). The deep side generalizes to unseen
combinations. They are trained jointly: the wide branch uses FTRL, the deep
branch uses Adam, and neither is trained in isolation. Stripe's Radar system uses
a supervised model over tabular features and entity embeddings that is
continuously retrained and must score each transaction in sub-100 ms.

Two points to state in an interview: (1) embedding tables dominate parameter
count, not the MLP layers, so the engineering cost of growing the embedding
dimension is in memory, not compute; (2) the joint logit distorts calibration
and must be recalibrated before using the cost-optimal threshold formula.

## Graph methods: RGCN for rings, graph-DB traversal for online latency

Individual transactions can look clean while the entity network screams. Fraud
rings share devices, cards, IPs, and addresses across many accounts. Two
approaches exploit this structure, at different latency points.

**RGCN (Relational Graph Convolutional Network).** Used by Uber and Grab. Each
edge type (shared card, shared device, shared city) gets its own transform
matrix so different relationship types carry different signal weights. Message
passing aggregates neighborhood information:

$$h_i^{(l+1)} = \sigma\!\left(W_0^{(l)} h_i^{(l)} + \sum_{r \in R}\sum_{j \in N_i^{r}} \frac{1}{|N_i^{r}|} W_r^{(l)} h_j^{(l)}\right)$$

Here $R$ is the set of relation types, $N_i^r$ is the set of neighbors of node
$i$ under relation $r$, and $W_r^{(l)}$ is the relation-specific weight matrix
at layer $l$. The key: a plain GCN would conflate a shared device with a shared
city; RGCN keeps them separate. Uber reported 15 percent better precision with
minimal added false positives feeding downstream risk models. Because training
and inference are batch jobs, RGCN scores are injected as features into the
online risk model rather than serving the GNN inline.

**Graph-DB traversal features.** Used by PayPal and Booking. A graph database
(PayPal's custom Gremlin-over-Aerospike; Booking's JanusGraph over Cassandra)
stores entity relationships and runs BFS or multi-hop traversal at request time.
PayPal links a new account to a known fraud ring within a sub-second at million-
QPS throughput. Booking extracts a `hops_to_fraud` scalar feature within a p99
of 300 ms. This path delivers online graph signal without training or serving a
GNN, at the cost of a specialized graph infrastructure.

## Anomaly detection methods

**Isolation Forest.** Builds an ensemble of random binary trees by splitting
features at random split points. Points that require fewer splits to isolate
score as more anomalous. Fast to train and serve, no labels, interpretable
anomaly scores. Suitable as a first pass for novel attacks or as a complement
to the supervised model.

**Autoencoder / GraphBEAN.** Grab's GraphBEAN is a bipartite-graph autoencoder
that reconstructs both node attributes and edge attributes on a consumer-merchant
graph. Points that reconstruct poorly are anomalous. The anomaly score is:

$$s(v) = \|x_v - \hat{x}_v\|^2 + \sum_{e \ni v}\|a_e - \hat{a}_e\|^2 + \text{BCE}(A, \hat{A})$$

The three terms are node reconstruction error, edge reconstruction error, and
structural reconstruction error (binary cross-entropy on the adjacency matrix).
Normal behavior reconstructs easily; novel fraud produces high combined error.
This catches novel fraud without labels, at the cost of high false positives on
unusual-but-legitimate behavior.

## Loss functions

### Focal loss

Standard cross-entropy treats every example equally. When 99.8 percent of
examples are easy legitimate transactions, the gradient is dominated by easy
negatives and the rare fraud class barely moves the model. Focal loss fixes this
by downweighting easy, well-classified examples:

$$\text{FL}(p_t) = -\,\alpha_t\,(1-p_t)^{\gamma}\log(p_t)$$

where $p_t = p$ when $y=1$ and $p_t = 1-p$ when $y=0$.

The modulating factor $(1 - p_t)^{\gamma}$ goes to zero for confident correct
predictions (easy legitimate transactions contribute almost nothing) while a
hard or misclassified fraud retains its full weight. $\alpha_t$ is a class-
weight term; $\gamma$ controls the focus strength (typically 2.0). Focal loss
is preferred over SMOTE when you want to handle skew inside the loss without
distorting the data distribution.

### Class-weighted binary cross-entropy

Simpler than focal loss: multiply the per-example BCE loss by a class weight.
If fraud is 0.2 percent, a weight of 499 on fraud examples restores a balanced
effective loss. Tune the weight as a hyperparameter against PR-AUC on the true
base rate.

### Airbnb three-action loss

When the action space is allow / friction / block (rather than binary), the
loss accounts for friction as a middle option:

$$L = \text{FP} \cdot G \cdot V + \text{FN} \cdot C + \text{TP} \cdot (1-F) \cdot C$$

Here $G$ is the good-user churn rate under friction, $V$ is the transaction
value, $C$ is the chargeback cost, and $F$ is the friction effectiveness
(fraction of fraud stopped by the friction step). The first term penalizes
false positives that drop good users through friction; the second penalizes
missed fraud; the third acknowledges that some fraudsters beat the friction
and still cause loss.

## Cost-sensitive thresholding

The model outputs a calibrated probability. The threshold that minimizes
expected cost in closed form is:

$$\tau^{\star} = \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}}$$

When a missed fraud costs 20 times a false positive ($c_{\text{FN}} = 20$,
$c_{\text{FP}} = 1$), this gives $\tau^{\star} = 1/21 \approx 0.048$. The
system blocks any transaction where the estimated fraud probability exceeds
4.8 percent. This is far below the default 0.5 and is why reporting a model
with a 0.5 threshold is almost always wrong in fraud.

For the full expected-cost curve across all thresholds:

$$L(\tau) = c_{\text{FP}} \cdot \text{FP}(\tau) + c_{\text{FN}} \cdot \text{FN}(\tau), \qquad \tau^{\star} = \arg\min_{\tau} L(\tau)$$

```python
def expected_cost(labels, scores, tau, c_fp, c_fn):   # labels 0/1, scores in [0, 1]
    import numpy as np
    labels, scores = np.asarray(labels), np.asarray(scores)
    pred = scores >= tau                              # block when score at or above threshold
    fp = int(((pred == 1) & (labels == 0)).sum())     # false positives: blocked good users
    fn = int(((pred == 0) & (labels == 1)).sum())     # false negatives: missed fraud
    return c_fp * fp + c_fn * fn                       # total expected cost at this threshold
# expected_cost([1,0,1,0], [0.9,0.6,0.2,0.1], 0.5, 1, 20) -> 1*1 + 20*1 = 21
```

The figure below shows this curve for symmetric and asymmetric cost ratios, and
the shift in optimal threshold.

![Threshold vs expected cost](assets/fig-threshold-vs-cost.png)

*Asymmetric costs (c_FN=20) shift the optimal threshold well to the left of the
symmetric case (c_FP=c_FN=1), catching more fraud at the cost of more false
alarms. The dashed verticals mark each optimum. Illustrative.*

The figure below shows the cost matrix that drives this calculation.

![Cost matrix](assets/fig-cost-matrix.png)

*TN (correct allow) costs zero. TP (prevented fraud) costs little. FP (blocked
good user) costs the lost sale plus support and churn. FN (missed fraud) is the
most expensive: the chargeback amount plus network fees. Illustrative.*

## Why resampling breaks the threshold you just derived

The closed-form threshold $\tau^{\star}$ above assumes the model outputs a
**calibrated** probability. Two of the imbalance fixes on this page quietly violate
that assumption. Downsampling the negatives (a common way to make training tractable
at a fraction-of-a-percent base rate) shifts the model's implied prior, so it
systematically over-predicts fraud probability; focal loss and heavy class weights
distort the score in the same direction. Feeding those inflated scores into
$\tau^{\star} = c_{\text{FP}}/(c_{\text{FP}} + c_{\text{FN}})$ blocks far more traffic
than the cost math intended. This is the mechanism behind the "ranking fine,
probabilities biased high" pitfall.

Because the mechanism is a clean prior shift, the correction is exact rather than a
re-fit. If negatives are kept at rate $r$ during downsampling, the model's odds are
inflated by $1/r$, and multiplying the predicted odds back by $r$ recovers the true
probability:

$$p_{\text{true}} = \frac{r\, p_s}{1 - p_s + r\, p_s}$$

where $p_s$ is the score from the model trained on the resampled data. This is the
sampling correction of Elkan (2001) and Dal Pozzolo et al. (2015). The senior
discipline: fit a monotone method (isotonic, Platt) or apply this closed-form
correction on a held-out set drawn from the true base rate **before** computing
$\tau^{\star}$, and recalibrate after every retrain, since the score distribution
moves each time.

## When to use which model family

| Reach for | When | Instead of |
|---|---|---|
| Gradient-boosted trees (XGBoost, LightGBM) | tabular features, explainability required, fast retrain cadence, regulatory audit | deep models as the default; trees often match or beat DNN on tabular fraud |
| Wide-and-Deep DNN | high-cardinality sparse categoricals (device, merchant) reward embeddings and you have scale | trees when the feature set is mostly dense and well-engineered |
| RGCN / GNN | fraud is coordinated rings over shared cards, devices, addresses | per-transaction models that miss collusion; costly if latency is inline |
| Graph-DB traversal features | you want ring signal inline at serving time without training a GNN | learned GNN, when the inline latency budget allows and you need richer representations |
| Isolation Forest | first-pass anomaly detection for novel attacks with no labels | supervised only, which is blind to novel patterns it was never trained on |
| GraphBEAN / autoencoder | bipartite or graph-structured data, novel fraud, no labels | supervised anomaly, which needs labels it does not have |
| Ensemble (supervised + anomaly) | mature production system with both known and novel fraud | single model that covers only one threat mode |

**Provenance.** The gradient-boosted-tree workhorses are XGBoost (Chen and Guestrin,
2016) and LightGBM (Microsoft, 2017); CatBoost (Yandex, 2017) is the alternative when
high-cardinality categoricals dominate. The unsupervised anomaly branch draws on
Isolation Forest (Liu et al., 2008) for label-free novel-attack detection and on
autoencoder anomaly detection (reconstruction error as the score).

**Tools.** Gradient-boosted trees are XGBoost and LightGBM (Microsoft), with
scikit-learn isotonic or Platt calibration on top and imbalanced-learn available for
resampling when focal or class-weighted loss is not enough. Wide-and-Deep DNNs are
built in PyTorch or TensorFlow with embedding tables for the sparse categoricals.
RGCN and other GNNs use PyG (PyTorch Geometric) or DGL and run as batch jobs whose
scores feed the online model. Graph-DB traversal uses a graph database such as
JanusGraph, Neo4j, or Aerospike queried with Gremlin or Cypher at request time.
Isolation Forest comes from scikit-learn; autoencoders and GraphBEAN-style bipartite
models are built in PyTorch or PyG.

**Worked example.** A payments company has hundreds of dense engineered features
under regulatory audit, so gradient-boosted trees (LightGBM) are the workhorse
baseline, calibrated post-training before the cost-optimal threshold is applied.
High-cardinality device and merchant IDs would reward embeddings, but it only reaches
for a Wide-and-Deep DNN once it has the scale to justify the extra memory and the
recalibration the joint logit forces. Because it also sees ring fraud sharing devices
and cards, it adds RGCN scores computed offline (PyG) as features rather than serving
a GNN inline, and where it needs ring signal within the request budget it falls back
to graph-DB traversal for a hops-to-fraud scalar. Isolation Forest and a GraphBEAN
autoencoder cover novel unlabeled attacks, and the mature system ensembles the
supervised and anomaly outputs so both known and novel fraud are covered rather than
leaving one threat mode exposed.

## Implementation and training pitfalls

Fraud models fail most often on label timing and evaluation setup, not the choice of
algorithm: labels mature slowly, blocked transactions never get a true outcome, and
the same entity leaking across the split makes an offline number look far better than
production ever will.

| Problem | Symptom | Fix |
|---|---|---|
| Label maturity (delayed chargebacks) | recent transactions marked legitimate are actually pending fraud | enforce a performance-window cutoff, exclude transactions too recent to be fully matured |
| Feature velocity leakage | a count or velocity feature includes the event being scored | compute aggregates strictly before the event timestamp |
| Selection bias and feedback loop | blocked transactions never get a true label, the model forgets that pattern | hold out a small unblocked control, use reject inference or counterfactual logging |
| Class imbalance | model predicts everything legitimate | class weights or focal loss, then recalibrate after any resampling |
| Miscalibration after downsampling | ranking fine but probabilities biased high | isotonic or Platt on a held-out set, or correct the intercept for the sampling rate |
| Adversarial concept drift | precision decays as fraudsters adapt | frequent retraining, monitor score drift, keep anomaly detectors for novel attacks |
| Entity leakage across split | same card or device in train and test inflates metrics | split by entity (card, device, account) and by time, never random rows |
| Threshold drift | fixed cutoff, alert volume swings with the base rate | re-derive the cost-optimal threshold on recent data, monitor the review-queue load |

The through-line: an offline PR-AUC that looks too good in fraud usually means an
immature label window or an entity that leaked across the split, so distrust it until
both are ruled out.
