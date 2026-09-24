# Papers, by topic

The repo's [production case studies](CASE-STUDIES.md) answer "how did a real team ship
this." This page answers the other half: **which papers does an interviewer assume you
have read** before they ask a follow-up.

Every topic caps at eight. That cap is the point. A list of two thousand papers is an
index of a field, not a study plan, and nobody reads one before a loop. These are the
ones whose result gets referenced by name: "that is the two-tower sampling bias", "that
is what DIN does", "trees still win on tabular data."

**How to read this page.** For each entry, the sentence after the year is the thing to
be able to say about it. If you can produce that sentence and the number attached to
it, you have what an interview will ask for. Read the abstract and the one figure that
carries the result; read the whole paper only for the topics you are being hired to own.

A paper appears under two topics when both interviews genuinely assume it. That
repetition is deliberate: this is a study list per topic, not a bibliography.

Not on this page: engineering blog posts and first-party writeups. Those are the
**Seen in production** section of every topic, rolled up in
[CASE-STUDIES.md](CASE-STUDIES.md) with per-system [teardowns](CASE-TEARDOWNS.md), and
for the platform topics they carry more of the real answer than the papers do.

---

## Start here

The handful an interviewer will not stop to explain, in any classic-ML loop.

- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)**
  (Google). Not a paper, and the most useful document on this page. The order of the
  rules is the argument: launch a simple model with good infrastructure first, and do
  not add a feature you cannot serve.
- **[Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)**
  (2015). The model is a small box in a large diagram. Every platform topic in this
  repo exists because of something in this paper.
- **[Ad Click Prediction: a View from the Trenches](https://research.google/pubs/pub41159/)**
  (2013). What a production system actually needs beyond the model: online learning,
  memory-saving tricks, calibration, and a confidence estimate that is cheap.
- **[Wide and Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)**
  (2016). Memorization and generalization as two jobs, one model. The shape most
  ranking systems still have.
- **[Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091)**
  (2019). Embedding tables plus feature interaction, with the system constraints
  (memory, sharding) stated as first-class. The reference architecture for scale.
- **[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)**
  (2017). Modern networks are overconfident, and reliability diagrams plus temperature
  scaling are the fix. Required wherever the probability is the product.

---

## Retrieval and representation

### [Candidate retrieval](topics/01-candidate-retrieval.md) and [embeddings](topics/07-embeddings-and-representation-learning.md)

- **[word2vec](https://arxiv.org/abs/1301.3781)** (2013). Where the whole embedding
  idea becomes cheap and useful. Read it for negative sampling, which is the ancestor
  of every in-batch-negatives trick you will use.
- **[item2vec](https://arxiv.org/abs/1603.04259)** (2016). The same machinery applied
  to items in a basket or a session, and the clean statement of why co-occurrence is a
  usable signal without any content features.
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)** (2019). Bi-encoders that make
  semantic similarity a nearest-neighbour lookup, which is what makes retrieval
  possible at all at scale.
- **[SimCSE](https://arxiv.org/abs/2104.08821)** (2021). Contrastive sentence
  embeddings with a startlingly simple positive pair. Also the clearest short read on
  alignment and uniformity as a diagnostic.
- **[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)** (2020). Dual encoders
  beating BM25, and the in-batch-negatives recipe that made it trainable.
- **[HNSW](https://arxiv.org/abs/1603.09320)** (2016). The index almost everything uses.
  Recall against latency against memory is tuned with the parameters in this paper.
- **[Faiss](https://arxiv.org/abs/1702.08734)** (2017). Billion-scale similarity search
  on GPUs, and the practical vocabulary (IVF, PQ) that vector-database conversations
  assume.
- **[ScaNN](https://arxiv.org/abs/1908.10396)** (2019). Anisotropic vector quantization:
  quantize for the inner product you will actually compute, not for reconstruction.

### [Search ranking](topics/09-search-ranking.md)

- **[Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468)**
  (2016). Clicks are biased by position, and propensity weighting is the correction.
  The single most important idea in this topic.
- **From RankNet to LambdaRank to LambdaMART: An Overview** (Burges, 2010). The
  canonical learning-to-rank reference, and the reason gradient-boosted trees still win
  many ranking benchmarks.
- **[ColBERT](https://arxiv.org/abs/2004.12832)** (2020). Late interaction: better
  quality than a single vector, at a storage cost you should be able to estimate.
- **[BEIR](https://arxiv.org/abs/2104.08663)** (2021). Zero-shot retrieval across
  domains, and the result to remember is that a model tuned on one corpus often does
  not transfer.
- **[MTEB](https://arxiv.org/abs/2210.07316)** (2022). How embedding models get
  compared, and why a leaderboard rank does not transfer to your corpus.

---

## Recommender systems

### [Ranking models](topics/02-ranking-model.md) and [ads CTR](topics/10-ads-ctr-prediction.md)

- **[DeepFM](https://arxiv.org/abs/1703.04247)** (2017). Factorization machines and a
  deep network sharing an embedding layer, so second-order interactions are learned
  rather than hand-built.
- **[Deep and Cross Network](https://arxiv.org/abs/1708.05123)** (2017) and
  **[DCN V2](https://arxiv.org/abs/2008.13535)** (2020). Explicit bounded-degree
  feature crosses, then the version rewritten for what actually worked at web scale.
  Read the second for its practical lessons section.
- **[xDeepFM](https://arxiv.org/abs/1803.05170)** (2018). Explicit interactions at the
  vector level, useful mostly as the contrast that makes DCN's design legible.
- **[Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)** (2017). Worth
  reading together with the later replication work that questioned it: a lesson about
  baselines as much as a method.
- **[Deep Interest Network](https://arxiv.org/abs/1706.06978)** (2017). An attention
  unit that adapts the user-interest vector to the candidate, which is where "one user
  embedding" stops being enough.
- **[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)**
  (2017). In an auction the calibrated probability is the product, so this is not
  optional here.
- **[Ad Click Prediction: a View from the Trenches](https://research.google/pubs/pub41159/)**
  (2013). Online learning, memory tricks and the operational reality of a CTR system.

### [Sequential recommendation](topics/03-sequential-recommendation.md) and [generative recommendation](topics/19-generative-recommendation.md)

- **[GRU4Rec](https://arxiv.org/abs/1511.06939)** (2015). Session-based recommendation
  with an RNN, the paper that made "order matters" a modeling decision.
- **[SASRec](https://arxiv.org/abs/1808.09781)** (2018). Causal self-attention over the
  behaviour sequence: parallel to train, and the last position is directly usable at
  serving time.
- **[BERT4Rec](https://arxiv.org/abs/1904.06690)** (2019). The bidirectional contrast.
  Know why masked training makes it a scorer rather than a generator.
- **[Behavior Sequence Transformer](https://arxiv.org/abs/1905.06874)** (2019). The
  same idea inside a production ranking model at Taobao, which is the version you would
  actually ship.
- **[PinnerFormer](https://arxiv.org/abs/2205.04507)** (2022). An all-action window
  loss so a batch-computed user embedding does not go stale between runs. The best
  worked example of freshness as a design constraint.
- **[TIGER](https://arxiv.org/abs/2305.05065)** (2023). Semantic IDs from a residual
  quantizer, and retrieval by decoding. The reference design for generative retrieval.
- **[Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121)**
  (2023). The same representation as ranking features, with the gains on cold-start and
  long-tail slices.
- **[HSTU](https://arxiv.org/abs/2402.17152)** (2024). Recommendation as sequential
  transduction at scale, and the clearest statement of the scaling-law bet.

### Graph recommendation

- **[node2vec](https://arxiv.org/abs/1607.00653)** (2016). Biased random walks as the
  bridge from graphs to embeddings, and the breadth-versus-depth knob.
- **[GraphSAGE](https://arxiv.org/abs/1706.02216)** (2017). Inductive embeddings by
  sampling and aggregating neighbours, which is what lets a new node get an embedding
  without retraining.
- **[PinSage](https://arxiv.org/abs/1806.01973)** (2018). GraphSAGE made to work on
  billions of nodes, with the engineering (producer-consumer minibatching, MapReduce
  inference) that made it deployable.
- **[LightGCN](https://arxiv.org/abs/2002.02126)** (2020). Removing the parts of a GCN
  that do not help collaborative filtering. The tabular-data lesson in graph form.
- **[PyTorch-BigGraph](https://arxiv.org/abs/1903.12287)** (2019). Graph embeddings when
  the graph does not fit in memory, which is the constraint that actually shapes these
  systems.

### [Cold start and exploration](topics/18-cold-start-and-exploration.md)

- **[LinUCB](https://arxiv.org/abs/1003.0146)** (2010). Contextual bandits for news
  recommendation, and where the explore-exploit vocabulary in this repo comes from.
- **[A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038)** (2017). The
  posterior-sampling alternative, and the best single explanation of why it works.
- **[Deep Bayesian Bandits Showdown](https://arxiv.org/abs/1802.09127)** (2018). An
  empirical comparison that is unusually honest about how fragile the fancy methods are.
- **[Degenerate Feedback Loops in Recommender Systems](https://arxiv.org/abs/1902.10730)**
  (2019). The system trains on what it showed, so exploration is not a nicety. The
  paper to cite when someone proposes removing the exploration arm.

---

## Prediction problems

### [Fraud, anomaly and imbalance](topics/08-fraud-and-anomaly-detection.md)

- **[SMOTE](https://arxiv.org/abs/1106.1813)** (2002). Synthetic minority oversampling.
  Know it, and know that in fraud the usual right answer is thresholds and
  cost-sensitive learning rather than resampling.
- **Isolation Forest** (Liu, Ting and Zhou, ICDM 2008). Anomalies are easier to isolate
  than normal points, so path length is the score. The standard unsupervised baseline.
- **[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)**
  (2017). A fraud threshold is a decision on a probability, so miscalibration is a
  business error rather than a modeling one.
- **[XGBoost](https://arxiv.org/abs/1603.02754)** (2016). Still the workhorse for this
  problem, and the paper explains why (sparsity awareness, the weighted quantile
  sketch, the system design).

### [Tabular data](topics/15-predictive-modeling-tabular.md)

- **[XGBoost](https://arxiv.org/abs/1603.02754)** (2016). The system paper as much as
  the algorithm paper.
- **[CatBoost](https://arxiv.org/abs/1706.09516)** (2017). Ordered boosting and target
  statistics for categorical features, which is the failure mode naive target encoding
  has.
- **LightGBM** (Ke et al., NeurIPS 2017). Histogram splits, leaf-wise growth, GOSS and
  EFB. The speed baseline everyone compares against.
- **[Tabular Data: Deep Learning is Not All You Need](https://arxiv.org/abs/2106.03253)**
  (2021) and **[Why do tree-based models still outperform deep learning on tabular
  data?](https://arxiv.org/abs/2207.08815)** (2022). Read as a pair: the first is the
  result, the second is the explanation (irregular targets, uninformative features,
  rotation invariance).

### [Forecasting](topics/14-demand-forecasting-and-time-series.md)

- **[DeepAR](https://arxiv.org/abs/1704.04110)** (2017). Probabilistic forecasts from an
  autoregressive RNN trained across many related series. The paper that made global
  models standard.
- **[N-BEATS](https://arxiv.org/abs/1905.10437)** (2019). A pure deep architecture that
  beat the M4 winner, with interpretable basis blocks.
- **[Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363)** (2019). Static,
  known-future and observed covariates handled explicitly, which is the shape real
  demand data has.
- **Prophet** (Taylor and Letham, 2017) and **the M4 and M5 competitions**. The
  baselines that keep winning. Know that a well-tuned seasonal-naive model is the bar,
  and that the M5 result made hierarchical reconciliation mainstream.

---

## Perception

### [Computer vision](topics/12-computer-vision.md)

- **[ResNet](https://arxiv.org/abs/1512.03385)** (2015). Residual connections, and the
  reason deep networks train at all. Still the default backbone in production.
- **[Faster R-CNN](https://arxiv.org/abs/1506.01497)** (2015) and
  **[YOLO](https://arxiv.org/abs/1506.02640)** (2015). Two-stage against one-stage
  detection: accuracy against latency, which is the tradeoff you will be asked to make.
- **[EfficientNet](https://arxiv.org/abs/1905.11946)** (2019). Compound scaling of
  depth, width and resolution. The paper to cite when asked how to fit a quality target
  into a serving budget.
- **[Vision Transformer](https://arxiv.org/abs/2010.11929)** (2020). Images as patch
  tokens, and the data-scale condition under which it beats convolutions.
- **[CLIP](https://arxiv.org/abs/2103.00020)** (2021). Contrastive image-text
  pretraining, which turned visual search and zero-shot classification into embedding
  problems.
- **[Segment Anything](https://arxiv.org/abs/2304.02643)** (2023). A promptable
  segmentation foundation model, and a case study in building a data engine.

### [Speech and audio](topics/17-speech-and-audio.md)

- **[Sequence Transduction with RNNs (RNN-T)](https://arxiv.org/abs/1211.3711)** (2012).
  The streaming-friendly loss that on-device ASR is built on.
- **[Conformer](https://arxiv.org/abs/2005.08100)** (2020). Convolution plus attention,
  the encoder most production ASR uses.
- **[wav2vec 2.0](https://arxiv.org/abs/2006.11477)** (2020). Self-supervised speech
  representations, and the reason a low-resource language is now tractable.
- **[Whisper](https://arxiv.org/abs/2212.04356)** (2022). Weak supervision at scale,
  robust across domains. Read it for what it gives up: latency and streaming.
- **[Tacotron 2](https://arxiv.org/abs/1712.05884)** (2017). The mel-spectrogram plus
  vocoder decomposition that TTS still follows.

### [Natural language processing](topics/13-natural-language-processing.md)

- **[BERT](https://arxiv.org/abs/1810.04805)** (2018). The fine-tuned encoder that most
  production classification and extraction still runs on, and the baseline an LLM has
  to beat on cost.
- **[RoBERTa](https://arxiv.org/abs/1907.11692)** (2019). Same architecture, better
  training. A lesson about ablations more than about models.
- **[DistilBERT](https://arxiv.org/abs/1910.01108)** (2019). Distillation to 40 percent
  fewer parameters at most of the quality, which is the serving answer for this topic.
- **[T5](https://arxiv.org/abs/1910.10683)** (2019). Every task as text-to-text, and the
  C4 corpus that came with it.
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)** (2019). Where NLP and retrieval
  meet, and the practical way to get sentence embeddings.

### Video and multimodal search

- **[CLIP](https://arxiv.org/abs/2103.00020)** (2021). The joint space that makes
  text-to-video retrieval possible without video-text labels for your own corpus.
- **[Frozen in Time](https://arxiv.org/abs/2104.00650)** (2021). Joint image and video
  training for end-to-end retrieval, and the curriculum that makes it affordable.

---

## The platform

### [Feature stores](topics/04-feature-store-and-training-serving-skew.md), [serving](topics/05-realtime-serving-and-deployment.md), [experimentation](topics/06-online-experimentation-and-ab-testing.md), [monitoring](topics/11-ml-monitoring-and-drift.md)

- **[Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)**
  (2015). Entanglement, undeclared consumers, pipeline jungles, glue code. Every one of
  these four topics is a named debt from this paper.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)**
  (Google). Read rules 1 to 15 before designing any of this.
- **[Clipper](https://arxiv.org/abs/1612.03079)** (2016). A prediction-serving system
  with batching, caching and model selection. The vocabulary of model serving, from
  before it was a product category.
- **[CUPED](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)**
  (2013). Use pre-experiment data to cut variance, which is the cheapest way to make an
  A/B test more sensitive. Ask for it by name.
- **Trustworthy Online Controlled Experiments** (Kohavi, Tang and Xu, 2020). The book.
  If you only take one thing: most ideas do not win, and the guardrail metrics are what
  stop a win from being a loss elsewhere.
- **[Failing Loudly](https://arxiv.org/abs/1810.11953)** (2018). An empirical study of
  dataset-shift detection, and the result that matters: multivariate two-sample tests on
  representations beat per-feature statistics.
- **[Degenerate Feedback Loops in Recommender Systems](https://arxiv.org/abs/1902.10730)**
  (2019). The monitoring version of the problem: the system changes the distribution it
  is measured on.

---

## Contributing

Add a paper only with the sentence that says what it changes for the reader, and only
where a topic has fewer than eight. If a topic is full, argue for a swap in the pull
request rather than making the list longer. Verify that an arXiv id resolves to the
title you wrote: a wrong id is the one error nobody catches by reading. See
[CONTRIBUTING.md](CONTRIBUTING.md).
