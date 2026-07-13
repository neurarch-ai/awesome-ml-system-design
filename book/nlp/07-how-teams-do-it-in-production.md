# 7. How teams do it in production

Every production NLP system normalizes and tokenizes text once, then fans out to a
task-specific model whose score a threshold either auto-acts on or routes to human
review, whose verdicts flow back as fresh labels. None puts a large LLM on the
inline firehose; volume forces a small, calibratable model on the hot path.

What actually differs between companies is the model era, the task, the latency
regime (inline vs batch), and the supervision strategy. The architecture everyone
converges on; the leverage is in the labeling pipeline and the human review loop.

## Where the real designs diverge

| System | Task | Model | Multilingual | Supervision | Latency regime | Key metric | Watch out |
|---|---|---|---|---|---|---|---|
| Uber Maps | classification (ticket routing) | Word2Vec plus WordCNN | English first | manual labels (~10-20K tickets) | weekly Spark batch | AUC-PR, routing accuracy | vocab/domain drift; non-English tickets before expansion |
| Airbnb LAEP | NER plus entity resolution | CNN NER plus word2vec cosine plus BERT scorer | English (language ID filter) | taxonomy-mapped labels (~30K spans) | batch enrichment | strict span F1, taxonomy coverage | entities outside taxonomy; negation ("no lockbox") missed without scorer |
| Meta hate speech | classification (toxicity) | RIO plus Linformer | many languages | proactive online sampling via RIO | near real-time inline firehose | proactive detection rate, false block rate | adversarial evasion; multimodal cases benign alone but hateful combined |
| Google GNMT | translation | deep LSTM seq2seq plus attention | high-resource language pairs | bilingual human ratings (0-6 scale) | online translation (TPU serving) | human error reduction, BLEU | dropped words on long sentences; sentence-isolation errors |
| Meta NMT | translation | LSTM plus attention, later CNN seq2seq | 2,000+ directions | bilingual corpora, human ratings | 4.5B translations/day (Caffe2 quantized) | BLEU, per-direction coverage | low-resource direction quality; quantization quality loss |
| LinkedIn KG | entity resolution | word2vec cosine, co-occurrence disambiguation, binary relationship classifiers | planned (MT for long tail) | member-confirmed relationships as free positives | batch pipeline | pairwise precision/recall, dedup rate | polysemy ignored without disambiguation; taxonomy staleness |
| Pinterest spam | classification (spam domain + user) | DNN plus clustering plus bipartite graph label propagation | not stated | synthetic labels, graph propagation | batch (PySpark) | spam catch rate at fixed precision | propagation errors spread; supervised model misses novel bot patterns |
| LinkedIn abuse | sequence classification (scraping/abuse) | LSTM over member activity sequences | not stated | isolation-forest bootstrap, then supervised | batch scoring, periodic | abuse/scraping detection rate | low-and-slow abuse; very long sequences |
| Uber COTA | classification/ranking (ticket resolution) | TF-IDF plus LSA plus random forest (later CNN/RNN) | not stated | historical ticket labels | batch pipeline (Michelangelo) | resolution time reduction, routing accuracy | multi-class solution space; offline-online metric alignment |
| Airbnb voice | classification (contact reason) | domain-tuned ASR plus intent classifier | not stated | contact-reason labels | inline, under 50 ms | self-serve deflection, routing accuracy | bad ASR transcript poisons all downstream stages |
| Grammarly GECToR | sequence tagging (GEC) | BERT-like encoder plus edit-tag heads | English | synthetic (9M pairs) plus real learner data | inline correction, up to 10x faster than seq2seq | F0.5 on CoNLL-2014 and BEA-2019 | missing g-transformations for morphology; training only on errors causes over-correction |

The core dividing line is whether the task maps text to a fixed decision (classify,
tag, resolve) or generates new text (translate, correct), which decides everything
downstream: encoder head vs seq2seq, discrete metrics vs BLEU or $F_{0.5}$, and how
much bilingual or sequential supervision the labels must carry.

## The systems (first-party write-ups)

- **Uber** [Applying Customer Feedback: NLP and Deep Learning Improve Uber's Maps](https://www.uber.com/gb/en/blog/nlp-deep-learning-uber-maps/): Word2Vec plus a WordCNN classify support tickets to find map-data errors. *(product design)*
- **Airbnb** [Building Airbnb's Listing Knowledge from big text data](https://medium.com/airbnb-engineering/wisdom-of-unstructured-data-building-airbnbs-listing-knowledge-from-big-text-data-7c533466a63c): A CNN-based NER extracts amenities from free-text listings into an 800-attribute taxonomy. *(product design)*
- **Meta** [How AI is getting better at detecting hate speech](https://ai.meta.com/blog/how-ai-is-getting-better-at-detecting-hate-speech/): RIO plus Linformer proactively detect toxic content at scale; 94.7% of removals are automated. *(deployment)*
- **Google** [A Neural Network for Machine Translation, at Production Scale](https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/): GNMT seq2seq cuts translation errors 55 to 85% over phrase-based. *(deployment)*
- **Meta** [Transitioning entirely to neural machine translation](https://engineering.fb.com/2017/08/03/ml-applications/transitioning-entirely-to-neural-machine-translation/): LSTM-plus-attention NMT deployed across 2,000+ directions, 4.5B daily translations. *(deployment)*
- **LinkedIn** [Building The LinkedIn Knowledge Graph](https://www.linkedin.com/blog/engineering/knowledge/building-the-linkedin-knowledge-graph): Entity resolution and standardization of user-generated entities into a canonical taxonomy. *(deployment)*
- **Pinterest** [How Pinterest Fights Spam Using Machine Learning](https://medium.com/pinterest-engineering/how-pinterest-fights-spam-using-machine-learning-d0ee2589f00a): DNN plus clustering plus graph label-propagation flag spam domains and users. *(deployment)*
- **LinkedIn** [Using deep learning to detect abusive sequences of member activity](https://www.linkedin.com/blog/engineering/trust-and-safety/using-deep-learning-to-detect-abusive-sequences-of-member-activi): An LSTM classifies member activity sequences as scraping or abuse. *(eval bar)*
- **Uber** [COTA: Improving Uber Customer Care with NLP and ML](https://www.uber.com/blog/cota/): An NLP model suggests top issue types and solutions to route and resolve tickets, cutting resolution time over 10%. *(product design)*
- **Airbnb** [How ML Transforms Airbnb's Voice Support Experience](https://airbnb.tech/ai-ml/listening-learning-and-helping-at-scale-how-machine-learning-transforms-airbnbs-voice-support-experience/): Contact-reason detection classifies issues to self-serve or route to a live agent, under 50 ms. *(product design)*
- **Grammarly** [Grammatical Error Correction: Tag, Not Rewrite](https://www.grammarly.com/blog/engineering/gec-tag-not-rewrite/): GECToR tags word-level edit transformations instead of generating, running 10x faster than seq2seq. *(eval bar)*
