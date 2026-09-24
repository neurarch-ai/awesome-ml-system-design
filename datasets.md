# Datasets, by what you would use them for

"Where would the data come from?" is asked in almost every ML system design loop, and
it is the question candidates answer worst. The usual answer names a model and waves at
the data. This page is the other half: the public datasets people actually build and
benchmark on, grouped by the problem that consumes them.

Same cap and same rule as [papers.md](papers.md): at most eight per section, and each
entry says what it is for and what is wrong with it. A dataset with no stated limit is
a dataset nobody has used in anger.

**Four questions to ask about any dataset, in this order.** They come up as follow-ups
more often than the names do.

1. **What is the licence, and does it cover what you want to do?** Several of the
   competition datasets below are research-only, and a few of the recommendation ones
   forbid redistribution. Check the card, not the paper.
2. **Is there a time dimension, and did you respect it?** Most of these are logs. A
   random train/test split on log data leaks the future, and the offline number that
   results is the single most common reason an offline win does not reproduce online.
3. **What was the logging policy?** Every interaction dataset is the output of some
   ranker, so it is biased by what that ranker showed. Nothing here is a random sample
   of user preference, and the missing-not-at-random problem is the reason
   [counterfactual evaluation](topics/18-cold-start-and-exploration.md) exists.
4. **How stale is it?** MovieLens and the Criteo logs are still the standard baselines
   and are also old enough that the distributions no longer look like a live product.

---

## Recommendation and CTR

- **[MovieLens](https://grouplens.org/datasets/movielens/)**. The default
  recommendation benchmark, from 100k up to 32M ratings. Use it to sanity-check an
  implementation, and know its limits: explicit ratings, a small item catalogue, and
  results on it have poor correlation with production behaviour.
- **[Criteo 1TB click logs](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)**.
  24 days of display-ad click logs with 13 numeric and 26 hashed categorical features,
  and the reason "high-cardinality embedding tables" is a systems problem. The
  benchmark most CTR papers report on, usually via a
  [smaller standardized split](https://huggingface.co/datasets/reczoo/Criteo_x1).
  The features are anonymized, so nothing about feature engineering transfers.
- **[Avazu](https://www.kaggle.com/c/avazu-ctr-prediction)**. Ten days of mobile ad
  click logs. Smaller than Criteo and with a real time ordering, which makes it the
  better one for demonstrating that you split by time.
- **[Taobao UserBehavior](https://tianchi.aliyun.com/dataset/649)**. Around 100M
  interactions with behaviour types (click, cart, favourite, buy) over a week. The
  standard set for sequential recommendation, and the one where behaviour-type
  modelling is actually testable.
- **[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)**. Reviews plus rich
  item metadata across categories, which is what makes it usable for content-based and
  cold-start work rather than only for collaborative filtering.
- **[MIND](https://msnews.github.io/)**. Microsoft News: impression logs with the full
  candidate list, not just clicks. That detail is rare and it is what makes an honest
  ranking evaluation possible.
- **[Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)**.
  Reviews, businesses and check-ins with a geographic dimension, useful for local
  recommendation and for cold start on new businesses.

## Search and retrieval

- **[MS MARCO](https://microsoft.github.io/msmarco/)**. Real Bing queries with passage
  relevance labels. The default training set for dense retrievers, and the reason most
  public embedding models are tuned to short web-search-shaped queries.
- **[BEIR](https://github.com/beir-cellar/beir)**. Zero-shot retrieval across many
  domains. The result to remember is that a model tuned on MS MARCO often does not
  transfer, which is why you evaluate retrieval on your own corpus.
- **[Amazon ESCI (shopping queries)](https://github.com/amazon-science/esci-data)**.
  Multilingual product search with graded relevance (exact, substitute, complement,
  irrelevant). The closest public thing to e-commerce search relevance, and the graded
  labels are what make it interesting.
- **[MTEB](https://huggingface.co/spaces/mteb/leaderboard)**. Not a dataset but the
  embedding leaderboard. Useful for shortlisting, heavily overfit as a target, so
  re-rank the shortlist on your data.

For a production search system none of these is your evaluation set. They are how you
calibrate components before you have your own judged queries, and a few hundred queries
labelled by your own domain experts beats all of them for the decision you are making.

## Fraud, risk and tabular

- **[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)**. Real
  e-commerce transactions with rich, partly anonymized features. The best public
  approximation of the actual problem, including the identity and transaction join.
- **[Credit Card Fraud (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**.
  284,807 transactions with 492 frauds, 0.172 percent positive. Use it to talk about
  class imbalance and threshold selection; do not use it to talk about features, since
  they are PCA components.
- **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)**.
  Credit scoring across several joined tables, which makes it the public dataset that
  most resembles a real feature-engineering problem.
- **[Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult)**. Small,
  ancient, and still the standard example for fairness and calibration discussions,
  partly because everyone already knows its columns.
- **[Criteo Uplift](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)**. 13M
  rows from a randomized treatment assignment, which is what uplift modelling needs and
  almost never gets publicly.

The pattern to notice: the tabular problems have public data with real features, and
the recommendation problems mostly do not. That asymmetry is why tabular papers argue
about feature engineering and recommendation papers argue about architectures.

## Forecasting

- **[M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy)**. Walmart unit sales,
  hierarchical across item, store and state, with prices and events. The competition
  that made hierarchical reconciliation and probabilistic forecasting mainstream.
- **[M4](https://github.com/Mcompetitions/M4-methods)**. 100,000 series across
  frequencies, and the source of the result every forecasting answer should carry: a
  well-tuned statistical baseline is very hard to beat, and the winner was a hybrid.
- **[UCI electricity and traffic](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)**.
  The two long multivariate series that most deep forecasting papers report on. Small,
  clean, and not representative of a demand-planning problem.

The honest note for this section: public forecasting data is mostly competitions, and
competitions reward accuracy on a fixed horizon rather than the decision the forecast
feeds. Say what the forecast is for before you pick a metric.

## Vision

- **[ImageNet](https://www.image-net.org/)**. The pretraining and benchmarking default.
  Know that its label noise and its single-label assumption are both real, and that
  ImageNet accuracy is a weak predictor of performance on your domain.
- **[COCO](https://cocodataset.org/)**. Detection, segmentation and captioning with 80
  common categories. The standard detection benchmark, and small-object performance is
  where its metrics hide the most.
- **[Open Images](https://storage.googleapis.com/openimages/web/index.html)**. Nine
  million images with box, segmentation and relationship annotations. Larger and
  sparser than COCO, with partial labels, which is closer to how a real annotation
  budget behaves.

## Speech and audio

- **[LibriSpeech](https://www.openslr.org/12)**. 1,000 hours of read English
  audiobooks. The ASR benchmark, and a poor proxy for conversational or noisy audio,
  which is what your product has.
- **[Common Voice](https://commonvoice.mozilla.org/en/datasets)**. Crowdsourced,
  multilingual, with speaker demographics. The best public set for accent and language
  coverage, and its recording conditions vary wildly by design.
- **[AudioSet](https://research.google.com/audioset/)**. Two million ten-second
  YouTube clips labelled with an ontology of 632 event classes. The reference for
  general audio tagging, distributed as identifiers, so it decays as videos disappear.

## Language

- **[GLUE](https://gluebenchmark.com/)**. The classification and inference suite that
  fine-tuned encoders are compared on. Saturated, and useful mostly as a smoke test now.
- **[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)**. Extractive question
  answering. Read v2 rather than v1: the unanswerable questions are the part that
  matters for a production system, which has to abstain.

For anything LLM-shaped (instruction data, preference data, LLM evaluation), the
companion [LLM System Design repo](https://github.com/neurarch-ai/awesome-llm-system-design)
has its own `datasets.md` covering instruction data, preference data and LLM evaluation,
rather than duplicating it here.

## Graphs

- **[Open Graph Benchmark](https://ogb.stanford.edu/)**. Node, link and graph tasks at
  a size where the systems constraints are real, with standardized splits that are
  usually temporal or scaffold-based rather than random. That split discipline is the
  reason to prefer it over the older citation benchmarks.

---

## Contributing

Same rules as [papers.md](papers.md): eight per section, one line on what it is for and
one on its limit, and a swap argued in the pull request when a section is full. Do not
add a dataset without checking its card for the licence, and do not restate a size you
have not seen stated on the card or in the paper. See [CONTRIBUTING.md](CONTRIBUTING.md).
