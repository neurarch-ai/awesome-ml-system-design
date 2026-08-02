# Reading paths

Twenty chapters is a lot to read front to back before a loop next week. These are
the orders that make sense for a specific interview, each with what the loop
actually tests.

Every path assumes you have read [the method](00-the-method.md) first. It is short
and it is the spine everything else hangs off.

## If you have one week

| Day | Read | Why |
|---|---|---|
| 1 | [The method](00-the-method.md), [candidate retrieval](candidate-retrieval/) | The frame, then the funnel that every recommendation question starts from |
| 2 | [Ranking](ranking/) | The most-probed chapter in the book: features, interactions, multi-task, calibration |
| 3 | [Feature store](feature-store/), [real-time serving](realtime-serving/) | Training-serving skew and the seam where offline wins die |
| 4 | [Experimentation](experimentation/) | Every design ends at "how would you know it worked" |
| 5 | Your domain chapter ([ads](ads-ctr/), [search](search-ranking/), [fraud](fraud-detection/), [CV](computer-vision/), [NLP](nlp/), [forecasting](forecasting/)) | The one the job description names |
| 6 | [Numbers to know](numbers-to-know.md), [monitoring](monitoring/) | Recall drills, then the "it is live and degrading" follow-up |
| 7 | [Mock interview](mock-interview.md), then re-read your weakest chapter's Q&A | Practice the delivery, not just the content |

## By role

**Recommendations and personalization.** [Candidate retrieval](candidate-retrieval/)
then [ranking](ranking/) then [sequential recommendation](sequential-recommendation/)
then [cold start](cold-start/) then [embeddings](embeddings/) then
[experimentation](experimentation/). The loop tests the retrieve-then-rank funnel,
what each stage optimizes, and whether you can name the failure that kills offline
wins (position bias, training-serving skew, feedback loops).

**Ads and monetization.** [Ads CTR](ads-ctr/) then [ranking](ranking/) then
[experimentation](experimentation/) then [feature store](feature-store/) then
[tabular](tabular/) for uplift. Calibration is the differentiator here: the score
feeds an auction, so being well-ordered is not enough, and the downsampling
correction is a question you should expect verbatim.

**Search and relevance.** [Search ranking](search-ranking/) then
[candidate retrieval](candidate-retrieval/) then [embeddings](embeddings/) then
[video and multimodal search](video-search/) if the product has video or images,
then [experimentation](experimentation/) (interleaving especially), then the LLM
companion's [semantic search](https://github.com/neurarch-ai/awesome-llm-system-design/tree/main/book/semantic-search/)
and [RAG](https://github.com/neurarch-ai/awesome-llm-system-design/tree/main/book/rag-serving/)
chapters if the role touches generative retrieval.

**Trust, safety, and risk.** [Fraud detection](fraud-detection/) then
[content moderation](content-moderation/) then [monitoring](monitoring/) then
[experimentation](experimentation/). The loop tests class imbalance, cost-sensitive
thresholds, label delay, and adversaries who adapt faster than your retraining
cadence.

**Perception (vision, speech, multimodal).** [Computer vision](computer-vision/)
then [speech](speech/) then [embeddings](embeddings/) then
[real-time serving](realtime-serving/). Expect labeling cost, transfer learning, GPU
serving economics, and streaming versus batch tradeoffs.

**Platform, MLOps, and infrastructure.** [Feature store](feature-store/) then
[real-time serving](realtime-serving/) then [monitoring](monitoring/) then
[experimentation](experimentation/) then [embeddings](embeddings/). The loop tests
whether you can make other teams' models reproducible, fresh, observable, and cheap.

**Forecasting and operations research adjacent.** [Forecasting](forecasting/) then
[tabular](tabular/) then [experimentation](experimentation/) then
[monitoring](monitoring/). Probabilistic forecasts, backtesting that respects time,
and the forecast-then-optimize seam are the recurring probes.

**Crossing into LLM work.** Read this book's [ranking](ranking/),
[experimentation](experimentation/), and [feature store](feature-store/), then the
LLM companion's [RAG](https://github.com/neurarch-ai/awesome-llm-system-design/tree/main/book/rag-serving/),
[evaluation](https://github.com/neurarch-ai/awesome-llm-system-design/tree/main/book/evaluation/),
and [benchmarking](https://github.com/neurarch-ai/awesome-llm-system-design/tree/main/book/benchmark-eval/)
chapters. Most LLM loops still ask a classic-ML question somewhere, and the
statistics transfer directly.

## By question you were told to expect

| They said | Read, in this order |
|---|---|
| "Design a recommendation system" | [Candidate retrieval](candidate-retrieval/), [ranking](ranking/), [sequential recommendation](sequential-recommendation/), [experimentation](experimentation/) |
| "Design ads ranking" | [Ads CTR](ads-ctr/), [ranking](ranking/), [experimentation](experimentation/) |
| "Design search" | [Search ranking](search-ranking/), [candidate retrieval](candidate-retrieval/), [embeddings](embeddings/) |
| "Design video or image search" | [Video and multimodal search](video-search/), [embeddings](embeddings/), [computer vision](computer-vision/), [speech](speech/) |
| "Detect fraud or abuse" | [Fraud detection](fraud-detection/), [content moderation](content-moderation/), [monitoring](monitoring/) |
| "Our model degraded in production" | [Monitoring](monitoring/), [feature store](feature-store/), [real-time serving](realtime-serving/) |
| "How would you test this" | [Experimentation](experimentation/), [ranking](ranking/) (offline metrics), [monitoring](monitoring/) |
| "Serving and infrastructure" | [Real-time serving](realtime-serving/), [feature store](feature-store/), [embeddings](embeddings/) |
| Unspecified, a general ML loop | The one-week path above |

## How to read a chapter under time pressure

Twenty minutes and a chapter on the list: read the **README** (framing and diagram),
then **08 Interview Q&A**, then **09 Summary** and attempt the test-yourself
questions, then the capstone table in **10**. That gives you the answers and the
numbers. The middle sections are where the understanding is, and they are what you
read when you have time to do it properly.
