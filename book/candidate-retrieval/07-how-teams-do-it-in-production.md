# 7. How teams do it in production

Every large recommender converges on the same two-tower skeleton: an offline item
tower embeds the whole catalog into an ANN index, and only the user tower runs
online, emitting one vector for a single nearest-neighbor lookup before ranking.
What actually differs between companies is two decisions: **which negatives they
train against**, and **how they keep the index fresh and fast**. The architecture
everyone shares; the leverage is in the negatives and the index.

## Where the real designs diverge

| System | Negatives | ANN index | Item cadence | Why this shape |
|---|---|---|---|---|
| YouTube / Google | in-batch + logQ correction | not the focus | batch | massive power-law catalog; logQ restores an unbiased softmax |
| Airbnb | impression-not-booked (journey) | IVF over HNSW | daily batch | listings churn on price and availability; IVF absorbs updates and geo filters |
| Snap | in-batch | HNSW | few-hours refresh | high-QPS feeds; request and retrieval split into independently scaled services |
| Etsy | hard-negative mining | HNSW with 4-bit PQ | batch | relevance-critical search; PQ fits the index in memory (+5.58% purchase rate) |
| Pinterest | sampled softmax, user-level masking | learned offline index | batch, versioned hosts | request-level dedup fixes a ~30% in-batch false-negative rate |
| Uber | shared UUID layer, bag-of-words history | not the focus | batch | one global model replaces thousands of per-city models |
| Expedia | in-batch + logQ | ScaNN / OpenSearch | batch | travel query-item matching with dot-product scoring |
| Spotify | in-batch | HNSW (Voyager) | batch, stateless in-memory | a production HNSW library, 10x faster than Annoy |
| Walmart | relevance reward model, typo-aware | not the focus | batch | query-heavy search where typos and relevance dominate |

The dividing line is simple: **data and negatives buy the quality ceiling; the
index and freshness buy the latency and the unit economics.** A complete answer
picks a point on both, and justifies it from the catalog's churn and skew.

## The systems (first-party write-ups)

- **YouTube / Google** [Sampling-Bias-Corrected Neural Modeling for Large Corpus Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/): in-batch negatives are biased under a power law; the logQ correction restores an unbiased softmax. The canonical reference for the correction.
- **Airbnb** [Embedding-Based Retrieval for Airbnb Search](https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/): chose IVF over HNSW for high listing-update volume; the listing tower is offline-computable.
- **Pinterest** [Establishing a Large Scale Learned Retrieval System](https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5) and [request-level deduplication](https://medium.com/pinterest-engineering/scaling-recommendation-systems-with-request-level-deduplication-93bd514142d9): offline item embeddings plus a request-time user tower; sampled softmax with popularity correction, and user-level masking to fix in-batch false negatives.
- **Etsy** [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833): hard-negative sampling plus unified embeddings; HNSW with 4-bit product quantization; +5.58% purchase rate.
- **Snap** [Embedding-based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval): in-batch negatives for video retrieval; request and retrieval split into independently scaled services.
- **Uber** [Innovative Recommendation Applications Using Two Tower Embeddings](https://www.uber.com/blog/innovative-recommendation-applications-using-two-tower-embeddings/): layer-sharing plus bag-of-words history; one global model replaces thousands of city models.
- **Expedia** [Candidate generation using a two-tower approach](https://medium.com/expedia-group-tech/candidate-generation-using-a-two-tower-approach-with-expedia-group-traveler-data-ca6a0dcab83e): two-tower query and item encoders with dot-product scoring for travel.
- **Spotify** [Introducing Voyager](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library): a production HNSW library, 10x faster than Annoy.
- **Walmart** [Enhancing relevance of embedding-based retrieval](https://arxiv.org/abs/2408.04884): neural retrieval improved with a relevance reward model and typo-aware training.

For the full comparison (divergence diagram, choices table, math, quadrant plot),
see the dense reference in [topics/01-candidate-retrieval.md](../../topics/01-candidate-retrieval.md).
