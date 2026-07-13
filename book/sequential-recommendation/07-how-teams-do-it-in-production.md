# 7. How teams do it in production

Every large sequential recommender shares the same skeleton: build a per-user
ordered sequence, run a self-attention or recurrent encoder to produce a
user-intent vector, and inject that vector into ranking or retrieval. What
actually differs between companies is four decisions: **which encoder they
chose**, **how much history they carry**, **how fresh the user state is**, and
**whether the model is per-surface or a shared foundation**. The architecture
everyone shares; the leverage is in freshness and scope.

## Where the real designs diverge

| System | Encoder | History | Freshness | Funnel position | Why this shape |
|---|---|---|---|---|---|
| Alibaba BST | Self-attention (1 block) | Recent window | Batch | Ranking (CTR) | Shallow attention fits CTR latency budget; one block was enough for significant lift over concat-features baseline |
| Alibaba DIN | Activation-unit pool (no order) | Recent history, per-candidate | Batch | Ranking (CTR) | Interest depends on the candidate ad; attention pool recomputed per candidate; no sequence order needed |
| Pinterest TransAct | Transformer encoder (multi-layer) | Last 100 real-time actions | Real-time streaming | Ranking (Homefeed) | Same-session responsiveness is the product value; GPU serving absorbs the Transformer cost |
| Pinterest PinnerFormer | Transformer with all-action loss | Longer horizon | Daily batch | Retrieval + ranking feature | All-action loss closes most of the freshness gap without streaming infrastructure |
| Kuaishou TWIN V2 | Two-stage attention (GSU + ESU) | Lifelong (~10^6 events) | Offline clustering, online inference | Ranking (CTR) | Lifelong histories carry signal that a short window throws away; two-stage keeps online cost tractable |
| Netflix Foundation Model | Large foundation Transformer | Long-horizon pretraining | Monthly pretrain + daily update | Retrieval + ranking (multiple surfaces) | One pretrained model amortizes across many personalization surfaces |
| Spotify CoSeRNN | Recurrent (RNN) per session | Session-level + long-term offset | Real-time at session start | Retrieval (ANN over tracks) | Session context dominates music intent; RNN per session plus stable long-term base |
| Instacart | BERT-style Transformer (masked-LM) | Recent 20 tokens | Centralized serving | Retrieval (search, browse, recs) | One model across many surfaces; bidirectional context; strong lift in cart additions |
| Etsy adSformers | Transformer encoder (short-term) | Short recent window | Batch short-term | Ranking (ad CTR, CVR) | Short-term intent for ads; enriched with visual/multimodal pretraining |
| Wayfair MARS | Self-attention (stacked layers) | Recent 100 views | Batch | Ranking / recs | Browsing sequences track taste shifts; significant recall lift over matrix factorization |
| LinkedIn Feed SR | Transformer sequential ranker | Recent feed interactions | Batch-trained ranker | Ranking (Feed) | Replaces a DCNv2 ranker with a sequence-native model at 1.2B-member scale |
| Airbnb | Word2vec-style session embeddings | Session-level (click + skip windows) | Real-time Kafka windows | Retrieval + search ranking | Co-occurrence embeddings over 800M sessions; EmbClickSim pushes similar listings up, EmbSkipSim pushes skipped-similar listings down |

The dividing line: **data and sequence quality buy the quality ceiling; freshness
and scope buy the business value**. A sequence model that is only updated daily
is a lifestyle-aggregate model with extra compute; the same model updated in
real time is a session-aware assistant.

## The systems (first-party write-ups)

- **Alibaba** [Behavior Sequence Transformer for E-commerce Recommendation](https://arxiv.org/abs/1905.06874): a Transformer over the user's ordered behavior sequence lifts CTR in Taobao ranking.
- **Alibaba** [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978): a candidate-conditioned attention pool adapts the user-interest vector per ad; intentionally has no softmax normalization.
- **Pinterest** [How Pinterest Leverages Realtime User Actions (TransAct)](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8): TransAct fuses real-time last-100 actions into Homefeed ranking; early fusion of the candidate was critical.
- **Pinterest** [PinnerFormer: Sequence Modeling for User Representation](https://arxiv.org/abs/2205.04507): daily batch Transformer with all-action loss avoids streaming costs while mostly closing the freshness gap.
- **Kuaishou** [TWIN V2: ultra-long user behavior sequence modeling](https://arxiv.org/abs/2407.16357): two-stage attention (GSU retrieve then ESU score) over lifelong histories up to 10^6 events.
- **Netflix** [Integrating Netflix Foundation Model into Personalization](https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb): three integration modes (embedding store, subgraph graft, full fine-tune) for one foundation model across many surfaces.
- **Spotify** [Contextual and sequential user embeddings for music (CoSeRNN)](https://research.atspotify.com/contextual-and-sequential-user-embeddings-for-music-recommendation/): per-session embeddings modeled as a recurrent sequence plus a stable long-term preference base.
- **Instacart** [Sequence models for contextual recommendations](https://tech.instacart.com/sequence-models-for-contextual-recommendations-at-instacart-93414a28e70c): centralized BERT-style next-product retrieval across search, browse, and recs surfaces with strong gains in cart additions.
- **Etsy** [adSformers: personalization from short-term sequences](https://arxiv.org/abs/2302.01255): Transformer encoder over recent actions enriched with visual/multimodal representations for ad CTR and CVR.
- **Wayfair** [MARS: Transformer networks for sequential recommendation](https://www.aboutwayfair.com/careers/tech-blog/mars-transformer-networks-for-sequential-recommendation): self-attention over browsed-item sequences tracks shifting tastes with significant recall improvement over matrix factorization.
- **LinkedIn** [An industrial-scale sequential recommender for feed ranking (Feed SR)](https://arxiv.org/abs/2602.12354): Transformer sequential ranker replacing DCNv2 at 1.2B-member scale with measured gains in time spent.
- **Airbnb** [Listing Embeddings in Search Ranking](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e): word2vec-style listing embeddings from 800M sessions for real-time in-session personalization.

For the full comparison (decision-tree divergence diagram, choices table, math,
quadrant plot), see the dense reference in
[topics/03-sequential-recommendation.md](../../topics/03-sequential-recommendation.md).
