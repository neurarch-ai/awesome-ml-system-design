# 7. How teams do it in production

Every large search system that added semantic retrieval converged on the same
skeleton: keep the lexical index, add a dense retriever, fuse the candidates, and let
a learned ranker arbitrate. What differs is **which modality carries the tail** (text
from transcripts, or the visual signal) and **how the document is represented** (one
vector, several fields, or discrete codes).

## Where the real designs diverge

| System | Tail signal | Document representation | Fusion | When it wins | Watch out |
|---|---|---|---|---|---|
| Podcast and long-form audio search (Spotify) | ASR transcript plus metadata, dense text retrieval | Episode-level text embedding | Dense candidates blended with keyword results by a final ranker | Descriptive queries over spoken content | ASR quality bounds the whole system; language coverage is uneven |
| Visual discovery (Pinterest) | Visual embedding | Image embedding per pin, plus text | Visual and text candidates merged in ranking | Queries that are about what something looks like | Visual similarity is not intent; needs engagement to disambiguate |
| Image-text aligned retrieval (CLIP lineage) | Joint query-image space | One vector per frame or image | Usually dense-only in research settings | Zero-shot coverage of concepts with no training data | Shallow per-frame semantics; weak on text-in-image and rare entities |
| Generative retrieval with semantic IDs (Spotify podcast discovery) | Discrete codes decoded by a sequence model | A short sequence of semantic ID tokens | Retrieval becomes decoding | Generalization to items with little interaction data | New paradigm: index updates and cold items behave differently from ANN |
| Classic lexical-first web and video search | Exact terms over title, description, transcript | Inverted index over fields | Lexical only, or lexical with a semantic layer added | Navigational and entity queries, instant freshness | Vocabulary mismatch on descriptive tail queries |

## What they share

Three properties show up in every one of them:

1. **Lexical retrieval is never removed.** It is instantly fresh, exact, explainable,
   and it answers the head. Semantic retrieval is added beside it.
2. **The document side is precomputed.** Whatever the representation, it must be
   producible at ingestion and storable in an index, which is what rules out
   cross-encoders in retrieval.
3. **Engagement arrives at ranking, not retrieval.** Otherwise new items are
   structurally unreachable and the feedback loop closes.

## The systems (first-party sources)

- **Spotify** [Introducing Natural Language Search for Podcast Episodes](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes): dense retrieval trained on (query, episode) pairs from search logs, with in-batch negatives, ANN serving, and a final ranker that blends semantic candidates with the existing keyword results.
- **Spotify** [Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery](https://arxiv.org/abs/2603.17540): representing items as discrete semantic-ID sequences so retrieval becomes decoding, deployed at scale.
- **Pinterest** [Visual Search at Pinterest](https://arxiv.org/abs/1505.07647): the production account of building visual retrieval, including the engineering tradeoffs of maintaining an image embedding index.
- **OpenAI** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020): the image-text aligned embedding space that makes text-to-frame retrieval possible without task-specific training.
- **OpenAI** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356): the ASR line that turns audio into the transcript field this chapter leans on.
- **Facebook AI** [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906): the dual-encoder retrieval recipe with in-batch negatives that the text half of this system is built on.
- **Meta** [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) and **Google** [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396): the ANN index implementations that make dense retrieval affordable, and the quantization that decides the recall-versus-memory curve.

## The dividing line

Two questions place any design in the table. **Where does the tail signal come
from?** If the content is spoken, transcripts dominate and this is a text retrieval
problem wearing a video costume. If it is visual or multilingual with poor ASR, the
visual tower has to carry it, and the cost profile changes completely. **How many
vectors per item can you afford?** One vector per video is a search product; twenty
is a moment-retrieval product; six hundred is a research demo.
