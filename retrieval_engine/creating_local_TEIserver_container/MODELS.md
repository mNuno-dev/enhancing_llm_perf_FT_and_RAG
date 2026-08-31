# Embedding Models Reference

This document lists the embedding models considered for use with the local TEI server container, grouped by TEI compatibility and language coverage.

---

## Open-Source Models

### TEI-Compatible Models

#### Multilingual

| Model | Parameters | Notes |
|---|---|---|
| `Qwen3/Qwen3-Embedding-8B` | ~8B | Large; high GPU memory requirement |
| `Qwen3/Qwen3-Embedding-0.6B` | ~0.6B | Requires ~28 GB GPU memory |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 278M | |
| `BAAI/bge-m3` | 568M | Also available via Ollama |
| `intfloat/multilingual-e5-large-instruct` | 560M | Top ranking on MTEB Multilingual v2 |

#### English-Only

| Model | Parameters | Notes |
|---|---|---|
| `nomic-ai/nomic-embed-text-v1.5` | — | Also available via Ollama |

---

### Not TEI-Compatible

#### Portuguese Sentence Encoders (PORTULAN/Serafim)

**General sentence embedders:**

| Model |
|---|
| `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder` |
| `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder` |
| `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder` |

**Information-retrieval (IR) optimised:**

| Model |
|---|
| `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir` |
| `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir` |
| `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir` |

---

### Models from MTEB Multilingual v2 Leaderboard (TEI-Compatible)

#### Small (recommended for constrained hardware)

| Model | Parameters |
|---|---|
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 278M |
| `BAAI/bge-m3` | 568M |
| `intfloat/multilingual-e5-large-instruct` | 560M |

#### Medium

| Model | Parameters |
|---|---|
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | ~1.5B |
| `Qwen3/Qwen3-Embedding-4B` | ~4B |

#### Large

| Model | Parameters | Notes |
|---|---|---|
| `Salesforce/SFR-Embedding-Mistral` | ~7B | High latency; significant GPU memory requirement |

---

## Proprietary / API-Based Models

| Model | Provider |
|---|---|
| `gemini-embedding-001` | Google |
| `Cohere-embed-multilingual-light-v3.0` | Cohere |
