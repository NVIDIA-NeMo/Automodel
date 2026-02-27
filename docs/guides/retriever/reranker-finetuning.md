# Reranker Model Fine-Tuning with NeMo AutoModel

:::{important}
Reranker fine-tuning support in NeMo AutoModel is **coming soon**. This page will be updated with full documentation when the feature is available.
:::

## Overview

Reranker models take a (query, document) pair and produce a relevance score, providing a more accurate ranking signal than embedding-based retrieval alone. They are typically used as a second-stage ranker in retrieval pipelines: an embedding model retrieves a candidate set, then the reranker re-scores and re-orders those candidates.

### Planned Features

- Cross-encoder reranker architecture using Llama-family backbones
- Pointwise and listwise training objectives
- Seamless integration with the existing biencoder pipeline for end-to-end retrieval fine-tuning
- LoRA/PEFT support for parameter-efficient adaptation
- FSDP2 distributed training

### Typical Retrieval Pipeline

```
Query → Embedding Model (fast retrieval) → Top-K candidates → Reranker (accurate re-scoring) → Final ranking
```

Fine-tuning both the embedding model and reranker on the same domain data produces the best end-to-end retrieval quality.

## Related Resources

- [Embedding Model Fine-Tuning](embedding-finetuning.md) — train a biencoder embedding model with NeMo AutoModel
- [Biencoder Retrieval Dataset](../llm/retrieval-dataset.md) — dataset formats for retrieval training
