# Additional RETRIEVAL fine-tuning examples

These configurations are extracted from #4102 for separate recipe review.
They have not been validated with GPU training or memory measurements.
They remain excluded from automatic GPU CI until that validation is recorded.

Set up the [AutoModel environment](https://docs.nvidia.com/nemo/automodel/nightly/get-started/installation),
then run from the repository root on one node with eight GPUs:

```bash
uv run automodel examples/retrieval/bi_encoder/llama_embed_nemotron_8b_finetune.yaml --nproc-per-node 8
```

Each configuration includes its launch command and the Hugging Face revision used
when inspecting the checkpoint configuration. Access to gated checkpoints requires
accepting the publisher terms and authenticating with Hugging Face.
When changing GPU count, keep the global batch divisible by the local batch times
the data-parallel size. Training and validation may download the referenced datasets.

| Checkpoint | Configuration |
| --- | --- |
| `nvidia/llama-embed-nemotron-8b` | [llama_embed_nemotron_8b_finetune.yaml](bi_encoder/llama_embed_nemotron_8b_finetune.yaml) |
| `mistralai/Ministral-3-3B-Base-2512` | [ministral_3_3b_base_2512_finetune.yaml](bi_encoder/ministral_3_3b_base_2512_finetune.yaml) |
| `nvidia/llama-nemotron-rerank-1b-v2` | [llama_nemotron_rerank_1b_v2_finetune.yaml](cross_encoder/llama_nemotron_rerank_1b_v2_finetune.yaml) |
