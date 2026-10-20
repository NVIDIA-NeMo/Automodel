# Additional VLM fine-tuning examples

These configurations are extracted from #4102 for separate recipe review.
They have not been validated with GPU training or memory measurements.
They remain excluded from automatic GPU CI until that validation is recorded.

Set up the [AutoModel environment](https://docs.nvidia.com/nemo/automodel/nightly/get-started/installation),
then run from the repository root on one node with eight GPUs:

```bash
uv run automodel examples/vlm_finetune/huggingfacetb/smolvlm_instruct_medpix_peft.yaml --nproc-per-node 8
```

Each configuration includes its launch command and the Hugging Face revision used
when inspecting the checkpoint configuration. Access to gated checkpoints requires
accepting the publisher terms and authenticating with Hugging Face.
When changing GPU count, keep the global batch divisible by the local batch times
the data-parallel size. Training and validation may download the referenced datasets.

| Checkpoint | Configuration |
| --- | --- |
| `HuggingFaceTB/SmolVLM-Instruct` | [smolvlm_instruct_medpix_peft.yaml](huggingfacetb/smolvlm_instruct_medpix_peft.yaml) |
| `llava-hf/llava-1.5-7b-hf` | [llava_1_5_7b_hf_medpix_peft.yaml](llava_hf/llava_1_5_7b_hf_medpix_peft.yaml) |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | [llama_4_scout_17b_16e_instruct_medpix_peft.yaml](meta_llama/llama_4_scout_17b_16e_instruct_medpix_peft.yaml) |
| `Qwen/Qwen3-VL-4B-Instruct` | [qwen3_vl_4b_instruct_medpix_peft.yaml](qwen/qwen3_vl_4b_instruct_medpix_peft.yaml) |
