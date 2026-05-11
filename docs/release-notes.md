# Release Notes

## 0.4.0 · 26.04 (2026-04-28) · [PyPI](https://pypi.org/project/nemo-automodel/0.4.0/) · [GH](https://github.com/NVIDIA-NeMo/Automodel/releases/tag/v0.4.0) · [NGC Docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/tags?version=26.04.00)

### Highlights

- **Discrete-diffusion LLMs (dLLM).** SFT and generation support for dLLM
  models, including Llada.
- **Embedding and retrieval training.** Reranker training, biencoder datasets
  loaded directly from the HuggingFace Hub, in-batch negative sampling, and
  ONNX export for biencoder models.
- **SkyPilot launcher.** Native multi-node launch on cloud (SkyPilot,
  including Kubernetes), in addition to local interactive runs. SkyPilot and
  NeMo-Run launchers are selected with YAML sections in the config; SLURM jobs
  use the `sbatch slurm.sub` workflow.
- **CLI install profile.** The `nemo-automodel[cli]` extra declares `pyyaml`
  beyond the package's base dependencies for job-submission configs.
- **Refreshed CLI.** `automodel <config.yaml>` (alias `am`) replaces the older
  `automodel <command> <domain> -c <config>` form.

### New Models

- **LLM:** GLM-5, MiniMax-M2.5, Nemotron Super v3, Nemotron Nano 4B/8B.
- **MoE / VLM:** Qwen3.5-MoE (397B-A17B, 35B-A3B).
- **VLM:** Gemma 4, Mistral Small 4, Qwen3.5 small dense models.
- **Diffusion:** FLUX.1-dev, Wan 2.1 T2V, HunyuanVideo 1.5; Wan
  multi-resolution and LoRA recipes for diffusion.

### Distributed Training

- Context parallelism for Qwen3.5-MoE and Nemotron v3.
- Pipeline parallelism for knowledge distillation.
- HybridEP and UCCL-EP as alternative expert-parallel dispatchers.
- FSDP2 weight prefetching and async TP optimization.
- TP > 1 in knowledge distillation.

### Performance and Kernels

- TE Linear layers enabled for PEFT/LoRA.
- `torch._grouped_mm` expert backend.
- fp32 RMSNorm backend and `cast_model_to_dtype` controls.
- TP-aware KD loss with distributed softmax and T² scaling.
- FlashOptim optimizer integration.
- Sequence-packing updates: Qwen3.5-MoE VLM neat-packing recipe with EP+PP;
   generic THD collation for chat datasets; CP/BSHD padding fixes.
  BSHD+CP, and padding fixes.

### PEFT

- MoE LoRA: rank scaling, `torch_mm` integration, expert-LoRA init using
  `config.expert_dim`.
- `merge_lora` tool for materializing adapters into the base model.
- QLoRA PEFT checkpoints saved with the HF adapter prefix.

### Recipes and Workflow

- New recipes for Gemma 4 (LoRA), Nemotron Nano 4B SQuAD, Mistral Small 4,
  Tulu-3 E2E convergence, GPT-OSS 20B / Moonlight 16B convergence, and
  reranker / biencoder training.
- MFU logging for LLM and dLLM train recipes.
- Native Comet ML experiment tracking.
- NEFTune noisy embeddings for instruction fine-tuning.
- Scheduler-driven manual garbage collection.
- Common inference utility and `.generate()` with KV cache for Nemotron v3.

### Checkpointing

- `v4_compatible` checkpoint format.
- Diffusion full fine-tuning and pretraining examples use safetensors
  checkpoint format; diffusion LoRA examples use `torch_save`.
- QLoRA / LoRA loading robustness; tied-weight handling moved out of
  `_init_model`.

### Notable Fixes

- FSDP2 meta-device crash for Qwen3.5 GatedDeltaNet fp32 params.
- Activation checkpointing silently skipped on registered VLMs (ModuleList
  flattening).
- Gradient checkpointing for MoE models on single GPU (`ep_size=1`).
- Gradient clipping with `torch_mm` + EP (GPT-OSS 120B recipe).
- Rotary embeddings for v4 models; `inputs_embeds` passthrough for Nano v3.

### Breaking Changes

A migration guide for the new CLI, the `recipe` YAML section, the SLURM
`sbatch`-script workflow, and the `nemo-automodel[cli]` install profile is in
[Breaking Changes](breaking-changes.md).

---

## 0.3.0 · 26.02 (2026-02-26) · [PyPI](https://pypi.org/project/nemo-automodel/0.3.0/) · [GH](https://github.com/NVIDIA-NeMo/Automodel/releases/tag/v0.3.0) · [NGC Docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/tags?version=26.02.00)

### Highlights

- **Transformers v4 / v5 alignment.** New `transformers v4` API support and a
  v5 refactor for device-mesh-only model init.
- **Streaming safetensors writer** for faster checkpoint export.
- **Faster fp8 dequant kernels** with DTensor dequantization fixes for DSv3.

### New Models

- **LLM:** DeepSeek V3.2, Step-3.5-Flash, MiniMax-M2.1,
  Nemotron-3-Nano-30B-A3B, Nemotron Flash 1B, GLM-4.7,
  Devstral-Small-2-24B.
- **MoE / VLM / Omni:** Qwen3-VL (4B/8B), Qwen3-VL-MoE (30B/235B),
  Kimi-VL, Kimi-K2.5 VL, Nemotron-Parse VLM, InternVL3.5-4B,
  Ministral3 (3B/8B/14B), Phi-4-multimodal.

### Distributed Training

- v5 refactor: device-mesh-only model init.
- TP plan for Ministral; Ministral3 ported to transformers v4.
- Pipeline-parallelism validation support.
- Parallel diffusers `generate`.

### Performance and Kernels

- TE fp8 for models that support it.
- `GroupedExpertsTE` backend (prerequisite for MoE fp8).
- TE RoPE fusion for custom MoE models; norm fusion and RoPE cache for dense
  models.
- Improved import time.

### PEFT

- DoRA implementation.
- LoRA support for custom MoEs.
- LoRA support in Biencoder.

### Datasets and Workflow

- Databricks Deltalake dataset support; consolidation for Databricks.
- Parquet file support; inline text dataset format.
- `ColumnMapped`: configurable special tokens, chat-template flags, and
  answer-only masking.
- Hard negative mining and biencoder + inline-dataset tests.
- nsys benchmark support and model-layer name scoping in the CLI.
- Updated checkpoint auto-loading with explicit `restore_from`.
- Dion optimizer.
- Functiongemma + xlam tool-calling recipes.

### Notable Fixes

- `inputs_embeds` passthrough for Nano v3.
- `from_pretrained` / `from_config` simplification with model-id pass-through.
- Tied-embedding detection improvements.

---

## 0.2.0 · 25.11 (2025-12-04) · [PyPI](https://pypi.org/project/nemo-automodel/0.2.0/) · [GH](https://github.com/NVIDIA-NeMo/Automodel/releases/tag/v0.2.0) · [NGC Docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/tags?version=25.11.00)

### Highlights

- **Async checkpointing.** Checkpoint refactor with async DCP and HF
  safetensors backport / consolidation.
- **Custom MoE optimizations.** FSDP optimizations, packed-sequence + context
  parallel via TE, configurable router precision, fp32 `lm_head` and
  fp32 `apply_rope`.
- **Performance documentation.** New performance-summary doc and benchmarking
  recipe with configs.
- **Multinode + cluster guidance.** Multinode configs and updated launcher
  docs.

### New Models

- **MoE:** Qwen3 MoE custom implementation, Qwen3 Next, GPT-OSS (custom
  implementation, dequantization, DGX Spark recipe), GLM 4 / 4.5 / 4.6 MoE,
  GLM 4.5 Air, Moonlight 2L test, Phi 4 (TP plan).
- **Omni / VLM:** Qwen3-Omni OOTB recipe and custom implementation.
- **DeepSeek v3** with fp8 base checkpoint loading.
- **Sequence classification:** Qwen3ForSequenceClassification registered;
  generic SFT sequence-classification recipe.

### Distributed Training

- VLM expert-parallel recipe support.
- PP for VLM; PEFT with PP.
- Sharding optimization for SP / LoRA.
- `clip_grad_norm` across all parallelism modes.
- `fully_shard_by_dtype` option.
- Out-of-tree (OOT) parallelism decorator.

### Performance and Kernels

- Mask creation moved into the data pipeline for better perf.
- TE attention for GPT-OSS.
- Faster fp8 dequant; auto-detect base-weights dequant.

### PEFT

- LoRA-aware `ColwiseParallel` / `RowwiseParallel`.
- LoRA + TE.
- MFU estimation for LoRA.
- Additional PEFT LoRA recipes.

### Datasets and Recipes

- Multiturn chat dataset; VLM multiturn chat support.
- Tool-calling dataset and recipe.
- Streaming dataset.
- Multiple validation datasets with per-dataset logging.
- ColumnMapped: surface truncating + padding options.
- Configurable max-clip-grad; configurable remote-logging frequency via
  `step_scheduler`.
- Validation-loss checkpoint, run-val-at-ckpt, best-ckpt symlink.
- InternVL recipe; Qwen3-VL 30B recipe; Llama-Embed-Nemotron-8B training.

### Logging and Observability

- MLflow integration.
- Metric logger with JSONL output.
- YAML logging-to-stdout improvements.

### Workflow

- Knowledge-distillation custom validation step; `ScopedModuleOffloading` to
  reduce memory.
- Model Registry component.
- SIGTERM handling.
- `NEMO_ENABLE_USER_MODULES` for user-extension modules.
- Rank-0 download for custom models.
- Dereference env vars in YAML.

---

## 0.1.2 (2025-10-23) · [PyPI](https://pypi.org/project/nemo-automodel/0.1.2/) · [GH](https://github.com/NVIDIA-NeMo/Automodel/releases/tag/v0.1.2)

Patch release.

- **Fix:** `max_steps` now set inside the constructor (#650).
- **Fix:** step scheduler switched to zero-based indexing (#627).
- **Fix:** sample-limit handling for `ColumnMapped` datasets (#521).

---

## 0.1.0 (2025-10-08) · [PyPI](https://pypi.org/project/nemo-automodel/0.1.0/) · [GH](https://github.com/NVIDIA-NeMo/Automodel/releases/tag/v0.1.0)

Initial public release of NeMo AutoModel.

### Highlights

- PyTorch-native training framework for LLMs and VLMs with HuggingFace
  Transformers integration via `NeMoAuto*` wrapper classes.
- YAML-driven recipes for SFT and PEFT.
- FSDP2 / HSDP / DDP distributed training with DTensor sharding.
- Megatron-FSDP available as the default heavy-duty sharding option (replaces
  the earlier nvFSDP path).
- Knowledge distillation recipe.
- MoE component with DeepSeek v3 model implementation.
- `ColumnMappedTextInstructionDataset` for instruction tuning.
- Gradient checkpointing.
- Slurm launcher.

---

For the list of newly supported models per release, see the
[Model Coverage Release Log](model-coverage/latest-models.md).
