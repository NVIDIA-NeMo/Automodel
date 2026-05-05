# Release Notes

## 0.4.0 · 26.04 (2026-04-28)

### Highlights

- **New CLI.** `automodel <config.yaml>` replaces the old
  `automodel <command> <domain> -c <config>` form, with a short alias `am`.
  See [Breaking Changes](breaking-changes.md) for the full migration guide.
- **SkyPilot and NeMo-Run launchers.** Native multi-node launch on cloud
  (SkyPilot, including Kubernetes) and via NeMo-Run, in addition to local and
  SLURM. Launcher selection is driven by YAML sections in the config.
- **Lightweight CLI install.** `pip install nemo-automodel[cli]` installs only
  `pyyaml` — enough to submit jobs from a login node without pulling in
  PyTorch/CUDA.
- **Discrete-diffusion LLMs (dLLM).** SFT and generation support for dLLM
  models, including Llada.

### New Models

- **LLM:** Gemma 4, Mistral 4, Mistral Medium 3.5, GLM 5, Minimax M2.5,
  Step-3.5-Flash, Devstral 24B, Nemotron Nano 4B/8B.
- **MoE:** Qwen3.5-MoE (30B-A3B, 35B), GPT-OSS 20B, Moonlight 16B.
- **Dense:** Qwen3.5 small dense models.
- **Multimodal / Omni:** Qwen3-Omni port, Nemotron-Omni, Phi4MM audio.
- **Diffusion:** Wan multi-resolution, LoRA for diffusion.

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
- Packed sequences for Qwen3.5-MoE with EP+PP; chat datasets with THD,
  BSHD+CP, and padding fixes.

### PEFT

- MoE LoRA: rank scaling, `torch_mm` integration, expert-LoRA init using
  `config.expert_dim`.
- `merge_lora` tool for materializing adapters into the base model.
- QLoRA PEFT checkpoints saved with the HF adapter prefix.

### Recipes and Workflow

- New recipes for Gemma 4 (LoRA), Nemotron Nano 4B SQuAD, Mistral 4, Tulu-3
  E2E convergence, GPT-OSS 20B / Moonlight 16B convergence, and reranker /
  biencoder training.
- MFU logging across train recipes.
- Native Comet ML experiment tracking.
- NEFTune noisy embeddings for instruction fine-tuning.
- Scheduler-driven manual garbage collection.
- Common inference utility and `.generate()` with KV cache for Nemotron v3.

### Checkpointing

- `v4_compatible` checkpoint format.
- Diffusion training jobs default to safetensors checkpoint format.
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

A migration guide for the new CLI, the required `recipe._target_` YAML
section, the SLURM `sbatch`-script workflow, and the `nemo-automodel[cli]`
install profile is in [Breaking Changes](breaking-changes.md).

---

## 0.3.0 (2026-02-26)

### Highlights

- **Transformers v4 / v5 alignment.** New `transformers v4` API support and a
  v5 refactor for device-mesh-only model init.
- **Streaming safetensors writer** for faster checkpoint export.
- **Faster fp8 dequant kernels** with DTensor dequantization fixes for DSv3.

### New Models

- **LLM:** DeepSeek 3.2, Step3p5, Minimax M2, Nano v3 custom, Nemotron Flash,
  GLM 4.7, Devstral (backported to v4).
- **MoE / VLM:** Qwen3-VL custom implementation (235B, 30B, 4B/8B configs),
  Kimi-VL, Kimi K2.5 VL, Qwen3-Omni port via `transformers omni`,
  Nemotron-Parse VLM.

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

## 0.2.0 (2025-12-04)

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
- **DeepSeek v3** with fp8 base checkpoint loading.
- **Sequence classification:** Qwen3ForSequenceClassification registered;
  generic SFT sequence-classification recipe.

### Distributed Training

- VLM EP and Qwen-Omni custom implementation.
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

## 0.1.2 (2025-10-23)

Patch release.

- **Fix:** `max_steps` now set inside the constructor (#650).
- **Fix:** step scheduler switched to zero-based indexing (#627).
- **Fix:** sample-limit handling for `ColumnMapped` datasets (#521).

---

## 0.1.0 (2025-10-08)

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
