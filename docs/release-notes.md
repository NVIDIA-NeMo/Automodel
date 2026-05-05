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

For the list of newly supported models per release, see the
[Model Coverage Release Log](model-coverage/latest-models.md).
