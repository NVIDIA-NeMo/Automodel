# Scalable AI guest lecture: NeMo Automodel materials (2026 September edition)

Configs, profiling scripts and Nsight Systems recipes used in the Automodel guest lecture. This edition moves the
material from DeepSeek-V3 / Moonlight-16B-A3B (January 2026 edition) to the **DeepSeek-V4 architecture**, using
[Moonlight-V4-16B-A3B](https://huggingface.co/akoumpa/Moonlight-V4-16B-A3B): the V4 architecture at Moonlight's
scale (2048 hidden, 27 layers, 64 routed + 1 shared experts, top-6), 16.5B total / 3.0B active parameters, with
hybrid Compressed Sparse / Heavily Compressed Attention, manifold-constrained hyper-connections (mHC), a hash-routed
first MoE layer, shared-KV MQA with a grouped low-rank output projection, sqrt-softplus routing, attention sinks and
clamped SwiGLU. The model repo carries the config, the tokenizer, a full comparison with Moonlight-16B-A3B
(`COMPARISON.md`) and a parameter / KV-cache / FLOP calculator. No trained weights exist; every run here starts from
random initialisation on mock or Megatron-format data.

Two ~1B siblings that train on two 48 GB GPUs are available for hands-on sessions:
[`akoumpa/Moonlight-V4-1B-h16d256`](https://huggingface.co/akoumpa/Moonlight-V4-1B-h16d256) and
[`akoumpa/Moonlight-V4-1B-h16d256-r8`](https://huggingface.co/akoumpa/Moonlight-V4-1B-h16d256-r8) (ratio-8
compression, no indexer); both ship a training recipe in their `training/` folder.

## Contents

| path | purpose |
| --- | --- |
| `configs/moonlight_v4_16b_hf.yaml` | baseline: stock transformers `DeepseekV4ForCausalLM` driven by the benchmark recipe |
| `configs/moonlight_v4_16b_torch.yaml` | NeMo Automodel native DeepSeek-V4, portable backends (eager attention, `torch._grouped_mm` experts, torch dispatcher) |
| `configs/moonlight_v4_16b_tilelang_deepep.yaml` | NeMo Automodel with TileLang sparse-attention / indexer / Sinkhorn kernels and DeepEP (Hopper-class GPUs) |
| `configs/pretrain_moonlight_v4_16b.yaml` | from-scratch pre-training on Megatron-format data (8 GPUs, EP 8) |
| `profile_layer.py` | per-layer fwd+bwd profiling (attention SWA/CSA/HCA, MoE, block, RMSNorm, mHC mixer), Automodel vs transformers |
| `nsys_profiles/` | the nsys commands behind the lecture's profiles and a script to regenerate them |

## Quick start

```bash
# Throughput benchmark, 8 GPUs (random init, mock data)
torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py --config examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml
torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py --config examples/scalable_ai/configs/moonlight_v4_16b_tilelang_deepep.yaml   # Hopper

# Small variants: keep the first N layers of the schedule [0, 0, 4, 128, ...] (2 = sliding-window only, 3 = + CSA)
torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py --config examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml \
    --model.config.num_hidden_layers 3 --distributed.ep_size 2 --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1

# One layer under nsys: CSA attention, Automodel eager vs TileLang, and the transformers sliding-window layer
python examples/scalable_ai/profile_layer.py --layer attn --compress-ratio 4 --no-nsys
python examples/scalable_ai/profile_layer.py --layer attn --compress-ratio 4 --backend-attn tilelang --no-nsys   # Hopper
python examples/scalable_ai/profile_layer.py --layer attn --compress-ratio 0 --use-hf --no-nsys

# Pre-training (replace the data paths in the yaml first)
automodel examples/scalable_ai/configs/pretrain_moonlight_v4_16b.yaml --nproc-per-node 8
```

## What the profiles show

- **Attention kinds.** `compress_ratios` selects, per layer, sliding-window attention (0), CSA (4: overlapped ratio-4
  compression, a lightning indexer that picks 512 compressed entries per query, plus the 128-token window) or HCA
  (128: ratio-128 compression, no indexer). `profile_layer.py --compress-ratio` profiles each kind.
- **Kernels.** NeMo Automodel runs DeepSeek-V4 attention either as dense masked attention with sinks (`attn: eager`,
  any GPU) or on the vendored TileLang sparse-attention and indexer kernels plus TileKernels' Sinkhorn
  (`attn: tilelang`). The kernels were written for Hopper's 227 KB of shared memory (147 KB at `head_dim` 512 for
  attention, 224 KB for the indexer); on 99 KB-per-block GPUs (Ada, consumer parts) only the eager path runs.
- **MoE.** Experts run as grouped GEMMs (`torch._grouped_mm`, `grouped_gemm` or TE GroupedLinear) with a torch,
  DeepEP or HybridEP token dispatcher; `fake_balanced_gate: true` in the benchmark configs removes load-imbalance
  noise from throughput numbers.
- **mHC.** Every block keeps 4 residual streams and mixes them through a Sinkhorn-projected doubly-stochastic matrix;
  the `hc` layer profile isolates this cost.
- **transformers baseline.** The stock implementation is inference-oriented (eager attention only; in training mode
  CSA gathers `S x k` keys per layer and compressed entries carry no causal mask), so end-to-end comparisons use the
  two sliding-window layers and the HF attention profiles are limited to ratio 0 (and 128 for timing only).

## Notes for reproducing the lecture numbers

- Automodel's native DeepSeek-V4 model keeps a few tensors in fp32 (attention sinks, compressor position biases,
  mHC mixers, `lm_head`). The recipes use `distributed.moe.reshard_after_forward: false` so their FSDP2 units stay
  resident between forward and backward, and the logits-based `MaskedCrossEntropy` (the fused linear cross-entropy
  needs a single dtype).
- Benchmark configs set `num_hash_layers: 0`: with `fake_balanced_gate` every layer routes with the fake balanced
  gate anyway, and it keeps the transformers baseline (whose from-config hash table is all zeros) balanced too.
- `benchmark.py` reports tokens/s and MFU against `benchmark.peak_tflops` (989 for H100 BF16 dense, 2500 for GB200).
- Kernel prerequisites for the Hopper configs: `tilelang`, `tile_kernels` (DeepSeek TileKernels, Sinkhorn),
  `deep_ep`; optional `transformer_engine` for the TE RMSNorm / GroupedLinear profiles.

## Changes from the January 2026 edition

- Model: DeepSeek-V3 / Moonlight-16B-A3B -> DeepSeek-V4 / Moonlight-V4-16B-A3B; layer profiles cover SWA/CSA/HCA
  attention, MoE, block, RMSNorm and the new mHC mixer (`mla` no longer exists in V4).
- `deepseekv4_flops` added to `nemo_automodel/components/utils/flops_utils.py` (registered for `DeepseekV4Config`), so the
  benchmark recipe reports MFU for V4 models instead of falling back to the dense transformer formula (which also
  crashed on Automodel's V4 config); unit-tested in `tests/unit_tests/utils/test_flops_utils_deepseek_v4.py`.
- Configs rewritten for the current recipe schema (`recipe:` key, `distributed.strategy: fsdp2`, `distributed.moe`,
  `models.common.BackendConfig`), and the new benchmark recipe options (`gate_precision`, `experts`, `dispatcher`).
- The edition's core-code tweaks (mesh axis names in the recipe, Makefile `python`, mock-dataset tokenizer argument,
  `trust_remote_code` in the Megatron preprocessor) are all either upstream or unnecessary now (transformers 5.8
  loads Moonshot's tokenizer without remote code), and the `pyproject.toml` / `uv.lock` pins were dropped in favour
  of main's environment.
