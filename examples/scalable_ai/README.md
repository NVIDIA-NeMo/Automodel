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
| `run_bench.sh` | 8-GPU benchmark runner behind the reference tables below; `run_bench.sh journey` is the stage-by-stage comparison on the 4-layer model |
| `pretrain.sh`, `prepare_fineweb.py`, `finite_nanogpt.py` | the 8-GPU pre-training run: FineWeb-edu tokenisation into NanogptDataset shards, bounded validation dataset, container launcher with auto-resume and Weights & Biases logging |

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

# Pre-training on 8 GPUs (inside the container: tokenises FineWeb-edu on first use, resumes from the latest checkpoint)
bash examples/scalable_ai/pretrain.sh                    # or: automodel examples/scalable_ai/configs/pretrain_moonlight_v4_16b.yaml --nproc-per-node 8
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

## Reference numbers (8x H100 80 GB, NeMo Automodel 26.08 container, 2026-09-19)

Moonlight-V4-16B-A3B from random init on mock data, sequence length 2048, global batch 256 sequences
(524k tokens per optimizer step), 12 steps with 4 warm-up, Adam. MFU uses the `deepseekv4_flops` formula against
989 TFLOPS (H100 dense BF16). Per-GPU memory is rank 0's `max_memory_allocated`.

| run (`run_bench.sh`) | micro-batch / GPU | step time | tokens/s (8 GPUs) | peak memory | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock transformers (`moonlight_v4_16b_hf.yaml`), 27 layers, seq 2048 / 1024 / 512 | 1 | OOM in the first forward at every length | - | > 79 GB | - |
| NeMo Automodel, eager attention, `torch_mm` experts, torch dispatcher (`moonlight_v4_16b_torch.yaml`) | 2 | OOM | - | > 79 GB | - |
| same | 1 | 19.75 s | 26.5k | 58.8 GB | 5.6% |
| NeMo Automodel, eager attention + DeepEP dispatcher | 1 | 18.05 s | 29.0k | 58.8 GB | 6.1% |
| NeMo Automodel, TileLang sparse attention + indexer + Sinkhorn, DeepEP (`moonlight_v4_16b_tilelang_deepep.yaml`) | 2 | 10.42 s | 50.3k | 69.8 GB | 10.6% |
| like-for-like 2 sliding-window layers: stock transformers | 1 | 2.37 s | 221k | 14.7 GB | 8.4% |
| like-for-like 2 sliding-window layers: NeMo Automodel eager | 4 | 1.51 s | 347k | 26.1 GB | 13.1% |

Reading the table: the stock implementation cannot train the full model on 80 GB GPUs (its CSA layers gather
`S x k` keys per query); the eager Automodel path fits at micro-batch 1; the TileLang kernels cut the attention
memory enough for micro-batch 2 and run 1.9x faster than the eager path. Reproduce with
`examples/scalable_ai/run_bench.sh` (see its header for the expected workspace layout).

### Stage by stage on a model every stage can run

Stock transformers never fits the full model, so the stage-by-stage comparison uses the first four layers of the
schedule (`--model.config.num_hidden_layers 4`: sliding-window, sliding-window, CSA, HCA, one of each attention
kind; 3.01B parameters), everything else unchanged: sequence 2048, global batch 256 sequences (524k tokens per
step), 8x H100, same recipe, data and optimizer. `run_bench.sh journey` produces the table; the Automodel rows use
the fake balanced gate like the full-model table above, stock transformers routes with its (random-init) learned gate.

| stage | micro-batch / GPU | step time | tokens/s (8 GPUs) | peak memory | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. stock transformers (`moonlight_v4_16b_hf.yaml`) | 1 | 4.36 s | 120k | 22.0 GB | 6.2% |
| 1. stock transformers | 4 | 2.58 s | 203k | 43.6 GB | 10.5% |
| 1. stock transformers | 8 | OOM in the first forward | - | > 79 GB | - |
| 2. NeMo Automodel, eager attention, torch dispatcher (`moonlight_v4_16b_torch.yaml`) | 1 | 3.35 s | 157k | 15.7 GB | 8.1% |
| 2. NeMo Automodel, eager attention, torch dispatcher | 4 | 2.56 s | 205k | 37.2 GB | 10.6% |
| 2. NeMo Automodel, eager attention, torch dispatcher | 8 | no result: did not finish iteration 0 in 11 min (both attempts) | - | - | - |
| 3. NeMo Automodel, eager attention + DeepEP dispatcher | 1 | 3.05 s | 172k | 15.7 GB | 8.9% |
| 3. NeMo Automodel, eager attention + DeepEP dispatcher | 4 | 2.26 s | 232k | 37.2 GB | 12.0% |
| 3. NeMo Automodel, eager attention + DeepEP dispatcher | 8 | OOM in the first iteration | - | > 79 GB | - |
| 4. NeMo Automodel, TileLang attention kernels + DeepEP (`moonlight_v4_16b_tilelang_deepep.yaml`) | 1 | 2.49 s | 210k | 14.2 GB | 10.9% |
| 4. NeMo Automodel, TileLang attention kernels + DeepEP | 4 | 1.76 s | 297k | 30.2 GB | 15.3% |
| 4. NeMo Automodel, TileLang attention kernels + DeepEP | 8 | 1.71 s | 308k | 52.4 GB | 15.9% |

Reading the table: memory is the first thing that moves. Stock transformers needs 22 GB at micro-batch 1 and
44 GB at 4, and does not fit 8; the Automodel stages need 14-16 GB and 30-38 GB for the same work. Speed follows:
at micro-batch 4 the portable Automodel path (stage 2) is as fast as stock, DeepEP takes 12% off the step time
(the torch dispatcher exchanges tokens with all-gathers over the EP group), and the TileLang kernels take another
22% off while saving 7 GB, so stage 4 is 1.47x faster than stock at micro-batch 4 and 1.75x at micro-batch 1.
Only the TileLang path fits micro-batch 8 (52 GB), where it reaches 308k tokens/s and 15.9% MFU, 1.5x the best
stock throughput (203k tokens/s at micro-batch 4). On the full 27-layer model the same steps are the difference
between "does not run" and 50k tokens/s.

The gate matters for a fair reading: rerunning the Automodel stages with the learned gate (`JOURNEY_GATE=false`)
gives 3.55-3.58 s (eager, micro-batch 1), 2.61 s (eager, 4), 2.40 s (DeepEP, 4), 2.55 s (TileLang, 1) and
1.88-1.90 s (TileLang, 4), i.e. random-init learned routing costs 2-8% over the fake balanced gate, so the
stock-to-Automodel gap in the table is overstated by about that much. Those learned-gate runs were also less stable
at EP 8 in the 26.08 container: of 15 runs, 4 died right after their first iteration on the DeepEP dispatcher
(`DeepEP timeout check failed`, then `CUDA error: unspecified launch failure`), one hit a cuBLAS execution failure
in backward and one hung in iteration 0; with the balanced gate all 4-layer runs at micro-batch 1 and 4 completed.
Rerun the stage when that happens; the cause was not investigated here.

Operational notes for containers: put `TILELANG_CACHE_DIR`, `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` on a
writable filesystem (`run_bench.sh` does); when tilelang cannot create its cache directory its import fails and
Automodel silently falls back to the transformers model class, which then rejects the `backend` argument.

## Pre-training run (700 steps on 8x H100, 2026-09-20)

`pretrain.sh` trains Moonlight-V4-16B-A3B from scratch with `configs/pretrain_moonlight_v4_16b.yaml`: FineWeb-edu
(sample-10BT, two parquet files, 568.6M tokens with the Moonshot tokenizer, tokenised on the node in 148 s by
`prepare_fineweb.py`), sequence 2048, global batch 256 sequences (524k tokens per step), micro-batch 1, TileLang
attention kernels + DeepEP, FSDP2 with data parallel 8 and expert parallel 8, AdamW at 4.2e-4 with 100 warm-up steps
and cosine decay to 4.2e-5 over 700 steps (367M tokens). One Slurm job on the `batch` partition (4 h limit) in the
26.08 container; the recipe wrote a 95 GB checkpoint every 200 steps (about 20 s each, two kept) and validated on a
held-out 8M-token shard every 100 steps.

| step | train loss | validation loss | learning rate |
| ---: | ---: | ---: | ---: |
| 0 | 12.50 | - | 4.6e-5 |
| 100 | 5.83 | 5.85 | 4.2e-4 |
| 200 | 4.78 | 4.85 | 3.9e-4 |
| 300 | 4.43 | 4.40 | 3.3e-4 |
| 400 | 4.07 | 4.08 | 2.3e-4 |
| 500 | 3.89 | 3.90 | 1.4e-4 |
| 600 | 3.78 | 3.82 | 6.7e-5 |
| 700 | 3.78 | 3.79 | 4.2e-5 |

Throughput was 19 s per step, 27.3k tokens/s averaged over steps 100-699 (about 5.8% MFU by the `deepseekv4_flops`
formula), with 50 GB peak memory per GPU; the whole job took 3 h 51 min. Micro-batch 2 reaches 69 GB in step 0
and runs out of memory in step 1, when the Adam states are allocated, so micro-batch 1 is the ceiling for this
recipe on 80 GB GPUs. Automodel warns that AdamW on bf16 parameters keeps bf16 optimizer states; that is fine for
this demonstration but not how one would run a real pre-training. Weights & Biases logging follows the credentials:
with `WANDB_API_KEY` or a netrc the run is live, otherwise the run is written offline under `$WORK/logs/wandb` and
uploaded later with `wandb sync <offline-run-dir>` using the same wandb major version that wrote it (this run was written
offline and uploaded afterwards); per-step metrics are also in
`checkpoints/moonlight_v4_16b/training.jsonl` and `validation.jsonl`.

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
