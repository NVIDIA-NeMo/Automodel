# Nsight Systems Profiles (2026 September edition, DeepSeek-V4 / Moonlight-V4)

This folder documents the Nsight Systems profiles used in the Automodel guest lecture. The model is
[Moonlight-V4-16B-A3B](https://huggingface.co/akoumpa/Moonlight-V4-16B-A3B), the DeepSeek-V4 architecture at
Moonlight scale (16.5B total / 3.0B active parameters, 27 layers, 64 experts, hybrid CSA/HCA attention,
manifold-constrained hyper-connections). All runs start from random weights with mock data.

Profiles are not committed (they may contain environment data). Regenerate them with

```bash
./examples/scalable_ai/nsys_profiles/regenerate_profiles.sh            # portable profiles (any CUDA GPU)
RUN_HOPPER=1 ./examples/scalable_ai/nsys_profiles/regenerate_profiles.sh  # also the TileLang / DeepEP / TE profiles
```

or run the individual commands below from the repository root.

## Why the small configs look the way they do

- `--model.config.num_hidden_layers N` keeps the first N entries of the 27-entry schedule
  `[0, 0, 4, 128, 4, 128, ...]`: 2 layers = sliding-window only, 3 layers = adds one CSA layer, 4 = adds an HCA layer.
- The transformers implementation of DeepSeek-V4 is inference-oriented: in training mode its CSA layers gather the
  per-query top-k entries into an `S x k` key axis and its compressed entries carry no causal mask. The HF baseline
  is therefore profiled on the 2 sliding-window layers only; the NeMo Automodel run uses the same 2 layers for a
  like-for-like comparison, plus a 3-layer run that includes CSA.
- `attn: tilelang` (TileLang sparse attention + indexer + Sinkhorn kernels) and DeepEP need Hopper-class GPUs; the
  vendored kernels request 150-230 KB of shared memory per block.

## End-to-end benchmark profiles (`nemo_automodel/recipes/llm/benchmark.py`)

### moonlight_v4_hf_small.nsys-rep
Stock transformers `DeepseekV4ForCausalLM`, 2 sliding-window layers.
```bash
nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true \
  --output=moonlight_v4_hf_small.nsys-rep \
  torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_hf.yaml \
  --benchmark.nsys_start 3 --benchmark.nsys_end 5 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 --dataset.seq_len 1024 --model.config.num_hidden_layers 2
```

### moonlight_v4_torch_small.nsys-rep
NeMo Automodel native DeepSeek-V4, portable backends (eager attention, `torch._grouped_mm` experts), same 2 layers.
```bash
nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true \
  --output=moonlight_v4_torch_small.nsys-rep \
  torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml \
  --benchmark.nsys_start 3 --benchmark.nsys_end 5 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024 \
  --distributed.ep_size 1 --model.config.num_hidden_layers 2
```

### moonlight_v4_torch_csa_small.nsys-rep
As above with 3 layers (adds a Compressed Sparse Attention layer with compressor + indexer), EP=2.
```bash
nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true \
  --output=moonlight_v4_torch_csa_small.nsys-rep \
  torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml \
  --benchmark.nsys_start 3 --benchmark.nsys_end 5 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024 \
  --distributed.ep_size 2 --model.config.num_hidden_layers 3
```

### moonlight_v4_tilelang_deepep_small.nsys-rep (Hopper-class GPUs)
TileLang sparse attention + indexer + Sinkhorn kernels and DeepEP dispatch, 3 layers, EP=2.
```bash
nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true \
  --output=moonlight_v4_tilelang_deepep_small.nsys-rep \
  torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_tilelang_deepep.yaml \
  --benchmark.nsys_start 3 --benchmark.nsys_end 5 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024 \
  --distributed.ep_size 2 --model.config.num_hidden_layers 3
```

## Layer-level profiles (`examples/scalable_ai/profile_layer.py`)

All layer profiles use `nsys profile --force-overwrite true -c cudaProfilerApi -t cuda,nvtx -o <name> python examples/scalable_ai/profile_layer.py ...`
with the arguments below (batch 1, 4096 tokens by default).

| profile | arguments | note |
| --- | --- | --- |
| `attn_swa_hf` | `--layer attn --compress-ratio 0 --use-hf` | sliding-window layer, transformers |
| `attn_swa_automodel` | `--layer attn --compress-ratio 0` | sliding-window layer, Automodel eager |
| `attn_csa_automodel_eager` | `--layer attn --compress-ratio 4 --backend-attn eager` | CSA: compressor + indexer + dense masked attention |
| `attn_csa_automodel_tilelang` | `--layer attn --compress-ratio 4 --backend-attn tilelang` | CSA on the TileLang kernels (Hopper) |
| `attn_hca_hf` | `--layer attn --compress-ratio 128 --use-hf` | HCA, transformers (timing only: no causal mask on compressed entries) |
| `attn_hca_automodel` | `--layer attn --compress-ratio 128` | HCA, Automodel eager |
| `moe_hf` | `--layer moe --use-hf` | 64 experts top-6 + shared, transformers |
| `moe_automodel_torchmm` | `--layer moe --backend-experts torch_mm` | grouped GEMM experts (`torch._grouped_mm`) |
| `moe_automodel_te` | `--layer moe --backend-experts te` | TE GroupedLinear experts (needs TE) |
| `rmsnorm_hf` | `--layer rmsnorm --use-hf` | |
| `rmsnorm_automodel_fp32` | `--layer rmsnorm --backend-rms-norm torch_fp32` | |
| `rmsnorm_automodel_te` | `--layer rmsnorm --backend-rms-norm te` | needs TE |
| `hc_hf` | `--layer hc --use-hf` | mHC mixer (Sinkhorn), transformers |
| `hc_automodel` | `--layer hc` | mHC mixer, Automodel (torch; TileKernels Sinkhorn when `tile_kernels` is installed and `--backend-attn tilelang`) |
| `block_csa_automodel` | `--layer block --compress-ratio 4` | full block: mHC + CSA attention + MoE |

## Viewing profiles

```bash
nsys-ui <profile_name>.nsys-rep
```
