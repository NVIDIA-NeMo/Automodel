# Blog benchmarks — Transformers v5 vs NeMo AutoModel

Standalone throughput (TPS) benchmarks comparing HuggingFace Transformers v5 against
NeMo AutoModel on MoE models — single node (8× H100 80GB) for the 30B models, and 16 nodes
(128× H100) for the 550B Nemotron-3-Ultra. TPS is measured as
`tokens / (forward + loss + backward)`; the optimizer step is excluded. Reported numbers
are the **mean over the timed steps** (after warmup).

## Container

```
nvcr.io/nvidia/nemo-automodel:26.04.00
```

The 30B benchmarks run on a single 8× H100 80GB node; the 550B Ultra benchmark runs across
16 such nodes (see [Multi-node](#multi-node-nemotron-3-ultra-550b-16-nodes--8-h100--128-gpus) below).
All commands below assume the repo is mounted at `/opt/Automodel` and you are inside the container.

## Versions (inside the 26.04.00 container)

| Component | Version |
|---|---|
| PyTorch | `2.11.0a0+eb65b36914.nv26.02` |
| CUDA | `13.1` |
| Transformers | `5.5.0` (container); bump to `5.10.2` for the v5 benchmarks |
| TransformerEngine | `2.11.0+c188b533` |
| flash-attn | `2.7.4.post1` |
| DeepEP | bundled (no `__version__`) |
| nemo-automodel | `0.4.0` |

The NeMo AutoModel benchmarks run on the container-native packages with no overrides. The
**v5 benchmarks require Transformers `5.10.2`**, which is newer than the container's bundled
`5.5.0` (qwen3's `ep_plan` and the v5 expert-parallel path only exist in 5.10.2). You must
bump it manually inside the 26.04 container before running `bench_v5.py`:

## Commands

Run with `torchrun` across all 8 GPUs. Methodology: sequence length 4096, local batch size 1,
bf16 (params + reduce), 10 warmup steps, 30 timed steps.

### NeMo AutoModel (EP=8 + DeepEP + TransformerEngine)

```bash
# Qwen3-30B-A3B
torchrun --nproc-per-node=8 blog_experiments/bench_automodel.py --model qwen3_30b

# Nemotron Nano v3 30B
torchrun --nproc-per-node=8 blog_experiments/bench_automodel.py --model nemotron_nano
```

### HuggingFace Transformers v5 (per-layer FSDP2, no EP)

```bash
# Qwen3-30B-A3B
torchrun --nproc-per-node=8 blog_experiments/bench_v5.py --model qwen3_30b

# Nemotron Nano v3 30B
torchrun --nproc-per-node=8 blog_experiments/bench_v5.py --model nemotron_nano
```

(`gptoss_20b` is also available as a `--model` choice in both scripts.)

## Multi-node: Nemotron-3-Ultra-550B (16 nodes × 8 H100 = 128 GPUs)

Full fine-tune of the 550B Nemotron-3-Ultra. Same container, launched across 16 nodes with
multi-node `torchrun` (one task per node, 8 ranks each). The Ultra config — EP=64, local batch
size 2, balanced gate, `torch_mm` experts, MTP, activation checkpointing, fused linear
cross-entropy — is baked into the `nemotron_ultra` entry, so no extra flags are needed.
Use the **default allocator** (do not set `PYTORCH_CUDA_ALLOC_CONF`).

Launch on **every** node, with `MASTER_ADDR` set to the rank-0 hostname and `NODE_RANK` the
0-based node index (e.g. `$SLURM_NODEID` under Slurm `srun --ntasks-per-node=1 --nodes=16`):

### NeMo AutoModel (EP=64 + DeepEP + TransformerEngine)

```bash
torchrun \
  --nnodes=16 --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 --node-rank=$NODE_RANK \
  blog_experiments/bench_automodel.py --model nemotron_ultra
```
