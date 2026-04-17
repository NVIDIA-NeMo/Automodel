# Benchmark Results — 2026-04-14

## Setup
- 8x H100 80GB GPUs (single node)
- Dummy data: random tokens
- TPS = tokens / (forward + loss + backward time), optimizer step excluded
- Warmup: 10 steps, Timed: 30 steps
- FSDP2 MixedPrecisionPolicy: param_dtype=bf16, reduce_dtype=bf16

## Results

### Qwen3-30B-A3B (seq_len=4096)

| Metric | v5 (HF + FSDP) | Automodel (EP=8) | Speedup |
|--------|-----------------|-------------------|---------|
| TPS/GPU (avg) | 2,982 | 7,879 | **2.64x** |
| TPS/GPU (std) | 229 | 122 | — |
| TPS/GPU (min) | 2,476 | 7,424 | — |
| TPS/GPU (max) | 3,817 | 8,092 | — |
| Peak Mem (GiB) | 67.5 | 49.5 | **-27%** |
| Avg Fwd+Loss (ms) | 572 | 206 | 2.78x |
| Avg Bwd (ms) | 809 | 314 | 2.58x |
| LBS | 1 | 1 | — |
| Attention | FA2 | TE | — |
| Experts | grouped_mm | torch_mm + DeepEP | — |

**v5 config**: `AutoModelForCausalLM.from_pretrained` + manual `fully_shard` per layer
**Automodel config**: `NeMoAutoModelForCausalLM.from_pretrained` with `BackendConfig(attn=te, linear=te, rms_norm=torch_fp32, experts=torch_mm, dispatcher=deepep, gate_precision=bf16)`

### Nemotron Nano v3 30B (seq_len=4096)

| Metric | v4 (HF hub) | v5 (HF native + FA2) | Automodel (EP=8) | v5→Automodel |
|--------|-------------|------------------------|-------------------|--------------|
| TPS/GPU (avg) | 1,807 | 4,771 | 11,477 | **2.41x** |
| TPS/GPU (std) | 205 | 6 | 1,303 | — |
| Peak Mem (GiB) | 61.9 | 62.0 | 44.5 | -28% |
| LBS | 1 | 1 | 1 | — |
| Attention | eager | FA2 | TE | — |
| Experts | for-loop (hub) | grouped_mm | gmm + DeepEP | — |

**v4**: Uses `trust_remote_code=True` (NVIDIA's hub code). Eager attention + eager expert for-loop.

**v5**: Native transformers NemotronH with FA2 + grouped_mm + Mamba2 CUDA kernels. Required patching `lazy_load_kernel` to fall back to regular `import mamba_ssm` when HF kernel binaries aren't available for the current CUDA/torch version. FA2 on 6 attention layers + grouped_mm experts + Mamba CUDA kernels give 2.64x speedup over v4 hub code.

**Automodel config**: `NeMoAutoModelForCausalLM.from_pretrained` with `BackendConfig(attn=te, linear=te, rms_norm=te, experts=gmm, dispatcher=deepep)`

## Raw JSON Summaries

### v5 Qwen3-30B
```json
{
  "benchmark": "v5",
  "model": "qwen3_30b",
  "world_size": 8,
  "local_batch_size": 1,
  "seq_len": 4096,
  "tokens_per_step": 32768,
  "warmup_steps": 10,
  "timed_steps": 30,
  "tps_per_gpu_avg": 2981.8,
  "tps_per_gpu_std": 229.3,
  "tps_per_gpu_min": 2475.6,
  "tps_per_gpu_max": 3817.4,
  "tps_total_avg": 23854.1,
  "avg_fwd_ms": 572.1,
  "avg_bwd_ms": 809.3,
  "peak_mem_gib": 67.53
}
```

### Automodel Qwen3-30B (EP=8)
```json
{
  "benchmark": "automodel",
  "model": "qwen3_30b",
  "world_size": 8,
  "local_batch_size": 1,
  "seq_len": 4096,
  "ep_size": 8,
  "tokens_per_step": 32768,
  "warmup_steps": 10,
  "timed_steps": 30,
  "tps_per_gpu_avg": 7879.0,
  "tps_per_gpu_std": 121.6,
  "tps_per_gpu_min": 7423.7,
  "tps_per_gpu_max": 8091.6,
  "tps_total_avg": 63031.7,
  "avg_fwd_ms": 205.8,
  "avg_bwd_ms": 314.2,
  "peak_mem_gib": 49.48
}
```

### v5 Nemotron Nano v3 (native + FA2 + grouped_mm)
```json
{
  "benchmark": "v5",
  "model": "nemotron_nano",
  "world_size": 8,
  "local_batch_size": 1,
  "seq_len": 4096,
  "tokens_per_step": 32768,
  "warmup_steps": 3,
  "timed_steps": 5,
  "tps_per_gpu_avg": 4771,
  "tps_per_gpu_std": 6,
  "peak_mem_gib": 62.0
}
```

### v4 Nemotron Nano v3 (hub code)
```json
{
  "benchmark": "v4",
  "model": "nemotron_nano",
  "world_size": 8,
  "local_batch_size": 1,
  "seq_len": 4096,
  "tokens_per_step": 32768,
  "warmup_steps": 2,
  "timed_steps": 3,
  "tps_per_gpu_avg": 1807,
  "tps_per_gpu_std": 205,
  "peak_mem_gib": 61.9
}
```

### Automodel Nemotron Nano v3 (EP=8)
```json
{
  "benchmark": "automodel",
  "model": "nemotron_nano",
  "world_size": 8,
  "local_batch_size": 1,
  "seq_len": 4096,
  "ep_size": 8,
  "tokens_per_step": 32768,
  "warmup_steps": 10,
  "timed_steps": 30,
  "tps_per_gpu_avg": 11477.1,
  "tps_per_gpu_std": 1302.9,
  "tps_per_gpu_min": 4516.6,
  "tps_per_gpu_max": 11946.5,
  "tps_total_avg": 91816.9,
  "avg_fwd_ms": 154.8,
  "avg_bwd_ms": 213.4,
  "peak_mem_gib": 44.46
}
```

## Notes
- Qwen3 v5 model loading: 302s (loading 531 safetensors shards to each GPU). Automodel: 44s (meta device init + DCP parallel load).
- Nemotron Nano v4/v5 loading: ~50-75s. Automodel: 27s.
- Both Qwen3 benchmarks use the same LBS=1, seq_len=4096, making TPS directly comparable.
- Nemotron Nano v5 uses native transformers code (`trust_remote_code=False`) with SDPA attention. Required patching `lazy_load_kernel` to fall back to regular `import mamba_ssm` when HF kernel binaries aren't available (CUDA 13.1 / PyTorch 2.11 has no pre-built `kernels-community/mamba-ssm` binary). Without this patch, v5 native OOMs (naive torch Mamba2 uses 7.4 GiB more memory).
- Nemotron Nano v4 uses `trust_remote_code=True` (hub code) with eager attention + eager expert for-loop.
- Qwen3 v4 deadlocks with FSDP: experts are stored as a `ModuleList` of 128 MLP modules, each FSDP-wrapped separately. The forward loop `for expert_idx in expert_hit` only iterates experts that received tokens — with different data per rank, different ranks skip different experts, causing mismatched AllGather/ReduceScatter collectives (indefinite hang). v5 fixes this by storing experts as fused 3D `nn.Parameter` tensors (no per-expert modules → no per-expert FSDP collectives).
- v4/v5 Nemotron benchmarks have 3 timed steps; Qwen3/automodel benchmarks have 30 timed steps.
