# Nsight Systems Profiles

This folder contains Nsight Systems profiles for the Automodel guest lecture demonstrations.

## Generating Profiles

Profiles cannot be committed to the repository as they may contain sensitive environment data.
To generate all profiles, run:

```bash
./examples/scalable_ai/nsys_profiles/regenerate_profiles.sh
```

Or run individual commands below to generate specific profiles.

## End-to-End Benchmark Profiles

### moonlight_hf_small.nsys-rep
HuggingFace baseline benchmark with Moonlight-16B model.
```bash
nsys profile --force-overwrite true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=moonlight_hf_small.nsys-rep \
  torchrun \
  --nproc-per-node 2 \
  nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_16b_hf.yaml \
  --benchmark.nsys_start 3 \
  --benchmark.nsys_end 5 \
  --step_scheduler.max_steps 6 \
  --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 \
  --dataset.seq_len 1024 \
  --model.config.num_hidden_layers 3
```

### moonlight_te_deepep-false_small.nsys-rep
Transformer Engine backend without DeepEP (Expert Parallelism).
```bash
nsys profile --force-overwrite true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=moonlight_te_deepep-false_small.nsys-rep \
  torchrun \
  --nproc-per-node 2 \
  nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_16b_te_deepep.yaml \
  --benchmark.nsys_start 3 \
  --benchmark.nsys_end 5 \
  --step_scheduler.max_steps 6 \
  --benchmark.warmup_steps 1 \
  --distributed.ep_size 2 \
  --step_scheduler.global_batch_size 16 \
  --model.backend.enable_deepep False \
  --model.config.num_hidden_layers 3
```

### moonlight_te_deepep_small.nsys-rep
Transformer Engine backend with DeepEP enabled.
```bash
nsys profile --force-overwrite true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=moonlight_te_deepep_small.nsys-rep \
  torchrun \
  --nproc-per-node 2 \
  nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_16b_te_deepep.yaml \
  --benchmark.nsys_start 3 \
  --benchmark.nsys_end 5 \
  --step_scheduler.max_steps 6 \
  --benchmark.warmup_steps 1 \
  --distributed.ep_size 2 \
  --step_scheduler.global_batch_size 16 \
  --model.config.num_hidden_layers 3
```

## Layer-Level Profiles

### mla_profile_hf.nsys-rep
Multi-head Latent Attention (MLA) layer profiling with HuggingFace implementation.
```bash
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o mla_profile_hf \
  python examples/scalable_ai/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer mla \
  --use-hf
```

### mla_profile_te.nsys-rep
Multi-head Latent Attention (MLA) layer profiling with Transformer Engine backend.
```bash
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o mla_profile_te \
  python examples/scalable_ai/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer mla \
  --backend-attn te
```

### rmsnorm_profile_hf.nsys-rep
RMSNorm layer profiling with HuggingFace implementation.
```bash
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o rmsnorm_hf \
  python examples/scalable_ai/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer rmsnorm \
  --use-hf
```

### rmsnorm_profile_te.nsys-rep
RMSNorm layer profiling with Transformer Engine backend.
```bash
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o rmsnorm_te \
  python examples/scalable_ai/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer rmsnorm \
  --backend-rms-norm te
```

## Viewing Profiles

Open profiles with Nsight Systems GUI:
```bash
nsys-ui <profile_name>.nsys-rep
```
