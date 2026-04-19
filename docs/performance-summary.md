# Performance Summary

This document provides performance benchmarks for various large language models using NeMo AutoModel with the PyTorch backend.

## Pre-Training Performance

The table below shows training performance for full sequences with no padding across different model architectures and scales.

### System: DGX-H100, Precision: BF16

| Model | #GPUs | GBS | MBS | LBS | GA | Seq Length | TP | PP | CP | EP | VP | FSDP | Kernel Optimizations | Time per Global Step (s) | Model TFLOPs/sec/GPU | Tokens/sec/GPU |
|-------|------:|----:|----:|----:|---:|-----------:|---:|---:|---:|---:|---:|-----:|---------|-------------------------:|---------------------:|---------------:|
| Nemotron V3 Super 120B (26.02) | 64 | 512 | 2 | 2 | 4 | 4096 | 1 | 1 | 1 | 64 | - | 64 | TE + DeepEP + TorchSDPA | 7.286 | 334 | 4,497 |
| Nemotron V3 Nano 30B (26.02) | 8 | 512 | 4 | 4 | 16 | 4096 | 1 | 1 | 1 | 8 | - | 8 | TE + DeepEP + TorchSDPA | 15.614 | 328 | 16,789 |
| DeepSeek V3 671B | 1024 | 8192 | 1 | 8 | 4 | 4096 | 1 | 4 | 1 | 64 | 8 | 256 | TE + DeepEP | 37.87 | 216 | 865 |
| DeepSeek V3 671B | 256 | 512 | 1 | 8 | 1 | 4096 | 1 | 4 | 1 | 64 | 8 | 64 | TE + DeepEP | 8.18 | 250 | 1,002 |
| Kimi K2 | 256 | 512 | 1 | 8 | 2 | 4096 | 1 | 8 | 1 | 32 | 4 | 32 | TE + DeepEP | 8.86 | 189 | 924 |
| Qwen3 MoE 30B | 8 | 512 | 4 | 4 | 16 | 4096 | 1 | 1 | 1 | 8 | - | 8 | TE + DeepEP | 21.773 | 277 | 12,040 |
| GPT-OSS 20B | 8 | 256 | 2 | 2 | 16 | 4096 | 1 | 1 | 1 | - | - | 8 | TE + DeepEP + FlexAttn | 10.04 | 279 | 13,058 |
| GPT-OSS 120B | 64 | 512 | 2 | 2 | 4 | 4096 | 1 | 1 | 1 | - | - | 64 | TE + DeepEP + FlexAttn | 4.30 | 231 | 7,626 |
| Llama3 70B | 64 | 128 | 1 | 1 | 4 | 8192 | 1 | 1 | 2 | - | - | 32 | TE + fsdp2_prefetch | 18.90 | 389 | 866.77 |


## Fine-Tuning (LoRA) Performance

The table below shows fine-tuning (LoRA) performance for full sequences with no padding across different model architectures and scales.

### System: DGX-H100, Precision: BF16

| Model | #GPUs | GBS | MBS | LBS | GA | Seq Length | TP | PP | CP | EP | VP | FSDP | Kernel Optimizations | Time per Global Step (s) | Model TFLOPs/sec/GPU | Tokens/sec/GPU |
|-------|------:|----:|----:|----:|---:|-----------:|---:|---:|---:|---:|---:|-----:|---------|-------------------------:|---------------------:|---------------:|
| Llama3 8B | 1 | 32 | 2 | 2 | 16 | 4096 | 1 | 1 | 1 | - | 1 | 1 | TE + triton | 10.51 | 402 | 12472.87 |
| Qwen2.5 7B | 1 | 32 | 2 | 2 | 16 | 4096 | 1 | 1 | 1 | - | 1 | 1 | TE + triton | 9.29 | 423 | 14110.05 |
| Llama3 70B | 8 | 32 | 2 | 2 | 4 | 4096 | 2 | 1 | 1 | - | 1 | 4 | TE + triton + fsdp2_prefetch | 15.00 | 316 | 1091.85 |
| Qwen2.5 32B | 8 | 32 | 2 | 2 | 4 | 4096 | 2 | 1 | 1 | - | 1 | 4 | TE + triton + fsdp2_prefetch | 7.28 | 301 | 2250.31 |
| Llama3 70B 2-node | 16 | 32 | 2 | 2 | 2 | 4096 | 2 | 1 | 1 | - | 1 | 8 | TE + triton + fsdp2_prefetch | 8.32 | 285 | 984.85 |
| Qwen2.5 32B 2-node | 16 | 32 | 2 | 2 | 2 | 4096 | 2 | 1 | 1 | - | 1 | 8 | TE + triton + fsdp2_prefetch | 3.95 | 277 | 2072.89 |

## Glossary

- **MFU**: Model FLOPs Utilization - ratio of achieved compute to peak hardware capability
- **TP**: Tensor Parallelism - splits individual layers across GPUs
- **PP**: Pipeline Parallelism - splits model layers into stages
- **EP**: Expert Parallelism - distributes MoE experts across GPUs
- **DP**: Data Parallelism - replicates model and splits data
- **VP**: Virtual Pipeline - number of pipeline stages per GPU for interleaving
- **MBS**: Micro-Batch Size - size of one forward pass in pipeline
- **LBS**: Local Batch Size - size of one step per GPU
- **GBS**: Global Batch Size - total batch size across all GPUs
- **GA**: Gradient Accumulation - number of local-batches before optimizer step
- **TE**: Transformer Engine kernel optimizations - RMSNorm, Linear and DotProductAttention
- **DeepEP**: Deep Expert Parallelism - advanced EP routing for MoE models
- **FlexAttn**: PyTorch's [Flex Attention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)

## Configuration Files

Pre-training and fine-tuning (LoRA) benchmark configurations are available in [`examples/llm_benchmark/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark):

- [`deepseek_v3_te_deepep.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/deepseek/deepseek_v3_te_deepep.yaml) - DeepSeek V3 with TE + DeepEP
- [`kimi_k2_te_deepep.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/kimi/kimi_k2_te_deepep.yaml) - Kimi K2 optimized configuration
- [`qwen3_moe_30b_te_deepep.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml) - Qwen3 MoE with TE + DeepEP
- [`gptoss_20b_te_deepep.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/gpt_oss/gptoss_20b_te_deepep.yaml) - GPT-OSS 20B with optimizations
- [`gptoss_120b_te_deepep.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/gpt_oss/gptoss_120b_te_deepep.yaml) - GPT-OSS 120B optimized
- [`custom_llama3_1_70b_pretrain_benchmark_8nodes.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/llama3_3/custom_llama3_1_70b_pretrain_benchmark_8nodes.yaml) - Llama3-70B optimized
- [`llama3_1_8b_peft_benchmark.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/llama3_1/llama3_1_8b_peft_benchmark.yaml) - Llama-8B fine-tuning (LoRA) optimized
- [`qwen2_5_7b_peft_benchmark.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/qwen/qwen2_5_7b_peft_benchmark.yaml) - Qwen2.5-7B fine-tuning (LoRA) optimized
- [`custom_llama3_3_70b_instruct_peft_benchmark.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/llama3_3/custom_llama3_3_70b_instruct_peft_benchmark.yaml) - Llama-70B fine-tuning (LoRA) optimized
- [`custom_qwen2_5_32b_peft_benchmark.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/qwen/custom_qwen2_5_32b_peft_benchmark.yaml) - Qwen2.5-32B fine-tuning (LoRA) optimized
- [`custom_llama3_3_70b_instruct_peft_benchmark_2nodes.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/llama3_3/custom_llama3_3_70b_instruct_peft_benchmark_2nodes.yaml) - Llama-70B fine-tuning (LoRA) optimized on 2 nodes
- [`custom_qwen2_5_32b_peft_benchmark_2nodes.yaml`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_benchmark/qwen/custom_qwen2_5_32b_peft_benchmark_2nodes.yaml) - Qwen2.5-32B fine-tuning (LoRA) optimized on 2 nodes

:::{note}
- All benchmarks use mock data for consistent performance measurement.
- Fake balanced gate is enabled to simulate ideal expert routing.
- No gradient clipping applied for pure performance measurement.
- MFU calculated using peak TFLOPs for the system (989 for BF16 H100).
- Step times include forward and backward passes + optimizer step for the global batch.
:::


## Version Information
- **Last Updated**: 2025-10-02
- **NeMo AutoModel Version**: `main` Branch
