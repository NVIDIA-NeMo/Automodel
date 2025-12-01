# Changelog

## NVIDIA NeMo-Automodel 0.2.0

- Fast Model Implementations
  - LLM
     - GPT-OSS 20B and 120B
     - Qwen3 next and Qwen3-235B
     - GLM-4.5-344BA32B, GLM-4.6, GLM-4.5-Air
  - VLM & OMNI
     - Qwen3-vl
     - Qwen2-5-vl
     - Qwen3-omni-30b-a3b
     - Intern-vl-4B (ootb)
- Parallelism
  - Improved support for CP and sequence packing with MoE models
  - Optimized TP plan for LoRA
- Dataset support for
  - Single-turn tool calling
  - Multi-turn tool calling
  - Streaming dataset
  - Chat dataset with OpenAI format
  - Improved support for truncation/padding
- Checkpointing & logging
  - Support for asynchronous checkpointing with DCP
  - Symbolic links (LATEST, LOWEST_VAL) pointing to the latest and lowest validation score checkpoints
  - MLFlow support
- Task support
  - QAT for SFT
  - Sequence classification
- Known issues / limitations
  - Minor perf regression with DSv3
  - Sequence parallel plan incorrect for Qwen3
  - Non-functional validation with MoE models utilizing TE, packed sequence and pipeline parallel size of 1
  - PEFT (LoRA) support is scheduled for follow-up release
  - CP support may be impacted by edge cases

### NeMo-Automodel 25.11 Container

The 0.2.0 release is also included the NeMo Automodel 25.11 container on NGC at https://registry.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel.
Here are the major software components included in the container:

| Software Component | Version     |
| -------------------|-------------|
| CUDA               | 13.0        |
| cuDNN              | 9.13.0.50-1 |
| Pytorch            | 2.9.0a0     |
| NeMo-Automodel     | 0.2.0       |
| Transformer Engine | 2.8.0       |
| Transformers       | 4.57.1      |

## NVIDIA NeMo-Automodel 0.1.1

- Features:
  - Included support for limiting the number of samples with the ColumnMappedDataset

- Bug Fixes (step scheduler):
  - Switched to zero-based indexing
  - Epoch length accounts for accumulation steps

## NVIDIA NeMo-Automodel 0.1.0

- Pretraining support for
  - Models under 40B with PyT FSDP2
  - Larger models by applying PyT PP
  - TP can also be used for models with a TP plan
  - Large MOE via custom implementations
- Knowledge distillation for LLMs (requires same tokenizer)
- FP8 with torchao (requires torch.compile)
- Parallelism
  - HSDP with FSDP2
  - Auto Pipelining Support
- Checkpointing
  - Pipeline support (load and save)
  - Parallel load with meta device
- Data
  - ColumnMapped Dataset for single-turn SFT
  - Pretrain Data: Megatron-Core and Nano-gpt compatible data
- Performance <https://docs.nvidia.com/nemo/automodel/latest/performance-summary.html>
  - Pretraining benchmark for Large MoE user-defined models
  - Fast DeepSeek v3 implementation with DeepEP

## NVIDIA NeMo-Automodel 0.1.0.a0

* Megatron FSDP support
* Packed sequence support
* Triton kernels for LoRA
