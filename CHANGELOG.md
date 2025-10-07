# Changelog

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
