# FP8 Training in NeMo-Automodel

NeMo-Automodel now supports FP8 quantization using [TorchAO](https://github.com/pytorch/ao) and `torch.compile` to accelerate training on compatible hardware.

FP8 (8-bit floating point) quantization can provide substantial speedups for models where the majority of GEMMs are sufficiently large. The speedup from using FP8 tensor cores must outweigh the overhead of dynamic quantization.

**Important**: `torch.compile` is required to achieve any meaningful speedup with TorchAO FP8 training.

### Hardware Requirements

- NVIDIA H100 or newer GPUs

## Installation

Make sure you have TorchAO installed. Follow the [installation guide](https://github.com/pytorch/ao?tab=readme-ov-file#-installation) for TorchAO.

## Usage

### Basic Configuration

To enable FP8 quantization with `torch.compile`, you need both FP8 and compilation enabled in your configuration:

```yaml
# Enable torch.compile (required for FP8 speedup)
compile:
  enabled: true
  mode: "default"
  fullgraph: false
  dynamic: false

# Enable FP8 quantization
fp8:
  enabled: true
  recipe_name: tensorwise
  enable_fsdp_float8_all_gather: true
  precompute_float8_dynamic_scale_for_fsdp: true
  force_recompute_fp8_weight_in_bwd: true
  filter_fqns: ["lm_head"]
  emulate: false
```

### FP8Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recipe_name` | str | None | FP8 recipe: "tensorwise", "rowwise", or "rowwise_with_gw_hp" |
| `enable_fsdp_fp8_all_gather` | bool | False | Enable FP8 all-gather in FSDP for bandwidth savings |
| `force_recompute_fp8_weight_in_bwd` | bool | False | Force recomputation of FP8 weights in backward pass |
| `precompute_fp8_dynamic_scale_for_fsdp` | bool | False | Precompute FP8 scales for FSDP optimization |
| `filter_fqns` | list[str] | [] | Module names to exclude from FP8 conversion |
| `emulate` | bool | False | Use emulation instead of hardware acceleration |

### Scaling Strategies

#### Tensorwise Scaling (Default)
- Single scale per tensor
- Good performance, moderate accuracy
- Recommended for most use cases


#### Rowwise Scaling
- Scale per row for better accuracy
- Slower than tensorwise
- Better numerical stability


For more on scaling strategies, refer to the [TorchAO FP8 documentation](https://github.com/pytorch/ao/tree/main/torchao/float8).

## Module Filtering

You can exclude specific modules from FP8 conversion using `filter_fqns`:

```yaml
fp8:
  enabled: true
  recipe_name: tensorwise
  filter_fqns: ["lm_head"]  # Skip these modules
```

### Speed and Convergence

FP8 quantization provides measurable performance improvements while maintaining model convergence:

- **Speed**: Over 1.2x training speedup on 8xH100 with tensorwise scaling.
- **Convergence**: FP8 training achieves loss parity with BF16 training.
- **Memory**: FP8 training achieves on par memory usage with BF16 baseline.

<img src="fp8_convergence.jpg" alt="FP8 Convergence Comparison" width="600px" />

*Figure: Loss curves comparing FP8 tensorwise scaling + torch.compile vs. BF16 + torch.compile training on 8xH100 with 8k sequence length, demonstrating virtually identical convergence behavior with 1.24x speedup*

## Ready to use recipes
We provide fp8 training configs for popular models:

- **Llama**: [Llama 3.1 8B](https://github.com/NVIDIA/NeMo-Automodel/blob/main/examples/llm_finetune/llama/llama3_1_8b_hellaswag_fp8.yaml)
- **Mistral**: [Mistral 7B](https://github.com/NVIDIA/NeMo-Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_hellaswag_fp8.yaml), [Mistral Nemo 2407](https://github.com/NVIDIA/NeMo-Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_hellaswag_fp8.yaml) 
- **Qwen**: [Qwen 2.5 7B](https://github.com/NVIDIA/NeMo-Automodel/blob/main/examples/llm_finetune/qwen/qwen2_5_7b_hellaswag_fp8.yaml)
- **Phi**: [Phi 4](https://github.com/NVIDIA/NeMo-Automodel/blob/main/examples/llm_finetune/phi/phi_4_hellaswag_fp8.yaml)

Check out our [examples directory](https://github.com/NVIDIA/NeMo-Automodel/tree/main/examples/llm_finetune/) for more recipes and configurations.

To run any of these FP8 training recipes, use the following command:

```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config <path-to-config.yaml>
```

For example, to train Llama 3.1 8B with FP8:
```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config examples/llm_finetune/llama/llama3_1_8b_hellaswag_fp8.yaml
```


## Performance Considerations

#### FP8 requires specific conditions to be effective:
- Input tensors must have dimensions divisible by 16 
- Using compatible hardware (H100+)
- Training with `torch.compile`

FP8 works best when the majority of GEMM operations are sufficiently large such that the speedup achieved by using FP8 tensor cores is greater than the overhead of dynamic quantization.

#### Ideal Conditions for FP8 Performance

- Linear layers are large and compute-intensive
- The model consists of fewer small operations and more large matrix multiplications
- You have modern (H100+) hardware optimized for FP8 acceleration
- Moderate numerical precision is acceptable and slight approximations won't affect outcomes



## References

- [TorchAO FP8 Documentation](https://github.com/pytorch/ao/tree/main/torchao/float8)
- [FP8 Performance Benchmarks](https://github.com/pytorch/ao/tree/main/torchao/float8#performance)
- [NVIDIA FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) 
