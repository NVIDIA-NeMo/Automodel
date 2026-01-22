# Quantization-Aware Training (QAT) in NeMo Automodel

NeMo Automodel supports Quantization-Aware Training (QAT) for Supervised Fine-Tuning (SFT) using [TorchAO](https://github.com/pytorch/ao). QAT simulates quantization effects during the training process, allowing models to adapt to lower precision representations while learning. This approach produces quantized models that maintain significantly higher accuracy compared to applying quantization after training is complete.

## What is Quantization-Aware Training?

Quantization-Aware Training simulates the effects of quantization during the training process. By introducing fake quantization operations in the forward pass, the model learns to adapt to lower precision representations, maintaining better accuracy when deployed with actual quantization.

### Benefits of QAT

- **Better accuracy**: Models trained with QAT maintain higher accuracy when quantized compared to post-training quantization
- **Efficient deployment**: Quantized models require less memory and compute resources
- **Edge device support**: Enables deployment on resource-constrained devices
- **Production optimization**: Reduces inference costs while maintaining model quality

### QAT vs. Post-Training Quantization

| Aspect | QAT | Post-Training Quantization |
|--------|-----|---------------------------|
| **Accuracy** | Higher - model adapts during training | Lower - no adaptation |
| **Training time** | Longer - requires retraining | None - applied after training |
| **Use case** | Production deployments requiring best accuracy | Quick prototyping or less critical applications |
| **Flexibility** | Can fine-tune quantization parameters | Limited to fixed quantization schemes |

## Requirements

To use QAT in NeMo Automodel, you need:

- **Software**: TorchAO library must be installed
- **Hardware**: Compatible NVIDIA GPU (recommended: A100 or newer)
- **Model**: Any supported model architecture for SFT

## Install TorchAO

Make sure you have TorchAO installed. Follow the [installation guide](https://github.com/pytorch/ao?tab=readme-ov-file#-installation) for TorchAO.

```bash
pip install torchao
```

## How QAT Works in NeMo Automodel

NeMo Automodel integrates TorchAO's QAT quantizers into the training pipeline. During training:

1. **Model preparation**: The quantizer prepares the model by inserting fake quantization operations
2. **Forward pass**: Weights and activations are quantized using fake quantization
3. **Backward pass**: Gradients flow through the fake quantization operations
4. **Weight updates**: Model learns to minimize loss while accounting for quantization effects

### Supported Quantization Schemes

NeMo Automodel supports two TorchAO QAT quantizers:

#### Int8 Dynamic Activation + Int4 Weight (8da4w-qat)
- **Quantizer**: `Int8DynActInt4WeightQATQuantizer`
- **Activations**: INT8 with dynamic quantization
- **Weights**: INT4 quantization
- **Use case**: Balanced accuracy and efficiency
- **Memory savings**: ~4x compared to FP16/BF16

#### Int4 Weight-Only (4w-qat)
- **Quantizer**: `Int4WeightOnlyQATQuantizer`
- **Activations**: Full precision
- **Weights**: INT4 quantization
- **Use case**: Maximum memory savings with minimal accuracy loss
- **Memory savings**: ~4x for weights only

## Configuration

To enable QAT in your training configuration, you need to specify the quantizer in your YAML configuration file.

### Basic Configuration

```yaml
# Enable QAT with Int8 Dynamic Activation + Int4 Weight quantization
qat:
  enabled: true
  quantizer:
    _target_: torchao.quantization.qat.Int8DynActInt4WeightQATQuantizer
    groupsize: 256
```

### Int4 Weight-Only Configuration

```yaml
# Enable QAT with Int4 Weight-Only quantization
qat:
  enabled: true
  quantizer:
    _target_: torchao.quantization.qat.Int4WeightOnlyQATQuantizer
    groupsize: 256
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | bool | Enable or disable QAT |
| `quantizer._target_` | str | Fully qualified class name of the TorchAO quantizer |
| `quantizer.groupsize` | int | Group size for weight quantization (typically 128 or 256) |

### Delayed Fake Quantization

You can optionally delay the activation of fake quantization to allow the model to train normally for a few steps before introducing quantization effects:

```yaml
qat:
  enabled: true
  quantizer:
    _target_: torchao.quantization.qat.Int8DynActInt4WeightQATQuantizer
    groupsize: 256
  delay_fake_quant_steps: 1000  # Enable fake quant after 1000 steps
```

## Training Workflow

### 1. Prepare Your Configuration

Create a YAML configuration file with QAT enabled:

```yaml
model:
  model_name: meta-llama/Llama-3.2-1B

task:
  type: sft
  
qat:
  enabled: true
  quantizer:
    _target_: torchao.quantization.qat.Int8DynActInt4WeightQATQuantizer
    groupsize: 256

trainer:
  max_steps: 10000
  val_check_interval: 500
```

### 2. Run Training

Launch training with your QAT-enabled configuration:

```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config your_qat_config.yaml
```

### 3. Monitor Training

During training, the model will:
- Apply fake quantization to weights and activations
- Learn to minimize loss while accounting for quantization effects
- Produce checkpoints that can be converted to actual quantized models

### 4. Deploy Quantized Model

After training, convert the QAT checkpoint to a fully quantized model for deployment:

```python
from torchao.quantization import quantize_

# Load your trained model
model = load_model_from_checkpoint(checkpoint_path)

# Apply actual quantization (not fake quantization)
quantize_(model, int8_dynamic_activation_int4_weight())

# Deploy the quantized model
model.eval()
```

## Performance Considerations

### Training Performance

- **Training time**: QAT adds overhead during training due to fake quantization operations
- **Memory usage**: Similar to full-precision training during the training phase
- **Convergence**: May require slightly more training steps to converge compared to full-precision training

### Inference Performance

After converting to actual quantization:

- **Speed**: 2-4x faster inference depending on hardware and model size
- **Memory**: ~4x reduction in model size
- **Accuracy**: Minimal degradation compared to full-precision models (typically <1% difference)

### When to Use QAT

QAT is most beneficial when:

- **Deploying to production**: Where inference efficiency is critical
- **Edge devices**: Resource-constrained environments
- **Large-scale serving**: Reducing infrastructure costs
- **Accuracy is important**: When post-training quantization causes unacceptable accuracy loss

### When Not to Use QAT

Consider alternatives when:

- **Quick prototyping**: Post-training quantization is faster
- **Small models**: Quantization overhead may not be worth it
- **Limited training resources**: QAT requires retraining the model
- **Accuracy is not critical**: Post-training quantization may be sufficient

## Best Practices

### 1. Start with Post-Training Quantization

Before investing in QAT, try post-training quantization to establish a baseline:

```python
# Quick post-training quantization test
from torchao.quantization import quantize_
quantize_(model, int8_dynamic_activation_int4_weight())
```

If accuracy is acceptable, you may not need QAT.

### 2. Choose the Right Quantization Scheme

- **8da4w-qat**: Best balance of accuracy and efficiency for most use cases
- **4w-qat**: Use when memory is the primary constraint and activations can remain full precision

### 3. Tune Group Size

The `groupsize` parameter affects the granularity of quantization:

- **Smaller groups (128)**: Better accuracy, slightly more memory
- **Larger groups (256)**: More efficient, may have minor accuracy impact

Start with 256 and reduce to 128 if accuracy is insufficient.

### 4. Monitor Validation Metrics

Track validation metrics closely during QAT training:

- Compare against full-precision baseline
- Watch for convergence issues
- Adjust learning rate if needed (QAT may benefit from slightly lower learning rates)

### 5. Use Delayed Fake Quantization

For better convergence, consider delaying fake quantization:

```yaml
qat:
  delay_fake_quant_steps: 1000  # Let model train normally first
```

This allows the model to learn basic patterns before introducing quantization constraints.

## Accuracy vs. Efficiency Trade-offs

### Expected Accuracy Impact

| Quantization Method | Typical Accuracy Loss | Memory Savings |
|---------------------|----------------------|----------------|
| Full Precision (BF16) | Baseline | Baseline |
| Post-Training Quantization | 1-3% | 4x |
| QAT (8da4w) | <1% | 4x |
| QAT (4w) | <1.5% | 4x (weights only) |

### Optimization Strategies

If accuracy is below expectations:

1. **Increase training steps**: QAT may need more training to converge
2. **Reduce learning rate**: Lower learning rates can help with quantization constraints
3. **Use 8da4w instead of 4w**: Better accuracy with minimal additional cost
4. **Reduce group size**: Smaller groups provide finer-grained quantization
5. **Delay fake quantization**: Give the model time to learn before quantizing

## Limitations and Known Issues

### Current Limitations

- **SFT only**: QAT is currently supported for Supervised Fine-Tuning tasks only
- **Model compatibility**: Not all model architectures may be compatible with TorchAO quantizers
- **Training overhead**: QAT adds computational overhead during training

### Troubleshooting

#### Issue: Training diverges or doesn't converge

**Solution**: Try these approaches:
- Reduce learning rate by 2-5x
- Increase `delay_fake_quant_steps` to 2000-5000
- Use a smaller group size (128 instead of 256)
- Verify your baseline model trains successfully without QAT

#### Issue: Accuracy is significantly worse than expected

**Solution**:
- Ensure you're comparing against the same baseline (same training steps, data, etc.)
- Try 8da4w quantization instead of 4w
- Reduce group size to 128
- Increase training steps by 20-30%

#### Issue: Out of memory during training

**Solution**:
- QAT should have similar memory usage to full-precision training
- Reduce batch size if needed
- Use gradient accumulation to maintain effective batch size

## References

- [TorchAO Documentation](https://github.com/pytorch/ao)
- [TorchAO QAT Guide](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat)
- [Quantization Fundamentals](https://pytorch.org/docs/stable/quantization.html)
- [INT8 Quantization for Deep Learning](https://arxiv.org/abs/1806.08342)
