# Custom Callbacks

## Introduction

Callbacks provide a flexible way to inject custom logic into the training loop without modifying recipe code. They enable integration with external systems, custom logging, metrics collection, and monitoring.

Callbacks work across **all NeMo AutoModel recipes**: LLM fine-tuning, VLM fine-tuning, sequence classification, and knowledge distillation.

## Key Features

- **PyTorch Lightning-style API**: Familiar hooks like `on_train_start`, `on_train_batch_end`, `on_validation_end`
- **Programmatic integration**: Pass callbacks directly to recipe constructors
- **Full training context**: Access to recipe state, metrics, and checkpoint information
- **Distributed training support**: Includes `@rank_zero_only` decorator for multi-GPU environments

## Available Hooks

| Hook | When Called | Key Arguments |
|------|------------|---------------|
| `on_train_start` | After setup, before training begins | `recipe` |
| `on_train_batch_end` | After each training step | `recipe`, `train_log_data` |
| `on_validation_end` | After validation completes | `recipe`, `val_results` |
| `on_save_checkpoint` | When checkpoint is saved | `recipe`, `checkpoint_info` |
| `on_exception` | When training fails | `recipe`, `exception` |
| `on_train_end` | When training completes successfully | `recipe` |

## Quick Example

### 1. Define a Custom Callback

```python
from nemo_automodel.components.callbacks import Callback

class MetricsReporterCallback(Callback):
    """Report metrics to external API."""
    
    def on_train_batch_end(self, recipe, **kwargs):
        train_log_data = kwargs['train_log_data']
        step = train_log_data.step
        loss = train_log_data.metrics['loss']
        
        # Send to your API, database, or monitoring system
        print(f"Step {step}: Loss = {loss:.4f}")
    
    def on_validation_end(self, recipe, **kwargs):
        val_results = kwargs['val_results']
        
        # val_results is a dict: {"validation": MetricsSample, ...}
        for name, log_data in val_results.items():
            val_loss = log_data.metrics['val_loss']
            print(f"Validation '{name}': Loss = {val_loss:.4f}")
    
    def on_save_checkpoint(self, recipe, **kwargs):
        checkpoint_info = kwargs['checkpoint_info']
        print(f"Checkpoint saved: {checkpoint_info['checkpoint_path']}")
```

### 2. Pass to Recipe

```python
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

# Instantiate your callbacks
metrics_callback = MetricsReporterCallback()

# Pass to recipe constructor
recipe = TrainFinetuneRecipeForNextTokenPrediction(
    cfg,
    callbacks=[metrics_callback]
)

recipe.setup()
recipe.run_train_validation_loop()
```

## Distributed Training

In multi-GPU training, callbacks run on **all ranks**. Python `logging` (e.g., `logger.info()`) is automatically filtered to rank 0, but other operations need explicit handling.

### Use `@rank_zero_only` Decorator

```python
from nemo_automodel.components.callbacks import Callback, rank_zero_only

class CustomizerCallback(Callback):
    @rank_zero_only
    def on_validation_end(self, recipe, **kwargs):
        # This only runs on rank 0
        val_results = kwargs['val_results']
        
        # Safe to do file I/O, API calls, etc.
        requests.post('https://api.example.com/metrics', json=val_results)
```

### Or Using Manual Rank Checking

```python
def on_train_batch_end(self, recipe, **kwargs):
    # Check if main rank before doing expensive operations
    if recipe.dist_env.is_main:
        # Do rank-0-only work (file I/O, API calls, etc.)
        save_metrics_to_file()
```

## Hook Details

For complete API reference, see the [API documentation](../apidocs/index.rst).

### `on_train_batch_end`

```python
def on_train_batch_end(self, recipe, **kwargs):
    train_log_data = kwargs['train_log_data']  # MetricsSample
    # Fields: train_log_data.step, .epoch, .metrics (dict), .timestamp
```

### `on_validation_end`

```python
def on_validation_end(self, recipe, **kwargs):
    val_results = kwargs['val_results']  # dict[str, MetricsSample]
    # For single validation set: {"validation": MetricsSample}
    # For multiple validation sets: {"squad": MetricsSample, "hellaswag": MetricsSample, ...}
```

### `on_save_checkpoint`

```python
def on_save_checkpoint(self, recipe, **kwargs):
    checkpoint_info = kwargs['checkpoint_info']  # dict
    # Fields: 'epoch', 'step', 'train_loss', 'val_losses', 'checkpoint_path', 'best_metric_key'
```

## Complete Example

See [`examples/llm_finetune/finetune_with_callback.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/finetune_with_callback.py) for a full working example demonstrating:
- Multiple callbacks
- Distributed training with `@rank_zero_only`
- Metrics collection for external reporting
- Custom logging with prefixes

### Running the Example

```bash
# Single GPU
uv run python examples/llm_finetune/finetune_with_callback.py

# Multi-GPU
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune_with_callback.py
```

## Use Cases

- **External integrations**: Report metrics to W&B, MLflow, Customizer, or custom APIs
- **Progress monitoring**: Send Slack/email notifications on training milestones or failures
- **Custom metrics collection**: Track and store domain-specific metrics beyond standard training logs
