# MLFlow Logging in NeMo Automodel

## Introduction

MLFlow is an open-source platform for managing the machine learning lifecycle, including experiment tracking, model versioning, and deployment. NeMo Automodel integrates with MLFlow to automatically log training metrics, parameters, and artifacts during model training.

With MLFlow integration, you can:
- Track and compare experiments across multiple runs
- Log hyperparameters and training configurations
- Monitor training and validation metrics in real-time
- Store model checkpoints and artifacts
- Visualize experiment results through the MLFlow UI
- Share results with team members

## Prerequisites

Before using MLFlow logging in NeMo Automodel, ensure you have:

1. **MLFlow installed**: MLFlow is included as a dependency in NeMo Automodel. If needed, install it manually:
   ```bash
   uv add mlflow
   ```

2. **MLFlow tracking server** (optional): For production use, set up a tracking server to centralize experiment data. For local development, MLFlow will use a local file-based store by default.

## Configuration

Enable MLFlow logging by adding a `mlflow` section to your recipe YAML configuration:

```yaml
mlflow:
  experiment_name: "automodel-llm-llama3_2_1b_squad-finetune"
  run_name: ""
  tracking_uri: null
  artifact_location: null
  tags:
    task: "squad-finetune"
    model_family: "llama3.2"
    model_size: "1b"
    dataset: "squad"
    framework: "automodel"
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | "automodel-experiment" | Name of the MLFlow experiment. All runs are grouped under this experiment. |
| `run_name` | str | "" | Optional name for the current run. If empty, MLFlow generates a unique name. |
| `tracking_uri` | str | null | URI of the MLFlow tracking server. If null, uses local file-based storage. |
| `artifact_location` | str | null | Location to store artifacts. If null, uses default MLFlow location. |
| `tags` | dict | {} | Dictionary of tags to attach to the run for organization and filtering. |

### Tracking URI Options

The `tracking_uri` parameter determines where MLFlow stores experiment data:

- **Local file storage** (default): `null` or `file:///path/to/mlruns`
- **Remote tracking server**: `http://your-mlflow-server:5000`
- **Database backend**: `postgresql://user:password@host:port/database`

For team collaboration, we recommend setting up a remote tracking server.

## What Gets Logged

NeMo Automodel automatically logs the following information to MLFlow:

### Metrics
- Training loss at each step
- Validation loss and metrics
- Learning rate schedule
- Gradient norms (if gradient clipping is enabled)

### Parameters
- Model configuration (architecture, size, pretrained checkpoint)
- Training hyperparameters (learning rate, batch size, optimizer settings)
- Dataset information
- Parallelism configuration (DP, TP, CP settings)

### Tags
- Custom tags from configuration
- Automatically added tags:
  - Model name from `pretrained_model_name_or_path`
  - Global and local batch sizes

### Artifacts
- Model checkpoints (if configured)
- Training configuration files

> **Note**: Only rank 0 in distributed training logs to MLFlow to avoid duplicate entries and reduce overhead.

## Usage Example

Here's a complete example of training with MLFlow logging enabled:

### 1. Configure Your Recipe

Add the MLFlow configuration to your YAML file (e.g., `llama3_2_1b_squad.yaml`):

```yaml
step_scheduler:
  global_batch_size: 64
  local_batch_size: 8
  ckpt_every_steps: 1000
  val_every_steps: 10
  num_epochs: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

mlflow:
  experiment_name: "llama3-squad-finetune"
  run_name: "baseline-run-1"
  tracking_uri: null  # Uses local storage
  tags:
    task: "question-answering"
    dataset: "squad"
    model: "llama-3.2-1b"
```

### 2. Run Training

```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

During training, you'll see MLFlow logging messages:

```
MLflow run started: abc123def456
View run at: file:///path/to/mlruns/#/experiments/1/runs/abc123def456
```

### 3. View Results in MLFlow UI

Launch the MLFlow UI to visualize your experiments:

```bash
mlflow ui
```

By default, the UI runs at `http://localhost:5000`. Open this URL in your browser to:
- Compare metrics across runs
- View parameter configurations
- Download artifacts
- Filter and search experiments by tags

## Integration with Other Loggers

MLFlow can be used alongside other logging tools like Weights & Biases (WandB). Simply enable both in your configuration:

```yaml
# Enable both MLFlow and WandB
mlflow:
  experiment_name: "my-experiment"
  tags:
    framework: "automodel"

wandb:
  project: "my-project"
  entity: "my-team"
  name: "my-run"
```

Both loggers will track the same metrics independently, allowing you to leverage the strengths of each platform.

## Best Practices

### Experiment Organization

1. **Use descriptive experiment names**: Group related runs under meaningful experiment names
   ```yaml
   experiment_name: "llama3-squad-ablation-study"
   ```

2. **Tag your runs**: Add tags for easy filtering and comparison
   ```yaml
   tags:
     model_size: "1b"
     learning_rate: "1e-5"
     optimizer: "adam"
   ```

3. **Use run names for variants**: Differentiate runs within an experiment
   ```yaml
   run_name: "lr-1e5-bs64"
   ```

### Remote Tracking Server

For team collaboration, set up a shared MLFlow tracking server:

```yaml
mlflow:
  tracking_uri: "http://mlflow-server.example.com:5000"
  experiment_name: "team-llm-experiments"
```

### Artifact Storage

For large-scale experiments, configure a dedicated artifact location:

```yaml
mlflow:
  artifact_location: "s3://my-bucket/mlflow-artifacts"
```

Supported storage backends include S3, Azure Blob Storage, Google Cloud Storage, and network file systems.

### Performance Considerations

- MLFlow logging adds minimal overhead since only rank 0 logs
- Metrics are logged asynchronously to avoid blocking training
- For very frequent logging (every step), consider increasing `val_every_steps` to reduce I/O

## Troubleshooting

### MLFlow Not Installed

If you see an import error:
```
ImportError: MLflow is not installed. Please install it with: uv add mlflow
```

Install MLFlow:
```bash
uv add mlflow
```

### Connection Issues

If you can't connect to a remote tracking server:
- Verify the `tracking_uri` is correct
- Check network connectivity and firewall rules
- Ensure the tracking server is running

### Missing Metrics

If metrics aren't appearing in MLFlow:
- Verify you're running on rank 0 or check rank 0 logs
- Ensure the MLFlow run started successfully (check for "MLflow run started" message)
- Check that metrics are being computed during training

## References

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLFlow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [NeMo Automodel Examples](https://github.com/NVIDIA/NeMo-Automodel/tree/main/examples)