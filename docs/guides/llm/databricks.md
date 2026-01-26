# Model Training on Databricks

Databricks is a widely-used platform for managing data, models, applications, and compute on the cloud. This guide shows how to use Automodel for scalable, performant model training on Databricks.

The specific example here fine-tunes a [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model using the [SQuAD dataset](https://huggingface.co/datasets/rajpurkar/squad) from Hugging Face, but any Automodel functionality (for example, {doc}`model pre-training <pretraining>`, {doc}`VLMs </model-coverage/vlm>`, {doc}`other supported models </model-coverage/overview>`) can also be run on Databricks.

## Compute

Let’s start by [provisioning](https://docs.databricks.com/aws/en/compute/configure) a Databricks classic compute cluster with the following setup:

- Databricks runtime: [18.0 LTS (Machine Learning version)](https://docs.databricks.com/aws/en/release-notes/runtime/18.0ml)
- Worker instance type: `g6e.12xlarge` on AWS (4x L40S GPU per node)  
- Number of workers: 2  
- Global [environment variable](https://docs.databricks.com/aws/en/compute/configure#environment-variables): `GLOO_SOCKET_IFNAME=eth0` (see [this](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor#gloo-failure-runtimeerror-connection-refused) for details)   
- Cluster-scoped [init script](https://docs.databricks.com/aws/en/init-scripts/cluster-scoped):

```bash
#!/bin/bash

# Install Automodel on all nodes
/databricks/python3/bin/pip install git+https://github.com/NVIDIA-NeMo/Automodel
```

This will provision three compute nodes – one driver node we’ll attach a notebook to, and two worker nodes we’ll use for multi-node training.

Note that we’ve selected a small number of instances for demo purposes, but you can adjust the specific instance type and number of workers for your actual use case.

## Training

With the above compute resources provisioned, we’re ready to fine-tune a model using Automodel.

Automodel uses YAML file recipes to configure various settings for the training process (for example, model, dataset, loss function, optimizer, etc.). Here we’ll use this [preconfigured recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml) for fine-tuning a Llama-3.2-1B model using the SQuAD dataset from Hugging Face. In a notebook connected to our compute resource, download the training script and configuration file with these `curl` commands:

```bash
# Download training script
!curl -O https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/refs/heads/main/examples/llm_finetune/finetune.py
# Download configuration file
!curl -O https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/refs/heads/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

Here’s what the model, dataset, and optimizer portions of the config file look like:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

...
```

See the full file for complete details (`!cat llama3_2_1b_squad.yaml`). 

Finally, we'll [authenticate](https://huggingface.co/docs/hub/en/security-tokens) the VM running the notebook with Hugging Face so we can download the model and dataset:

```python
from getpass import getpass

hf_token = getpass("HF token: ")
```
```bash
!hf auth login --token {hf_token}
```

### Single-node

To run fine-tuning, we’ll use the `finetune.py` script from the Automodel repository and our config file.

To run training on a single GPU, use this command:

```bash
!python finetune.py \
    --config llama3_2_1b_squad.yaml \
    --step_scheduler.max_steps 20 \
    --checkpoint.checkpoint_dir /Volumes/<catalog_name>/<schema_name>/<volume_name>/checkpoints_single/ \
    --checkpoint.staging_dir /local_disk0/checkpoints_single/ \
    --checkpoint.is_async True
```

In addition to specifying the configuration file, we also use these options:

- `--step_scheduler.max_steps`: Limits the number of training steps taken. Again, this is for example purposes – adapt for your actual use case as needed.
- `--checkpoint.checkpoint_dir`: Tells Automodel where to {doc}`save model checkpoints </guides/checkpointing>` from training. We recommend saving model checkpoints in a Databricks Unity Catalog [volume](https://docs.databricks.com/aws/en/volumes/).
- `--checkpoint.staging_dir`: Specifies a temporary staging location for model checkpoints. Files will be temporarily saved to this location before being moved to the final `checkpoint_dir` location. This is needed when saving checkpoints in Unity Catalog. 
- `--checkpoint.is_async`: Uses asynchronous checkpointing. 

Looking at GPU metrics in Databricks, we see our single GPU is being well utilized (\~95% utilization).

:::{figure} ./databricks-gpu-metrics-single.png
:name: databricks-gpu-metrics-single
:alt: Single GPU utilization of ~95% during model training.
:align: center

Single GPU utilization of ~95% during model training.
:::

To utilize all four GPUs available on this `g6e.12xlarge` instance, use `torchrun --nproc-per-node=4` with our same training script and config file: 

```bash
!torchrun --nproc-per-node=4 finetune.py \
    --config llama3_2_1b_squad.yaml \
    --step_scheduler.max_steps 20 \
    --checkpoint.checkpoint_dir /Volumes/<catalog_name>/<schema_name>/<volume_name>/checkpoints_multi/ \
    --checkpoint.staging_dir /local_disk0/checkpoints_multi/ \
    --checkpoint.is_async True
```

This uses PyTorch’s [Elastic Launch](https://docs.pytorch.org/docs/stable/elastic/run.html) functionality to spawn and coordinate multiple training processes on the VM. Each training process runs on a separate GPU, and we can now see all four GPUs are being used (\~95% utilization for each GPU).

:::{figure} ./databricks-gpu-metrics-multi.png
:name: databricks-gpu-metrics-multi
:alt: Multi-GPU, single-node utilization of ~95% during model training.
:align: center

Multi-GPU, single-node utilization of ~95% during model training.
:::


### Multi-node

To scale further to multi-node training, we need to submit training jobs to all instances in our Databricks cluster.

First, each instance needs to be authenticated with Hugging Face to download the model and dataset:

```python
# Ensure workers are authenticated with Hugging Face

import subprocess
import shlex

def run_command(cmd):
    p = subprocess.run(shlex.split(cmd), capture_output=True)
    return p.stdout.decode()
 
rdd = sc.parallelize(range(sc.defaultParallelism))
rdd.mapPartitions(lambda _: [run_command("hf auth login --token " + hf_token)]).collect();
```

Next, we use PySpark's [TorchDistributor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html) to run the same training job across multiple instances like this:

```py
from pyspark.ml.torch.distributor import TorchDistributor

num_executor = 2            # Number of workers in cluster
num_gpus_per_executor = 4   # Number of GPUs per worker
distributor = TorchDistributor(
    num_processes=num_executor * num_gpus_per_executor,
    local_mode=False,
    use_gpu=True,
)

train_file = "finetune.py"
args = [
    "--config", "llama3_2_1b_squad.yaml",
    "--step_scheduler.max_steps", "20",
    "--checkpoint.checkpoint_dir", "/Volumes/<catalog_name>/<schema_name>/<volume_name>/checkpoints_dist/",
    "--checkpoint.staging_dir", "/local_disk0/checkpoints_dist/",
    "--checkpoint.is_async", "True",
]
distributor.run(train_file, *args)
```

`TorchDistributor` uses `torchrun` internally and also handles constructing and submitting training jobs to the cluster.

We now see GPU utilization is \~95% for all GPUs on all worker nodes during training (8 GPUs in this particular case).


## Conclusion

This guide showed how to use Automodel for model training on Databricks-managed compute. It’s relatively straightforward to scale from a single-GPU to multi-GPU to multi-node training to best suit your needs. 

While the example here fine-tunes a Llama-3.2-1B model using the SQuAD dataset, any supported Automodel functionality (like model pre-training, VLMs, etc.) can also run, and scale, on Databricks. Check out {doc}`additional recipes and end-to-end examples </guides/overview>` to learn more. 
