# Full Model Fine-Tuning on HellaSwag

This document explains how to perform SFT on HellaSwag in NeMo Automodel, the default finetuning recipe. It outlines key operations, including initiating SFT runs and managing experiment configurations using YAML.

## Launch an SFT Run

The script, [recipes/llm/finetune.py](https://github.com/NVIDIA-NeMo/Automodel/blob/main/recipes/llm/finetune.py), can be used to launch an experiment. This script can be launched either locally or on a cluster (TODO).
<!-- For details on how to launch a job on a cluster, refer to the [cluster documentation](../environment/cluster.md). -->

Be sure to launch the job using `uv`. The command to launch an SFT job is as follows:

```bash
uv run recipes/llm/finetune.py
```

> **_NOTE:_**  The default config launches a finetune run with Llama-3.2-1B. Llama models require the user to provide a HF token, which can be done using `export HF_TOKEN=<YOUR HF TOKEN>`

## Example Configuration File

NeMo Automodel allows users to configure experiments using `yaml` config files. To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run recipes/llm/finetune.py \
    --step_scheduler.ckpt_every_steps 50 \
    --rng.seed 1234
```

## Logging

By default, metrics like losses and peak GPU memory usage will be printed out. If you'd like to also use Weights&Biases logging, pass in your Weights&Biases key using `export WANDB_API_KEY=<YOUR WANDB API KEY>`.


## Training

The default execution is for the training to be launched on a single GPU. If you wish to launch on multiple GPUs, you can additionally launch using `torchrun` to initialize the distributed strategy. We use FSDP2 as a default. For example, to launch on 2 GPUs you can use:

```bash
uv run torchrun --nproc-per-node=2 recipes/llm/finetune.py
```

We also allow for tensor-parallel and context-parallel training strategies. These can also be passed through the command-line interface. For example, to use tensor-parallel on 2 GPUs you can run

```bash
uv run torchrun --nproc-per-node=2 recipes/llm/finetune.py \
    --distributed.tp_size 2
```

TODO: include screenshots of training metrics, wandb loss curve

## Checkpointing

We allow training state checkpointing to be done in either [Safetensors](https://huggingface.co/docs/safetensors/en/index) or [PyTorch DCP](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) format.