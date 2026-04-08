# Checkpointing in NeMo AutoModel

## Introduction

During machine-learning experiments, the model-training routine regularly saves checkpoints. A checkpoint is a complete snapshot of a run that includes model weights, optimizer states, and other metadata required to resume training exactly where it left off. Writing these snapshots at regular intervals lets you recover quickly from crashes or pauses without losing progress.

NeMo AutoModel checkpoints capture the complete state of a distributed training run across multiple GPUs or nodes. This reduces memory overhead, improves GPU utilization, and allows training to be resumed with a different parallelism strategy.

NeMo AutoModel writes checkpoints in two formats: [Hugging Face Safetensors](https://github.com/huggingface/safetensors) and [PyTorch Distributed Checkpointing (DCP)](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html). It also supports two layouts:

- **Consolidated Checkpoints**: The complete model state is saved as a Hugging Face-compatible bundle, typically in a single file or a compact set of files with an index. Because tensors are not split across GPUs (unsharded), tools like Hugging Face, vLLM, and SGLang can load these checkpoints directly.

- **Sharded Checkpoints**: During distributed training with parameter sharing, each GPU holds a subset (or "shard") of the full state, such as model weights and optimizer states. When checkpointing, each GPU writes its own shard independently without reconstructing the full model state.

We provide an overview of the different types of available checkpoint formats in the table below.

Task | Model domain  | DCP (sharded) | Safetensors (sharded) | Safetensors (consolidated) |
-----|----------------------|:-----------:|:-------------------:|:------------------------:|
SFT  | LLM                  | ✅          | ✅                   | ✅                      |
SFT  | VLM                  | ✅          | ✅                   | ✅                      |
PEFT | LLM / VLM            | 🚧          | 🚧                   | ✅                      | 


Changing between output formats can be done seamlessly through the recipe's `yaml` configuration file:
```yaml
checkpoint:
    ...
    model_save_format: safetensors # Format for saving (torch_save or safetensors)
    save_consolidated: true # Change to false if you want to save sharded checkpoints.
    ...
```
::: {important}
For optimal compatibility with the Hugging Face ecosystem, including downstream tools such as vLLM and SGLang, we recommend using the checkpoint configuration provided above.
:::
The optimizer states are _always_ saved in DCP (`.distcp` extension) format.
:::

## Checkpoint Symbolic Links

NeMo AutoModel automatically creates symbolic links in the checkpoint directory to provide convenient access to important checkpoints:

- **LATEST**: Points to the most recently saved checkpoint. This is useful for resuming training from the last saved state.
- **LOWEST_VAL**: Points to the checkpoint with the lowest validation score/loss. This provides easy access to the best-performing checkpoint based on validation metrics, making it ideal for model evaluation or deployment.

These symbolic links eliminate the need to manually track checkpoint names or search through directories to find the best model. When validation is enabled in your training run, both links are automatically maintained and updated as training progresses.

## Safetensors
To ensure seamless integration with the Hugging Face ecosystem, NeMo AutoModel saves checkpoints in the [Safetensors](https://github.com/huggingface/safetensors) format. Safetensors is a memory-safe, zero-copy alternative to Python's pickle (PyTorch .bin), natively supported by Hugging Face Transformers, offering both safety and performance advantages over Python pickle-based approaches.

### Key Benefits
- **Native Hugging Face Compatibility**: Checkpoints can be loaded directly into Hugging Face-compatible tools, including vLLM, SGLang, and others.
- **Memory Safety and Speed**: The Safetensors format prohibits saving serialized Python code, ensuring memory safety, and supports zero-copy loading for improved performance.
- **Optional Consolidation**: Sharded checkpoints can be merged into a standard Hugging Face model format for easier downstream use.

**Most importantly**, this format offers the added advantage of optionally consolidating multiple shards into a complete Hugging Face format model.

### Example

The following command runs the LLM fine-tuning recipe on two GPUs and saves the resulting checkpoint in the Safetensors format:
```bash
automodel --nproc-per-node=2 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --step_scheduler.ckpt_every_steps 20 \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated True
```

::: {note}
In the above command we used the [`llama3_2_1b_squad.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/492add84a2b9d495946fe211c28973cd00051f3e/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml) config as a running example, adjust as necessary to your case.
More config examples can be found in our [`examples/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples) directory.
:::

If you're running on a single GPU, you can run:
```bash
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --step_scheduler.ckpt_every_steps 20 \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated True
```

After running for a few seconds, the standard output should be:
```
...
> Saving checkpoint to checkpoints/epoch_0_step_20
...
```

The `checkpoints/` directory should have the following contents:
```
checkpoints/
├── LATEST -> epoch_0_step_20
├── LOWEST_VAL -> epoch_0_step_20
└── epoch_0_step_20
   ├── model
   │   ├── consolidated
   │   │   ├── config.json
   │   │   ├── generation_config.json
   │   │   ├── model-00001-of-00001.safetensors
   │   │   ├── model.safetensors.index.json
   │   │   ├── special_tokens_map.json
   │   │   ├── tokenizer.json
   │   │   └── tokenizer_config.json
   │   ├── shard-00001-model-00001-of-00001.safetensors
   │   └── shard-00002-model-00001-of-00001.safetensors
   └── optim
       ├── __0_0.distcp
       └── __1_0.distcp
...
```

The `epoch_0_step_20/` directory stores the full training state from step `20` of the first epoch, including both the model and optimizer states.

We can load and run the consolidated checkpoint using the Hugging Face Transformers API directly:
```python
import torch
from transformers import pipeline

model_id = "checkpoints/epoch_0_step_20/model/consolidated/"
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

print(pipe("The key to life is"))

>>> [{'generated_text': 'The key to life is to be happy. The key to happiness is to be kind. The key to kindness is to be'}]
```

Although this example uses the Hugging Face Transformers API, the `consolidated/` checkpoint is compatible with any Hugging Face-compatible tool, such as vLLM, SGLang, and others.


## PEFT
When training with Parameter-Efficient Fine-Tuning (PEFT) techniques, only a small subset of model weights are updated — the rest of the model remains frozen. This dramatically reduces the size of the checkpoint, often to just a few megabytes.

### Why Consolidated Checkpoints?
Because the PEFT state is so lightweight, sharded checkpointing adds unnecessary overhead. Instead, NeMo AutoModel automatically saves a single, consolidated Hugging Face–compatible checkpoint when using PEFT. This makes it:

- easier to manage and share (just the adapters),
- compatible with Hugging Face Transformers out of the box,
- ideal for deployment and downstream evaluation.

### Example: PEFT Fine-Tuning on Two GPUs

To fine-tune a model using PEFT and save a Hugging Face–ready checkpoint:
```bash
automodel --nproc-per-node=2 examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml --step_scheduler.ckpt_every_steps 20 --checkpoint.model_save_format safetensors
```

After training, you'll get a compact, consolidated Safetensors checkpoint that can be loaded directly with Hugging Face tools:

```
checkpoints/
├── LATEST -> epoch_0_step_20
├── LOWEST_VAL -> epoch_0_step_20
├── epoch_0_step_20
│   ├── config.yaml
│   ├── dataloader
│   │   ├── dataloader_dp_rank_0.pt
│   │   └── dataloader_dp_rank_1.pt
│   ├── losses.json
│   ├── model
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── automodel_peft_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── optim
│   │   ├── __0_0.distcp
│   │   └── __1_0.distcp
│   ├── rng
│   │   ├── rng_dp_rank_0.pt
│   │   └── rng_dp_rank_1.pt
│   └── step_scheduler.pt
├── training.jsonl
└── validation.jsonl
```

The example below showcases the direct compatibility of NeMo AutoModel with Hugging Face and PEFT:
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

checkpoint_path = "checkpoints/epoch_0_step_20/model/"
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model = model.to("cuda")
model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

>>> Preheat the oven to 350 degrees and place the cookie dough in a large bowl. Roll the dough into 1-inch balls and place them on a cookie sheet. Bake the cookies for 10 minutes. While the cookies are baking, melt the chocolate chips in the microwave for 30 seconds.
```

## PyTorch DCP
NeMo AutoModel also offers native PyTorch DCP checkpointing support (`.distcp` extension). Like Safetensors, it supports load-time resharding and parallel saving.

As a simple example, we can run the following command to launch the training recipe on two GPUs.
```bash
automodel --nproc-per-node=2 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --step_scheduler.ckpt_every_steps 20 \
    --checkpoint.model_save_format torch_save

...
> Saving checkpoint to checkpoints/epoch_0_step_20
...
```
After 20 steps, the following checkpoint will be saved:

```
checkpoints/
├── LATEST -> epoch_0_step_20
├── LOWEST_VAL -> epoch_0_step_20
└── epoch_0_step_20
   ├── config.yaml
   ├── dataloader
   │   ├── dataloader_dp_rank_0.pt
   │   └── dataloader_dp_rank_1.pt
   ├── losses.json
   ├── model
   │   ├── __0_0.distcp
   │   └── __1_0.distcp
   └── optim
       ├── __0_0.distcp
       └── __1_0.distcp
...
```

If you rerun the script, NeMo AutoModel automatically detects and restores the most recent checkpoint.
```bash
automodel --nproc-per-node=2 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --step_scheduler.ckpt_every_steps 20 \
    --checkpoint.model_save_format torch_save

...
> Loading checkpoint from checkpoints/epoch_0_step_20
...
```

## Saving Checkpoints When Using Docker

When training inside a Docker container (see [Installation Guide](installation.md)), any files written to the container's filesystem are lost when the container exits (especially with `--rm`). To keep your checkpoints, you must **bind-mount a host directory** to the checkpoint path before starting the container:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v "$(pwd)"/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:25.11.00
```

You can also set a custom checkpoint directory in the YAML config or as a CLI override:
```yaml
checkpoint:
  checkpoint_dir: /mnt/shared/my_checkpoints
```
```bash
# Or via CLI override:
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --checkpoint.checkpoint_dir /mnt/shared/my_checkpoints
```

When using a custom path, make sure the corresponding host directory is mounted into the container with `-v`.

::: {tip}
Mount additional host directories for datasets and the Hugging Face model cache to avoid re-downloading large models across container restarts. See the [Installation Guide](installation.md) for a complete `docker run` example with all recommended mounts.
:::

## Asynchronous Checkpointing

NeMo AutoModel can write checkpoints asynchronously to reduce training stalls caused by I/O. When enabled, checkpoint writes are scheduled in the background using PyTorch Distributed Checkpointing's async API while training continues.

- **Enable** (YAML):
  ```yaml
  checkpoint:
    is_async: true
  ```
- **Enable** (CLI): add `--checkpoint.is_async True` to your run command.
- **Requirements**: PyTorch ≥ 2.9.0. If an older version is detected, async mode is automatically disabled.
- **Behavior**: At most one checkpoint uploads at a time; the next save waits for the previous upload to finish. The `LATEST` symlink is updated after the async save completes (may be deferred until the next save call). During PEFT, adapter model files are written synchronously on rank 0; optimizer states can still use async.

## Advanced Usage: Save Additional States
You can also save additional states in NeMo AutoModel. By default, we automatically checkpoint the `dataloader`, `rng`, and `step_scheduler` states that are necessary to resume training accurately. In full, a Safetensors consolidated checkpoint will look like this:

```
checkpoints/
├── LATEST -> epoch_0_step_20
├── LOWEST_VAL -> epoch_0_step_20
├── epoch_0_step_20
│   ├── config.yaml
│   ├── dataloader
│   │   ├── dataloader_dp_rank_0.pt
│   │   └── dataloader_dp_rank_1.pt
│   ├── losses.json
│   ├── model
│   │   ├── consolidated
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   ├── model-00001-of-00001.safetensors
│   │   │   ├── model.safetensors.index.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   └── tokenizer_config.json
│   │   ├── shard-00001-model-00001-of-00001.safetensors
│   │   └── shard-00002-model-00001-of-00001.safetensors
│   ├── optim
│   │   ├── __0_0.distcp
│   │   └── __1_0.distcp
│   ├── rng
│   │   ├── rng_dp_rank_0.pt
│   │   └── rng_dp_rank_1.pt
│   └── step_scheduler.pt
├── training.jsonl
└── validation.jsonl
```

If you want to define a new state to be checkpointed in the recipe, the easiest way is to create a new attribute in the recipe class (defined using `self.` inside the recipe). Just make sure that the new attribute uses both the `load_state_dict` and `state_dict` methods.

Here is an example of what it might look like:

```python

class NewState:

    def __init__(self, ...):
        self.state_value = ...
        self.another_value = ...
        ...
    
    def state_dict(self) -> dict[str, Any]:
        return {
            "<some state you're tracking>": self.state_value,
            "<another state you're tracking>": self.another_value,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state_value = state_dict["<some state you're tracking>"]
        self.another_value = state_dict["<another state you're tracking>"]
```

Inside your recipe class, define the new state as an instance attribute using `self.new_state = NewState(...)`.

## Cloud Storage Checkpoints (MSC)

NeMo AutoModel supports saving and loading checkpoints directly to cloud object storage
using NVIDIA's [Multi-Storage Client (MSC)](https://nvidia.github.io/multi-storage-client/).
This is useful when training on cloud clusters where local disk is ephemeral or too small
to hold full distributed checkpoints.

### Installation  

```bash
pip install multi-storage-client --index-url https://pypi.nvidia.com
```

### MSC Profile Configuration

MSC authenticates with your storage provider using a profile configuration file at
`~/.msc_config.yaml`. The profile name **must match the bucket name** in your
`msc://` path — this is the most common source of errors when first setting up MSC.

For example, if your checkpoint path is `msc://my-bucket/checkpoints`, your config
must have a profile named `my-bucket`:
```yaml
profiles:
  my-bucket:
    storage_provider:
      type: s3
      options:
        region_name: us-east-1
    credentials:
      type: s3
      options:
        access_key: YOUR_ACCESS_KEY
        secret_key: YOUR_SECRET_KEY
```

MSC supports AWS S3, Azure Blob Storage, Google Cloud Storage, and NVIDIA AIStore.
See the [MSC documentation](https://nvidia.github.io/multi-storage-client/) for
provider-specific configuration.

### Usage

Set `checkpoint_dir` to an `msc://` path — everything else works the same as local
checkpointing:
```yaml
checkpoint:
  checkpoint_dir: msc://my-bucket/checkpoints
```
```bash
automodel --nproc-per-node=2 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --checkpoint.checkpoint_dir msc://my-bucket/checkpoints \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated true
```

After 20 steps, you should see:

> Saving checkpoint to msc://my-bucket/checkpoints/epoch_0_step_20


The checkpoint layout in cloud storage is identical to the local layout described
above. Resume works the same way — rerunning the command picks up from the
`LATEST` symlink automatically:

> Loading checkpoint from msc://my-bucket/checkpoints/epoch_0_step_20


::: {note}
Asynchronous checkpointing (`is_async: true`) is supported with MSC paths and is
recommended for large models where synchronous cloud writes would stall training.
:::
