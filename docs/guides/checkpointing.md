# Checkpointing in NeMo AutoModel

## Introduction

In model training, checkpoints are used to periodically save intermediate model states (including model weights, optimizer states, and other necessary metadata). This allows for easy recovery if the training process is interrupted.

Checkpointing in NeMo AutoModel refers to saving the state of a distributed training job across multiple GPUs or nodes. This approach aims to reduce memory overhead and improve GPU utilization. It also provides users with the flexibility to resume training using different parallelism strategies.

NeMo AutoModel offers state checkpointing across [HuggingFace Safetensors](https://huggingface.co/docs/safetensors/en/index) and [PyTorch Distributed Checkpointing (DCP)](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) formats. State checkpointing can be done either sharded or consolidated.

- **Sharded** checkpoints: distributed training is done by splitting states (model weights, optimizer states, etc.) over many GPUs and each GPU will save its own "shard" of the full state. As such, it allows checkpointing to be done in parallel across all the GPUs thus speeding up checkpointing tremendously.
- **Consolidated** checkpoints: the multiple sharded files get _consolidated_ from a sharded checkpoint into one complete HuggingFace compatible checkpoint. This can then be loaded into downstream applications using the HuggingFace API (e.g., HuggingFace, vLLM, SGLang, etc.).

We provide an overview of the different types of available checkpoint formats in the table below.

| Model type           | DCP (sharded) | Safetensors (sharded) | Safetensors (consolidated) |
|----------------------|:-----------:|:-------------------:|:------------------------:|
| LLM                  | ✅          | ✅                   | ✅                      |
| VLM                  | ✅          | ✅                   | ✅                      |
| LLM / VLM – PEFT     | ❌          | ❌                   | ✅                      | 

The user can seamlessly switch between output formats through the recipe `yaml` file
```
checkpoint:
    ...
    model_save_format: torch_save # Format for saving (torch_save or safetensors)
    save_consolidated: false # Change to true if you want the checkpoint to have HuggingFace compatibility.
                             # Requires model_save_format to be safetensors.
    ...
```
> **Note:** The optimizer states are _always_ saved in DCP (`.distcp` extension) format.

## Safetensors
To ensure a smooth integration with the HuggingFace ecosystem, we make the Safetensors format (`.safetensors` extension) available to the user.

The sharded Safetensors format leverages the PyTorch DCP API under the hood for saving. PyTorch DCP supports loading and saving training states from multiple ranks in parallel, which makes checkpointing far more efficient as all GPUs can contribute their "shard" of the state. We can also benefit from features like load-time resharding which allows the user to save in one hardware setup and load it back in another. For example, the user can save with 2 GPUs at train time and still be able to load it back in with 1 GPU. 

**Most importantly**, the added advantage of this format is that we can optionally consolidate the various shards into a full HuggingFace format model.

To showcase this, we can run the following command on 2 GPUs and we get the following checkpoint:
```bash
uv run torchrun --nproc-per-node=2 examples/llm/finetune.py --step_scheduler.ckpt_every_steps 20 --checkpoint.model_save_format safetensors --checkpoint.save_consolidated True

...
> Saving checkpoint to checkpoints/epoch_0_step_20
...
```
:::{tip}
If you're running on a single GPU, you can run
```
uv run examples/llm/finetune.py --step_scheduler.ckpt_every_steps 20 --checkpoint.model_save_format safetensors --checkpoint.save_consolidated True
```
:::
```
checkpoints/
└── epoch_0_step_20/
    ├── model/
    │   ├── consolidated/
    │   │   ├── config.json
    │   │   ├── model-00001-of-00001.safetensors
    │   │   ├── model.safetensors.index.json
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   └── tokenizer.json
    │   ├── shard-00001-model-00001-of-00002.safetensors
    │   └── shard-00002-model-00001-of-00002.safetensors
    └── optim/
        ├── __0_0.distcp
        ├── __1_0.distcp
        └── .metadata
    ...
```
The `shard-*` files will be used by the PyTorch DCP API when resuming from the checkpoint for training. The consolidated model is only stored for downstream usage.

The consolidation step happens on the main process as the final step of checkpointing. This step is done solely through file I/O operations which means we will never have to materialize the sharded weights in GPU memory.

We can load and run the consolidated model using HuggingFace API:

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

### PEFT
When a user performs training using PEFT techniques, the trainable model weights are only a fraction of the full model. All remaining model weights are treated to be frozen.

This means that the state to checkpoint is very small (usually a few MB), so it's unnecessary to have multiple small sharded states. Consequently, NeMo AutoModel enforces consolidated HuggingFace compatible checkpoints when training with PEFT techniques. To run an example on 2 GPUs, the user can run the following example:
```
uv run torchrun --nproc-per-node=2 examples/llm/finetune.py --config examples/llm/llama_3_2_1b_hellaswag_peft.yaml --step_scheduler.ckpt_every_steps 20 --checkpoint.model_save_format safetensors
```

A HuggingFace compatible PEFT checkpoint gets saved
```
checkpoints/
└── epoch_0_step_20/
    ├── model/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── automodel_peft_config.json
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── tokenizer.json
    └── optim/
        ├── __0_0.distcp
        ├── __1_0.distcp
        └── .metadata
    ...
```

The example below showcases the direct compatibility of NeMo AutoModel with HuggingFace and PEFT
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
NeMo AutoModel also offers native PyTorch DCP checkpointing support (`.distcp` extension). Similar to Safetensors, it also provides the same features of load-time resharding and parallel saving.

As a simple example, we can run the following command to launch the training recipe on 2 GPUs.
```bash
uv run torchrun --nproc-per-node=2 examples/llm/finetune.py --step_scheduler.ckpt_every_steps 20 --checkpoint.model_save_format torch_save

...
> Saving checkpoint to checkpoints/epoch_0_step_20
...
```
After 20 steps, the following checkpoint will be saved

```
checkpoints/
└── epoch_0_step_20/
    ├── model/
    │   ├── __0_0.distcp
    │   ├── __1_0.distcp
    │   └── .metadata
    └── optim/
        ├── __0_0.distcp
        ├── __1_0.distcp
        └── .metadata
        ...
```

If you run the same script again, NeMo AutoModel will automatically find and load the latest checkpoint back in
```bash
uv run torchrun --nproc-per-node=2 examples/llm/finetune.py --step_scheduler.ckpt_every_steps 20 --checkpoint.model_save_format torch_save

...
> Loading checkpoint from checkpoints/epoch_0_step_20
...
```

## Saving Additional States
You can also save additional states in NeMo AutoModel. By default, we also automatically checkpoint the `dataloader`, `rng`, and `step_scheduler` states which are necessary to resume training accurately. In full, a Safetensors consolidated checkpoint will look like this:

```
checkpoints/
└── epoch_0_step_20/
    ├── model/
    │   ├── consolidated/
    │   │   ├── config.json
    │   │   ├── model-00001-of-00001.safetensors
    │   │   ├── model.safetensors.index.json
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   └── tokenizer.json
    │   ├── shard-00001-model-00001-of-00002.safetensors
    │   └── shard-00002-model-00001-of-00002.safetensors
    ├── optim/
    │   ├── __0_0.distcp
    │   ├── __1_0.distcp
    │   └── .metadata
    ├── dataloader.pt
    ├── rng.pt
    └── step_scheduler.pt
```

If the user wants to define a new state to be checkpointed in the recipe, the easiest way to do this is create a new attribute in the recipe class (i.e., defined using `self.` inside the recipe) and make sure the new attribute has `load_state_dict` and `state_dict` methods.

Here is a skeleton of what it might look like

```python

class NewState:

    def __init__(self, ...):
        self.state_value = ...
        self.another_value = ...
        ...
    
    def state_dict() -> dict[str, Any]:
        return {
            "<some state you're tracking>": self.state_value,
            "<another state you're tracking>": self.another_value,
        }
    
    def load_state_dict(state_dict: dict[str, Any]) -> None:
        self.state_value = state_dict["<some state you're tracking>"]
        self.another_value = state_dict["<another state you're tracking>"]
```

Then inside the recipe, you can define it using `self.new_state = NewState(...)`.