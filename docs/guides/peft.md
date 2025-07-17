# Parameter-Efficient Fine-Tuning (PEFT) with NeMo AutoModel

## Introduction

As large language models (LLMs) continue to grow in size, the ability to
efficiently customize them for specific tasks has become increasingly
important. Fine-tuning allows a general pre-trained model to be adapted
to a particular domain or dataset, improving accuracy and relevance
while leveraging the extensive knowledge captured during pretraining.

However, full-parameter fine-tuning can be computationally expensive,
requiring significant hardware resources. To address this, PEFT
techniques such as [Low-Rank Adapters (LoRA)](https://arxiv.org/abs/2106.09685) have emerged as a
lightweight approach that updates only a small subset of parameters
while keeping the base model weights frozen. This approach reduces the
number of trainable parameters, often to less than 1%, while
achieving performance comparable to full fine-tuning.

This guide offers a comprehensive overview of fine-tuning models
available on the Hugging Face Hub using NeMo AutoModel. In addition to
fine-tuning, it illustrates how effortlessly you can generate text with
the trained adapters, evaluate model performance with the [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness),
publish your adapters to the Hugging Face Model Hub, and export
them to [vLLM](https://docs.vllm.ai/en/latest/) for seamless deployment.

<!-- In addition to this user guide, you can also explore our Quickstart,
which features a [standalone Python3
script](https://github.com/NVIDIA/NeMo/blob/main/examples/llm/peft/automodel.py)
and a guided [Jupyter
Notebook](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/automodel/peft.ipynb),
offering hands-on demonstrations for quickly getting started with NeMo
AutoModel. -->

## Run PEFT with NeMo AutoModel

### Prerequisites: install NeMo Automodel
Before proceeding with this guide, please ensure that you have NeMo Automodel installed on your
machine. This can be achieved by running:
```bash
pip3 install nemo-automodel
```
For a complete guide and additional options please consult the [`Automodel installation guide`](installation.md).

### Model and Dataset Context
In this guide, we will fine-tune Meta’s `LLaMA 3.2 1B` model on the popular [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset).

#### 🔍 About LLaMA 3.2 1B
**LLaMA** is a family of decoder-only transformer models developed by Meta. The **LLaMA 3.2 1B** variant is a compact, lightweight model ideal for research and edge deployment. Despite its size, it maintains architectural features consistent with its larger siblings:

- **Decoder-only architecture**: Follows a GPT-style, autoregressive design—optimized for generation tasks.

- **Rotary positional embeddings (RoPE)**: Efficient and extendable positional encoding technique.

- **Grouped-query attention (GQA)**: Enhances scalability by decoupling key/value heads from query heads.

- **SwiGLU activation**: A variant of the GLU activation, offering improved convergence and expressiveness.

- **Multi-layer residual connections**: Enhances training stability and depth scaling.

These design choices make LLaMA models highly competitive across various benchmarks, and their open weights make them a strong base for task-specific fine-tuning.

> [!TIP]
> In this guide, `meta-llama/Llama-3.2-1B` is used only as a placeholder
> model ID. You can replace it with any valid Hugging Face model ID, such
> as `Qwen/Qwen2.5-1.5B`, or any other checkpoint you have access to on
> the Hugging Face Hub.

> [!IMPORTANT]
> Some Hugging Face model repositories are **gated**, you must explicitly
> request permission before you can download their files. If the model
> page shows a "Request access" or "Agree and access" button:
>
> 1.  Log in with your Hugging Face account.
> 2.  Click the button and accept the license terms.
> 3.  Wait for approval (usually instant; occasionally manual).
> 4.  Ensure the token you pass to your script (via `huggingface-cli login` or the `HF_TOKEN` environment variable)
>    belongs to the account that was approved.
>
> Trying to pull a gated model without an authorized token will trigger a 403 "permission denied" error.


#### 📚 About SQuAD
Stanford Question Answering Dataset (SQuAD) is a **reading comprehension dataset**, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

There are two major versions:

- **SQuAD v1.1**: All answers are guaranteed to be present in the context.

- **SQuAD v2.0**: Introduces unanswerable questions, adding complexity and realism.

In this tutorial, we’ll focus on **SQuAD v1.1**, which is more suitable for straightforward supervised fine-tuning without requiring additional handling of null answers.

Here’s a glimpse of what the data looks like:
``` json
{

    "id": "5733be284776f41900661182",
    "title": "University_of_Notre_Dame",
    "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend Venite Ad Me Omnes. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "answers": {
        "text": [
            "Saint Bernadette Soubirous"
        ],
        "answer_start": [
            515
        ]
    }
}
```
This structure is ideal for training models in context-based question answering, where the model learns to answer questions based on the input context.

> [!TIP]
> In this guide, we use the `SQuAD v1.1` dataset, but you can specify your own data as needed.


## Automodel CLI

The easiest way to run PEFT training is with the recipe files. You can
find the list of supported models and their predefined recipes
[here](https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/llm/recipes).

<!-- > [!NOTE]
> **Prerequisite**: Before proceeding, please follow the example in
> `nemo-2-quickstart-nemo-run`{.interpreted-text role="ref"} to
> familiarize yourself with NeMo-Run first. -->

``` python
from nemo.collections import llm
import nemo_run as run

nodes = 1
gpus_per_node = 1
recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
    model_name="meta-llama/Llama-3.2-1B",  # The Hugging Face model-id or path to a local checkpoint (HF-native format).
    dir="/checkpoints/llama3.2_1b", # Path to store checkpoints
    name="llama3_lora",
    num_nodes=nodes,
    num_gpus_per_node=gpus_per_node,
    peft_scheme="lora",
)
# Note: "lora" is the default peft_scheme.
# Supported values are "lora", "none"/None (full fine-tuning).

# Override your PEFT configuration here, if needed. Regexp-like format is also supported,
# to match all modules ending in `_proj` use `*_proj`. For example:
recipe.peft.target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "*_proj"]
recipe.peft.dim = 16
recipe.peft.alpha = 32

# Add other overrides here:
...

run.run(recipe)
```

### NeMo-Run CLI

You can use PEFT recipes via the NeMo-Run CLI (See [NeMo-Run\'s
docs](https://github.com/NVIDIA/NeMo-Run) for more details). LoRA are
registered as factory classes, so you can specify `peft=<lora/none>`
directly in the terminal. This provides a quick and easy way to launch
training jobs when you do not need to override any configuration from
the default recipes.

``` bash
nemo llm finetune -f hf_auto_model_for_causal_lm model_name="meta-llama/Llama-3.2-1B" peft=lora  # acceptable values are lora/none
```

### llm.finetune API

This example uses the [finetune
API](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/api.py)
from the NeMo Framework LLM collection. This is a lower-level API that
allows you to lay out the various configurations in a Pythonic fashion.
This gives you the greatest amount of control over each configuration.

``` yaml
step_scheduler:
  grad_acc_steps: 4
  ckpt_every_steps: 1000
  val_every_steps: 10  # will run every x number of gradient steps
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false

loss_fn: nemo_automodel.components.loss.masked_ce.masked_cross_entropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8
  shuffle: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
  limit_dataset_samples: 64

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

# Uncomment and configure for W&B logging
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
#   save_dir: <your_wandb_save_dir>
```

> [!TIP]
> To avoid using unnessary storage space and enable faster sharing, the
> adapter checkpoint only contains the adapter weights. As a result, when
> running inference, the adapter and base model weights need to match
> those used for training.


> [!TIP]
> In the above example, we used the FSDP2 strategy with 8 GPUs. Depending
> on the size of your model, you may need to adjust the number of GPUs or
> use a different strategy.


> [!NOTE]
> The FSDP2Strategy is introduced in NeMo\'s lightning strategy and
> requires an active NeMo installation. See the NeMo documentation for
> installation details.


## Run PEFT Inference with NeMo AutoModel-trained Adapters

Inference with adapters is supported using Hugging Face\'s generate API.
Simply replace the path to the full model with the path to a PEFT
checkpoint. The required configurations, including the model type,
adapter type, and base model checkpoint path, should be properly set up
within the PEFT checkpoint.

The following is an example script using Hugging Face\'s transformers
library:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
base_model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load PEFT adapter
adapter_path = "path/to/nemo-trained/peft/adapter"
model = PeftModel.from_pretrained(model, adapter_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate text
input_text = "Your input prompt here"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=100)

# Decode and print the output
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Publish PEFT Adapters to Hugging Face Hub

After fine-tuning a Hugging Face model using NeMo, the resulting PEFT
adapters are stored in Hugging Face-native format, making them easy to
share and deploy. To make these adapters publicly accessible, we can
upload them to the Hugging Face Model Hub, allowing seamless integration
with the Hugging Face ecosystem.

Using the Hugging Face Hub API, we can push the PEFT adapter to a
repository, ensuring that others can easily load and use it with
[peft.AutoPeftModel](https://huggingface.co/docs/peft/package_reference/auto_class#peft.AutoPeftModel).
The following steps outline how to publish the adapter:

1.  Install the Hugging Face Hub library (if not already installed):

``` bash
pip3 install huggingface_hub
```

2.  Log in to Hugging Face using your authentication token:

``` bash
huggingface-cli login
```

3.  Upload the PEFT adapter using the [huggingface_hub](https://github.com/huggingface/huggingface_hub)
    Python API:

``` python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="path/to/nemo-trained/peft/adapter",
    repo_id="your-username/peft-adapter-name",
    repo_type="model"
)
```

Once uploaded, the adapter can be loaded directly using:

``` python
from peft import PeftModel, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("base-model")
peft_model = PeftModel.from_pretrained(model, "your-username/peft-adapter-name")
```

By publishing the adapters to the Hugging Face Hub, we enable easy
sharing, reproducibility, and integration with downstream applications.

<!-- ## Evaluate with the LM Evaluation Harness

After fine-tuning the pretrained model on a domain-specific dataset
using NeMo AutoModel, the process generates Hugging Face-native PEFT
adapters. These adapters are fully compatible with the Hugging Face
ecosystem, allowing seamless integration with evaluation tools.

To assess the performance of the fine-tuned model, we will use the [LM
Evaluation
Harness](https://github.com/EleutherAI/lm-evaluation-harness), a
standardized framework for benchmarking language models. The PEFT
Hugging Face-native adapters can be directly loaded into the evaluation
pipeline without additional conversion.

In the following setup, we utilize the LM Evaluation Harness to evaluate
the fine-tuned model on HellaSwag, a benchmark designed to measure
commonsense reasoning capabilities.

``` bash
ckpt="path/to/meta-llama/Llama-3.2-1B"
adapter="path/to/nemo-trained/peft/adapter"
python3 -m lm_eval --model hf \
    --model_args pretrained=$ckpt,peft=$adapter \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

This command will run lm_eval on hellaswag using [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and the NeMo AutoModel-trained HF
adapters. -->

## Export to vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference
engine designed to optimize the deployment of large language models
(LLMs) for production use. By utilizing advanced techniques like
parallel processing and optimized memory management, vLLM accelerates
inference while maintaining model accuracy.

NeMo AutoModel provides support for exporting PEFT adapters for use with
vLLM, enabling optimized inference without requiring model architecture
changes. The [vLLMHFExporter](https://github.com/NVIDIA/NeMo/blob/main/nemo/export/vllm_hf_exporter.py) utility facilitates this
process, ensuring compatibility with Hugging Face-based models.

The following script demonstrates how to export a PEFT adapter for vLLM,
allowing seamless deployment and efficient inference.

> [!NOTE]
> Make sure vLLM is installed (pip install vllm, or use the environment
> that includes it) before proceeding with vLLMHFExporter.

``` python
from nemo.export.vllm_hf_exporter import vLLMHFExporter

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="Local path of the base model")
    parser.add_argument('--lora-model', required=True, type=str, help="Local path of the lora model")
    args = parser.parse_args()

    lora_model_name = "lora_model"

    exporter = vLLMHFExporter()
    exporter.export(model=args.model, enable_lora=True)
    exporter.add_lora_models(lora_model_name=lora_model_name, lora_model=args.lora_model)

    print("vLLM Output: ", exporter.forward(input_texts=["How are you doing?"], lora_model_name=lora_model_name))
```
