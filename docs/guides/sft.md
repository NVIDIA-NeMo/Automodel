# Supervised Fine-Tuning (SFT) with NeMo AutoModel

## Introduction

As large language models (LLMs) become more powerful, adapting them to
specific tasks through fine-tuning has become essential for achieving
high accuracy and relevance. Supervised Fine-Tuning (SFT) enables a
pre-trained model to specialize in a given domain by training it on
labeled data, refining its responses while preserving the broad
knowledge acquired during pretraining.

Unlike Parameter-Efficient Fine-Tuning (PEFT), which optimizes a small
subset of parameters, SFT updates a larger portion, or even all, of
the model's weights. While this requires more computational resources,
it allows for deeper adaptation, making it particularly useful for
complex or high-precision applications.

NeMo AutoModel simplifies the fine-tuning process by offering seamless
integration with Hugging Face Transformers. It allows you to fine-tune
models without converting checkpoints, ensuring full compatibility with
the Hugging Face ecosystem.

This guide walks you through the end-to-end process of fine-tuning
models from the Hugging Face Hub using NeMo AutoModel. You'll learn how
to prepare datasets, train models, generate text with fine-tuned
checkpoints, evaluate performance using the LM Eval Harness, share your
models on the Hugging Face Model Hub, and deploy them efficiently with
vLLM.

<!-- In addition to this user guide, you can also explore our Quickstart,
which features a [standalone python3
recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/finetune.py),
offering hands-on demonstrations for quickly getting started with NeMo AutoModel. -->

## Run SFT with NeMo AutoModel

In this guide, we will run supervised fine-tuning (SFT) on Meta’s `LLaMA 3.2 1B` model with
the popular [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset).

> [!IMPORTANT]
> Before proceeding with this guide, please ensure that you have NeMo Automodel installed on your
> machine. This can be achieved by running:
> ```bash
> pip3 install nemo-automodel
> ```
> For a complete guide and additional options please consult the Automodel [installation guide](installation.md).

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


### Finetune recipe and configuration

This example demonstrates how to fine-tune a large language model using NVIDIA's NeMo Automodel library.
Specifically, we use the LLM [finetune recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/finetune.py),and in particular the `FinetuneRecipeForNextTokenPrediction` class to orchestrate the fine-tuning process end-to-end: model loading, dataset preparation, optimizer setup, distributed training, checkpointing, and logging.

#### 🧠 What is a Recipe?

A recipe in NeMo Automodel is a **self-contained orchestration module** that wires together all
components needed to perform a specific task (e.g., fine-tuning for next-token prediction or instruction tuning).
Think of it as the equivalent of a Trainer class, but highly modular, stateful, and reproducible.

The `FinetuneRecipeForNextTokenPrediction` class is one such recipe. It inherits from `BaseRecipe` and implements:

- `setup()`: builds all training components from the config

- `run_train_validation_loop()`: executes training + validation steps

- Misc: Checkpoint handling, logging, and RNG setup.

> [!NOTE]
> Key Insight: The recipe ensures stateless config-driven orchestration, meaning no component is hardcoded: key items, such as the model, dataset and optimizer are loaded via Hydra-style `instantiate()` calls.

#### Recipe Config
``` yaml
# The model section is responsible for configuring the model we want to finetune.
# Since we want to use the Llama 3 1B model, we pass `meta-llama/Llama-3.2-1B` to the
# `pretrained_model_name_or_path` option.
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# As mentioned earlier, we are using the SQuAD dataset. NeMo Automodel provides the make_squad_dataset
# function which formats the prepares the dataset (e.g., formatting). We are using the "train"
# split for training.
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

# Similarly, for validation we use the "validation" split, and limit the number of samples to 64.
validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
  limit_dataset_samples: 64

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

# For distributed processing, we will FSDP2.
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false

loss_fn: nemo_automodel.components.loss.masked_ce.masked_cross_entropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8

# We will use the standard Adam optimizer, but you can specify any optimizer you want, by changing
# the import path using the _target_ option.
optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

# If you want to log your experiment on wandb, uncomment and configure the following section
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


## Run the finetune recipe
Assuming the above `yaml` is saved in a file named `peft_guide.yaml`, we can run the finetune workflow
with either the automodel CLI or by invoking the recipe python script directrly.

### Automodel CLI

When NeMo Automodel is installed on your system, it includes the `automodel` CLI program that you
can use to run jobs, locally or on distributed environments.

<!-- You can use PEFT recipes via the NeMo-Run CLI (See [NeMo-Run\'s
docs](https://github.com/NVIDIA/NeMo-Run) for more details). LoRA are
registered as factory classes, so you can specify `peft=<lora/none>`
directly in the terminal. This provides a quick and easy way to launch
training jobs when you do not need to override any configuration from
the default recipes. -->

``` bash
automodel finetune llm -c peft_guide.yaml
```

### Invoking the recipe script directly

Alternatively, you can run the recipe [script](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/finetune.py) directly using [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html), as shown bellow.

``` bash
torchrun --nproc-per-node=8 examples/llm/finetune.py --config peft_guide.yaml
```


## Run Inference with the NeMo AutoModel Fine-Tuned Checkpoint

Inference on the fine-tuned checkpoint is supported using Hugging Face's generate API.
Simply replace the path of the full model with the path to a SFT checkpoint.

The following is an example script using Hugging Face's transformers library:

``` python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load finetuned checkpoint
finetuned_ckpt_path = "/ft_checkpoints/llama3.2_1b"
tokenizer = AutoTokenizer.from_pretrained(finetuned_ckpt_path)
model = AutoModelForCausalLM.from_pretrained(finetuned_ckpt_path)

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

## Publish the SFT Checkpoint to the Hugging Face Hub

After fine-tuning a Hugging Face model using NeMo AutoModel, the
resulting checkpoint is stored in a Hugging Face-native format, making
it easy to share and deploy. To make these checkpoints publicly
accessible, we can upload them to the Hugging Face Model Hub, allowing
seamless integration with the Hugging Face ecosystem.

Using the Hugging Face Hub API, we can push the fine-tuned checkpoint to
a repository, ensuring that others can easily load and use it with
transformer's [AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM).
The following steps outline how to publish the fine-tuned checkpoint:

1.  Install the Hugging Face Hub library (if not alredy installed):

``` bash
pip3 install huggingface_hub
```

2.  Log in to Hugging Face using your authentication token:

``` bash
huggingface-cli login
```

3.  Upload the fine-tuned checkpoint using the
    [huggingface_hub](https://github.com/huggingface/huggingface_hub) Python API:

``` python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/ft_checkpoints/llama3.2_1b",
    repo_id="your-username/llama3.2_1b-finetuned-name",
    repo_type="model"
)
```

Once uploaded, the fine-tuned checkpoint can be loaded directly using:

``` python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-username/llama3.2_1b-finetuned-name")
```

By publishing the fine-tuned checkpoint to the Hugging Face Hub, we
enable easy sharing, reproducibility, and integration with downstream
applications.
<!-- 
## Evaluate with the LM Evaluation Harness

After fine-tuning the pretrained model on a domain-specific dataset
using NeMo AutoModel, the process generates one (or more) Hugging
Face-native checkpoint(s). The checkpoint(s) is/are fully compatible
with the Hugging Face ecosystem, allowing seamless integration with
evaluation tools.

To assess the performance of the fine-tuned model, we will use the [LM
Evaluation
Harness](https://github.com/EleutherAI/lm-evaluation-harness), a
standardized framework for benchmarking language models. The finetuned
checkpoint can be directly loaded into the evaluation pipeline without
additional conversion.

In the following setup, we utilize the LM Evaluation Harness to evaluate
the fine-tuned model on HellaSwag, a benchmark designed to measure
commonsense reasoning capabilities.

``` bash
ckpt="/ft_checkpoints/llama3.2_1b"
python3 -m lm_eval --model hf \
    --model_args pretrained=$ckpt \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

This command will run lm_eval on hellaswag using NeMo
AutoModel-finetuned checkpoint of [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).

Before running this command, make sure you have specified the checkpoint
path that you used during fine-tuning, we will use
[/ft_checkpoints/llama3.2_1b]{.title-ref} as in the fine-tuning section. -->

## Export to vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference
engine designed to optimize the deployment of large language models
(LLMs) for production use. By utilizing advanced techniques like
parallel processing and optimized memory management, vLLM accelerates
inference while maintaining model accuracy.

NeMo AutoModel provides support for exporting a fine-tuned checkpoint
for use with vLLM, enabling optimized inference without requiring model
architecture changes. The [vLLMHFExporter](https://github.com/NVIDIA/NeMo/blob/main/nemo/export/vllm_hf_exporter.py)  utility
facilitates this process, ensuring compatibility with Hugging Face-based
models.

The following script demonstrates how to export a fine-tuned checkpoint
to vLLM and deploy it using PyTriton, allowing seamless deployment and
efficient inference:

> [!NOTE]
> Make sure vLLM is installed (pip install vllm, or use the environment
> that includes it) before proceeding with vLLMHFExporter.


``` python
from nemo.deploy import DeployPyTriton
from nemo.deploy.nlp import NemoQueryLLM

try:
    from nemo.export.vllm_hf_exporter import vLLMHFExporter
except Exception:
    raise Exception(
        "vLLM should be installed in the environment or import "
        "the vLLM environment in the NeMo FW container using "
        "source /opt/venv/bin/activate command"
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="Local path or model name on Hugging Face")
    parser.add_argument('--triton-model-name', required=True, type=str, help="Name for the service")
    args = parser.parse_args()

    exporter = vLLMHFExporter()
    exporter.export(model=args.model)

    nm = DeployPyTriton(
        model=exporter,
        triton_model_name=args.triton_model_name,
        triton_model_version=1,
        max_batch_size=64,
        http_port=8000,
        address="0.0.0.0",
    )

    nm.deploy()
    nm.run()

    nq = NemoQueryLLM(url="localhost:8000", model_name=args.triton_model_name)
    output_deployed = nq.query_llm(
        prompts=["How are you doing?"],
        max_output_len=128,
        top_k=1,
        top_p=0.2,
        temperature=1.0,
    )

    print(" Output: ", output_deployed)
    nm.stop()
```
