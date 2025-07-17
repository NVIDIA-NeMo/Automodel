# Supervised Fine-Tuning (SFT) with NeMo AutoModel

## Introduction

As large language models (LLMs) become more powerful, adapting them to
specific tasks through fine-tuning has become essential for achieving
high accuracy and relevance. Supervised Fine-Tuning (SFT) enables a
pre-trained model to specialize in a given domain by training it on
labeled data, refining its responses while preserving the broad
knowledge acquired during pretraining.

Unlike Parameter-Efficient Fine-Tuning (PEFT), which optimizes a small
subset of parameters, SFT updates a larger portion---or even all---of
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

In addition to this user guide, you can also explore our Quickstart,
which features a [standalone python3
script](https://github.com/NVIDIA/NeMo/blob/main/examples/llm/sft/automodel.py)
and a guided [Jupyter
Notebook](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/automodel/sft.ipynb),
offering hands-on demonstrations for quickly getting started with NeMo
AutoModel.

## Run SFT with NeMo AutoModel

Below are three examples of running a simple SFT training loop for the
Llama 3.2 1B model using NeMo AutoModel. These examples showcase
different levels of abstraction provided by the NeMo Framework. Once you
have set up your environment following the instructions in
`install-nemo-framework`{.interpreted-text role="ref"}, you are ready to
run the simple SFT tuning script.

::: hint
::: title
Hint
:::

In this guide, "meta-llama/Llama-3.2-1B" is used only as a placeholder
model ID. You can replace it with any valid Hugging Face model ID, such
as "Qwen/Qwen2.5-1.5B", or any other model available on the Hugging Face
Hub.
:::

::: tip
::: title
Tip
:::

Some Hugging Face model repositories are **gated**---you must explicitly
request permission before you can download their files. If the model
page shows a "Request access" or "Agree and access" button:

1.  Log in with your Hugging Face account.
2.  Click the button and accept the license terms.
3.  Wait for approval (usually instant; occasionally manual).
4.  Ensure the token you pass to your script (via [huggingface-cli
    login]{.title-ref} or the \$\$HF_TOKEN\$\$ environment variable)
    belongs to the account that was approved.

Trying to pull a gated model without an authorized token will trigger a
403 "permission denied" error.
:::

::: tab-set
::: tab-item
NeMo-Run Recipes

The easiest way to run SFT training is with the recipe files. You can
find the list of supported models and their predefined recipes available
on NeMo\'s [GitHub
repository](https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/llm/recipes).

::: note
::: title
Note
:::

**Prerequisite**: Before proceeding, please follow the example in
`nemo-2-quickstart-nemo-run`{.interpreted-text role="ref"} to
familiarize yourself with NeMo-Run first.
:::

``` python
from nemo.collections import llm
import nemo_run as run

nodes = 1
gpus_per_node = 1
# Note: we use peft=None to enable full fine-tuning.
recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
    model_name="meta-llama/Llama-3.2-1B",  # The Hugging Face model-id or path to a local checkpoint (HF-native format).
    dir="/ft_checkpoints/llama3.2_1b", # Path to store checkpoints
    name="llama3_sft",
    num_nodes=nodes,
    num_gpus_per_node=gpus_per_node,
    peft_scheme=None, # Setting peft_scheme=None disables parameter-efficient tuning and triggers full fine-tuning.


)

# Add other overrides here
...

run.run(recipe)
```
:::

::: tab-item
NeMo-Run CLI

You can use SFT recipes via the NeMo-Run CLI (See [NeMo-Run\'s
docs](https://github.com/NVIDIA/NeMo-Run) for more details). This
provides a quick and easy way to launch training jobs when you do not
need to override any configuration from the default recipes.

``` bash
nemo llm finetune -f hf_auto_model_for_causal_lm model_name="meta-llama/Llama-3.2-1B" peft=none
```
:::

::: tab-item
llm.finetune API

This example uses the [finetune
API](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/api.py)
from the NeMo Framework LLM collection. This is a lower-level API that
allows you to lay out the various configurations in a Pythonic fashion.
This gives you the greatest amount of control over each configuration.

``` python
import fiddle as fdl
import lightning.pytorch as pl
from nemo import lightning as nl
from nemo.collections import llm
from lightning.pytorch.loggers import WandbLogger

def make_squad_hf_dataset(tokenizer, batch_size):
    def formatting_prompts_func(example):
        formatted_text = [
            f"Context: {example['context']} Question: {example['question']} Answer:",
            f" {example['answers']['text'][0].strip()}",
        ]
        context_ids, answer_ids = list(map(tokenizer.text_to_ids, formatted_text))
        if len(context_ids) > 0 and context_ids[0] != tokenizer.bos_id and tokenizer.bos_id is not None:
            context_ids.insert(0, tokenizer.bos_id)
        if len(answer_ids) > 0 and answer_ids[-1] != tokenizer.eos_id and tokenizer.eos_id is not None:
            answer_ids.append(tokenizer.eos_id)

        return dict(
            labels=(context_ids + answer_ids)[1:],
            input_ids=(context_ids + answer_ids)[:-1],
            loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
        )

    datamodule = llm.HFDatasetDataModule(
        "rajpurkar/squad", split="train",
        micro_batch_size=batch_size, pad_token_id=tokenizer.eos_id or 0
    )
    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )
    return datamodule


model = llm.HFAutoModelForCausalLM(model_name="meta-llama/Llama-3.2-1B")
strategy = nl.FSDP2Strategy(
    data_parallel_size=8,
    tensor_parallel_size=1,
    checkpoint_io=model.make_checkpoint_io(adapter_only=True),
)

wandb = WandbLogger(
    project="nemo-automodel-sft",
    name="user-guide-llama",
)

llm.api.finetune(
    model=model,
    data=make_squad_hf_dataset(model.tokenizer, batch_size=1),
    trainer=nl.Trainer(
        devices=8,
        num_nodes=1,
        max_steps=100,
        accelerator='gpu',
        strategy=strategy,
        log_every_n_steps=1,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=10,
        gradient_clip_val=1.0,
        use_distributed_sampler=False,
        logger=wandb,
        precision="bf16-mixed",
    ),
    optim=fdl.build(llm.adam.te_adam_with_flat_lr(lr=1e-5)),
    peft=None, # Setting peft=None disables parameter-efficient tuning and triggers full fine-tuning.
)
```
:::
:::

::: hint
::: title
Hint
:::

In the above example, we used the FSDP2 strategy with 8 GPUs. Depending
on the size of your model, you may need to adjust the number of GPUs or
use a different strategy.
:::

::: note
::: title
Note
:::

The FSDP2Strategy is introduced in NeMo\'s lightning strategy and
requires an active NeMo installation. See the NeMo documentation for
installation details.
:::

## Run Inference with the NeMo AutoModel Fine-Tuned Checkpoint

Inference on the fine-tuned checkpoint is supported using Hugging
Face\'s generate API. Simply replace the path of the full model with the
path to a SFT checkpoint.

The following is an example script using Hugging Face\'s transformers
library:

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
transformer\'s
[AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM).
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
    [huggingface_hub]{.title-ref} Python API:

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
AutoModel-finetuned checkpoint of [meta-llama/Llama-3.2-1B]{.title-ref}.

Before running this command, make sure you have specified the checkpoint
path that you used during fine-tuning, we will use
[/ft_checkpoints/llama3.2_1b]{.title-ref} as in the fine-tuning section.

## Export to vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference
engine designed to optimize the deployment of large language models
(LLMs) for production use. By utilizing advanced techniques like
parallel processing and optimized memory management, vLLM accelerates
inference while maintaining model accuracy.

NeMo AutoModel provides support for exporting a fine-tuned checkpoint
for use with vLLM, enabling optimized inference without requiring model
architecture changes. The [vLLMHFExporter]{.title-ref} utility
facilitates this process, ensuring compatibility with Hugging Face-based
models.

The following script demonstrates how to export a fine-tuned checkpoint
to vLLM and deploy it using PyTriton, allowing seamless deployment and
efficient inference:

::: note
::: title
Note
:::

Make sure vLLM is installed (pip install vllm, or use the environment
that includes it) before proceeding with vLLMHFExporter.
:::

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
