# Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) with NeMo AutoModel

Fine-tune Hugging Face models using NeMo AutoModel with full-parameter SFT or lightweight LoRA adapters. No checkpoint conversion required.

In this guide you will:
- Fine-tune `meta-llama/Llama-3.2-1B` on the SQuAD dataset
- Configure a recipe via YAML (model, dataset, optimizer, distributed)
- Run training with the AutoModel CLI or `torchrun`
- Run inference, publish to the Hugging Face Hub, and export to vLLM

In addition to this guide, you can also explore our Quickstart,
which features a [standalone python3 recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/finetune.py),
offering hands-on demonstrations for quickly getting started with NeMo AutoModel. 

## Run SFT and PEFT with NeMo AutoModel

:::{important}
Before proceeding with this guide, please ensure that you have NeMo AutoModel installed on your
machine. This can be achieved by running:
```bash
pip3 install nemo-automodel
```
For a complete guide and additional options please consult the AutoModel [installation guide](../installation.md).
:::

### Model and Dataset Context
In this guide, we will fine-tune Meta's `LLaMA 3.2 1B` model on the popular [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset).

:::{tip}
`meta-llama/Llama-3.2-1B` is used only as a placeholder model ID. You can replace it with any valid Hugging Face model ID, such
as `Qwen/Qwen2.5-1.5B`, or any other checkpoint you have access to on
the Hugging Face Hub that is supported as per [model coverage](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/llm.md) list.
:::

::::{dropdown} About LLaMA 3.2 1B
**LLaMA** is a family of decoder-only transformer models developed by Meta. The **LLaMA 3.2 1B** variant is a compact, lightweight model ideal for research and edge deployment. Key architectural features:

- **Decoder-only architecture**: GPT-style, autoregressive design optimized for generation.
- **Rotary positional embeddings (RoPE)**: Efficient and extendable positional encoding.
- **Grouped-query attention (GQA)**: Decouples key/value heads from query heads for scalability.
- **SwiGLU activation**: Improved convergence and expressiveness.
- **Multi-layer residual connections**: Better training stability and depth scaling.
::::

#### Access Gated Models

Some Hugging Face model repositories are **gated**. To download their files:

1.  Log in with your Hugging Face account.
2.  Click "Request access" or "Agree and access" and accept the license terms.
3.  Wait for approval (usually instant; occasionally manual).
4.  Ensure the token you pass to your script (via `huggingface-cli login` or the `HF_TOKEN` environment variable) belongs to the approved account.

:::{note}
Trying to pull a gated model without an authorized token will trigger a 403 "permission denied" error.
:::

::::{dropdown} About SQuAD
The Stanford Question Answering Dataset (SQuAD) is a **reading comprehension dataset** where the answer is a text span from a Wikipedia passage. We use **SQuAD v1.1** (all questions answerable).

Sample entry:
```json
{
    "id": "5733be284776f41900661182",
    "title": "University_of_Notre_Dame",
    "context": "Architecturally, the school has a Catholic character. ...",
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "answers": { "text": ["Saint Bernadette Soubirous"], "answer_start": [515] }
}
```
::::

:::{tip}
You can use your own dataset instead of SQuAD. Edit the YAML `dataset` / `validation_dataset` sections (for example `dataset._target_`, `dataset_name`/`path_or_dataset`, and `split`). See [Integrate Your Own Text Dataset](dataset.md) and [Dataset Overview](../dataset-overview.md).
:::

## Configure the Recipe

A recipe in NeMo AutoModel is a **self-contained orchestration module** that wires together model loading, dataset preparation, optimizer setup, distributed training, checkpointing, and logging. The `TrainFinetuneRecipeForNextTokenPrediction` class used here inherits from `BaseRecipe` and is configured entirely through YAML.

:::{note}
The recipe is stateless and config-driven: components are instantiated dynamically using Hydra-style `instantiate()` calls, avoiding hardcoded dependencies.
:::

### Model and PEFT

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  is_meta_device: false

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"   # match all linear layers with ".proj" in their FQN
  dim: 8                     # low-rank dimension
  alpha: 32                  # scales the learned weights
  use_triton: True           # optimized LoRA kernel
```

### Dataset

```yaml
# Training split
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

# Validation split (limited to 64 samples)
validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
  limit_dataset_samples: 64
```

### Training Schedule and Distributed

```yaml
step_scheduler:
  grad_acc_steps: 4
  ckpt_every_steps: 10
  val_every_steps: 10
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
```

### Loss, Dataloader, Checkpoint, Optimizer

```yaml
loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_consolidated: True

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

# Optional: wandb logging
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
#   save_dir: <your_wandb_save_dir>
```

:::{tip}
The PEFT adapter checkpoint only contains adapter weights. When running inference, the adapter and base model weights must match those used for training.
:::

## QLoRA: Quantized Low-Rank Adaptation

[QLoRA](https://arxiv.org/abs/2305.14314) combines LoRA with **4-bit NormalFloat (NF4)** quantization, reducing memory by up to 75% while maintaining model quality comparable to full-precision LoRA.

- **Memory Efficiency**: Up to 75% reduction vs. full-precision fine-tuning
- **Hardware Accessibility**: Fine-tune large models on consumer-grade GPUs
- **Performance Preservation**: Quality comparable to full-precision LoRA

### QLoRA Configuration

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.1-8B

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true
  dim: 16
  alpha: 32
  dropout: 0.1

quantization:
  load_in_4bit: True
  load_in_8bit: False
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: True
  bnb_4bit_quant_type: nf4
  bnb_4bit_quant_storage: bfloat16
```

## Loading Large Models

For models larger than single-GPU memory (e.g., 70B = 140 GB in BF16), set `is_meta_device: true` in the model config. This uses [PyTorch's Meta device](https://docs.pytorch.org/docs/stable/meta.html) to instantiate the model without loading data, then populates weights only for each GPU's shard after FSDP sharding.

## Run the Fine-Tune Recipe

Save your YAML as `sft_guide.yaml` (or `peft_guide.yaml` for PEFT), then run:

### AutoModel CLI

```bash
automodel finetune llm -c sft_guide.yaml
```

Where `finetune` maps to the LLM training recipe and `llm` is the model domain. You can also use PEFT recipes via the NeMo Run CLI (See [NeMo-Run documentation](https://github.com/NVIDIA/NeMo-Run) for more details), specifying `peft=<lora/none>` directly in the terminal.

### Recipe Script with Torchrun

```bash
torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config sft_guide.yaml
```

### Sample Output

```
$ automodel finetune llm -c sft_guide.yaml
INFO:root:Domain:  llm
INFO:root:Command: finetune
INFO:root:Config:  /mnt/4tb/auto/Automodel/sft_guide.yaml
INFO:root:Running job using source from: /mnt/4tb/auto/Automodel
INFO:root:Launching job locally on 2 devices
cfg-path: /mnt/4tb/auto/Automodel/sft_guide.yaml
INFO:root:step 4 | epoch 0 | loss 1.5514 | grad_norm 102.0000 | mem: 11.66 GiB | tps 6924.50
INFO:root:step 8 | epoch 0 | loss 0.7913 | grad_norm 46.2500 | mem: 14.58 GiB | tps 9328.79
Saving checkpoint to checkpoints/epoch_0_step_10
INFO:root:step 12 | epoch 0 | loss 0.4358 | grad_norm 23.8750 | mem: 15.48 GiB | tps 9068.99
INFO:root:step 16 | epoch 0 | loss 0.2057 | grad_norm 12.9375 | mem: 16.47 GiB | tps 9148.28
INFO:root:step 20 | epoch 0 | loss 0.2557 | grad_norm 13.4375 | mem: 12.35 GiB | tps 9196.97
Saving checkpoint to checkpoints/epoch_0_step_20
INFO:root:[val] step 20 | epoch 0 | loss 0.2469
```

### Checkpoint Structure

**SFT checkpoint:**
```bash
$ tree checkpoints/epoch_0_step_10/
checkpoints/epoch_0_step_10/
├── config.yaml
├── dataloader.pt
├── model
│   ├── consolidated
│   │   ├── config.json
│   │   ├── model-00001-of-00001.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── generation_config.json
│   ├── shard-00001-model-00001-of-00001.safetensors
│   └── shard-00002-model-00001-of-00001.safetensors
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt

4 directories, 11 files
```

**PEFT checkpoint:**
```bash
$ tree checkpoints/epoch_0_step_10/
checkpoints/epoch_0_step_10/
├── dataloader.pt
├── config.yaml
├── model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── automodel_peft_config.json
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt

2 directories, 8 files
```

## Run Inference

Load the fine-tuned checkpoint or PEFT adapter using the Hugging Face generate API:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel #for PEFT

#For SFT, Load finetuned checkpoint
finetuned_ckpt_path = "checkpoints/epoch_0_step_10/model/consolidated"
tokenizer = AutoTokenizer.from_pretrained(finetuned_ckpt_path)
model = AutoModelForCausalLM.from_pretrained(finetuned_ckpt_path)

# For PEFT, Load base model, tokenizer and PEFT adapter
base_model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
adapter_path = "checkpoints/epoch_0_step_10/model/"
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

## Publish to the Hugging Face Hub

Checkpoints and PEFT adapters are stored in Hugging Face-native format, ready to upload:

1.  Install and log in:
```bash
pip3 install huggingface_hub
huggingface-cli login
```

2.  Upload:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="checkpoints/epoch_0_step_10/model/consolidated",
    repo_id="your-username/llama3.2_1b-finetuned-name" or "your-username/peft-adapter-name",
    repo_type="model"
)
```

3.  Load from the Hub:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-username/llama3.2_1b-finetuned-name")
```

For PEFT adapters: 
```python
from peft import PeftModel, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("base-model")
peft_model = PeftModel.from_pretrained(model, "your-username/peft-adapter-name")
```
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
checkpoint or PEFT adapters can be directly loaded into the evaluation pipeline without
additional conversion.

In the following setup, we utilize the LM Evaluation Harness to evaluate
the fine-tuned model on HellaSwag, a benchmark designed to measure
commonsense reasoning capabilities.

```bash
ckpt="/ft_checkpoints/llama3.2_1b"
python3 -m lm_eval --model hf \
    --model_args pretrained=$ckpt \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

This command will run lm_eval on hellaswag using the NeMo
AutoModel fine-tuned checkpoint of [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).

Before running this command, make sure you have specified the checkpoint
path that you used during fine-tuning, we will use
[/ft_checkpoints/llama3.2_1b]{.title-ref} as in the fine-tuning section. -->

## Export to vLLM

:::{note}
Make sure vLLM is installed (`pip install vllm`).
:::

**SFT checkpoint:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="checkpoints/epoch_0_step_10/model/consolidated/", model_impl="transformers")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("Toronto is a city in Canada.", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```
```text
>>> Generated text:  It is the capital of Ontario. Toronto is a global hub for cultural tourism. The City of Toronto
```

**PEFT adapter:**
```python
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
