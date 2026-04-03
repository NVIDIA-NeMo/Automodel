# Fine-Tuning Gemma 4 31B on CORD-v2 Receipts — End-to-End Guide

**A step-by-step guide for fine-tuning Gemma 4 31B to extract structured receipt data
from scanned images using [NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel).**

---

## What is Gemma 4 31B?

Gemma 4 31B is a dense vision-language model with a 60-layer transformer decoder,
SigLIP vision encoder, and support for multimodal inputs (images, audio, text).

Key architectural details:
- Mixed attention: sliding window (512 tokens) + full attention (every 6th layer)
- 32 attention heads, 16 KV heads (GQA)
- Hidden dim 5376, vocab size 262,144
- bfloat16, final logit softcapping at 30.0
- Thinking-channel support (`<|channel>thought\n<channel|>` prefix)

## The Task

We fine-tune Gemma 4 31B on the **CORD-v2** (Consolidated Receipt Dataset) to extract
structured fields from scanned receipts:

| Field | Example |
|-------|---------|
| `menu` | Item names, quantities, unit prices, sub-totals |
| `sub_total` | Subtotal details (subtotal price, discount, tax, etc.) |
| `total` | Total price, cash price, change price, etc. |
| `void_menu` | Voided items (if any) |

The **base model** produces free-form descriptions. After fine-tuning, it outputs
**structured XML-like token sequences** matching the receipt fields.

## Guide Overview

| Step | Description |
|------|-------------|
| **Step 0** | Environment setup |
| **Step 1** | Explore the CORD-v2 dataset |
| **Step 2** | Evaluate the base model (before fine-tuning) |
| **Step 3** | Training configuration |
| **Step 4** | Launch fine-tuning |
| **Step 5** | Evaluate the fine-tuned model |
| **Step 6** | Compare results |

## Hardware Requirements

- **8x A100 80 GB** (or 8x H100) GPUs required for 31B with FSDP2 + activation checkpointing
- **Estimated training time**: ~45 min on 8x H100 (800 training samples, 500 steps)

---

## Step 0 — Environment Setup

This guide runs **inside** the NeMo Automodel Docker container:

```bash
docker run -it --rm --gpus all --ipc=host --network host \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/nemo-automodel:26.02

# Inside the container:
huggingface-cli login          # needed for gated model access
cd /opt/Automodel
```

> **Note**: Gemma 4 requires the transformers version that include the model implementation. Please make sure proper transformers is installed.

---

## Step 1 — Explore the CORD-v2 Dataset

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) is a Consolidated
Receipt Dataset for Post-OCR Parsing containing scanned receipts with structured
ground-truth JSON labels.

```python
import json
from datasets import load_dataset

dataset = load_dataset("naver-clova-ix/cord-v2")

print(f"Train      : {len(dataset['train'])} samples")
print(f"Validation : {len(dataset['validation'])} samples")
print(f"Test       : {len(dataset['test'])} samples")

# Inspect a sample
ex = dataset["train"][0]
gt = json.loads(ex["ground_truth"])["gt_parse"]
print(f"\nGround-truth keys: {list(gt.keys())}")

for key in gt:
    if isinstance(gt[key], list):
        print(f"\n  {key} ({len(gt[key])} items):")
        for item in gt[key][:2]:
            print(f"    {item}")
    else:
        print(f"\n  {key}: {gt[key]}")
```

Expected output:
```
Train      : 800 samples
Validation : 100 samples
Test       : 100 samples

Ground-truth keys: ['menu', 'sub_total', 'total', 'void_menu']

  menu (7 items):
    {'nm': 'ABRA KADABRA FLAME GRILLED', 'num': '1', 'unitprice': '39,000', 'cnt': '1', 'price': '39,000'}
    {'nm': 'Lemon Tea', 'num': '1', 'unitprice': '7,000', 'cnt': '1', 'price': '7,000'}

  sub_total: {'subtotal_price': '87,000', 'discount_price': '0', 'tax_price': '7,909'}

  total: {'total_price': '87,000', 'cashprice': '100,000', 'changeprice': '13,000'}

  void_menu: []
```

### Target format: JSON-to-token conversion

NeMo Automodel converts structured JSON into an XML-like **token sequence** using
the `json2token()` function. This is the format the model is trained to produce:

```python
from nemo_automodel.components.datasets.vlm.utils import json2token

token_seq = json2token(gt, sort_json_key=True)
print(f"Token sequence (first 300 chars):\n  {token_seq[:300]}...")
print(f"\nTotal length: {len(token_seq)} chars")
```

Expected output:
```
Token sequence (first 300 chars):
  <s_menu><s_cnt>1</s_cnt><s_nm>ABRA KADABRA FLAME GRILLED</s_nm><s_num>1</s_num>
  <s_price>39,000</s_price><s_unitprice>39,000</s_unitprice><sep/><s_cnt>1</s_cnt>
  <s_nm>Lemon Tea</s_nm><s_num>1</s_num><s_price>7,000</s_price><s_unitprice>7,000
  </s_unitprice><sep/>...

Total length: 827 chars
```

---

## Step 2 — Evaluate the Base Model (Before Fine-Tuning)

Load the pretrained Gemma 4 31B model and run it on receipt images. The base model
will produce free-form descriptions instead of structured token sequences.

```python
import os
import json
import torch
from transformers import AutoProcessor
from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.datasets.vlm.utils import json2token
from datasets import load_dataset

# --- Helpers ---

def compute_ned(pred: str, target: str) -> float:
    """Normalized Edit Distance (0 = perfect match, 1 = completely different)."""
    m, n = len(pred), len(target)
    if max(m, n) == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if pred[i - 1] == target[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = tmp
    return dp[n] / max(m, n)


def run_gemma4_inference(model, processor, pil_image, prompt="Describe this image.",
                         max_new_tokens=1024):
    """Run Gemma 4 inference on a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True))
    return generated_text[prompt_length:].strip()


def evaluate_receipts(model, processor, test_dataset, n_samples=20):
    """Evaluate model on receipt test set, return avg NED and per-sample results."""
    model.eval()
    results = []
    n = min(n_samples, len(test_dataset))
    for i in range(n):
        ex = test_dataset[i]
        gt = json.loads(ex["ground_truth"])["gt_parse"]
        target = json2token(gt, sort_json_key=True)
        pred = run_gemma4_inference(model, processor, ex["image"])
        ned = compute_ned(pred, target)
        results.append({"idx": i, "ned": ned, "pred": pred, "target": target, "gt": gt})
        print(f"  Sample {i:2d}: NED = {ned:.4f}")
    avg_ned = sum(r["ned"] for r in results) / len(results)
    print(f"\n  Average NED: {avg_ned:.4f}")
    return avg_ned, results

# --- Load base model ---

MODEL_PATH = "/path/to/gemma-4-31b-it"   # or HuggingFace model ID

processor = AutoProcessor.from_pretrained(MODEL_PATH)
base_model = NeMoAutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    use_liger_kernel=True,
    attn_implementation="flash_attention_2",
    text_config={"use_cache": False},
).eval().to("cuda")

print(f"Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

# --- Evaluate ---

dataset = load_dataset("naver-clova-ix/cord-v2")
print("\nEvaluating base model on receipt test set:")
base_avg_ned, base_results = evaluate_receipts(base_model, processor, dataset["test"])
```

Expected base model output (receipt image):
```
  Sample  0: NED = 0.8734
  Sample  1: NED = 0.9012
  ...
  Average NED: 0.8850
```

**Example base model prediction** (free-form, not structured):
```
The image shows a receipt from a restaurant. The total amount is 87,000 with items
including ABRA KADABRA FLAME GRILLED for 39,000 and Lemon Tea for 7,000...
```

> The base model produces readable descriptions but not the structured token format
> we need. Fine-tuning teaches it to output `<s_menu><s_nm>...</s_nm>...` sequences.

---

## Step 3 — Training Configuration
### YAML config

You can save the yaml below as `gemma4_31b_cord_v2.yaml` for training cord_v2 dataset. 

```yaml

step_scheduler:
  global_batch_size: 8
  local_batch_size: 1
  ckpt_every_steps: 100
  val_every_steps: 100
  max_steps: 500

dist_env:
  backend: nccl
  timeout_minutes: 60

model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: /path/to/gemma-4-31b-it
  torch_dtype: torch.bfloat16
  use_liger_kernel: true
  use_sdpa_patching: false
  attn_implementation: flash_attention_2
  text_config:
    use_cache: false

checkpoint:
  enabled: true
  checkpoint_dir: vlm_checkpoints/gemma4_31b_cord_v2/
  model_save_format: safetensors
  save_consolidated: true

distributed:
  strategy: fsdp2
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: train

dataloader:
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.gemma4_prefix_collate_fn

validation_dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: validation

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-5
  weight_decay: 0.01
  betas: [0.9, 0.95]

lr_scheduler:
  lr_decay_style: cosine

freeze_config:
  freeze_embeddings: true
  freeze_vision_tower: true
  freeze_audio_tower: true
  freeze_language_model: false
```

### Why `gemma4_prefix_collate_fn`?

Gemma 4 31B instruction-tuned models always emit a thinking-channel prefix
(`<|channel>thought\n<channel|>`) before the actual response. When this prefix
is absent from training sequences, the model predicts `<|channel>` but the label
says answer text, inflating initial loss to ~9. The `gemma4_prefix_collate_fn`
injects this prefix (masked as -100 in labels so the model is not penalized for it)
and brings initial loss down to ~3.

---

## Step 4 — Launch Fine-Tuning

```bash
torchrun --nproc-per-node=8 \
    examples/vlm_finetune/finetune.py \
    -c gemma4_31b_cord_v2.yaml \
    2>&1 | tee logs/train_gemma4_31b_cord_v2.log
```

### What to watch

- **Loss** drops rapidly from ~0.73 to ~0.04 in the first 50 steps, then stabilizes around 0.005
- **Validation loss** reaches ~0.018 by step 199 (best checkpoint)
- Training takes ~15 min on 8x H100 (300 steps, 800 training samples)

### Training log

```
step    0 | loss 0.7350 | grad_norm  35.65 | lr 1.18e-06 | mem 60.90 GiB | tps/gpu  45
step   10 | loss 0.5489 | grad_norm  26.19 | lr 2.98e-06 | mem 40.36 GiB | tps/gpu 425
step   20 | loss 0.1455 | grad_norm  10.53 | lr 4.78e-06 | mem 40.42 GiB | tps/gpu 438
step   50 | loss 0.0406 | grad_norm  27.16 | lr 1.00e-05 | mem 40.34 GiB | tps/gpu 377
step  100 | loss 0.0148 | grad_norm   7.02 | lr 9.70e-06 | mem 40.36 GiB | tps/gpu 449
step  200 | loss 0.0065 | grad_norm   2.28 | lr 7.52e-06 | mem 40.44 GiB | tps/gpu 441
step  300 | loss 0.0041 | grad_norm   2.10 | lr 3.16e-06 | mem 40.53 GiB | tps/gpu 448

Validation:
  step  99 | val_loss 0.0225
  step 199 | val_loss 0.0183  <-- LOWEST_VAL (best checkpoint)
  step 299 | val_loss 0.0192
```

### Checkpoints saved

```
vlm_checkpoints/gemma4_31b_cord_v2/
  epoch_0_step_99/
  epoch_0_step_199/
  epoch_0_step_299/
    model/
      consolidated/          <-- HF-compatible checkpoint for inference
        config.json
        model.safetensors.index.json
        model-00001-of-00013.safetensors
        ...
    optim/
    rng/
    dataloader/
  LATEST -> epoch_0_step_299
  LOWEST_VAL -> epoch_0_step_199
  training.jsonl             <-- per-step metrics
  validation.jsonl           <-- per-validation metrics
```

> **Tip**: `LOWEST_VAL` symlink points to the checkpoint with the best validation loss.
> Use this for inference evaluation.

---

## Step 5 — Evaluate the Fine-Tuned Model

### Load consolidated checkpoint with HF AutoModelForMultimodalLM

Because we set `save_consolidated: true` in the config, each checkpoint contains
an HF-compatible `model/consolidated/` directory. Use HF's `AutoModelForMultimodalLM`
for inference (generation), and load the processor from the **base model** path.

```python
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForMultimodalLM
from nemo_automodel.components.datasets.vlm.utils import json2token

# Paths
BASE_MODEL = "/path/to/gemma-4-31b-it"
CKPT_DIR = "vlm_checkpoints/gemma4_31b_cord_v2"
best_ckpt = os.path.realpath(os.path.join(CKPT_DIR, "LOWEST_VAL"))
consolidated = os.path.join(best_ckpt, "model", "consolidated")

# Load processor from base model, model from fine-tuned checkpoint
processor = AutoProcessor.from_pretrained(BASE_MODEL)
model = AutoModelForMultimodalLM.from_pretrained(
    consolidated,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

# Evaluate on test set
dataset = load_dataset("naver-clova-ix/cord-v2")
print("Evaluating fine-tuned model:")
ft_avg_ned, ft_results = evaluate_receipts(model, processor, dataset["test"])
```

### Fine-tuned output (test sample 1 -- perfect NED=0.0)

```
<s_total><s_total_price>91000</s_total_price><s_cashprice>91000</s_cashprice>
</s_total><s_menu><s_price>17500</s_price><s_nm>J.STB PROMO</s_nm><sep/>
<s_price>46000</s_price><s_nm>Y.B.BAT</s_nm><sep/><s_price>27500</s_price>
<s_nm>Y.BASO PROM</s_nm></s_menu>
```

### Parsing the structured output

You can convert the token sequence back to a structured dict:

```python
import re

def token2json(token_seq):
    """Convert a token sequence back to a JSON-like dict."""
    result = {}
    pattern = r"<s_(\w+)>(.*?)</s_\1>"
    matches = re.findall(pattern, token_seq, re.DOTALL)
    for key, value in matches:
        if "<sep/>" in value:
            items = value.split("<sep/>")
            result[key] = [token2json(item) if "<s_" in item else item for item in items]
        elif "<s_" in value:
            result[key] = token2json(value)
        else:
            result[key] = value
    return result

parsed = token2json(prediction)
print(json.dumps(parsed, indent=2))
```

Example parsed output (test sample 4):
```json
{
  "total": {"total_price": "174,600", "changeprice": "25,400", "cashprice": "200,000"},
  "sub_total": {"subtotal_price": "194,000", "discount_price": "19,400"},
  "menu": [
    {"price": "82,000", "nm": "ICE BLACKCOFFE"},
    {"price": "44,000", "nm": "C.Capuccino (L)"},
    {"price": "30,000", "nm": "C.WHITE COFFE"},
    {"price": "38,000", "nm": "C.Capuccino (L)"}
  ]
}
```

---

## Step 6 — Results Comparison

### Metrics (20 test samples)

| Metric | Fine-Tuned (epoch_1_step_199) |
|--------|-------------------------------|
| **Average NED** | **0.0601** |
| **Field-Level Accuracy** | **92.6%** |
| Perfect matches (NED=0.0) | 10/20 (50%) |
| Near-perfect (NED<0.05) | 14/20 (70%) |

### Field-level extraction accuracy (actual)

```
Field                 Correct / Total  Accuracy
--------------------------------------------------
total_price                18 /    19     94.7%
subtotal_price             13 /    14     92.9%
tax_price                   7 /     8     87.5%
cashprice                  13 /    15     86.7%
changeprice                12 /    12    100.0%
OVERALL                    63 /    68     92.6%
```
