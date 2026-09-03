# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Greedy-decode CORD-v2 receipts with a (fine-tuned) Nemotron 3.5 Super VL checkpoint and score them.

Mirrors the inference step of the NemotronOmni CORD-v2 guide, sized for the 67B model:
the consolidated HF checkpoint is spread over all visible GPUs with ``device_map="auto"``.

Metrics per sample: exact match against ``json2token(gt_parse, sort_json_key=True)`` and
a normalized sequence similarity (difflib ratio, 1.0 = identical). Aggregates (exact-match
rate, mean similarity) are printed and, optionally, logged to Weights & Biases together
with a per-sample table.

Example (one node, 8x H100):
    python examples/vlm_finetune/nemotron_3_5_super_vl/eval_cord_v2.py \
        --checkpoint vlm_checkpoints/nemotron_3_5_super_vl_cord_v2/LATEST/model/consolidated \
        --num-samples 20 --wandb-project huiyingl_workspace --wandb-entity Nemo-automodel
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

from nemo_automodel.components.datasets.vlm.utils import json2token

# v3 processors return placeholder-expansion metadata that is NOT a generate() kwarg.
PROCESSOR_METADATA_KEYS = ("num_patches", "num_tokens", "imgs_sizes")
PROMPT = "<image>\nDescribe this image."  # identical to the training prompt in make_cord_v2_dataset


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Consolidated HF checkpoint dir (or base model id/path).")
    parser.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", default=None, help="Optional JSON file for per-sample predictions.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def build_prompt(tokenizer) -> str:
    """Render the single-turn user prompt with the checkpoint's chat template (thinking disabled)."""
    messages = [{"role": "user", "content": PROMPT}]
    try:
        # enable_thinking=False -> the template emits an empty <think></think> block, which is
        # exactly the assistant prefix the collate function trained on.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(checkpoint: str):
    """Load processor and bf16 model, spreading the weights over all visible GPUs."""
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # RADIO's `summary_idxs` is a non-persistent buffer; it can come back as a meta tensor.
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None and hasattr(vision_model, "radio_model"):
        vision_model.radio_model.summary_idxs = None
    model.eval()
    print(f"loaded {checkpoint} in {time.perf_counter() - t0:.0f}s; device map: {getattr(model, 'hf_device_map', {})}")
    return model, processor


@torch.no_grad()
def predict(model, processor, prompt: str, image, max_new_tokens: int) -> str:
    """Greedy-decode one receipt image and return the generated text (special tokens stripped)."""
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    for key in PROCESSOR_METADATA_KEYS:
        inputs.pop(key, None)
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    """Decode ``--num-samples`` receipts, print/score them, and optionally log to W&B."""
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in redirected logs as it happens
    model, processor = load_model(args.checkpoint)
    prompt = build_prompt(processor.tokenizer)
    dataset = load_dataset(args.dataset, split=args.split)
    n = min(args.num_samples, len(dataset))

    rows = []
    for i in range(n):
        sample = dataset[i]
        image = sample["image"].convert("RGB")
        gt_text = json2token(json.loads(sample["ground_truth"])["gt_parse"], sort_json_key=True)
        t0 = time.perf_counter()
        pred = predict(model, processor, prompt, image, args.max_new_tokens)
        seconds = time.perf_counter() - t0
        exact = pred == gt_text
        similarity = difflib.SequenceMatcher(None, pred, gt_text).ratio()
        rows.append(
            {
                "index": i,
                "exact_match": exact,
                "similarity": similarity,
                "seconds": seconds,
                "ground_truth": gt_text,
                "prediction": pred,
            }
        )
        print(f"\n=== Sample {i} ({seconds:.1f}s) exact={exact} similarity={similarity:.3f}")
        print(f"Ground truth: {gt_text}")
        print(f"Prediction:   {pred}")

    exact_rate = sum(r["exact_match"] for r in rows) / n
    mean_sim = sum(r["similarity"] for r in rows) / n
    print(
        f"\n{args.split}[:{n}] exact-match {sum(r['exact_match'] for r in rows)}/{n} = {exact_rate:.3f} | mean similarity {mean_sim:.4f}"
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "exact_match_rate": exact_rate,
                    "mean_similarity": mean_sim,
                    "samples": rows,
                },
                indent=2,
            )
        )

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            job_type="eval",
            config=vars(args),
        )
        table = wandb.Table(columns=["index", "exact_match", "similarity", "seconds", "ground_truth", "prediction"])
        for r in rows:
            table.add_data(*(r[c] for c in table.columns))
        run.log(
            {
                "cord_v2/exact_match_rate": exact_rate,
                "cord_v2/mean_similarity": mean_sim,
                "cord_v2/num_samples": n,
                "cord_v2/predictions": table,
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
