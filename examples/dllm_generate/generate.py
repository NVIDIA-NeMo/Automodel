# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Inference script for Automodel LLaDA checkpoints using NeMo Automodel utilities.

Works with any consolidated Automodel checkpoint. Uses NeMoAutoModelForCausalLM
for model loading and NeMoAutoTokenizer for tokenizer setup.

Usage
-----
# Basic generation with a prompt:
  python examples/dllm_generate/generate.py \
      --checkpoint checkpoints/llada-tulu3-500/LATEST/model/consolidated \
      --prompt "Explain what a neural network is."

# Multiple prompts:
  python examples/dllm_generate/generate.py \
      --checkpoint checkpoints/llada-tulu3-500/LATEST/model/consolidated \
      --prompt "Lily runs 12 km/h for 4 hours. How far in 8 hours?" \
      --prompt "Please write an educational python function."

# Adjust sampling parameters:
  python examples/dllm_generate/generate.py \
      --checkpoint checkpoints/llada-tulu3-1k/LATEST/model/consolidated \
      --prompt "Write a haiku about programming." \
      --steps 256 --max_new_tokens 256 --temperature 0.2

# Raw mode (no chat template, just continue the text):
  python examples/dllm_generate/generate.py \
      --checkpoint checkpoints/llada-tulu3-500/LATEST/model/consolidated \
      --prompt "Once upon a time" \
      --raw

# Infilling mode (replace [MASK] placeholders):
  python examples/dllm_generate/generate.py \
      --checkpoint checkpoints/llada-tulu3-500/LATEST/model/consolidated \
      --prompt "The capital of France is [MASK] and it is known for [MASK]." \
      --infill

Checkpoint path resolution
--------------------------
The --checkpoint flag accepts flexible paths:
  - .../consolidated                    (direct HF-format dir)
  - .../model                           (finds consolidated/ inside)
  - .../LATEST                          (finds model/consolidated/ inside)
  - .../epoch_0_step_312/model/consolidated  (intermediate steps)
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

# Automodel utilities — use the installed/editable nemo_automodel package.
# If running from a standalone checkout, set AUTOMODEL_ROOT to the Automodel repo root.
_automodel_root = os.environ.get("AUTOMODEL_ROOT", "/opt/Automodel")
if _automodel_root not in sys.path:
    sys.path.insert(0, _automodel_root)
from nemo_automodel import NeMoAutoModelForCausalLM, NeMoAutoTokenizer

# LLaDA chat template (matches the reference dllm codebase)
LLADA_CHAT_TEMPLATE = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""


def resolve_checkpoint(path: str) -> str:
    """Resolve a checkpoint path, checking for consolidated/ subdirectory."""
    if os.path.isdir(os.path.join(path, "consolidated")):
        return os.path.join(path, "consolidated")
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    for sub in ["LATEST/model/consolidated", "LATEST/model", "model/consolidated", "model"]:
        candidate = os.path.join(path, sub)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(
        f"Could not find a valid HF checkpoint at {path}. Expected a directory with config.json and model safetensors."
    )


def load_model_and_tokenizer(checkpoint_path: str):
    """Load LLaDA model and tokenizer from an Automodel consolidated checkpoint."""
    tokenizer = NeMoAutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # LLaDA requires mask token for diffusion sampling
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    tokenizer.eot_token = "<|eot_id|>"

    # Ensure chat template is set correctly
    tokenizer.chat_template = LLADA_CHAT_TEMPLATE

    model = NeMoAutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="bfloat16",
        trust_remote_code=True,
        use_liger_kernel=False,
        use_sdpa_patching=False,
    ).eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# MDLM Sampling (self-contained, no dllm dependency)
# ---------------------------------------------------------------------------


def _add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    noise = torch.zeros_like(logits).uniform_(1e-20, 1.0)
    return logits + temperature * -torch.log(-torch.log(noise))


def _get_num_transfer_tokens(mask_index, steps):
    """Linear schedule: spread unmasking evenly across steps."""
    mask_num = mask_index.sum(dim=-1, keepdim=True)
    B = mask_index.shape[0]
    base = mask_num // steps
    remainder = mask_num % steps
    tokens_per_step = base.repeat(1, steps)
    for j in range(B):
        r = int(remainder[j].item())
        if r > 0:
            tokens_per_step[j, :r] += 1
    col_sums = tokens_per_step.sum(dim=0)
    last_nonzero = 0
    for c in range(col_sums.shape[0]):
        if col_sums[c] > 0:
            last_nonzero = c
    return tokens_per_step[:, : last_nonzero + 1]


@torch.no_grad()
def generate(
    model, tokenizer, inputs, steps=128, max_new_tokens=128, block_size=32, temperature=0.0, remasking="low_confidence"
):
    """Generate text via iterative masked-diffusion unmasking (MDLM)."""
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    if isinstance(inputs[0], list):
        inputs = [torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs]
    prompt_lens = [p.shape[0] for p in inputs]
    max_length = max_new_tokens + max(prompt_lens)
    B, T = len(inputs), max_length

    # Build canvas: prompt tokens + mask tail + eos padding
    x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
    for i, p in enumerate(inputs):
        x[i, : prompt_lens[i]] = p
        x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, pl in enumerate(prompt_lens):
        attention_mask[i, : min(pl + max_new_tokens, T)] = 1

    # Block-wise iterative unmasking
    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = math.ceil(steps / num_blocks)

    for b in range(num_blocks):
        block_mask = torch.zeros((B, block_size), dtype=torch.bool, device=device)
        for j in range(B):
            start = prompt_lens[j] + b * block_size
            end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                block_mask[j, : end - start] = x[j, start:end] == mask_id

        transfer_schedule = _get_num_transfer_tokens(block_mask, steps_per_block)
        for i in range(transfer_schedule.size(1)):
            mask_index = x == mask_id
            logits = model(x, attention_mask=attention_mask).logits

            x0 = torch.argmax(_add_gumbel_noise(logits, temperature), dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((B, T), device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            for j in range(B):
                x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -float("inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float("inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(transfer_schedule[j, i].item())
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def infill(model, tokenizer, inputs, steps=128, block_size=32, temperature=0.0, remasking="low_confidence"):
    """Fill [MASK] tokens in-place via MDLM unmasking."""
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    if isinstance(inputs[0], list):
        inputs = [torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs]

    B = len(inputs)
    seq_lens = [t.shape[0] for t in inputs]
    T = max(seq_lens)
    if block_size is None:
        block_size = T

    x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
    for i, t in enumerate(inputs):
        x[i, : seq_lens[i]] = t
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, L in enumerate(seq_lens):
        if L > 0:
            attention_mask[i, :L] = 1

    num_blocks = math.ceil(T / block_size)
    steps_per_block = math.ceil(steps / num_blocks)

    for b in range(num_blocks):
        start = b * block_size
        stop = min(start + block_size, T)
        block_mask = torch.zeros((B, block_size), dtype=torch.bool, device=device)
        widths = []
        for j in range(B):
            width = max(0, min(seq_lens[j], stop) - start)
            widths.append(width)
            if width > 0:
                block_mask[j, :width] = x[j, start : start + width] == mask_id

        transfer_schedule = _get_num_transfer_tokens(block_mask, steps_per_block)
        for s in range(transfer_schedule.size(1)):
            mask_full = x == mask_id
            logits = model(x, attention_mask=attention_mask).logits

            x0 = torch.argmax(_add_gumbel_noise(logits, temperature), dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((B, T), device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            for j in range(B):
                x0_p[j, :start] = -float("inf")
                x0_p[j, start + widths[j] :] = -float("inf")

            x0 = torch.where(mask_full, x0, x)
            confidence = torch.where(mask_full, x0_p, -float("inf"))

            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for j in range(B):
                k = int(transfer_schedule[j, s].item())
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    return x


def trim_response(tokenizer, seq_ids_list, input_ids_list):
    """Extract the generated text after the prompt, truncated at first EOS/EOT."""
    results = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = list(seq_ids)
        start = len(list(input_ids))
        end = len(full)

        eos_id = tokenizer.eos_token_id
        eot_id = getattr(tokenizer, "eot_token_id", None)
        for i in range(start, len(full)):
            if full[i] in (eos_id, eot_id):
                end = i
                break

        text = tokenizer.decode(full[start:end], skip_special_tokens=True)
        for stop in [tokenizer.eos_token, getattr(tokenizer, "eot_token", None)]:
            if stop:
                text = text.split(stop)[0]
        results.append(text)
    return results


def main():
    """Run diffusion-language-model text generation from the CLI."""

    parser = argparse.ArgumentParser(
        description="Generate text from Automodel LLaDA checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint (run dir, step dir, or consolidated dir)"
    )
    parser.add_argument(
        "--prompt", action="append", required=True, help="Prompt(s) to generate from (repeat for multiple)"
    )
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps (default: 128)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate (default: 128)")
    parser.add_argument("--block_size", type=int, default=32, help="Block size (default: 32)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature, 0=greedy (default: 0.0)")
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--raw", action="store_true", help="Raw mode: no chat template")
    parser.add_argument("--infill", action="store_true", help="Infilling mode: replace [MASK] in prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    if args.infill:
        print("\n" + "=" * 80)
        print("INFILLING MODE".center(80))
        print("=" * 80)

        messages_list = []
        for prompt in args.prompt:
            parts = prompt.split("[MASK]")
            content = (tokenizer.mask_token * 20).join(parts)
            messages_list.append([{"role": "user", "content": content}])

        encoded = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
        )
        inputs = encoded["input_ids"]

        outputs = infill(
            model,
            tokenizer,
            inputs,
            steps=args.steps,
            block_size=args.block_size,
            temperature=args.temperature,
            remasking=args.remasking,
        )

        for i, prompt in enumerate(args.prompt):
            print(f"\n{'─' * 80}")
            print(f"[Prompt {i}] {prompt}")
            print(f"{'─' * 80}")
            print(f"[Filled] {tokenizer.decode(outputs[i], skip_special_tokens=True)}")
    else:
        print("\n" + "=" * 80)
        print(("RAW GENERATION" if args.raw else "CHAT GENERATION").center(80))
        print("=" * 80)

        if args.raw:
            inputs = [tokenizer.encode(p, add_special_tokens=True) for p in args.prompt]
        else:
            messages_list = [[{"role": "user", "content": p}] for p in args.prompt]
            encoded = tokenizer.apply_chat_template(
                messages_list,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            inputs = encoded["input_ids"]

        outputs = generate(
            model,
            tokenizer,
            inputs,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            temperature=args.temperature,
            remasking=args.remasking,
        )

        sequences = trim_response(tokenizer, outputs.tolist(), inputs)
        for i, (prompt, response) in enumerate(zip(args.prompt, sequences)):
            print(f"\n{'─' * 80}")
            print(f"[Prompt {i}] {prompt}")
            print(f"{'─' * 80}")
            print(response.strip() or "<empty>")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
