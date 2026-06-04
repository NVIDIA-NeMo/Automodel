#!/usr/bin/env python3
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

"""Extract per-layer activations from HF Transformers as the reference implementation.

Loads the model via AutoModelForCausalLM with device_map="auto" and registers
forward hooks on each decoder layer to capture outputs. Saves activations and
final logits for comparison against NeMo AutoModel.

Usage (called automatically by compare_activations.py, or standalone):
    python extract_hf_activations.py \
        --model Qwen/Qwen3-30B-A3B \
        --output-file /tmp/hf_activations.pt \
        --num-prompts 3
"""

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Solve for x: 3x + 7 = 22. Show your work step by step.",
    "What is the derivative of f(x) = x^3 * ln(x)?",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the difference between a stack and a queue with code examples.",
    "Tell me about the history of the Silk Road in three paragraphs.",
    "What are the main differences between classical and operant conditioning?",
    "Summarize the following in exactly two bullet points: Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with many layers.",
    "Translate the following English sentence to French: 'The weather is beautiful today and I would like to go for a walk.'",
    "Describe how the transformer attention mechanism works, including the role of queries, keys, and values.",
    "Write a haiku about a neural network learning to see.",
]


def main():
    """Extract Hugging Face model activations for comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output-file", required=True, help="Path to save activations (.pt)")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--prompts-file", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading HF tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info("Loading HF model: %s (device_map=auto)", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # --- Load prompts ---
    if args.prompts_file:
        text = Path(args.prompts_file).read_text().strip()
        try:
            prompts = json.loads(text)
        except json.JSONDecodeError:
            prompts = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        prompts = list(DEFAULT_PROMPTS)
    prompts = prompts[: args.num_prompts]

    # --- Find decoder layers ---
    # Most HF models: model.model.layers[i] or model.transformer.h[i]
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        layers = inner.layers
    elif hasattr(inner, "h"):
        layers = inner.h
    else:
        logger.error("Cannot find decoder layers in model architecture")
        sys.exit(1)

    num_layers = len(layers)
    logger.info("Found %d decoder layers", num_layers)

    # --- Register hooks to capture per-layer hidden states ---
    all_results = []

    for prompt_idx, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)

        layer_outputs = OrderedDict()
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                # Most decoder layers return (hidden_states, ...) or a tuple
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Save last-position hidden state only (saves memory)
                layer_outputs[name] = hidden[0, -1, :].float().detach().cpu()

            return hook_fn

        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(make_hook(f"layer_{i}"))
            hooks.append(h)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Final logits at last position
        logits = outputs.logits[0, -1, :].float().detach().cpu()

        all_results.append(
            {
                "prompt": prompt,
                "num_tokens": len(token_ids),
                "layer_outputs": dict(layer_outputs),  # {layer_name: tensor(hidden_dim)}
                "logits": logits,  # tensor(vocab_size)
            }
        )

        logger.info("  Prompt %d: %d tokens, %d layers captured", prompt_idx, len(token_ids), len(layer_outputs))

    torch.save(all_results, args.output_file)
    logger.info("Saved activations to %s (%d prompts, %d layers)", args.output_file, len(all_results), num_layers)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
