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

"""Extract per-layer activations from NeMo AutoModel using the training code path.

Uses the same distributed setup as training (EP, FSDP, backend config) and
registers forward hooks on decoder layers to capture hidden states. Saves
activations and final logits for comparison against HF Transformers.

Run via torchrun:
    torchrun --nproc-per-node 8 extract_nemo_activations.py \
        --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml \
        --output-file /tmp/nemo_activations.pt \
        --num-prompts 3
"""

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist

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
    """Extract NeMo AutoModel activations for comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--output-file", required=True, help="Path to save activations (.pt)")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--gate-precision", type=str, default=None)
    parser.add_argument("--lm-head-precision", type=str, default=None)
    args, extra = parser.parse_known_args()

    _dtype_alias = {
        "fp16": "torch.float16",
        "fp32": "torch.float32",
        "fp64": "torch.float64",
        "bf16": "torch.bfloat16",
    }
    if args.gate_precision:
        val = _dtype_alias.get(args.gate_precision, args.gate_precision)
        extra += [f"--model.backend.gate_precision={val}"]
    if args.lm_head_precision:
        val = _dtype_alias.get(args.lm_head_precision, args.lm_head_precision)
        extra += [f"--distributed.moe.lm_head_precision={val}"]

    # --- Load config ---
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    sys.argv = ["extract_nemo_activations.py", "--config", args.config] + extra
    cfg = parse_args_and_load_config()

    # --- Distributed setup ---
    from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
    from nemo_automodel.components.loggers.log_utils import setup_logging
    from nemo_automodel.recipes._dist_setup import setup_distributed
    from nemo_automodel.recipes.llm.train_ft import build_distributed, build_model
    from nemo_automodel.shared.te_patches import apply_te_patches

    dist_env = build_distributed(cfg.get("dist_env", {}))
    setup_logging()
    apply_cache_compatibility_patches()
    apply_te_patches()
    dist_setup = setup_distributed(cfg, world_size=dist_env.world_size)

    if dist_setup.cp_size > 1 and cfg.get("model.backend.rope_fusion", False):
        cfg.model.backend.rope_fusion = False

    # --- Build model ---
    model = build_model(
        cfg.model,
        cfg_peft=None,
        seed=cfg.get("seed", 42),
        device_mesh=dist_setup.device_mesh,
        moe_mesh=dist_setup.moe_mesh,
        distributed_config=dist_setup.strategy_config,
        pipeline_config=dist_setup.pipeline_config,
        cfg_moe=dist_setup.moe_config,
        activation_checkpointing=dist_setup.activation_checkpointing,
    )
    model.eval()

    # --- Load tokenizer ---
    from transformers import AutoTokenizer

    model_name = cfg.model.get("pretrained_model_name_or_path", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    # --- Find decoder layers by name pattern (works with FSDP-wrapped models) ---
    decoder_layers = []
    for name, module in model.named_modules():
        # Match patterns like "model.layers.0", "model.model.layers.0", etc.
        # but not sub-modules like "model.layers.0.self_attn"
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            decoder_layers.append((name, module))

    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(torch.cuda.current_device())
    device = torch.device("cuda", local_rank)

    if rank == 0:
        logger.info("Found %d decoder layers", len(decoder_layers))

    # --- Forward pass with hooks ---
    all_results = []

    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

            layer_outputs = OrderedDict()
            hooks = []

            def make_hook(layer_key):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    layer_outputs[layer_key] = hidden[0, -1, :].float().detach().cpu()

                return hook_fn

            for i, (layer_name, layer_module) in enumerate(decoder_layers):
                h = layer_module.register_forward_hook(make_hook(f"layer_{i}"))
                hooks.append(h)

            outputs = model(input_ids=input_ids)
            if hasattr(outputs, "logits"):
                logits_out = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits_out = outputs
            else:
                logits_out = outputs[0]

            for h in hooks:
                h.remove()

            final_logits = logits_out[0, -1, :].float().detach().cpu()

            all_results.append(
                {
                    "prompt": prompt,
                    "num_tokens": len(token_ids),
                    "layer_outputs": dict(layer_outputs),
                    "logits": final_logits,
                }
            )

            if rank == 0:
                logger.info(
                    "  Prompt %d: %d tokens, %d layers captured", prompt_idx, len(token_ids), len(layer_outputs)
                )

    # --- Save on rank 0 ---
    if rank == 0:
        torch.save(all_results, args.output_file)
        logger.info("Saved activations to %s (%d prompts)", args.output_file, len(all_results))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
