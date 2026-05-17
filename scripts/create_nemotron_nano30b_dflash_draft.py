#!/usr/bin/env python3
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
"""Initialise a DFlash draft model for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Target architecture:
  hidden_size=2688, num_attention_heads=32, head_dim=128, num_hidden_layers=52,
  vocab_size=131072, bos=1, eos=2

Draft architecture (Qwen3 full-attention, 7 layers):
  hidden_size=2688, num_attention_heads=21 (21*128=2688), num_key_value_heads=3,
  head_dim=128, intermediate_size=8064 (3x), vocab_size=131072

Conditioning: 5 target layers uniformly sampled between layer 1 and layer 49.

Usage:
    python scripts/create_nemotron_nano30b_dflash_draft.py \
        --output_dir checkpoints/nemotron-nano-30b-dflash-b16/init
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


TARGET_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DFLASH_CODE_SOURCE = "z-lab/Qwen3-8B-DFlash-b16"

# Draft config — Qwen3 full-attention backbone tuned for Nano-30B target dims.
DRAFT_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
    "model_type": "qwen3",
    # Dimensions matching target hidden_size=2688, head_dim=128
    "hidden_size": 2688,
    "num_attention_heads": 21,       # 21 * 128 = 2688
    "num_key_value_heads": 3,        # 7 queries per KV head
    "head_dim": 128,
    "intermediate_size": 8064,       # 3x hidden_size
    "num_hidden_layers": 7,
    "layer_types": ["full_attention"] * 7,
    # Target conditioning
    "num_target_layers": 52,         # Nano-30B has 52 layers
    "dflash_config": {
        # 5 layers uniformly from layer 1 to layer 49 (second to third-to-last)
        "target_layer_ids": [1, 13, 25, 37, 49],
        "mask_token_id": 18,         # <SPECIAL_18> — safe unused special token
    },
    "block_size": 16,
    "dtype": "bfloat16",
    # Tokenizer fields from Nemotron-30B
    "vocab_size": 131072,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    # Qwen3 defaults
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "max_position_embeddings": 131072,
    "max_window_layers": 7,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 10000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "transformers_version": "4.51.0",
}

# Files to copy from the z-lab DFlash source (trust_remote_code model code)
DFLASH_CODE_FILES = ["dflash.py", "modeling_dflash.py", "utils.py"]


def get_dflash_code_dir() -> Path:
    from huggingface_hub import snapshot_download

    print(f"Downloading DFlash model code from {DFLASH_CODE_SOURCE} ...")
    path = snapshot_download(DFLASH_CODE_SOURCE, ignore_patterns=["*.safetensors", "*.bin"])
    return Path(path)


def init_draft_model(output_dir: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModel

    print("Initialising draft model weights from config ...")
    cfg = AutoConfig.from_pretrained(str(output_dir), trust_remote_code=True)
    model = AutoModel.from_config(cfg, trust_remote_code=True).to(torch.bfloat16)
    print(f"  Draft parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Optionally warm-start the embedding + lm_head from the target model so the
    # draft starts with the correct token representations.  This is optional —
    # the SFT training will quickly adapt them — but it gives a better loss at
    # step 0 and avoids a large embedding mismatch at the start.
    try:
        from transformers import AutoModelForCausalLM

        print(f"Copying embeddings from {TARGET_MODEL_ID} ...")
        target = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # Nemotron-H uses backbone.embed_tokens; fall back to model.embed_tokens
        src_embed = (
            getattr(getattr(target, "backbone", None), "embed_tokens", None)
            or target.get_input_embeddings()
        )
        src_head = target.get_output_embeddings()

        if src_embed is not None and src_embed.weight.shape == model.model.embed_tokens.weight.shape:
            model.model.embed_tokens.weight.data.copy_(src_embed.weight.data)
            print("  embed_tokens copied.")
        else:
            print("  embed_tokens shape mismatch — leaving random init.")

        if src_head is not None and src_head.weight.shape == model.lm_head.weight.shape:
            model.lm_head.weight.data.copy_(src_head.weight.data)
            print("  lm_head copied.")
        else:
            print("  lm_head shape mismatch — leaving random init.")

        del target
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  Warning: could not copy target embeddings ({exc}). Using random init.")

    print(f"Saving draft model to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="checkpoints/nemotron-nano-30b-dflash-b16/init",
        help="Directory to save the initialised draft model.",
    )
    parser.add_argument(
        "--skip_target_embed",
        action="store_true",
        help="Skip copying embeddings from the target (faster, fully random init).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy DFlash model code files
    code_dir = get_dflash_code_dir()
    for fname in DFLASH_CODE_FILES:
        src = code_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname}")
        else:
            print(f"  WARNING: {fname} not found in {code_dir}", file=sys.stderr)

    # 2. Write config.json
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(DRAFT_CONFIG, f, indent=2)
    print(f"Wrote {config_path}")

    # 3. Initialise weights
    if args.skip_target_embed:
        import torch
        from transformers import AutoConfig, AutoModel

        print("Initialising draft model with random weights ...")
        # No weight files exist yet — use from_config to build with random init.
        cfg = AutoConfig.from_pretrained(str(output_dir), trust_remote_code=True)
        model = AutoModel.from_config(cfg, trust_remote_code=True).to(torch.bfloat16)
        print(f"  Draft parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        model.save_pretrained(str(output_dir))
        print("Done.")
    else:
        init_draft_model(output_dir)


if __name__ == "__main__":
    main()
