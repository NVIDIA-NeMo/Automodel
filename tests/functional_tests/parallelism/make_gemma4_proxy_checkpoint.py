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

"""Build a randomly-initialized, Gemma4-31B-shaped proxy checkpoint.

The ``L2_Parallelism`` tests need a model small enough for the 2-GPU PR runners
that still exercises the code paths of
``examples/vlm_finetune/gemma4/gemma4_31b_tp4_pp2.yaml``. This writes one to a
temporary directory at test time, so nothing has to be staged in ``TEST_DATA_DIR``.

The checkpoint is produced rather than loaded through ``from_config`` for two
reasons. It lets the proxy use ``from_pretrained``, matching the real recipes.
And ``from_config`` currently yields an all-zero Gemma4: the model is built under
``no_init_weights()`` on meta device, and ``Checkpointer.initialize_model_weights``
does not re-run HF's initializer for custom NeMo classes that wrap an HF model
without defining their own ``init_weights`` (qwen3_moe defines one and
initializes correctly; gemma4_moe does not). A zero model emits uniform logits,
so its loss is pinned at ``ln(vocab_size)`` and every gradient is exactly zero --
which silently turns a parity test into a comparison of zero against zero.

The processor is copied from an existing staged Gemma4 checkpoint. Its
``image_seq_length``/``patch_size``/``pooling_kernel_size`` are identical across
Gemma4 31B, E4B, and the staged mini checkpoint, so the pairing is exact.

Usage:
    python make_gemma4_proxy_checkpoint.py --output-dir DIR --processor-dir DIR
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

# Shrunk from google/gemma-4-31B-it's config.json. The fields NOT scaled down are
# the ones the parallelism code branches on, so that TP shards and PP splits the
# same structure they do in production:
#   * layer_types      -- 31B repeats [sliding x5, full]; one full period is kept
#                         so a pp_size=2 split lands mid-period as it does at
#                         60 layers / 4 stages.
#   * attention_k_eq_v, global_head_dim != head_dim,
#     num_global_key_value_heads != num_key_value_heads
#                      -- the 31B-specific attention layout TP has to shard.
#   * vocab_size       -- must stay 262144: the Gemma4 processor emits
#                         image_token_id 258880, so a smaller vocab puts image
#                         tokens out of range.
#   * patch_size, pooling_kernel_size, position_embedding_size,
#     default_output_length, vision_soft_tokens_per_image
#                      -- must match the processor's image_seq_length of 280 or
#                         the image-embedding scatter raises a shape error.
TEXT_CONFIG: dict = {
    "vocab_size": 262144,
    "vocab_size_per_layer_input": 262144,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "global_head_dim": 128,
    "num_global_key_value_heads": 2,
    "attention_k_eq_v": True,
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
    "enable_moe_block": False,
    "use_double_wide_mlp": False,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "final_logit_softcapping": 30.0,
    "use_bidirectional_attention": "vision",
    "use_cache": False,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "tie_word_embeddings": True,
    "rope_parameters": {
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
}

VISION_CONFIG: dict = {
    "model_type": "gemma4_vision",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    # 31B keeps num_key_value_heads == num_attention_heads and
    # global_head_dim == head_dim in the vision tower; both must be set
    # explicitly, since their defaults (12 and 64) are sized for the untouched
    # 768-wide tower and do not follow num_attention_heads down.
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "global_head_dim": 64,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "default_output_length": 280,
    "rms_norm_eps": 1e-6,
    "standardize": True,
    "use_clipped_linears": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
}

# Files that make a directory loadable by AutoProcessor for Gemma4.
PROCESSOR_FILES = (
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "special_tokens_map.json",
)


def build_config():
    """Build the shrunk Gemma4 composite config.

    Returns:
        A ``Gemma4Config`` with the 31B's structural fields preserved.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    return Gemma4Config(
        text_config=dict(TEXT_CONFIG),
        vision_config=dict(VISION_CONFIG),
        architectures=["Gemma4ForConditionalGeneration"],
        vision_soft_tokens_per_image=280,
        image_token_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        video_token_id=258884,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def main() -> None:
    """Write a randomly-initialized proxy checkpoint plus its processor."""
    parser = argparse.ArgumentParser(description="Build a Gemma4 31B proxy checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to write the checkpoint into")
    parser.add_argument(
        "--processor-dir",
        required=True,
        help="Staged Gemma4 checkpoint to copy processor/tokenizer files from",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for weight initialization")
    args = parser.parse_args()

    # Built with the stock HF class, not the NeMo subclass: this script only
    # produces an on-disk artifact, and NeMo's ``save_pretrained`` override
    # requires a live checkpointer. The saved ``architectures`` entry still
    # routes ``from_pretrained`` to the NeMo class through MODEL_ARCH_MAPPING.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    output_dir = Path(args.output_dir)
    processor_dir = Path(args.processor_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    # Plain construction (no no_init_weights / meta device) so HF's initializer
    # actually runs and the weights are random rather than zero.
    config = build_config()
    model = Gemma4ForConditionalGeneration(config).to(torch.bfloat16)

    embed_absmax = float(model.get_input_embeddings().weight.detach().abs().max())
    if embed_absmax == 0.0:
        raise RuntimeError(
            "Proxy checkpoint would be all zeros: HF initialization did not run. "
            "A zero model emits uniform logits and zero gradients, which makes the "
            "parity comparison vacuous."
        )

    model.save_pretrained(str(output_dir))

    copied = []
    for name in PROCESSOR_FILES:
        source = processor_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    if "processor_config.json" not in copied:
        raise RuntimeError(f"No processor_config.json found in {processor_dir}; cannot build a loadable proxy.")

    print(f"Wrote Gemma4 proxy checkpoint to {output_dir} (embed absmax {embed_absmax:.6f})")
    print(f"Copied processor files: {', '.join(copied)}")


if __name__ == "__main__":
    main()
