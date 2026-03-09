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

"""Create a minimal 2-layer GPT-OSS checkpoint with mxfp4-quantized expert weights.

Run standalone:
    python tests/functional_tests/checkpoint/create_gptoss_2l_mxfp4.py \
        --output-dir /tmp/gptoss_2l_mxfp4 \
        --tokenizer-dir $TEST_DATA_DIR/hf_mixtral_2l/
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

_VOCAB_SIZE = 16000
_HIDDEN = 128
_HEADS = 4
_KV_HEADS = 4
_HEAD_DIM = 32
_INTER = 256
_EXPERTS = 4
_LAYERS = 2

# mxfp4 block/scale geometry chosen so dequantization produces correct shapes:
#   gate_up_proj  -> (experts, 2*inter, hidden):  G_g*B_g*2 = 2*_INTER = 512
#   down_proj     -> (experts, hidden, inter):     G_d*B_d*2 = _HIDDEN  = 128
_G_GATE, _B_GATE = 16, 16  # 16*16*2 = 512
_G_DOWN, _B_DOWN = 4, 16  # 4*16*2  = 128


def _build_config() -> dict:
    return {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "vocab_size": _VOCAB_SIZE,
        "hidden_size": _HIDDEN,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _KV_HEADS,
        "head_dim": _HEAD_DIM,
        "num_hidden_layers": _LAYERS,
        "intermediate_size": _INTER,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "sliding_window": None,
        "layer_types": ["full_attention", "sliding_attention"],
        "num_local_experts": _EXPERTS,
        "num_experts_per_tok": 2,
        "router_aux_loss_coef": 0.01,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 512,
        },
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
        },
        "tie_word_embeddings": False,
    }


def _build_tensors() -> dict[str, torch.Tensor]:
    """Return a state-dict with mxfp4 expert weights and bf16 dense weights."""
    t: dict[str, torch.Tensor] = {}
    bf = torch.bfloat16

    t["model.embed_tokens.weight"] = torch.randn(_VOCAB_SIZE, _HIDDEN, dtype=bf)
    t["lm_head.weight"] = torch.randn(_VOCAB_SIZE, _HIDDEN, dtype=bf)
    t["model.norm.weight"] = torch.ones(_HIDDEN, dtype=bf)

    kv_dim = _KV_HEADS * _HEAD_DIM

    for li in range(_LAYERS):
        p = f"model.layers.{li}"
        t[f"{p}.self_attn.q_proj.weight"] = torch.randn(_HEADS * _HEAD_DIM, _HIDDEN, dtype=bf)
        t[f"{p}.self_attn.k_proj.weight"] = torch.randn(kv_dim, _HIDDEN, dtype=bf)
        t[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_dim, _HIDDEN, dtype=bf)
        t[f"{p}.self_attn.o_proj.weight"] = torch.randn(_HIDDEN, _HEADS * _HEAD_DIM, dtype=bf)

        t[f"{p}.mlp.router.weight"] = torch.randn(_EXPERTS, _HIDDEN, dtype=bf)
        t[f"{p}.mlp.router.bias"] = torch.zeros(_EXPERTS, dtype=bf)

        t[f"{p}.input_layernorm.weight"] = torch.ones(_HIDDEN, dtype=bf)
        t[f"{p}.post_attention_layernorm.weight"] = torch.ones(_HIDDEN, dtype=bf)

        # mxfp4 expert tensors — random blocks, neutral exponent (127)
        t[f"{p}.mlp.experts.gate_up_proj_blocks"] = torch.randint(
            0, 256, (_EXPERTS, _HIDDEN, _G_GATE, _B_GATE), dtype=torch.uint8
        )
        t[f"{p}.mlp.experts.gate_up_proj_scales"] = torch.full((_EXPERTS, _HIDDEN, _G_GATE), 127, dtype=torch.uint8)
        t[f"{p}.mlp.experts.down_proj_blocks"] = torch.randint(
            0, 256, (_EXPERTS, _INTER, _G_DOWN, _B_DOWN), dtype=torch.uint8
        )
        t[f"{p}.mlp.experts.down_proj_scales"] = torch.full((_EXPERTS, _INTER, _G_DOWN), 127, dtype=torch.uint8)

    return t


def _build_index(tensors: dict[str, torch.Tensor], filename: str) -> dict:
    total_bytes = 0
    weight_map: dict[str, str] = {}
    for fqn, tensor in tensors.items():
        total_bytes += tensor.numel() * tensor.element_size()
        weight_map[fqn] = filename
    return {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}


def create_checkpoint(output_dir: str, tokenizer_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    config = _build_config()
    tensors = _build_tensors()
    safetensors_name = "model.safetensors"

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    save_file(tensors, os.path.join(output_dir, safetensors_name))

    index = _build_index(tensors, safetensors_name)
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    for fname in os.listdir(tokenizer_dir):
        src = os.path.join(tokenizer_dir, fname)
        if os.path.isfile(src) and ("token" in fname.lower() or fname == "special_tokens_map.json"):
            shutil.copy2(src, os.path.join(output_dir, fname))

    print(f"Created GPT-OSS 2L mxfp4 checkpoint in {output_dir}")
    print(f"  config keys:  {len(config)} entries")
    print(f"  tensor keys:  {len(tensors)} entries")
    print(f"  mxfp4 keys:   {sum(1 for k in tensors if '_blocks' in k or '_scales' in k)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    args = parser.parse_args()
    create_checkpoint(args.output_dir, args.tokenizer_dir)
