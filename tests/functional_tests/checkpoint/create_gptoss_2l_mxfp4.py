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

The mxfp4 block/scale geometry uses the same hardcoded (G=90, B=16) that the
GPTOSSStateDictAdapter.convert_single_tensor_to_hf produces, so that the DCP
planner sees matching shapes when loading.

Dequantization produces:
    blocks (E, dim, G, B)  →  (E, G*B*2, dim)  [after reshape + transpose]

where G*B*2 = 90*16*2 = 2880.

For both projections the adapter reads  dim = tensor.shape[-1]:
    gate_and_up_projs (E, hidden, 2*inter)  →  dim = 2*inter  →  2*inter = 2880  →  inter = 1440?
    down_projs        (E, inter, hidden)    →  dim = hidden    →  hidden  = 2880

BUT the *dequanted* tensor must match the internal shape:
    gate_up_proj  →  (E, 2880, 2*inter)  needs  2880 = hidden  ✓
    down_proj     →  (E, 2880, hidden)    needs  2880 = inter

So both hidden_size AND intermediate_size must equal 2880.

Run standalone:
    python tests/functional_tests/checkpoint/create_gptoss_2l_mxfp4.py \\
        --output-dir /tmp/gptoss_2l_mxfp4 \\
        --tokenizer-dir $TEST_DATA_DIR/hf_mixtral_2l/
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

_G, _B = 90, 16  # hardcoded in GPTOSSStateDictAdapter.convert_single_tensor_to_hf

_VOCAB_SIZE = 32000
_HIDDEN = 2880  # forced: dequant produces (E, 2880, dim), must equal hidden for down_proj
_INTER = 2880  # forced: dequant produces (E, 2880, dim), must equal inter for down_proj
_HEADS = 90  # 90 * 32 = 2880
_KV_HEADS = 2
_HEAD_DIM = 32
_EXPERTS = 2
_LAYERS = 2


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
        "sliding_window": 256,
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

    up_proj_dim = 2 * _INTER  # gated activation → gate + up concatenated
    kv_dim = _KV_HEADS * _HEAD_DIM

    for li in range(_LAYERS):
        p = f"model.layers.{li}"

        # ── Attention ──
        t[f"{p}.self_attn.q_proj.weight"] = torch.randn(_HEADS * _HEAD_DIM, _HIDDEN, dtype=bf)
        t[f"{p}.self_attn.q_proj.bias"] = torch.zeros(_HEADS * _HEAD_DIM, dtype=bf)
        t[f"{p}.self_attn.k_proj.weight"] = torch.randn(kv_dim, _HIDDEN, dtype=bf)
        t[f"{p}.self_attn.k_proj.bias"] = torch.zeros(kv_dim, dtype=bf)
        t[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_dim, _HIDDEN, dtype=bf)
        t[f"{p}.self_attn.v_proj.bias"] = torch.zeros(kv_dim, dtype=bf)
        t[f"{p}.self_attn.o_proj.weight"] = torch.randn(_HIDDEN, _HEADS * _HEAD_DIM, dtype=bf)
        t[f"{p}.self_attn.o_proj.bias"] = torch.zeros(_HIDDEN, dtype=bf)
        t[f"{p}.self_attn.sinks"] = torch.zeros(_HEADS, dtype=torch.float32)

        # ── Router ──
        t[f"{p}.mlp.router.weight"] = torch.randn(_EXPERTS, _HIDDEN, dtype=bf)
        t[f"{p}.mlp.router.bias"] = torch.zeros(_EXPERTS, dtype=bf)

        # ── Layer norms ──
        t[f"{p}.input_layernorm.weight"] = torch.ones(_HIDDEN, dtype=bf)
        t[f"{p}.post_attention_layernorm.weight"] = torch.ones(_HIDDEN, dtype=bf)

        # ── Expert biases (bf16, not quantized) ──
        t[f"{p}.mlp.experts.gate_up_proj_bias"] = torch.zeros(_EXPERTS, up_proj_dim, dtype=bf)
        t[f"{p}.mlp.experts.down_proj_bias"] = torch.zeros(_EXPERTS, _HIDDEN, dtype=bf)

        # ── MXFP4 blocks / scales ──
        # gate_and_up_projs internal shape: (E, hidden, 2*inter)
        # adapter: dim = tensor.shape[-1] = 2*inter = up_proj_dim
        # blocks = (E, up_proj_dim, G, B)
        t[f"{p}.mlp.experts.gate_up_proj_blocks"] = torch.randint(
            0, 256, (_EXPERTS, up_proj_dim, _G, _B), dtype=torch.uint8
        )
        t[f"{p}.mlp.experts.gate_up_proj_scales"] = torch.full((_EXPERTS, up_proj_dim, _G), 127, dtype=torch.uint8)

        # down_projs internal shape: (E, inter, hidden)
        # adapter: dim = tensor.shape[-1] = hidden
        # blocks = (E, hidden, G, B)
        t[f"{p}.mlp.experts.down_proj_blocks"] = torch.randint(0, 256, (_EXPERTS, _HIDDEN, _G, _B), dtype=torch.uint8)
        t[f"{p}.mlp.experts.down_proj_scales"] = torch.full((_EXPERTS, _HIDDEN, _G), 127, dtype=torch.uint8)

    return t


def _build_index(tensors: dict[str, torch.Tensor], filename: str) -> dict:
    total_bytes = 0
    weight_map: dict[str, str] = {}
    for fqn, tensor in tensors.items():
        total_bytes += tensor.numel() * tensor.element_size()
        weight_map[fqn] = filename
    return {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}


def _verify_mxfp4(output_dir: str, config: dict, expected_mxfp4_keys: list[str]) -> None:
    """Re-open the saved checkpoint and verify mxfp4 tensors are present and correct."""
    from safetensors import safe_open

    st_path = os.path.join(output_dir, "model.safetensors")
    with safe_open(st_path, framework="pt", device="cpu") as f:
        saved_keys = set(f.keys())

    for k in expected_mxfp4_keys:
        assert k in saved_keys, f"mxfp4 key missing from saved checkpoint: {k}"

    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in expected_mxfp4_keys:
            t = f.get_tensor(k)
            assert t.dtype == torch.uint8, f"{k}: expected uint8 but got {t.dtype}"
            if "_blocks" in k:
                assert t.shape[-2:] == (_G, _B), f"{k}: expected last dims ({_G}, {_B}) but got {t.shape[-2:]}"
            elif "_scales" in k:
                assert t.shape[-1] == _G, f"{k}: expected last dim {_G} but got {t.shape[-1]}"

    with open(os.path.join(output_dir, "config.json")) as f:
        saved_config = json.load(f)
    qcfg = saved_config.get("quantization_config", {})
    assert qcfg.get("quant_method") == "mxfp4", (
        f"config.json quant_method should be 'mxfp4', got {qcfg.get('quant_method')!r}"
    )

    print(f"  ✓ verified {len(expected_mxfp4_keys)} mxfp4 keys (uint8, correct shapes, config.json)")


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

    total_mb = sum(t.numel() * t.element_size() for t in tensors.values()) / (1 << 20)
    mxfp4_keys = [k for k in tensors if "_blocks" in k or "_scales" in k]
    print(f"Created GPT-OSS 2L mxfp4 checkpoint in {output_dir}")
    print(f"  tensor keys:  {len(tensors)} entries")
    print(f"  mxfp4 keys:   {len(mxfp4_keys)}")
    print(f"  total size:   {total_mb:.1f} MB")

    _verify_mxfp4(output_dir, config, mxfp4_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    args = parser.parse_args()
    create_checkpoint(args.output_dir, args.tokenizer_dir)
