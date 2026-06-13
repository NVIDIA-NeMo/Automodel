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

"""mxfp4 passthrough load path for DeepSeek-V4: experts load packed, never bf16."""

import json
import os
from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4

# Real DeepSeek-V4-Flash checkpoint (fp4 experts). Tests using it self-skip when absent.
_REAL_CKPT = "/raid0/data/models/DeepSeek-V4-Flash"

HIDDEN = 64
MOE_INTER = 32
N_EXPERTS = 4


def _make_adapter(expert_storage_format="bf16"):
    config = DeepseekV4Config(
        vocab_size=256,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=MOE_INTER,
        num_nextn_predict_layers=0,
    )
    moe_config = Mock(spec=MoEConfig)
    moe_config.n_routed_experts = N_EXPERTS
    moe_config.moe_inter_dim = MOE_INTER
    return DeepSeekV4StateDictAdapter(
        config, moe_config, BackendConfig(), dtype=torch.bfloat16, expert_storage_format=expert_storage_format
    )


def _synthetic_fp4_checkpoint(seed=0):
    """Per-expert packed fp4 (int8) weights + e8m0 scales for one MoE layer.

    w1/w3 (gate/up): [moe_inter, hidden] checkpoint orientation; w2 (down):
    [hidden, moe_inter]. Returns the HF-format dict plus the bf16 weights each
    packed tensor decodes to (for reference).
    """
    torch.manual_seed(seed)
    sd = {}
    ref = {}  # (expert, which) -> dequantized bf16 weight [out, in]
    for e in range(N_EXPERTS):
        for which, (out_dim, in_dim) in {
            "w1": (MOE_INTER, HIDDEN),
            "w3": (MOE_INTER, HIDDEN),
            "w2": (HIDDEN, MOE_INTER),
        }.items():
            w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16)
            packed, scales = quantize_mxfp4(w)
            base = f"layers.0.ffn.experts.{e}.{which}"
            sd[base + ".weight"] = packed
            sd[base + ".scale"] = scales
            ref[(e, which)] = dequantize_mxfp4(packed, scales, torch.bfloat16)
    return sd, ref


def test_passthrough_emits_packed_keys_no_bf16():
    sd, _ = _synthetic_fp4_checkpoint()
    out = _make_adapter("mxfp4").from_hf(dict(sd))

    gu_packed = out["model.layers.0.mlp.experts.gate_and_up_projs_packed"]
    gu_scales = out["model.layers.0.mlp.experts.gate_and_up_projs_scales"]
    dn_packed = out["model.layers.0.mlp.experts.down_projs_packed"]
    dn_scales = out["model.layers.0.mlp.experts.down_projs_scales"]

    # Packed storage, never materialized to bf16.
    assert gu_packed.dtype == torch.int8 and gu_scales.dtype == torch.float8_e8m0fnu
    assert dn_packed.dtype == torch.int8 and dn_scales.dtype == torch.float8_e8m0fnu
    # gate||up concatenated along the output dim: [E, 2*moe_inter, hidden//2].
    assert tuple(gu_packed.shape) == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // 2)
    assert tuple(gu_scales.shape) == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // 32)
    assert tuple(dn_packed.shape) == (N_EXPERTS, HIDDEN, MOE_INTER // 2)
    # No bf16 expert weight keys, and no orphaned scale keys leaked through.
    assert not any(k.endswith("gate_and_up_projs") or k.endswith("down_projs") for k in out)
    assert not any(".ffn.experts." in k for k in out)


def test_passthrough_decodes_to_same_weights_as_bf16_path():
    """Unpacking the passthrough output must equal the bf16 dequant+aggregate path."""
    sd, _ = _synthetic_fp4_checkpoint()
    out_mx = _make_adapter("mxfp4").from_hf(dict(sd))
    out_bf16 = _make_adapter("bf16").from_hf(dict(sd))

    # bf16 path: gate_and_up_projs is compute layout [E, in=hidden, 2*moe_inter].
    gu_bf16 = out_bf16["model.layers.0.mlp.experts.gate_and_up_projs"]
    dn_bf16 = out_bf16["model.layers.0.mlp.experts.down_projs"]

    # passthrough: unpack [E, 2*moe_inter, hidden] then transpose to compute layout.
    gu_unpacked = dequantize_mxfp4(
        out_mx["model.layers.0.mlp.experts.gate_and_up_projs_packed"],
        out_mx["model.layers.0.mlp.experts.gate_and_up_projs_scales"],
        torch.bfloat16,
    ).transpose(-2, -1)
    dn_unpacked = dequantize_mxfp4(
        out_mx["model.layers.0.mlp.experts.down_projs_packed"],
        out_mx["model.layers.0.mlp.experts.down_projs_scales"],
        torch.bfloat16,
    ).transpose(-2, -1)

    assert torch.equal(gu_unpacked, gu_bf16)
    assert torch.equal(dn_unpacked, dn_bf16)


def test_passthrough_roundtrip_to_hf_recovers_checkpoint_keys():
    """from_hf (aggregate) -> to_hf (split) must reproduce the per-expert packed
    checkpoint keys with matching dtypes and bit-exact values — this is the path
    the DCP loader uses to enumerate destination tensors."""
    sd, _ = _synthetic_fp4_checkpoint()
    adapter = _make_adapter("mxfp4")
    model_sd = adapter.from_hf(dict(sd))
    hf = adapter.to_hf(model_sd, quantization=True)

    for e in range(N_EXPERTS):
        for w in ("w1", "w2", "w3"):
            wk = f"layers.0.ffn.experts.{e}.{w}.weight"
            sk = f"layers.0.ffn.experts.{e}.{w}.scale"
            assert wk in hf and sk in hf, f"missing {wk}/{sk}"
            assert hf[wk].dtype == torch.int8
            assert hf[sk].dtype == torch.float8_e8m0fnu
            # cat-then-split is identity: recovered packed bytes equal the original.
            assert torch.equal(hf[wk], sd[wk]), f"{wk} value mismatch"
            assert torch.equal(hf[sk].view(torch.uint8), sd[sk].view(torch.uint8)), f"{sk} value mismatch"


@pytest.mark.skipif(
    not os.path.isdir(_REAL_CKPT) or not os.path.isfile(f"{_REAL_CKPT}/model.safetensors.index.json"),
    reason=f"real DeepSeek-V4-Flash checkpoint not present at {_REAL_CKPT}",
)
def test_passthrough_against_real_checkpoint():
    """Validate passthrough on real fp4 expert tensors: correct dtypes/shapes and
    bit-exact decode vs the bf16 dequant path on actual checkpoint bytes."""
    from safetensors import safe_open

    layer, n_exp = 3, 8
    weight_map = json.load(open(f"{_REAL_CKPT}/model.safetensors.index.json"))["weight_map"]
    needed = [
        f"layers.{layer}.ffn.experts.{e}.{w}.{suffix}"
        for e in range(n_exp)
        for w in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    ]
    by_file: dict[str, list[str]] = {}
    for k in needed:
        by_file.setdefault(weight_map[k], []).append(k)
    sd = {}
    for fname, keys in by_file.items():
        with safe_open(f"{_REAL_CKPT}/{fname}", framework="pt") as h:
            for k in keys:
                sd[k] = h.get_tensor(k)

    # Real layout: int8 packed weights + float8_e8m0fnu scales.
    w1 = sd[f"layers.{layer}.ffn.experts.0.w1.weight"]
    assert w1.dtype == torch.int8
    assert sd[f"layers.{layer}.ffn.experts.0.w1.scale"].dtype == torch.float8_e8m0fnu

    cfg = DeepseekV4Config.from_pretrained(_REAL_CKPT)
    moe_config = Mock(spec=MoEConfig)
    moe_config.n_routed_experts = n_exp
    moe_config.moe_inter_dim = cfg.moe_intermediate_size

    def adapter(fmt):
        return DeepSeekV4StateDictAdapter(
            cfg, moe_config, BackendConfig(), dtype=torch.bfloat16, expert_storage_format=fmt
        )

    out_mx = adapter("mxfp4").from_hf(dict(sd))
    out_bf16 = adapter("bf16").from_hf({k: v.clone() for k, v in sd.items()})

    base = f"model.layers.{layer}.mlp.experts"
    assert out_mx[f"{base}.gate_and_up_projs_packed"].dtype == torch.int8
    assert out_mx[f"{base}.gate_and_up_projs_packed"].shape == (
        n_exp,
        2 * cfg.moe_intermediate_size,
        cfg.hidden_size // 2,
    )

    for proj in ("gate_and_up_projs", "down_projs"):
        unpacked = dequantize_mxfp4(
            out_mx[f"{base}.{proj}_packed"], out_mx[f"{base}.{proj}_scales"], torch.bfloat16
        ).transpose(-2, -1)
        assert torch.equal(unpacked, out_bf16[f"{base}.{proj}"]), f"{proj} decode mismatch vs bf16 path"
