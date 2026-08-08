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

from types import SimpleNamespace

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.state_dict_adapter import (
    KimiK3StateDictAdapter,
    _route_kda_fp32_holder,
    _strip_kda_fp32_holder,
    dequantize_mxfp4,
)
from nemo_automodel.components.moe.config import MoEConfig


def _tiny_adapter():
    moe = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=16,
        norm_topk_prob=True,
    )
    backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
    return KimiK3StateDictAdapter(SimpleNamespace(), moe, backend)


def _peft_lora_state_dict(rank=8, n_experts=4, dim=64, inter=16):
    # grouped expert LoRA params exactly as ModelState.state_dict() emits them on a
    # PEFT save, shapes matching GroupedExpertsLoRA (lora_experts.py): A is
    # [experts, in_features, rank], B is [experts, rank, out_features]. Plus one
    # attention LoRA key for contrast.
    base = "base_model.model.model.layers.5.mlp.experts"
    attn = "base_model.model.model.layers.5.self_attn.q_proj"
    return {
        f"{base}.lora_gate_and_up_A": torch.randn(n_experts, dim, rank),
        f"{base}.lora_gate_and_up_B": torch.randn(n_experts, rank, 2 * inter),
        f"{base}.lora_down_A": torch.randn(n_experts, inter, rank),
        f"{base}.lora_down_B": torch.randn(n_experts, rank, dim),
        f"{attn}.lora_A.weight": torch.randn(rank, dim),
    }


def test_peft_expert_lora_keys_get_hf_renames():
    """Expert LoRA keys must use the checkpoint's real layout, like the weights do.

    Before the fix they skipped every rename and came out as
    ``base_model.model.model.layers.N.mlp.experts.E.gate_proj.lora_A.weight`` —
    names that exist nowhere in the actual model, so PEFT/vLLM could not attach
    the saved adapter.
    """
    adapter = _tiny_adapter()
    out = adapter.to_hf(_peft_lora_state_dict())

    expert_keys = [k for k in out if ".lora_" in k and "self_attn" not in k]
    assert expert_keys, "expert LoRA keys missing from the save"
    for key in expert_keys:
        assert key.startswith("base_model.model.language_model.model.layers."), key
        assert ".block_sparse_moe.experts." in key, key
    assert any(".w1.lora_A.weight" in k for k in expert_keys)
    assert any(".w2.lora_B.weight" in k for k in expert_keys)
    # attention LoRA gets the language_model. namespace inside the PEFT prefix too
    attn_keys = [k for k in out if "self_attn" in k]
    assert attn_keys == ["base_model.model.language_model.model.layers.5.self_attn.q_proj.lora_A.weight"]


def test_peft_lora_save_round_trips_for_resume():
    """from_hf must rebuild the exact grouped keys ModelState expects on resume."""
    adapter = _tiny_adapter()
    sd = _peft_lora_state_dict()
    out = adapter.to_hf({k: v.clone() for k, v in sd.items()})
    back = adapter.from_hf(dict(out))

    assert set(back) == set(sd)
    for key in sd:
        torch.testing.assert_close(back[key], sd[key])


def test_expert_weight_save_layout_unchanged():
    """Full-fine-tune expert weights keep their existing rename behavior."""
    adapter = _tiny_adapter()
    out = adapter.to_hf({"model.layers.5.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 32)})

    assert len(out) == 8  # 4 experts x (w1 + w3)
    assert all(k.startswith("language_model.model.layers.5.block_sparse_moe.experts.") for k in out)
    assert all(k.endswith((".w1.weight", ".w3.weight")) for k in out)


def test_kda_fp32_holder_keys_round_trip_to_hf_layout():
    hf_to_native = {
        "model.layers.9.self_attn.A_log": "model.layers.9.self_attn._fp32_params.A_log",
        "model.layers.9.self_attn.dt_bias": "model.layers.9.self_attn._fp32_params.dt_bias",
        "model.layers.9.self_attn.q_conv1d.weight": ("model.layers.9.self_attn.q_conv1d._fp32_params.weight"),
        "model.layers.9.self_attn.k_conv1d.weight": ("model.layers.9.self_attn.k_conv1d._fp32_params.weight"),
        "model.layers.9.self_attn.v_conv1d.weight": ("model.layers.9.self_attn.v_conv1d._fp32_params.weight"),
        "model.layers.9.self_attn.o_norm.weight": "model.layers.9.self_attn.o_norm._fp32_params.weight",
    }

    for hf_key, native_key in hf_to_native.items():
        assert _route_kda_fp32_holder(hf_key) == native_key
        assert _route_kda_fp32_holder(native_key) == native_key
        assert _strip_kda_fp32_holder(native_key) == hf_key


def test_mxfp4_load_dequantizes_directly_into_noncontiguous_model_view():
    adapter = object.__new__(KimiK3StateDictAdapter)
    adapter.dtype = torch.float32

    base = "model.layers.1.block_sparse_moe.experts.0.w1.weight"
    packed = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    expected = dequantize_mxfp4(packed, scales, dtype=torch.float32)

    storage = torch.empty(32, 2)
    destination = storage.t()
    assert not destination.is_contiguous()
    adapter._mxfp4_load_views = {base: destination}
    state_dict = {
        f"{base}_packed": packed,
        f"{base}_scale": scales,
    }

    with torch.no_grad():
        adapter._dequantize_packed_experts(state_dict)

    assert list(state_dict) == [base]
    assert state_dict[base] is destination
    assert state_dict[base].data_ptr() == storage.data_ptr()
    torch.testing.assert_close(destination, expected)
    assert not hasattr(adapter, "_mxfp4_load_views")


def test_mxfp4_load_without_model_view_returns_decoded_tensor():
    adapter = object.__new__(KimiK3StateDictAdapter)
    adapter.dtype = torch.bfloat16

    base = "model.layers.1.block_sparse_moe.experts.0.w2.weight"
    packed = torch.zeros((1, 16), dtype=torch.uint8)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    state_dict = {
        f"{base}_packed": packed,
        f"{base}_scale": scales,
    }

    adapter._dequantize_packed_experts(state_dict)

    assert list(state_dict) == [base]
    assert state_dict[base].shape == (1, 32)
    assert state_dict[base].dtype == torch.bfloat16
