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

import torch

from nemo_automodel.components.models.kimi_k3.state_dict_adapter import (
    KimiK3StateDictAdapter,
    _route_kda_fp32_holder,
    _strip_kda_fp32_holder,
    dequantize_mxfp4,
)


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
