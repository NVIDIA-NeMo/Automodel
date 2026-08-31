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


def test_grouped_expert_layout_round_trip_and_nonexperts_are_identity(tiny_hy_v4_model):
    """Checkpoint/native expert views round-trip without changing values."""
    adapter = tiny_hy_v4_model.state_dict_adapter
    gate_up_hf = torch.randn(4, 8, 8, dtype=torch.bfloat16)
    down_hf = torch.randn(4, 8, 4, dtype=torch.bfloat16)
    passthrough = torch.randn(8, 8, dtype=torch.bfloat16)
    sink = torch.randn(2, dtype=torch.float32)
    hf_state = {
        "model.layers.1.mlp.experts.gate_up_proj": gate_up_hf,
        "model.layers.1.mlp.experts.down_proj": down_hf,
        "model.layers.1.self_attn.q_a_proj.weight": passthrough,
        "model.layers.1.self_attn.learnable_sink_param": sink,
    }

    native = adapter.from_hf(dict(hf_state))

    gate_up_native = native["model.layers.1.mlp.experts.gate_and_up_projs"]
    down_native = native["model.layers.1.mlp.experts.down_projs"]
    assert gate_up_native.shape == (4, 8, 8)
    assert down_native.shape == (4, 4, 8)
    torch.testing.assert_close(gate_up_native, gate_up_hf.transpose(-2, -1))
    torch.testing.assert_close(down_native, down_hf.transpose(-2, -1))
    assert native["model.layers.1.self_attn.q_a_proj.weight"] is passthrough
    assert native["model.layers.1.self_attn.learnable_sink_param.weight"] is sink

    restored = adapter.to_hf(native)
    assert restored.keys() == hf_state.keys()
    for key, value in hf_state.items():
        torch.testing.assert_close(restored[key], value, rtol=0.0, atol=0.0)


def test_mtp_grouped_experts_use_the_same_checkpoint_conversion(tiny_hy_v4_model):
    """MTP grouped weights use the same transposed tensor layout as the backbone."""
    adapter = tiny_hy_v4_model.state_dict_adapter
    hf_tensor = torch.randn(4, 8, 4, dtype=torch.bfloat16)
    key = "model.mtp_layers.0.mlp.experts.down_proj"

    native = adapter.from_hf({key: hf_tensor})

    native_key = "model.mtp_layers.0.mlp.experts.down_projs"
    torch.testing.assert_close(native[native_key], hf_tensor.transpose(-2, -1))
    torch.testing.assert_close(adapter.to_hf(native)[key], hf_tensor)


def test_checkpoint_views_write_through_native_storage(tiny_hy_v4_model):
    """Nonexpert, transposed expert, and sink checkpoint tensors alias native storage."""
    adapter = tiny_hy_v4_model.state_dict_adapter
    gate_up_native = torch.zeros(4, 8, 8, dtype=torch.bfloat16)
    down_native = torch.zeros(4, 4, 8, dtype=torch.bfloat16)
    passthrough = torch.zeros(8, 8, dtype=torch.bfloat16)
    sink_native = torch.zeros(2, dtype=torch.float32)
    native = {
        "model.layers.1.mlp.experts.gate_and_up_projs": gate_up_native,
        "model.layers.1.mlp.experts.down_projs": down_native,
        "model.layers.1.self_attn.q_a_proj.weight": passthrough,
        "model.layers.1.self_attn.learnable_sink_param.weight": sink_native,
    }

    checkpoint = adapter.to_hf(native)
    gate_up_checkpoint = checkpoint["model.layers.1.mlp.experts.gate_up_proj"]
    down_checkpoint = checkpoint["model.layers.1.mlp.experts.down_proj"]
    passthrough_checkpoint = checkpoint["model.layers.1.self_attn.q_a_proj.weight"]
    sink_checkpoint = checkpoint["model.layers.1.self_attn.learnable_sink_param"]

    assert gate_up_checkpoint.untyped_storage().data_ptr() == gate_up_native.untyped_storage().data_ptr()
    assert down_checkpoint.untyped_storage().data_ptr() == down_native.untyped_storage().data_ptr()
    assert passthrough_checkpoint is passthrough
    assert sink_checkpoint is sink_native

    gate_up_checkpoint[1, 2, 3] = 11
    down_checkpoint[1, 2, 3] = 13
    passthrough_checkpoint[2, 3] = 17
    sink_checkpoint[1] = 19

    assert gate_up_native[1, 3, 2].item() == 11
    assert down_native[1, 3, 2].item() == 13
    assert passthrough[2, 3].item() == 17
    assert sink_native[1].item() == 19


def test_checkpoint_fast_path_is_limited_to_view_preserving_expert_backends(tiny_hy_v4_model):
    """Allocating expert implementations cannot advertise write-through loading."""
    adapter = tiny_hy_v4_model.state_dict_adapter

    assert adapter.supports_write_through_checkpoint_load
    assert adapter.supports_checkpoint_load_without_full_copy

    adapter.backend.experts = "te"
    assert not adapter.supports_write_through_checkpoint_load
    assert not adapter.supports_checkpoint_load_without_full_copy

    adapter.backend.experts = "torch_mm"
    adapter.backend.dispatcher = "mok"
    assert not adapter.supports_write_through_checkpoint_load
    assert not adapter.supports_checkpoint_load_without_full_copy
