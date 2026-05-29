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

from nemo_automodel.components.models.minimax_m3_vl.state_dict_adapter import dequantize_mxfp8


def test_native_to_hf_key_mapping(model):
    adapter = model.state_dict_adapter
    hf = adapter.to_hf(model.state_dict())

    # Routed experts split into per-expert w1/w2/w3 under block_sparse_moe.
    assert "model.layers.1.block_sparse_moe.experts.0.w1.weight" in hf
    assert "model.layers.1.block_sparse_moe.experts.0.w2.weight" in hf
    assert "model.layers.1.block_sparse_moe.experts.0.w3.weight" in hf
    # Router + correction bias.
    assert "model.layers.1.block_sparse_moe.gate.weight" in hf
    assert "model.layers.1.block_sparse_moe.e_score_correction_bias" in hf
    # Shared expert lives under block_sparse_moe in HF layout.
    assert "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight" in hf
    # Dense layer keeps plain mlp.* naming (no block_sparse_moe).
    assert "model.layers.0.mlp.gate_proj.weight" in hf
    assert not any("layers.0.block_sparse_moe" in k for k in hf)


def test_state_dict_round_trip_exact(model):
    adapter = model.state_dict_adapter
    native = {k: v.clone() for k, v in model.state_dict().items()}
    back = adapter.from_hf(adapter.to_hf(native))

    assert set(back.keys()) == set(native.keys()), (
        set(native) - set(back),
        set(back) - set(native),
    )
    for key in native:
        a, b = native[key].float(), back[key].float()
        assert a.shape == b.shape, (key, a.shape, b.shape)
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (key, (a - b).abs().max().item())


def test_from_hf_drops_index_and_mtp_keys(model):
    adapter = model.state_dict_adapter
    hf = adapter.to_hf(model.state_dict())
    # Inject Stage-2/4 tensors that the Stage-1 model does not own yet.
    hf["model.layers.1.self_attn.index_q_proj.weight"] = torch.zeros(8, model.config.hidden_size)
    hf["model.layers.1.self_attn.index_q_proj.weight_scale_inv"] = torch.zeros(8, 2, dtype=torch.uint8)
    hf["model.mtp.layers.0.eh_proj.weight"] = torch.zeros(model.config.hidden_size, 2 * model.config.hidden_size)

    native = adapter.from_hf(hf)
    assert not any("index_" in k for k in native)
    assert not any(".mtp." in k for k in native)


def test_mxfp8_dequant_e8m0():
    out_dim, in_dim, block = 2, 64, 32
    w_fp8 = (torch.randn(out_dim, in_dim) * 0.1).to(torch.float8_e4m3fn)
    # e8m0 exponents: scale = 2 ** (uint8 - 127). Row 0 -> scale 1, row 1 -> scale 2.
    scale_inv = torch.tensor([[127] * (in_dim // block), [128] * (in_dim // block)], dtype=torch.uint8)
    deq = dequantize_mxfp8(w_fp8, scale_inv, dtype=torch.float32)
    assert torch.allclose(deq[0], w_fp8[0].float() * 1.0)
    assert torch.allclose(deq[1], w_fp8[1].float() * 2.0)
