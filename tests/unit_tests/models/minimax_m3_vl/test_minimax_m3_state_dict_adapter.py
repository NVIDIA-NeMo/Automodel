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


def test_from_hf_maps_index_and_drops_mtp(model):
    adapter = model.state_dict_adapter
    hidden = model.config.hidden_size
    hf = adapter.to_hf(model.state_dict())
    # Sparse-attention index branch (Stage 2): mapped under indexer.*, FP8 dequantized.
    hf["model.layers.1.self_attn.index_q_proj.weight"] = torch.zeros(8, hidden).to(torch.float8_e4m3fn)
    hf["model.layers.1.self_attn.index_q_proj.weight_scale_inv"] = torch.full((8, hidden // 32), 127, dtype=torch.uint8)
    hf["model.layers.1.self_attn.index_q_norm.weight"] = torch.zeros(8)
    # MTP (Stage 4): still dropped.
    hf["model.mtp.layers.0.eh_proj.weight"] = torch.zeros(hidden, 2 * hidden)

    native = adapter.from_hf(hf)
    assert "model.layers.1.self_attn.indexer.index_q_proj.weight" in native
    assert "model.layers.1.self_attn.indexer.index_q_norm.weight" in native
    assert not any(k.endswith("_scale_inv") for k in native)
    assert not any(".mtp." in k for k in native)


def test_mxfp8_dequant_e8m0():
    out_dim, in_dim, block = 2, 64, 32
    w_fp8 = (torch.randn(out_dim, in_dim) * 0.1).to(torch.float8_e4m3fn)
    # e8m0 exponents: scale = 2 ** (uint8 - 127). Row 0 -> scale 1, row 1 -> scale 2.
    scale_inv = torch.tensor([[127] * (in_dim // block), [128] * (in_dim // block)], dtype=torch.uint8)
    deq = dequantize_mxfp8(w_fp8, scale_inv, dtype=torch.float32)
    assert torch.allclose(deq[0], w_fp8[0].float() * 1.0)
    assert torch.allclose(deq[1], w_fp8[1].float() * 2.0)


def test_to_hf_quantization_emits_scale_inv_for_quantized_keys(model):
    """to_hf(quantization=True) must re-emit the MXFP8 key-set (e4m3 weight +
    e8m0 scale_inv) so the load planner requests the checkpoint's scales -- but
    only for quantized projections, never for norms/embed/lm_head/router gate."""
    adapter = model.state_dict_adapter
    hf = adapter.to_hf(model.state_dict(), quantization=True)

    # Quantized projections: a companion _scale_inv exists; weight is e4m3, scale uint8.
    proj_keys = [
        k
        for k in hf
        if k.endswith(".weight") and (".self_attn.q_proj" in k or ".mlp.gate_proj" in k or ".experts." in k)
    ]
    assert proj_keys, "expected some quantized projection keys"
    for k in proj_keys:
        assert k + "_scale_inv" in hf, f"missing scale_inv for {k}"
        assert hf[k].dtype == torch.float8_e4m3fn
        si = hf[k + "_scale_inv"]
        assert si.dtype == torch.uint8
        # e8m0 block scale: [out, ceil(in/32)]
        assert si.shape == (hf[k].shape[-2], (hf[k].shape[-1] + 31) // 32)

    # Non-quantized keys must NOT get a scale_inv.
    for k in hf:
        if k.endswith(".weight") and (
            "norm" in k or k.endswith("embed_tokens.weight") or "lm_head" in k or ".block_sparse_moe.gate.weight" in k
        ):
            assert k + "_scale_inv" not in hf, f"unexpected scale_inv for non-quantized {k}"

    # No scale_inv leaks when quantization is off.
    hf_plain = adapter.to_hf(model.state_dict(), quantization=False)
    assert not any(k.endswith("_scale_inv") for k in hf_plain)


def test_from_hf_dequantizes_loaded_scale_inv(model):
    """from_hf must dequantize the e4m3 weight + scale_inv pairs the load produces,
    yielding a finite native weight (not the raw ~e4m3-magnitude values)."""
    adapter = model.state_dict_adapter
    native = model.state_dict()
    hf_q = adapter.to_hf(native, quantization=True)  # e4m3 + scale_inv (as the load delivers)
    back = adapter.from_hf({k: v.clone() if hasattr(v, "clone") else v for k, v in hf_q.items()})

    qkey = next(k for k in native if k.endswith(".self_attn.q_proj.weight"))
    assert qkey in back
    w = back[qkey]
    assert w.dtype == adapter.dtype  # dequantized to model dtype, not float8
    assert torch.isfinite(w.float()).all()
    # placeholder scale_inv is 127 (scale 1.0), so dequant ~= the e4m3-rounded weight,
    # i.e. bounded -- emphatically NOT left as raw ~±448 e4m3 magnitudes.
    assert w.float().abs().max().item() < 50.0
