# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config

from nemo_automodel.components.models.bagel.backend import resolve_bagel_backend
from nemo_automodel.components.models.bagel.hf_backbone_loader import _qwen_projection_fusion_layout
from nemo_automodel.components.models.bagel.modeling_qwen2_packed import (
    PackedAttention,
    Qwen2ForCausalLM,
    Qwen2MLP,
    _apply_qk_norm,
)
from nemo_automodel.components.models.common import BackendConfig

_TORCH_BACKEND = BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32", rope_fusion=False)


def _config(
    *,
    fused: bool = False,
    fused_qkv: bool | None = None,
    fused_gate_up: bool | None = None,
) -> Qwen2Config:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    config.qk_norm = True
    config.fused_projections = fused
    config.fused_qkv_projections = fused if fused_qkv is None else fused_qkv
    config.fused_gate_up_projections = fused if fused_gate_up is None else fused_gate_up
    return config


def test_mlp_projection_fusion_is_configurable_and_numerically_equivalent() -> None:
    torch.manual_seed(7)
    split = Qwen2MLP(_config(fused=False), backend=_TORCH_BACKEND)
    fused = Qwen2MLP(_config(fused_gate_up=True), backend=_TORCH_BACKEND)
    with torch.no_grad():
        fused.gate_up_proj.weight.copy_(torch.cat([split.gate_proj.weight, split.up_proj.weight], dim=0))
        fused.down_proj.weight.copy_(split.down_proj.weight)

    hidden_states = torch.randn(3, 5, split.hidden_size)

    torch.testing.assert_close(fused(hidden_states), split(hidden_states), atol=1e-6, rtol=1e-5)
    assert hasattr(split, "gate_proj") and not hasattr(split, "gate_up_proj")
    assert hasattr(fused, "gate_up_proj") and not hasattr(fused, "gate_proj")


def test_attention_projection_fusion_uses_q_k_v_row_order() -> None:
    torch.manual_seed(11)
    split = PackedAttention(_config(fused=False), layer_idx=0, backend=_TORCH_BACKEND)
    fused = PackedAttention(_config(fused_qkv=True), layer_idx=0, backend=_TORCH_BACKEND)
    with torch.no_grad():
        fused.qkv_proj.weight.copy_(torch.cat([split.q_proj.weight, split.k_proj.weight, split.v_proj.weight], dim=0))
        fused.qkv_proj.bias.copy_(torch.cat([split.q_proj.bias, split.k_proj.bias, split.v_proj.bias], dim=0))

    hidden_states = torch.randn(9, split.hidden_size)
    qkv = fused.qkv_proj(hidden_states)
    fused_q, fused_k, fused_v = qkv.split([fused.q_size, fused.k_size, fused.v_size], dim=-1)

    torch.testing.assert_close(fused_q, split.q_proj(hidden_states), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused_k, split.k_proj(hidden_states), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused_v, split.v_proj(hidden_states), atol=1e-6, rtol=1e-5)
    assert hasattr(split, "q_proj") and not hasattr(split, "qkv_proj")
    assert hasattr(fused, "qkv_proj") and not hasattr(fused, "q_proj")


@pytest.mark.parametrize("fused_qkv,fused_gate_up", [(False, False), (True, False), (False, True), (True, True)])
def test_hf_backbone_loader_detects_target_projection_layout(fused_qkv, fused_gate_up) -> None:
    model = Qwen2ForCausalLM(
        _config(fused_qkv=fused_qkv, fused_gate_up=fused_gate_up),
        backend=_TORCH_BACKEND,
    )

    assert _qwen_projection_fusion_layout(model) == (fused_qkv, fused_gate_up)


def test_partial_backend_mapping_inherits_bagel_defaults() -> None:
    backend = resolve_bagel_backend({"rms_norm": "te"})

    assert backend.attn == "flex"
    assert backend.linear == "torch"
    assert backend.rms_norm == "te"
    assert backend.rope_fusion is False


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"attn": "te"}, "attn must be 'flex'"),
        ({"rope_fusion": True}, "rope_fusion=True"),
        ({"compile_attn": True}, "compile_attn=True"),
        ({"linear": "te", "te_fp8": {"recipe": "current"}}, "te_fp8 is set"),
    ],
)
def test_bagel_backend_rejects_accepted_but_unimplemented_options(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_bagel_backend(override)


def test_lm_head_uses_configured_linear_backend(monkeypatch) -> None:
    import nemo_automodel.components.models.bagel.modeling_qwen2_packed as qwen_module

    def _fake_initialize_linear_module(_backend, in_features, out_features, *, bias, dtype):
        del dtype
        module = nn.Linear(in_features, out_features, bias=bias)
        module.test_backend = _backend
        return module

    monkeypatch.setattr(qwen_module, "initialize_linear_module", _fake_initialize_linear_module)
    backend = resolve_bagel_backend({"linear": "te"})

    model = Qwen2ForCausalLM(_config(fused=False), backend=backend)

    assert model.lm_head.test_backend == "te"


def test_te_qk_norm_keeps_explicit_fp32_inference_path() -> None:
    class _UnexpectedTENorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

        def forward(self, hidden_states):
            raise AssertionError("FP32 QK norm should not dispatch to TE")

    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    backend = resolve_bagel_backend({"rms_norm": "te"})

    output = _apply_qk_norm(_UnexpectedTENorm(), hidden_states, backend=backend, eps=1e-6)

    expected = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("fused_qkv,fused_gate_up", [(False, False), (True, False), (False, True), (True, True)])
def test_te_mot_forward_backward_smoke(fused_qkv, fused_gate_up) -> None:
    pytest.importorskip("transformer_engine")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2MoTDecoderLayer

    config = _config(fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
    config.freeze_und = False
    backend = resolve_bagel_backend({"linear": "te", "rms_norm": "te"})
    layer = Qwen2MoTDecoderLayer(config, layer_idx=0, backend=backend).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer.train()

    hidden_states = torch.randn(8, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    und_indexes = torch.arange(0, 4, device="cuda")
    gen_indexes = torch.arange(4, 8, device="cuda")
    head_dim = config.hidden_size // config.num_attention_heads
    cos = torch.ones(8, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    attention_mask = [torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = layer(
            packed_sequence=hidden_states,
            sample_lens=[4, 4],
            attention_mask=attention_mask,
            packed_position_embeddings=(cos, sin),
            packed_und_token_indexes=und_indexes,
            packed_gen_token_indexes=gen_indexes,
        )
        loss = output.float().square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
