# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config

from nemo_automodel.components.models.bagel.configuration import resolve_bagel_backend
from nemo_automodel.components.models.bagel.modeling_qwen2_packed import (
    Qwen2ForCausalLM,
    _apply_qk_norm,
)


def _config() -> Qwen2Config:
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
    return config


def test_partial_backend_mapping_inherits_bagel_defaults() -> None:
    backend = resolve_bagel_backend({"rms_norm": "te"})

    assert backend.linear == "torch"
    assert backend.rms_norm == "te"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"linear": "invalid"}, "linear backend"),
        ({"rms_norm": "invalid"}, "RMSNorm backend"),
    ],
)
def test_bagel_backend_rejects_invalid_values(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_bagel_backend(override)


def test_bagel_backend_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match="Unknown BAGEL backend field"):
        resolve_bagel_backend({"unsupported": True})


def test_lm_head_uses_configured_linear_backend(monkeypatch) -> None:
    import nemo_automodel.components.models.bagel.modeling_qwen2_packed as qwen_module

    def _fake_initialize_linear_module(_backend, in_features, out_features, *, bias, dtype):
        del dtype
        module = nn.Linear(in_features, out_features, bias=bias)
        module.test_backend = _backend
        return module

    monkeypatch.setattr(qwen_module, "initialize_linear_module", _fake_initialize_linear_module)
    backend = resolve_bagel_backend({"linear": "te"})

    model = Qwen2ForCausalLM(_config(), backend=backend)

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


def test_te_mot_forward_backward_smoke() -> None:
    pytest.importorskip("transformer_engine")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2MoTDecoderLayer

    config = _config()
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
