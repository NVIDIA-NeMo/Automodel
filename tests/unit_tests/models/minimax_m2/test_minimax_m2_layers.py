# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.layers import MiniMaxM2Attention


@dataclass
class MockMiniMaxM2Config:
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True


@pytest.fixture
def config():
    return MockMiniMaxM2Config()


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def test_attention_forward_shape(config, backend):
    attn = MiniMaxM2Attention(config, backend)

    batch, seq = 2, 7
    x = torch.randn(batch, seq, config.hidden_size, dtype=torch.bfloat16)
    # Non-fused rope path expects freqs_cis as concatenated [cos, sin] and bshd branch uses [S, D].
    freqs_cis = torch.randn(seq, config.head_dim, dtype=torch.bfloat16)

    out = attn(x, freqs_cis=freqs_cis, attention_mask=None)

    assert out.shape == (batch, seq, config.hidden_size)


def test_attention_without_qk_norm(backend):
    config = MockMiniMaxM2Config(use_qk_norm=False)
    attn = MiniMaxM2Attention(config, backend)
    assert attn.q_norm is None
    assert attn.k_norm is None
