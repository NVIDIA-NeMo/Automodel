# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.model import Block, MiniMaxM2ForCausalLM, MiniMaxM2Model
from nemo_automodel.components.moe.layers import MoE, MoEConfig


@dataclass
class MockMiniMaxM2Config:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rotary_dim: int = 8
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: dict = None
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    scoring_func: str = "sigmoid"
    use_qk_norm: bool = True
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}


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


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=32,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


class TestMiniMaxM2Block:
    def test_block_uses_moe(self, config, backend, moe_config):
        block = Block(0, config, moe_config, backend)
        assert isinstance(block.mlp, MoE)


class TestMiniMaxM2Model:
    def test_initialization(self, config, backend):
        model = MiniMaxM2Model(config, backend)
        assert len(model.layers) == config.num_hidden_layers
        assert model.config.num_experts == config.num_local_experts

    def test_forward_runs_all_layers(self, config, backend):
        model = MiniMaxM2Model(config, backend)
        batch, seq = 2, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        with patch.object(
            Block,
            "forward",
            side_effect=lambda *args, **kwargs: torch.randn(batch, seq, config.hidden_size, dtype=torch.bfloat16),
        ) as mock_block:
            out = model(input_ids)

        assert out.shape == (batch, seq, config.hidden_size)
        assert mock_block.call_count == config.num_hidden_layers


class TestMiniMaxM2ForCausalLM:
    def test_initialization(self, config, backend):
        model = MiniMaxM2ForCausalLM(config, backend=backend)
        assert model.model is not None
        assert model.lm_head is not None

    def test_forward_returns_logits(self, config, backend):
        model = MiniMaxM2ForCausalLM(config, backend=backend)
        input_ids = torch.randint(0, config.vocab_size, (2, 6))

        with patch.object(model.model, "forward", return_value=torch.randn(2, 6, config.hidden_size, dtype=torch.bfloat16)):
            logits = model(input_ids)

        assert logits.shape == (2, 6, config.vocab_size)
