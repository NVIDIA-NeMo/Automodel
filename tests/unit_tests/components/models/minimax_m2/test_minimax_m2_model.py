# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
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
        dispatcher="torch",
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
        gate_bias_update_factor=1e-3,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


@dataclass
class MockMiniMaxM2ConfigNoRopeParams:
    """Simulates a real HF MiniMaxM2Config that lacks a rope_parameters attribute."""

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
    rope_theta: float = 5000000.0
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    scoring_func: str = "sigmoid"
    use_qk_norm: bool = True
    torch_dtype: str = "bfloat16"


class TestRopeParametersFallback:
    """Tests for the rope_parameters fallback when the HF config lacks it."""

    def test_rope_parameters_constructed_when_missing(self, backend):
        """When config has no rope_parameters attr, MiniMaxM2Model should construct it."""
        cfg = MockMiniMaxM2ConfigNoRopeParams()
        assert not hasattr(cfg, "rope_parameters")
        MiniMaxM2Model(cfg, backend)
        assert hasattr(cfg, "rope_parameters")
        assert cfg.rope_parameters["rope_theta"] == 5000000.0
        assert cfg.rope_parameters["rope_type"] == "default"
        assert cfg.rope_parameters["partial_rotary_factor"] == pytest.approx(8 / 16)

    def test_rope_parameters_constructed_when_none(self, backend):
        """When config.rope_parameters is explicitly None, it should be populated."""
        cfg = MockMiniMaxM2Config(rope_theta=500000.0)
        cfg.rope_parameters = None
        MiniMaxM2Model(cfg, backend)
        assert cfg.rope_parameters is not None
        assert cfg.rope_parameters["rope_theta"] == 500000.0

    def test_rope_parameters_preserved_when_present(self, backend):
        """When config already has rope_parameters, they should not be overwritten."""
        custom_params = {"rope_theta": 42.0, "rope_type": "custom", "partial_rotary_factor": 0.25}
        cfg = MockMiniMaxM2Config()
        cfg.rope_parameters = custom_params
        MiniMaxM2Model(cfg, backend)
        assert cfg.rope_parameters is custom_params

    def test_partial_rotary_factor_computation(self, backend):
        """partial_rotary_factor should be rotary_dim / head_dim."""
        cfg = MockMiniMaxM2ConfigNoRopeParams(rotary_dim=4, head_dim=16)
        MiniMaxM2Model(cfg, backend)
        assert cfg.rope_parameters["partial_rotary_factor"] == pytest.approx(0.25)

    def test_defaults_when_rotary_dim_missing(self, backend):
        """When rotary_dim is absent, partial_rotary_factor should default to 1.0 (head_dim/head_dim)."""
        cfg = SimpleNamespace(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            num_local_experts=4,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            use_qk_norm=True,
            torch_dtype="bfloat16",
        )
        MiniMaxM2Model(cfg, backend)
        assert cfg.rope_parameters["partial_rotary_factor"] == pytest.approx(1.0)

    def test_defaults_when_rope_theta_missing(self, backend):
        """When rope_theta is absent, it should default to 5000000.0."""
        cfg = SimpleNamespace(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rotary_dim=8,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            num_local_experts=4,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            use_qk_norm=True,
            torch_dtype="bfloat16",
        )
        MiniMaxM2Model(cfg, backend)
        assert cfg.rope_parameters["rope_theta"] == 5000000.0


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

        with patch.object(
            model.model, "forward", return_value=torch.randn(2, 6, config.hidden_size, dtype=torch.bfloat16)
        ):
            out = model(input_ids)

        # forward returns a CausalLMOutputWithPast; downstream reads getattr(out, "logits", out).
        logits = getattr(out, "logits", out)
        assert logits.shape == (2, 6, config.vocab_size)

    def test_forward_thd_hidden_states_match_logits_layout(self, config, backend):
        model = MiniMaxM2ForCausalLM(config, backend=backend).to(torch.float32)
        batch, seq = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))
        position_ids = torch.arange(seq).unsqueeze(0)
        hidden = torch.randn(seq, config.hidden_size)
        with patch.object(model.model, "forward", return_value=hidden):
            out = model(
                input_ids,
                position_ids=position_ids,
                qkv_format="thd",
                output_hidden_states=True,
            )
        assert out.logits.shape == (batch, seq, config.vocab_size)
        assert out.hidden_states.shape == (batch, seq, config.hidden_size)

    def test_router_correction_bias_stays_fp32_after_bf16_cast(self, config, backend):
        model = MiniMaxM2ForCausalLM(config, backend=backend)
        expected = {}
        for name, buf in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                values = torch.linspace(0.00123, 0.00456, buf.numel(), dtype=torch.float32).reshape_as(buf)
                buf.copy_(values)
                expected[name] = values

        assert expected, "MiniMax M2 should create router correction-bias buffers"

        cast_model_to_dtype(model, torch.bfloat16)

        buffers = dict(model.named_buffers())
        for name, values in expected.items():
            assert buffers[name].dtype == torch.float32
            torch.testing.assert_close(buffers[name], values)
        assert model.lm_head.weight.dtype == torch.bfloat16


class TestMoeOverrides:
    """Tests for the moe_overrides dict pattern."""

    def test_moe_overrides_applied(self, config, backend):
        """moe_overrides should merge into the default MoEConfig."""
        model = MiniMaxM2Model(config, backend, moe_overrides={"gate_bias_update_factor": 0.5})
        assert model.moe_config.gate_bias_update_factor == 0.5

    def test_moe_overrides_preserves_defaults(self, config, backend):
        """Unspecified fields should keep their defaults."""
        model = MiniMaxM2Model(config, backend, moe_overrides={"gate_bias_update_factor": 0.5})
        assert model.moe_config.aux_loss_coeff == 0

    def test_moe_config_and_moe_overrides_raises(self, config, backend, moe_config):
        """Passing both moe_config and moe_overrides should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot pass both"):
            MiniMaxM2Model(config, backend, moe_config=moe_config, moe_overrides={"gate_bias_update_factor": 0.5})

    def test_moe_overrides_via_causal_lm(self, config, backend):
        """moe_overrides should flow from ForCausalLM to inner model."""
        model = MiniMaxM2ForCausalLM(config, backend=backend, moe_overrides={"gate_bias_update_factor": 0.5})
        assert model.model.moe_config.gate_bias_update_factor == 0.5
