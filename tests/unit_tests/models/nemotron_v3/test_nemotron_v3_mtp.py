# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for Nemotron-V3 Multi-Token Prediction (MTP) wiring.

Covers the model-agnostic scaffolding (``roll_tensor``, pattern parsing,
loss aggregation) and the NemotronV3-specific glue (forward emits
``mtp_loss``, gradients flow into MTP params, state-dict carries the
expected ``mtp.*`` keys).
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.mtp import (
    MTPConfig,
    parse_mtp_layer_pattern,
    roll_tensor,
)


class MockNemotronV3Config:
    """Minimal NemotronV3-compatible config for CPU unit testing."""

    def __init__(self, **overrides):
        # Core dims
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 16
        self.hidden_size = 64
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP / MoE
        self.intermediate_size = 64
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.moe_intermediate_size = 32
        self.moe_shared_expert_intermediate_size = 32

        # Mamba (unused when no mamba layers, but referenced by config)
        self.mamba_num_heads = 2
        self.mamba_head_dim = 16
        self.ssm_state_size = 8
        self.n_groups = 1
        self.chunk_size = 64
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # General
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 2
        self.vocab_size = 64
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False
        self.layers_block_type = ["attention", "moe"]

        # MoE routing
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.norm_topk_prob = False

        # MTP fields default to disabled; tests opt in via overrides.
        self.num_nextn_predict_layers = 0
        self.mtp_hybrid_override_pattern = ""

        for key, value in overrides.items():
            setattr(self, key, value)

    def to_dict(self):
        return vars(self)


# ---------------------------------------------------------------------------
# Pattern parsing & rolling
# ---------------------------------------------------------------------------


class TestPatternParsing:
    def test_parse_star_e(self):
        assert parse_mtp_layer_pattern("*E") == ["attention", "moe"]

    def test_parse_complex_pattern(self):
        assert parse_mtp_layer_pattern("M*-E") == ["mamba", "attention", "mlp", "moe"]

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="Unknown MTP layer symbol"):
            parse_mtp_layer_pattern("X")

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_mtp_layer_pattern("")


class TestRollTensor:
    def test_left_shift_zeros_trailing(self):
        t = torch.arange(8).unsqueeze(0)  # [1, 8]
        rolled = roll_tensor(t, shifts=-1, dim=-1)
        assert rolled.tolist() == [[1, 2, 3, 4, 5, 6, 7, 0]]

    def test_double_left_shift_zeros_two_trailing(self):
        t = torch.arange(8).unsqueeze(0)
        rolled1 = roll_tensor(t, shifts=-1, dim=-1)
        rolled2 = roll_tensor(rolled1, shifts=-1, dim=-1)
        assert rolled2.tolist() == [[2, 3, 4, 5, 6, 7, 0, 0]]

    def test_zero_shift_is_identity(self):
        t = torch.arange(8).unsqueeze(0)
        assert torch.equal(roll_tensor(t, shifts=0, dim=-1), t)

    def test_right_shift_zeros_leading(self):
        t = torch.arange(8).unsqueeze(0)
        rolled = roll_tensor(t, shifts=1, dim=-1)
        assert rolled.tolist() == [[0, 0, 1, 2, 3, 4, 5, 6]]


class TestMTPConfig:
    def test_disabled_when_zero_layers(self):
        c = MTPConfig(num_layers=0, layer_pattern="*E")
        assert not c.enabled

    def test_disabled_when_empty_pattern(self):
        c = MTPConfig(num_layers=1, layer_pattern="")
        assert not c.enabled

    def test_enabled_and_total_sublayers(self):
        c = MTPConfig(num_layers=2, layer_pattern="*E")
        assert c.enabled
        assert c.pattern_length == 2
        assert c.total_sublayers == 4


# ---------------------------------------------------------------------------
# End-to-end forward / backward
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
    )


def _make_model(backend, *, mtp_layers=0, mtp_pattern="", **cfg_overrides):
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    config = MockNemotronV3Config(
        num_nextn_predict_layers=mtp_layers,
        mtp_hybrid_override_pattern=mtp_pattern,
        **cfg_overrides,
    )
    model = NemotronHForCausalLM(config, backend=backend)
    return model.to(torch.bfloat16), config


class TestMTPDisabled:
    def test_no_mtp_when_config_omits_fields(self, backend):
        """When num_nextn_predict_layers is 0, self.mtp must be None and
        forward must not emit per-depth hidden states."""
        model, config = _make_model(backend)
        assert model.mtp is None

        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert getattr(out, "mtp_per_depth_h", None) is None
        assert getattr(out, "mtp_loss_scaling_factor", None) is None
        assert out.loss is not None
        assert out.logits.shape == (2, 8, config.vocab_size)


class TestMTPEnabled:
    def test_module_built_with_correct_layer_count(self, backend):
        model, _ = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        assert model.mtp is not None
        # D=1, P=2 -> 2 flat sublayers; fusion on idx 0, final_layernorm on idx 1.
        assert len(model.mtp.layers) == 2
        assert model.mtp.layers[0].has_fusion is True
        assert model.mtp.layers[0].has_final_norm is False
        assert model.mtp.layers[1].has_fusion is False
        assert model.mtp.layers[1].has_final_norm is True
        assert model.mtp.layers[0].block_type == "attention"
        assert model.mtp.layers[1].block_type == "moe"

    def test_fusion_modules_present_on_first_sublayer_only(self, backend):
        model, _ = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        first = model.mtp.layers[0]
        last = model.mtp.layers[1]
        assert hasattr(first, "enorm") and hasattr(first, "hnorm") and hasattr(first, "eh_proj")
        assert not hasattr(last, "enorm")
        assert hasattr(last, "final_layernorm")
        assert not hasattr(first, "final_layernorm")

    def test_forward_train_emits_mtp_per_depth_h(self, backend):
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert out.mtp_per_depth_h is not None
        assert isinstance(out.mtp_per_depth_h, list)
        # D=1 in this fixture
        assert len(out.mtp_per_depth_h) == 1
        # Each per-depth tensor must be on the autograd graph so the
        # recipe-side ``calculate_mtp_loss`` produces trainable gradients.
        assert all(h.requires_grad for h in out.mtp_per_depth_h)
        assert out.mtp_loss_scaling_factor == 0.1
        # Main loss is unaffected.
        assert out.loss is not None
        assert out.loss.requires_grad

    def test_forward_eval_skips_mtp_branch(self, backend):
        """At eval (or without labels) MTP must not run."""
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert out.mtp_per_depth_h is None
        assert out.mtp_loss_scaling_factor is None

    def test_mtp_backward_populates_mtp_grads(self, backend):
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        # Backprop through a sum of per-depth hidden states to isolate MTP grads
        # (the production path runs them through the configured loss class
        # via the recipe-side ``calculate_mtp_loss``).
        sum(h.sum() for h in out.mtp_per_depth_h).backward()
        assert model.mtp.layers[0].enorm.weight.grad is not None
        assert model.mtp.layers[0].hnorm.weight.grad is not None
        assert model.mtp.layers[0].eh_proj.weight.grad is not None
        assert model.mtp.layers[1].final_layernorm.weight.grad is not None


# ---------------------------------------------------------------------------
# State-dict adapter MTP key handling
# ---------------------------------------------------------------------------


class TestMTPStateDictAdapter:
    def test_mtp_keys_in_state_dict(self, backend):
        """Internal model state_dict must contain the expected mtp.* keys
        (matches the HF flat layout used by Super V3)."""
        model, _ = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        sd = model.state_dict()
        # Fusion modules on first sublayer (layer index 0 of the depth).
        assert "mtp.layers.0.enorm.weight" in sd
        assert "mtp.layers.0.hnorm.weight" in sd
        assert "mtp.layers.0.eh_proj.weight" in sd
        # final_layernorm on last sublayer (layer index 1).
        assert "mtp.layers.1.final_layernorm.weight" in sd
        # Per-sublayer block norm.
        assert "mtp.layers.0.norm.weight" in sd
        assert "mtp.layers.1.norm.weight" in sd
