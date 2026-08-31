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

"""DeepSeek V3 and V3.2 must share one router precision policy.

V3.2 skips DeepseekV3ForCausalLM.__init__, so a default installed there does
not reach it. Parametrizing over both classes keeps the two construction
paths from drifting apart silently.
"""

import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# Mock fast_hadamard_transform before importing deepseek_v32 modules
if "fast_hadamard_transform" not in sys.modules:
    mock_hadamard = types.ModuleType("fast_hadamard_transform")
    mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
    mock_hadamard.hadamard_transform = lambda x, scale: x
    sys.modules["fast_hadamard_transform"] = mock_hadamard

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate

_DEEPSEEK_MODEL_CASES = (
    (
        DeepseekV3ForCausalLM,
        "nemo_automodel.components.models.deepseek_v3.model.DeepseekV3Model",
    ),
    (
        DeepseekV32ForCausalLM,
        "nemo_automodel.components.models.deepseek_v32.model.DeepseekV32Model",
    ),
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        tie_word_embeddings=False,
        hidden_size=16,
        vocab_size=32,
        torch_dtype="bfloat16",
    )


def _backend(*, gate_precision: torch.dtype | None = None) -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        gate_precision=gate_precision,
    )


@pytest.mark.parametrize(("model_cls", "inner_model_path"), _DEEPSEEK_MODEL_CASES)
def test_deepseek_gate_precision_defaults_to_fp32_without_mutating_backend(model_cls, inner_model_path):
    backend = _backend()

    with patch(inner_model_path) as inner_model_cls:
        model = model_cls(_config(), backend=backend)

    assert backend.gate_precision is None
    assert model.backend.gate_precision is torch.float32
    assert inner_model_cls.call_args.kwargs["backend"] is model.backend


@pytest.mark.parametrize(("model_cls", "inner_model_path"), _DEEPSEEK_MODEL_CASES)
def test_deepseek_gate_precision_respects_explicit_override(model_cls, inner_model_path):
    backend = _backend(gate_precision=torch.bfloat16)

    with patch(inner_model_path) as inner_model_cls:
        model = model_cls(_config(), backend=backend)

    assert model.backend is backend
    assert model.backend.gate_precision is torch.bfloat16
    assert inner_model_cls.call_args.kwargs["backend"] is backend


@pytest.mark.parametrize("model_cls", [DeepseekV3ForCausalLM, DeepseekV32ForCausalLM])
def test_deepseek_preserves_score_correction_bias_in_fp32(model_cls):
    assert "e_score_correction_bias" in model_cls._keep_in_fp32_modules_strict


# The selected-weight policy lives on the MoEConfig the inner model builds, so these
# cases construct for real. first_k_dense_replace >= num_hidden_layers keeps every
# layer dense: no Gate or expert weights are allocated, but moe_config still is.
_COMMON = dict(
    vocab_size=100,
    hidden_size=64,
    num_attention_heads=4,
    num_hidden_layers=1,
    first_k_dense_replace=1,
    intermediate_size=128,
    qk_rope_head_dim=16,
    v_head_dim=16,
    qk_nope_head_dim=16,
)


def _v3_config() -> DeepseekV3Config:
    return DeepseekV3Config(**_COMMON)


def _v32_config() -> DeepseekV32Config:
    return DeepseekV32Config(
        **_COMMON,
        moe_intermediate_size=64,
        qk_head_dim=32,
        kv_lora_rank=32,
        q_lora_rank=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=16,
    )


def _router_config(**overrides) -> MoEConfig:
    """Tiny grouped router matching the DeepSeek policy (n_expert_groups > 1)."""
    base = dict(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=2,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        aux_loss_coeff=0,
        score_func="sigmoid",
        route_scale=2.5,
        norm_topk_prob=True,
        router_weights_fp32=True,
        dtype=torch.bfloat16,
    )
    base.update(overrides)
    return MoEConfig(**base)


def _run_gate(moe_config, gate_precision):
    torch.manual_seed(0)
    gate = Gate(moe_config, gate_precision=gate_precision)
    gate.weight.data.normal_(std=0.02)
    x = torch.randn(16, moe_config.dim, dtype=torch.bfloat16)
    token_mask = torch.ones(16, dtype=torch.bool)
    return gate, gate(x, token_mask, None)


def test_gate_hands_fp32_weights_to_expert_compute():
    """Out stage: with the DeepSeek policy, bf16 in still yields fp32 weights."""
    gate, (weights, indices, _) = _run_gate(_router_config(), torch.float32)

    assert gate.score_dtype is torch.float32
    assert weights.dtype is torch.float32
    assert indices.shape == (16, 2)


def test_gate_without_policy_downcasts_weights():
    """The bug this PR fixes: type_as(x) hands bf16 weights to expert compute."""
    _, (weights, _, _) = _run_gate(_router_config(router_weights_fp32=False), None)

    assert weights.dtype is torch.bfloat16


def test_explicit_gate_precision_also_moves_scoring():
    """Score has no separate knob: overriding Proj to bf16 takes Score with it."""
    gate, _ = _run_gate(_router_config(), torch.bfloat16)

    assert gate.score_dtype is torch.bfloat16


_DEEPSEEK_MOE_CONFIG_CASES = (
    pytest.param(DeepseekV3ForCausalLM, _v3_config, id="v3"),
    pytest.param(DeepseekV32ForCausalLM, _v32_config, id="v32"),
)


@pytest.mark.parametrize(("model_cls", "config_fn"), _DEEPSEEK_MOE_CONFIG_CASES)
def test_deepseek_selected_router_weights_default_to_fp32(model_cls, config_fn):
    model = model_cls(config_fn(), backend=_backend())
    assert model.model.moe_config.router_weights_fp32 is True


@pytest.mark.parametrize(("model_cls", "config_fn"), _DEEPSEEK_MOE_CONFIG_CASES)
def test_deepseek_selected_router_weights_are_overridable(model_cls, config_fn):
    model = model_cls(
        config_fn(),
        backend=_backend(),
        moe_overrides={"router_weights_fp32": False},
    )
    assert model.model.moe_config.router_weights_fp32 is False


def test_kimi_k2_routes_to_the_deepseek_v3_construction_path():
    """Kimi K2 is config only: it inherits V3's router precision policy.

    If Kimi K2 ever gains its own ForCausalLM, it will also need its own copy of
    the fp32 gate_precision default and router_weights_fp32, the same way V3.2
    does. This test fails at that moment.
    """
    from nemo_automodel._transformers.registry import ModelRegistry, resolve_custom_config_cls
    from nemo_automodel.components.models.kimi_k2.config import KimiK2Config

    assert resolve_custom_config_cls("kimi_k2") is KimiK2Config
    assert issubclass(KimiK2Config, DeepseekV3Config)
    assert ModelRegistry.get_model_cls_from_model_arch("DeepseekV3ForCausalLM") is DeepseekV3ForCausalLM
