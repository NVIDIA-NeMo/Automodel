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
