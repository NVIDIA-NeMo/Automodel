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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm4_moe.model import Glm4MoeForCausalLM
from nemo_automodel.components.models.glm4_moe_lite.model import Glm4MoeLiteForCausalLM
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM

_GLM_MODEL_CASES = (
    (
        Glm4MoeForCausalLM,
        "nemo_automodel.components.models.glm4_moe.model.Glm4MoeModel",
    ),
    (
        Glm4MoeLiteForCausalLM,
        "nemo_automodel.components.models.glm4_moe_lite.model.Glm4MoeLiteModel",
    ),
    (
        GlmMoeDsaForCausalLM,
        "nemo_automodel.components.models.glm_moe_dsa.model.GlmMoeDsaModel",
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


@pytest.mark.parametrize(("model_cls", "inner_model_path"), _GLM_MODEL_CASES)
def test_glm_gate_precision_defaults_to_fp32_without_mutating_backend(model_cls, inner_model_path):
    backend = _backend()

    with patch(inner_model_path) as inner_model_cls:
        model = model_cls(_config(), backend=backend)

    assert backend.gate_precision is None
    assert model.backend.gate_precision is torch.float32
    assert inner_model_cls.call_args.kwargs["backend"] is model.backend


@pytest.mark.parametrize(("model_cls", "inner_model_path"), _GLM_MODEL_CASES)
def test_glm_gate_precision_respects_explicit_override(model_cls, inner_model_path):
    backend = _backend(gate_precision=torch.bfloat16)

    with patch(inner_model_path) as inner_model_cls:
        model = model_cls(_config(), backend=backend)

    assert model.backend is backend
    assert model.backend.gate_precision is torch.bfloat16
    assert inner_model_cls.call_args.kwargs["backend"] is backend


@pytest.mark.parametrize("model_cls", [Glm4MoeForCausalLM, Glm4MoeLiteForCausalLM, GlmMoeDsaForCausalLM])
def test_glm_preserves_score_correction_bias_in_fp32(model_cls):
    assert "e_score_correction_bias" in model_cls._keep_in_fp32_modules_strict
