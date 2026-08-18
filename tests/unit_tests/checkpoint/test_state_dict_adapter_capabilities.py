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

import pytest

from nemo_automodel.components.models.bagel.state_dict_adapter import BagelStateDictAdapter
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_automodel.components.models.ernie4_5.state_dict_adapter import (
    Ernie4_5_MoeStateDictAdapter,
    Ernie4_5StateDictAdapter,
)
from nemo_automodel.components.models.gemma4_moe.state_dict_adapter import Gemma4MoEStateDictAdapter
from nemo_automodel.components.models.gemma4_unified.state_dict_adapter import Gemma4UnifiedStateDictAdapter
from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter
from nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter import GlmMoeDsaStateDictAdapter
from nemo_automodel.components.models.hy_mt2.state_dict_adapter import HyMT2StateDictAdapter
from nemo_automodel.components.models.hy_v3.state_dict_adapter import HYV3StateDictAdapter
from nemo_automodel.components.models.kimi_k25_vl.state_dict_adapter import KimiK25VLStateDictAdapter
from nemo_automodel.components.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from nemo_automodel.components.models.kimi_linear.state_dict_adapter import KimiLinear48BStateDictAdapter
from nemo_automodel.components.models.laguna.state_dict_adapter import LagunaStateDictAdapter
from nemo_automodel.components.models.ling_v2.state_dict_adapter import BailingMoeV2StateDictAdapter
from nemo_automodel.components.models.llava_onevision.state_dict_adapter import LlavaOneVisionStateDictAdapter
from nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter import MiMoV2FlashStateDictAdapter
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.models.minimax_m3_vl.state_dict_adapter import MiniMaxM3StateDictAdapter
from nemo_automodel.components.models.muse_glimmer.state_dict_adapter import MuseGlimmerStateDictAdapter
from nemo_automodel.components.models.nemotron_omni.state_dict_adapter import NemotronOmniStateDictAdapter
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.models.qwen2_5_omni.state_dict_adapter import Qwen2_5OmniStateDictAdapter
from nemo_automodel.components.models.qwen3_5.state_dict_adapter import Qwen3_5DenseStateDictAdapter
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.models.qwen3_next.state_dict_adapter import Qwen3NextStateDictAdapter
from nemo_automodel.components.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter
from nemo_automodel.components.models.qwen3_vl_moe.state_dict_adapter import Qwen3VLMoeStateDictAdapter


@pytest.mark.parametrize(
    "adapter_type",
    [
        BagelStateDictAdapter,
        Ernie4_5StateDictAdapter,
        Gemma4UnifiedStateDictAdapter,
        LlavaOneVisionStateDictAdapter,
        MuseGlimmerStateDictAdapter,
        Qwen2_5OmniStateDictAdapter,
        Qwen3_5DenseStateDictAdapter,
        Qwen3VLMoeStateDictAdapter,
    ],
)
def test_aliasing_adapters_opt_into_direct_checkpoint_load(adapter_type):
    adapter = object.__new__(adapter_type)

    assert adapter.supports_inplace_checkpoint_load is True


@pytest.mark.parametrize(
    "adapter_type",
    [
        Ernie4_5_MoeStateDictAdapter,
        Glm4MoeStateDictAdapter,
        HYV3StateDictAdapter,
        HyMT2StateDictAdapter,
        LagunaStateDictAdapter,
        NemotronV3StateDictAdapter,
        Qwen3MoeStateDictAdapter,
        Qwen3NextStateDictAdapter,
        Qwen3OmniMoeStateDictAdapter,
    ],
)
def test_aliasing_grouped_expert_adapters_require_aliasing_backend(adapter_type):
    adapter = object.__new__(adapter_type)
    adapter.moe_config = object()
    adapter.backend = SimpleNamespace(experts="torch", dispatcher="torch")
    assert adapter.supports_inplace_checkpoint_load is True

    adapter.backend = SimpleNamespace(experts="te", dispatcher="torch")
    assert adapter.supports_inplace_checkpoint_load is False

    adapter.backend = SimpleNamespace(experts="torch", dispatcher="mok")
    assert adapter.supports_inplace_checkpoint_load is False


@pytest.mark.parametrize(
    "adapter_type",
    [
        BailingMoeV2StateDictAdapter,
        DeepSeekV3StateDictAdapter,
        GlmMoeDsaStateDictAdapter,
        KimiK25VLStateDictAdapter,
        KimiK3StateDictAdapter,
        KimiLinear48BStateDictAdapter,
        MiMoV2FlashStateDictAdapter,
        MiniMaxM2StateDictAdapter,
        MiniMaxM3StateDictAdapter,
    ],
)
def test_materializing_grouped_expert_adapters_keep_frugal_load(adapter_type):
    adapter = object.__new__(adapter_type)
    adapter.moe_config = object()
    adapter.backend = SimpleNamespace(experts="torch", dispatcher="torch")

    assert adapter.supports_inplace_checkpoint_load is False


def test_materializing_gemma4_moe_adapter_keeps_frugal_load():
    adapter = object.__new__(Gemma4MoEStateDictAdapter)

    assert adapter.supports_inplace_checkpoint_load is False


def test_dense_grouped_adapter_does_not_require_an_expert_backend():
    adapter = object.__new__(NemotronV3StateDictAdapter)
    adapter.moe_config = None
    adapter.backend = SimpleNamespace(experts="te", dispatcher="mok")

    assert adapter.supports_inplace_checkpoint_load is True


def test_nemotron_omni_delegates_direct_load_support_to_language_adapter():
    adapter = object.__new__(NemotronOmniStateDictAdapter)
    adapter._llm_adapter = SimpleNamespace(supports_inplace_checkpoint_load=True)
    assert adapter.supports_inplace_checkpoint_load is True

    adapter._llm_adapter.supports_inplace_checkpoint_load = False
    assert adapter.supports_inplace_checkpoint_load is False
