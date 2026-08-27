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
import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel._transformers.models.bagel.state_dict_adapter import BagelStateDictAdapter
from nemo_automodel._transformers.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_automodel._transformers.models.ernie4_5.state_dict_adapter import (
    Ernie4_5_MoeStateDictAdapter,
    Ernie4_5StateDictAdapter,
)
from nemo_automodel._transformers.models.gemma4_moe.state_dict_adapter import Gemma4MoEStateDictAdapter
from nemo_automodel._transformers.models.gemma4_unified.state_dict_adapter import Gemma4UnifiedStateDictAdapter
from nemo_automodel._transformers.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter
from nemo_automodel._transformers.models.glm_moe_dsa.state_dict_adapter import GlmMoeDsaStateDictAdapter
from nemo_automodel._transformers.models.hy_mt2.state_dict_adapter import HyMT2StateDictAdapter
from nemo_automodel._transformers.models.hy_v3.state_dict_adapter import HYV3StateDictAdapter
from nemo_automodel._transformers.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from nemo_automodel._transformers.models.kimi_k25_vl.state_dict_adapter import KimiK25VLStateDictAdapter
from nemo_automodel._transformers.models.kimi_linear.state_dict_adapter import KimiLinear48BStateDictAdapter
from nemo_automodel._transformers.models.laguna.state_dict_adapter import LagunaStateDictAdapter
from nemo_automodel._transformers.models.ling_v2.state_dict_adapter import BailingMoeV2StateDictAdapter
from nemo_automodel._transformers.models.llava_onevision.state_dict_adapter import LlavaOneVisionStateDictAdapter
from nemo_automodel._transformers.models.mimo_v2_flash.state_dict_adapter import MiMoV2FlashStateDictAdapter
from nemo_automodel._transformers.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel._transformers.models.minimax_m3_vl.state_dict_adapter import MiniMaxM3StateDictAdapter
from nemo_automodel._transformers.models.muse_glimmer.state_dict_adapter import MuseGlimmerStateDictAdapter
from nemo_automodel._transformers.models.nemotron_omni.state_dict_adapter import NemotronOmniStateDictAdapter
from nemo_automodel._transformers.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel._transformers.models.qwen2_5_omni.state_dict_adapter import Qwen2_5OmniStateDictAdapter
from nemo_automodel._transformers.models.qwen3_5.state_dict_adapter import Qwen3_5DenseStateDictAdapter
from nemo_automodel._transformers.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel._transformers.models.qwen3_next.state_dict_adapter import Qwen3NextStateDictAdapter
from nemo_automodel._transformers.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter
from nemo_automodel._transformers.models.qwen3_vl_moe.state_dict_adapter import Qwen3VLMoeStateDictAdapter


def _assert_destinations_write_through(
    adapter: StateDictAdapter,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Verify that checkpoint destinations alias and mutate final model storage.

    Args:
        adapter: Adapter whose ``to_hf`` method constructs checkpoint-load destinations.
        state_dict: Native model state mapping. Tensor values may have arbitrary shapes and axis order; each value
            represents final model storage and must retain its dtype, device, and storage through conversion.
    """
    source_tensors = list(state_dict.values())
    destinations = adapter.to_hf(dict(state_dict))

    assert destinations
    for fill_value, (key, destination) in enumerate(destinations.items(), start=1):
        assert isinstance(destination, torch.Tensor), f"checkpoint destination {key!r} is not a tensor"
        destination_storage = destination.untyped_storage().data_ptr()
        source = next(
            (tensor for tensor in source_tensors if tensor.untyped_storage().data_ptr() == destination_storage),
            None,
        )
        assert source is not None, f"checkpoint destination {key!r} does not alias final model storage"

        before = source.clone()
        destination.fill_(fill_value)
        assert not torch.equal(source, before), f"writes to checkpoint destination {key!r} do not reach model storage"


@pytest.mark.parametrize(
    ("adapter_type", "adapter_attrs"),
    [
        pytest.param(BagelStateDictAdapter, {}, id="bagel"),
        pytest.param(Ernie4_5StateDictAdapter, {}, id="ernie4_5_dense"),
        pytest.param(Gemma4UnifiedStateDictAdapter, {}, id="gemma4_unified"),
        pytest.param(LlavaOneVisionStateDictAdapter, {}, id="llava_onevision"),
        pytest.param(MuseGlimmerStateDictAdapter, {"uses_canonical_layout": True}, id="muse_glimmer"),
        pytest.param(Qwen2_5OmniStateDictAdapter, {"_uses_thinker_prefix": True}, id="qwen2_5_omni"),
        pytest.param(Qwen3_5DenseStateDictAdapter, {}, id="qwen3_5_dense"),
        pytest.param(Qwen3VLMoeStateDictAdapter, {}, id="qwen3_vl_moe"),
    ],
)
def test_write_through_adapters_expose_aliasing_destinations(
    adapter_type: type[StateDictAdapter], adapter_attrs: dict[str, object]
) -> None:
    adapter = object.__new__(adapter_type)
    adapter.__dict__.update(adapter_attrs)

    assert adapter.supports_write_through_checkpoint_load is True

    _assert_destinations_write_through(
        adapter,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
            "model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(2, dtype=torch.float32),
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.zeros(2, 2, 4),
        },
    )


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
def test_write_through_grouped_adapters_preserve_non_expert_storage_and_require_aliasing_backend(
    adapter_type: type[StateDictAdapter],
) -> None:
    moe_config = SimpleNamespace(n_routed_experts=2, moe_inter_dim=2, expert_activation="silu")
    adapter = adapter_type(
        SimpleNamespace(num_hidden_layers=1),
        moe_config,
        SimpleNamespace(experts="torch", dispatcher="torch"),
    )
    assert adapter.supports_write_through_checkpoint_load is True

    _assert_destinations_write_through(
        adapter,
        {
            "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.bfloat16),
            "model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(2, dtype=torch.float32),
            "model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros(2, dtype=torch.float32),
        },
    )

    adapter.backend = SimpleNamespace(experts="te", dispatcher="torch")
    assert adapter.supports_write_through_checkpoint_load is False

    adapter.backend = SimpleNamespace(experts="torch", dispatcher="mok")
    assert adapter.supports_write_through_checkpoint_load is False


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

    assert adapter.supports_write_through_checkpoint_load is False


def test_gemma4_moe_adapter_loads_without_a_full_checkpoint_copy():
    adapter = object.__new__(Gemma4MoEStateDictAdapter)

    assert adapter.supports_write_through_checkpoint_load is False
    assert adapter.supports_checkpoint_load_without_full_copy is True


def test_dense_grouped_adapter_does_not_require_an_expert_backend():
    adapter = object.__new__(NemotronV3StateDictAdapter)
    adapter.moe_config = None
    adapter.backend = SimpleNamespace(experts="te", dispatcher="mok")

    assert adapter.supports_write_through_checkpoint_load is True


def test_nemotron_omni_delegates_direct_load_support_to_language_adapter():
    adapter = object.__new__(NemotronOmniStateDictAdapter)
    adapter._llm_adapter = SimpleNamespace(supports_write_through_checkpoint_load=True)
    assert adapter.supports_write_through_checkpoint_load is True

    adapter._llm_adapter.supports_write_through_checkpoint_load = False
    assert adapter.supports_write_through_checkpoint_load is False
