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

from nemo_automodel.components.models.gemma4_moe.state_dict_adapter import Gemma4MoEStateDictAdapter

N_EXPERTS = 4
HIDDEN = 64
EXPERT_INTER = 32


@pytest.fixture
def adapter() -> Gemma4MoEStateDictAdapter:
    return Gemma4MoEStateDictAdapter(
        config=SimpleNamespace(),
        moe_config=SimpleNamespace(n_routed_experts=N_EXPERTS),
        backend=SimpleNamespace(),
        dtype=torch.float32,
    )


def test_large_destinations_alias_model_storage_and_scale_is_bounded(adapter: Gemma4MoEStateDictAdapter) -> None:
    gate_and_up = torch.zeros(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
    down = torch.zeros(N_EXPERTS, EXPERT_INTER, HIDDEN)
    state_dict = {
        "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
        "model.language_model.layers.0.moe.experts.down_projs": down,
    }

    destinations = adapter.to_hf(state_dict, load_into_empty_destinations=True)

    gate_destination = destinations["model.language_model.layers.0.experts.gate_up_proj"]
    down_destination = destinations["model.language_model.layers.0.experts.down_proj"]
    scale_destination = destinations["model.language_model.layers.0.router.per_expert_scale"]
    assert gate_destination.untyped_storage().data_ptr() == gate_and_up.untyped_storage().data_ptr()
    assert down_destination.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    assert scale_destination.untyped_storage().data_ptr() not in {
        gate_and_up.untyped_storage().data_ptr(),
        down.untyped_storage().data_ptr(),
    }
    assert scale_destination.shape == (N_EXPERTS,)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_from_hf_finalizes_aliased_destinations_in_place(
    adapter: Gemma4MoEStateDictAdapter, dtype: torch.dtype
) -> None:
    adapter.dtype = dtype
    expected_gate_hf = torch.randn(N_EXPERTS, 2 * EXPERT_INTER, HIDDEN, dtype=dtype)
    expected_down_hf = torch.randn(N_EXPERTS, HIDDEN, EXPERT_INTER, dtype=dtype)
    expected_scale = torch.arange(1, N_EXPERTS + 1, dtype=dtype)
    reference = adapter.from_hf(
        {
            "model.language_model.layers.0.experts.gate_up_proj": expected_gate_hf.clone(),
            "model.language_model.layers.0.experts.down_proj": expected_down_hf.clone(),
            "model.language_model.layers.0.router.per_expert_scale": expected_scale.clone(),
        }
    )
    gate_and_up = torch.zeros(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER, dtype=dtype)
    down = torch.zeros(N_EXPERTS, EXPERT_INTER, HIDDEN, dtype=dtype)
    state_dict = {
        "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
        "model.language_model.layers.0.moe.experts.down_projs": down,
    }
    destinations = adapter.to_hf(state_dict, load_into_empty_destinations=True)
    destinations["model.language_model.layers.0.experts.gate_up_proj"].copy_(expected_gate_hf)
    destinations["model.language_model.layers.0.experts.down_proj"].copy_(expected_down_hf)
    destinations["model.language_model.layers.0.router.per_expert_scale"].copy_(expected_scale)

    converted = adapter.from_hf(destinations)

    converted_gate = converted["model.language_model.layers.0.moe.experts.gate_and_up_projs"]
    converted_down = converted["model.language_model.layers.0.moe.experts.down_projs"]
    assert converted_gate.untyped_storage().data_ptr() == gate_and_up.untyped_storage().data_ptr()
    assert converted_down.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    torch.testing.assert_close(
        converted_gate,
        reference["model.language_model.layers.0.moe.experts.gate_and_up_projs"],
    )
    torch.testing.assert_close(
        converted_down,
        reference["model.language_model.layers.0.moe.experts.down_projs"],
    )


def test_export_destinations_remain_independent_contiguous_tensors(adapter: Gemma4MoEStateDictAdapter) -> None:
    gate_and_up = torch.zeros(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
    down = torch.zeros(N_EXPERTS, EXPERT_INTER, HIDDEN)

    exported = adapter.to_hf(
        {
            "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
            "model.language_model.layers.0.moe.experts.down_projs": down,
        }
    )

    exported_gate = exported["model.language_model.layers.0.experts.gate_up_proj"]
    exported_down = exported["model.language_model.layers.0.experts.down_proj"]
    assert exported_gate.is_contiguous()
    assert exported_down.is_contiguous()
    assert exported_gate.untyped_storage().data_ptr() != gate_and_up.untyped_storage().data_ptr()
    assert exported_down.untyped_storage().data_ptr() != down.untyped_storage().data_ptr()
