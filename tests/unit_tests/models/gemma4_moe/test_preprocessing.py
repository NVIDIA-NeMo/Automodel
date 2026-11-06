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

"""Configuration-only Gemma4 positions."""

import pytest
import torch
from transformers import Gemma4Config

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.models.gemma4_moe.preprocessing import Gemma4PositionState


def _config():
    return Gemma4Config(
        text_config={"enable_moe_block": False, "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0},
        audio_config=None,
    )


def test_scalar_positions_preserve_excluded_slots_and_ignore_backend():
    config = _config()
    config._attn_implementation = "flash_attention_2"
    model = torch.nn.Module()
    model.config = config
    state = resolve_image_position_state(model)
    ids = torch.tensor([2, 9, 9, 3, 4])
    positions = state.positions(
        ids,
        mm_token_type_ids=torch.tensor([[0, 1, 1, 0, 0]]),
        image_grid_thw=None,
        attention_mask=torch.tensor([1, 0, 0, 1, 1]),
    )
    assert isinstance(state, Gemma4PositionState)
    assert positions.tolist() == [0, 1, 2, 3, 4]
    assert not isinstance(state, torch.nn.Module)


@pytest.mark.parametrize("field", ["enable_moe_block", "hidden_size_per_layer_input", "num_kv_shared_layers"])
def test_uncertified_variants_rejected(field):
    config = _config()
    assert hasattr(config.text_config, field)
    setattr(config.text_config, field, True if field == "enable_moe_block" else 1)
    with pytest.raises(ValueError, match="dense31B"):
        Gemma4PositionState(config)


def test_bidirectional_text_objective_rejected():
    config = _config()
    config.text_config.use_bidirectional_attention = "all"
    with pytest.raises(ValueError, match="causal text attention"):
        Gemma4PositionState(config)
