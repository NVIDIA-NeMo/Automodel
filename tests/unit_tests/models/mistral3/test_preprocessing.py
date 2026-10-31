# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Mistral3 config-only positions and model capability rejection."""

import pickle

import pytest
import torch
from transformers import Mistral3Config

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.models.mistral3.preprocessing import Mistral3PositionState


def test_scalar_state_serializes_without_model_and_preserves_excluded_offsets():
    model = torch.nn.Module()
    model.config = Mistral3Config(text_config={"model_type": "ministral3", "sliding_window": None})
    model.config._attn_implementation = "sdpa"
    state = pickle.loads(pickle.dumps(resolve_image_position_state(model)))  # noqa: S301 - local config
    assert isinstance(state, Mistral3PositionState)
    positions = state.positions(
        torch.tensor([2, 10, 10, 3, 4]),
        mm_token_type_ids=torch.zeros(1, 5),
        image_grid_thw=None,
        attention_mask=torch.tensor([1, 0, 0, 1, 1]),
    )
    assert positions.tolist() == [0, 1, 2, 3, 4]
    assert not isinstance(state, torch.nn.Module)


@pytest.mark.parametrize("mode", ["sliding", "other_text"])
def test_unsupported_variant_fails_before_worker_construction(mode):
    config = Mistral3Config(text_config={"model_type": "ministral3", "sliding_window": None})
    if mode == "sliding":
        config.text_config.sliding_window = 32
    else:
        config.text_config.model_type = "mistral4"
    with pytest.raises(ValueError, match="Mistral/Ministral3"):
        Mistral3PositionState(config)


def test_weight_tying_does_not_change_processor_position_state():
    config = Mistral3Config(text_config={"model_type": "ministral3", "sliding_window": None})
    config.tie_word_embeddings = True
    config.text_config.tie_word_embeddings = True
    assert Mistral3PositionState(config).config is config
