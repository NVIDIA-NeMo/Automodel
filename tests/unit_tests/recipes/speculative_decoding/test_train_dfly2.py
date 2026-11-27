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

"""Tests for the DFly2 recipe seams over the DSpark recipe.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
overridden seams are exercised: the unsupported-option gate, the model-owned draft
options, and the FSDP units of the FP32 heads.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

from nemo_automodel.recipes.speculative_decoding.train_dfly2 import TrainDFly2Recipe
from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe, _DraftArgs

pytestmark = pytest.mark.cpu_only


def _recipe(**recipe_args) -> TrainDFly2Recipe:
    recipe = TrainDFly2Recipe.__new__(TrainDFly2Recipe)
    recipe.cfg = SimpleNamespace(recipe_args=dict(recipe_args))
    return recipe


def _margs(**overrides) -> _DraftArgs:
    values = dict(
        num_draft_layers=3,
        target_layer_ids=[28, 32, 36],
        block_size=6,
        num_anchors=512,
        mask_token_id=128799,
        markov_rank=256,
        markov_head_type="vanilla",
        confidence_head_alpha=0.0,
        confidence_head_with_markov=True,
        confidence_head_stop_gradient=False,
    )
    values.update(overrides)
    return _DraftArgs(**values)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("cached_target_path", "/cache"),
        ("target_model_backend", "remote"),
        ("packed_sequence_size", 4096),
        ("multimodal", True),
    ],
)
def test_setup_rejects_options_without_a_colocated_unpacked_target(option, value, monkeypatch) -> None:
    called = Mock()
    monkeypatch.setattr(TrainDSparkRecipe, "setup", called)
    with pytest.raises(NotImplementedError, match=option):
        _recipe(**{option: value}).setup()
    called.assert_not_called()


def test_model_owned_draft_is_built_from_the_dfly2_options_and_the_target() -> None:
    draft = Mock()
    target_config = SimpleNamespace(build_dfly2_draft=Mock(return_value=draft))
    recipe = _recipe(confidence_head_stop_gradient=False, draft_attn_backend="tilelang")
    recipe.target_model = object()
    assert recipe._build_model_owned_draft(target_config, _margs(confidence_head_alpha=1.0)) is draft
    options = target_config.build_dfly2_draft.call_args.args[0]
    assert options.target_layer_ids == [28, 32, 36]
    assert (options.block_size, options.num_anchors, options.mask_token_id) == (6, 512, 128799)
    assert options.enable_confidence_head is True
    assert options.confidence_head_stop_gradient is False
    assert options.attn_backend == "tilelang"
    draft.initialize_from_target.assert_called_once_with(recipe.target_model)


def test_confidence_head_is_off_and_detached_by_default() -> None:
    target_config = SimpleNamespace(build_dfly2_draft=Mock(return_value=Mock()))
    recipe = _recipe()
    recipe.target_model = object()
    recipe._build_model_owned_draft(target_config, _margs())
    options = target_config.build_dfly2_draft.call_args.args[0]
    assert options.enable_confidence_head is False
    assert options.confidence_head_stop_gradient is True
    assert options.attn_backend == "sdpa"


def test_fsdp_units_include_the_heads_holding_fp32_tensors() -> None:
    draft = SimpleNamespace(
        layers=nn.ModuleList([nn.Linear(1, 1)]),
        mtp_fusion=nn.Linear(1, 1),
        hidden_correction=nn.Linear(1, 1),
        confidence_head=None,
    )
    assert TrainDFly2Recipe._draft_fsdp_units(draft) == [draft.layers[0], draft.mtp_fusion, draft.hidden_correction]
    # The FP32 confidence head must not fall into the root unit, which all-gathers in BF16.
    draft.confidence_head = nn.Linear(1, 1)
    assert TrainDFly2Recipe._draft_fsdp_units(draft)[-1] is draft.confidence_head
