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
overridden seams are exercised: the unsupported-option gate, the confidence default, the
warm-start and rank-count checks, and the FSDP units of the FP32 heads. The draft's start
from its paired target layers is tested with a real target in the model tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.speculative_decoding.train_dfly2 import TrainDFly2Recipe, check_fsdp_expert_split
from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe, _DraftArgs

pytestmark = pytest.mark.cpu_only


def _recipe(*, world_size: int = 1, strategy: str = "fsdp2", **recipe_args) -> TrainDFly2Recipe:
    recipe = TrainDFly2Recipe.__new__(TrainDFly2Recipe)
    recipe.cfg = ConfigNode({"recipe_args": dict(recipe_args), "distributed": {"strategy": strategy}})
    recipe.dist_env = SimpleNamespace(world_size=world_size)
    recipe.target_model = object()
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


def _target_config(routed_experts: int = 128) -> SimpleNamespace:
    draft = Mock()
    draft.moe_config.n_routed_experts = routed_experts
    return SimpleNamespace(build_dfly2_draft=Mock(return_value=draft))


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


def test_confidence_head_is_off_and_detached_by_default() -> None:
    recipe = _recipe()
    # setup() resolves the weight through this method: DFly2 drafts without a head unless asked,
    # so the released drafter (which has none) loads; DSpark keeps its default of 1.0.
    alpha = recipe._resolve_confidence_head_alpha(recipe.cfg.recipe_args)
    assert alpha == 0.0
    dspark = TrainDSparkRecipe.__new__(TrainDSparkRecipe)
    assert dspark._resolve_confidence_head_alpha(ConfigNode({})) == 1.0
    assert recipe._resolve_confidence_head_alpha(ConfigNode({"confidence_head_alpha": 0.5})) == 0.5
    target_config = _target_config()
    recipe._build_model_owned_draft(target_config, _margs(confidence_head_alpha=alpha))
    options = target_config.build_dfly2_draft.call_args.args[0]
    assert options.enable_confidence_head is False
    assert options.confidence_head_stop_gradient is True


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


@pytest.mark.parametrize(
    ("location", "setting", "value"),
    [
        ("config.json", "target_layer_ids", [24, 28, 32]),
        ("config.json", "mask_token_id", 1),
        # A step checkpoint's model/ directory keeps its config in .hf_metadata/.
        (".hf_metadata/config.json", "block_size", 7),
    ],
)
def test_warm_start_refuses_other_dfly2_settings_before_building(tmp_path, location, setting, value) -> None:
    released = {"block_size": 6, "target_kv_layers": [28, 32, 36], "mask_token_id": 128799}
    (tmp_path / location).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / location).write_text(json.dumps({"dfly_config": released}))
    target_config = _target_config()
    recipe = _recipe(draft_weights_path=str(tmp_path))
    recipe._build_model_owned_draft(target_config, _margs())
    with pytest.raises(ValueError, match="other DFly2 settings"):
        recipe._build_model_owned_draft(target_config, _margs(**{setting: value}))
    assert target_config.build_dfly2_draft.call_count == 1


def test_world_sizes_that_cannot_checkpoint_the_experts_are_refused() -> None:
    usable = []
    for world_size in range(1, 129):
        try:
            check_fsdp_expert_split(128, world_size)
            usable.append(world_size)
        except ValueError:
            pass
    # FSDP2 splits like torch.chunk; the export expects the remainder spread over the first ranks.
    assert usable == [1, 2, 3, 4, 8, 16, 32, 43, 64, 128]
    with pytest.raises(ValueError, match="world size of 12"):
        _recipe(world_size=12)._build_model_owned_draft(_target_config(), _margs())
    _recipe(world_size=16)._build_model_owned_draft(_target_config(), _margs())
    # DDP replicates the draft, so any world size saves.
    _recipe(world_size=12, strategy="ddp")._build_model_owned_draft(_target_config(), _margs())
