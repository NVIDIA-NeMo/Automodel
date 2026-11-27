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
overridden seams are exercised: the unsupported-option gate, the target wrapper, the
model-owned draft, the trainer module, the FSDP units, and the batch forward.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.speculative.dfly2 import DFly2TargetFeatures, DFly2TargetModel, DFly2TrainerModule
from nemo_automodel.components.speculative.dspark.core import DSparkTrainerModule
from nemo_automodel.recipes.speculative_decoding import train_dfly2
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
    with pytest.raises(NotImplementedError, match=option.split("=")[0]):
        _recipe(**{option: value}).setup()
    called.assert_not_called()


def test_setup_delegates_to_the_dspark_recipe(monkeypatch) -> None:
    called = Mock()
    monkeypatch.setattr(TrainDSparkRecipe, "setup", called)
    _recipe(target_model_backend="colocated", packed_sequence_size=0).setup()
    called.assert_called_once()


class _FeatureTarget(nn.Module):
    def dfly2_target_features(self, input_ids, *, attention_mask, layer_ids):
        return DFly2TargetFeatures(
            context_keys=tuple(torch.zeros(1, 4, 2) for _ in layer_ids),
            context_indices=tuple(torch.zeros(1, 2, 3, dtype=torch.int32) for _ in layer_ids),
            last_hidden_states=torch.zeros(1, 2, 5),
        )


def test_target_wrapper_captures_the_paired_layers() -> None:
    recipe = _recipe()
    recipe.target_model = _FeatureTarget()
    wrapper = recipe._build_target_wrapper([28, 32, 36], None)
    assert isinstance(wrapper, DFly2TargetModel) and wrapper.target_layer_ids == [28, 32, 36]


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


def test_model_owned_draft_requires_a_dfly2_provider() -> None:
    recipe = _recipe()
    recipe.target_model = object()
    with pytest.raises(ValueError, match="does not provide a DFly2 draft"):
        recipe._build_model_owned_draft(SimpleNamespace(build_dspark_draft=Mock()), _margs())


def test_trainer_module_threads_the_objective_options() -> None:
    recipe = _recipe()
    recipe.draft_model = nn.Linear(1, 1)
    module = recipe._build_trainer_module({"dpace_alpha": 0.25, "lk_eta": 5.0}, 0.5)
    assert isinstance(module, DFly2TrainerModule)
    assert (module.dpace_alpha, module.lk_eta, module.confidence_head_alpha) == (0.25, 5.0, 0.5)
    defaults = recipe._build_trainer_module({}, 1.0)
    assert (defaults.dpace_alpha, defaults.lk_eta) == (0.5, 3.0)


def test_dspark_seams_keep_their_defaults() -> None:
    recipe = TrainDSparkRecipe.__new__(TrainDSparkRecipe)
    recipe.draft_model = nn.Linear(1, 1)
    assert isinstance(recipe._build_trainer_module({}, 1.0), DSparkTrainerModule)
    draft = SimpleNamespace(layers=nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)]))
    assert TrainDSparkRecipe._draft_fsdp_units(draft) == list(draft.layers)


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


def test_forward_batch_feeds_the_target_features_to_the_trainer() -> None:
    recipe = _recipe()
    recipe.device = torch.device("cpu")
    recipe.target_wrapper = DFly2TargetModel(_FeatureTarget(), [28])
    recipe.trainer_module = Mock(return_value="metrics")
    batch = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2),
    }
    assert recipe._forward_batch(batch) == "metrics"
    kwargs = recipe.trainer_module.call_args.kwargs
    assert len(kwargs["context_keys"]) == 1 and len(kwargs["context_indices"]) == 1
    assert kwargs["target_last_hidden_states"].shape == (1, 2, 5)
    assert torch.equal(kwargs["loss_mask"], batch["loss_mask"])


def test_main_runs_setup_then_loop(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(train_dfly2, "parse_args_and_load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(TrainDFly2Recipe, "setup", lambda self: calls.append("setup"))
    monkeypatch.setattr(TrainDFly2Recipe, "run_train_validation_loop", lambda self: calls.append("loop"))
    train_dfly2.main("config.yaml")
    assert calls == ["setup", "loop"]
