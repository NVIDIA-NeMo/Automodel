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

"""Exercise retrieval setup up to the optimizer boundary without GPU infrastructure."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.utils.model_utils import FreezeConfig, apply_parameter_freezing
from nemo_automodel.recipes.retrieval import train_bi_encoder
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe


class _ReachedOptimizer(Exception):
    """Stop setup after model construction and validation, before external resources."""


class _VisualRetriever(torch.nn.Module):
    effective_score_temperature: float = 0.02

    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = torch.nn.Linear(2, 2)
        self.multi_modal_projector = torch.nn.Linear(2, 2)
        self.language_model = torch.nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a tiny vision/projector/language path.

        Args:
            inputs: Tensor of shape [batch, hidden], with hidden size 2.

        Returns:
            Tensor of shape [batch, hidden], with hidden size 2.
        """
        return self.language_model(self.multi_modal_projector(self.vision_tower(inputs)))


@pytest.fixture
def setup_model(monkeypatch: pytest.MonkeyPatch) -> _VisualRetriever:
    """Replace GPU/infrastructure construction while retaining real recipe setup."""
    model = _VisualRetriever()
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        train_bi_encoder, "initialize_distributed", lambda **kwargs: SimpleNamespace(world_size=1, is_main=False)
    )
    for name in ("setup_logging", "apply_cache_compatibility_patches", "apply_te_patches"):
        monkeypatch.setattr(train_bi_encoder, name, lambda: None)
    monkeypatch.setattr(train_bi_encoder, "StatefulRNG", lambda **kwargs: object())
    monkeypatch.setattr(train_bi_encoder, "ScopedRNG", lambda **kwargs: nullcontext())
    monkeypatch.setattr(train_bi_encoder, "create_distributed_setup_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        TrainBiEncoderRecipe,
        "_distributed_setup_attributes",
        lambda *args: (None, None, None, None, None, False, None, None, None),
    )
    for name in ("_log_experiment_details", "_log_library_versions"):
        monkeypatch.setattr(TrainBiEncoderRecipe, name, lambda self: None)
    for name in ("_get_dp_rank", "_get_tp_rank", "_get_pp_rank"):
        monkeypatch.setattr(TrainBiEncoderRecipe, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(CheckpointingConfig, "build", lambda *args, **kwargs: object())

    def instantiate(self: ConfigNode, **kwargs: Any) -> _VisualRetriever:
        freeze_config = kwargs.get("freeze_config")
        if freeze_config is not None:
            assert isinstance(freeze_config, FreezeConfig)
            apply_parameter_freezing(model, freeze_config)
        return model

    def reached_optimizer(self: TrainBiEncoderRecipe) -> None:
        raise _ReachedOptimizer

    monkeypatch.setattr(ConfigNode, "instantiate", instantiate)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_build_optimizer_param_groups", reached_optimizer)
    return model


@pytest.mark.parametrize("recipe_class", [TrainBiEncoderRecipe, TrainCrossEncoderRecipe])
def test_setup_preserves_freezing_through_optimizer_step(
    setup_model: _VisualRetriever, recipe_class: type[TrainBiEncoderRecipe]
) -> None:
    """Actual setup forwards the policy and SGD updates only the projector."""
    config = ConfigNode(
        {
            "model": {"_target_": "nemo_automodel._transformers.retrieval.BiEncoderModel"},
            "freeze_config": {"freeze_vision_tower": True, "freeze_language_model": True},
        }
    )
    recipe = recipe_class(config)
    with pytest.raises(_ReachedOptimizer):
        recipe.setup()
    model = recipe.model_parts[0]
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {"multi_modal_projector.weight", "multi_modal_projector.bias"}
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(2, 2)).square().sum().backward()
    optimizer.step()
    changed = {name for name, param in model.named_parameters() if not torch.equal(before[name], param)}
    assert changed == trainable


def test_setup_rejects_double_temperature_before_optimizer(setup_model: _VisualRetriever) -> None:
    """The validation hook must run from setup, not only when called by a test."""
    recipe = TrainCrossEncoderRecipe(
        ConfigNode(
            {"model": {"_target_": "nemo_automodel._transformers.retrieval.CrossEncoderModel"}, "temperature": 0.3}
        )
    )
    with pytest.raises(ValueError, match="temperature scaling is configured twice"):
        recipe.setup()


def test_setup_allows_model_owned_temperature(setup_model: _VisualRetriever) -> None:
    recipe = TrainCrossEncoderRecipe(
        ConfigNode(
            {"model": {"_target_": "nemo_automodel._transformers.retrieval.CrossEncoderModel"}, "temperature": 1.0}
        )
    )
    with pytest.raises(_ReachedOptimizer):
        recipe.setup()
