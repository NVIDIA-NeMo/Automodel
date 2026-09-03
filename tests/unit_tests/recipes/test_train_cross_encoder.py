# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the cross-encoder training recipe."""

import pytest
import torch

from nemo_automodel._transformers.retrieval import CrossEncoderModel
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalConfig,
    LlamaBidirectionalForSequenceClassification,
)
from nemo_automodel.components.models.qwen3_reranker.model import (
    Qwen3RerankerConfig,
    Qwen3RerankerForCausalReranking,
)
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe, accuracy, batch_mrr

# ---------------------------------------------------------------------------
# temperature configuration
# ---------------------------------------------------------------------------


def _tiny_llama_cross_encoder(*, temperature: float) -> CrossEncoderModel:
    config = LlamaBidirectionalConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=32,
        num_labels=1,
        temperature=temperature,
    )
    return CrossEncoderModel(LlamaBidirectionalForSequenceClassification(config))


def _tiny_qwen3_cross_encoder(*, temperature: float) -> CrossEncoderModel:
    config = Qwen3RerankerConfig(
        yes_token_id=5,
        no_token_id=7,
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        head_dim=8,
        temperature=temperature,
    )
    return CrossEncoderModel(Qwen3RerankerForCausalReranking(config))


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"temperature": 0.3},
        {"model": {"temperature": 0.02}},
        {"temperature": 0.3, "model": {"temperature": 1.0}},
        {"temperature": 1.0, "model": {"temperature": 0.02}},
        {"temperature": 0.3, "model": {"temperature": 0.02}},
        {"temperature": 0.3, "model": {"temperature": None}},
    ],
)
def test_recipe_defers_temperature_validation_until_model_is_built(config):
    recipe = TrainCrossEncoderRecipe(ConfigNode(config))

    assert recipe.temperature == config.get("temperature", 1.0)


def test_recipe_rejects_two_active_non_unit_temperatures():
    config = ConfigNode({"temperature": 0.3, "model": {"temperature": 0.02}})
    recipe = TrainCrossEncoderRecipe(config)
    model = _tiny_llama_cross_encoder(temperature=0.02)

    assert model.effective_score_temperature == 0.02
    with pytest.raises(
        ValueError,
        match="Set either top-level temperature or model.temperature to 1.0",
    ):
        recipe._validate_model(model)


def test_recipe_allows_configured_temperature_when_qwen3_does_not_apply_it():
    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": 0.3, "model": {"temperature": 0.02}}))
    model = _tiny_qwen3_cross_encoder(temperature=0.02)

    assert model.effective_score_temperature == 1.0
    recipe._validate_model(model)


def test_recipe_allows_null_override_when_loaded_model_uses_unit_temperature():
    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": 0.3, "model": {"temperature": None}}))
    model = _tiny_llama_cross_encoder(temperature=1.0)

    assert model.effective_score_temperature == 1.0
    recipe._validate_model(model)


def test_recipe_allows_active_model_temperature_when_recipe_temperature_is_unit():
    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": 1.0, "model": {"temperature": 0.02}}))
    model = _tiny_llama_cross_encoder(temperature=0.02)

    assert model.effective_score_temperature == 0.02
    recipe._validate_model(model)


@pytest.mark.parametrize("explicit_null", [False, True], ids=["omitted", "null"])
def test_recipe_rejects_checkpoint_temperature_when_model_override_is_absent(tmp_path, explicit_null):
    checkpoint_config = LlamaBidirectionalConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=32,
        num_labels=1,
        temperature=0.02,
    )
    LlamaBidirectionalForSequenceClassification(checkpoint_config).save_pretrained(tmp_path)
    model_build_kwargs = {"temperature": None} if explicit_null else {}
    recipe_model_config = {"pretrained_model_name_or_path": str(tmp_path)}
    if explicit_null:
        recipe_model_config["temperature"] = None

    model = CrossEncoderModel.build(model_name_or_path=str(tmp_path), **model_build_kwargs)
    recipe = TrainCrossEncoderRecipe(
        ConfigNode(
            {
                "temperature": 0.3,
                "model": recipe_model_config,
            }
        )
    )

    assert model.config.temperature == 0.02
    assert model.effective_score_temperature == 0.02
    with pytest.raises(
        ValueError,
        match="Set either top-level temperature or model.temperature to 1.0",
    ):
        recipe._validate_model(model)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_all_correct():
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    num_correct, total = accuracy(output, target)
    assert num_correct.item() == 2
    assert total == 2


def test_accuracy_partial():
    output = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    target = torch.tensor([1, 0])
    num_correct, total = accuracy(output, target)
    assert num_correct.item() == 1
    assert total == 2


def test_accuracy_all_wrong():
    output = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    target = torch.tensor([1, 0])
    num_correct, total = accuracy(output, target)
    assert num_correct.item() == 0
    assert total == 2


def test_accuracy_single_example():
    output = torch.tensor([[0.3, 0.7]])
    target = torch.tensor([1])
    num_correct, total = accuracy(output, target)
    assert num_correct.item() == 1
    assert total == 1


# ---------------------------------------------------------------------------
# batch_mrr
# ---------------------------------------------------------------------------


def test_batch_mrr_perfect_ranking():
    """All targets are at rank 1 -> each RR = 1.0, sum = batch_size."""
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    mrr_sum = batch_mrr(output, target)
    torch.testing.assert_close(mrr_sum, torch.tensor(2.0))


def test_batch_mrr_worst_ranking():
    """3-class output, targets at the last rank -> each RR = 1/3."""
    # For each row the target class has the lowest score.
    output = torch.tensor(
        [
            [0.1, 0.5, 0.9],  # target=0, score 0.1 is lowest -> rank 3
            [0.9, 0.5, 0.1],  # target=2, score 0.1 is lowest -> rank 3
        ]
    )
    target = torch.tensor([0, 2])
    mrr_sum = batch_mrr(output, target)
    torch.testing.assert_close(mrr_sum, torch.tensor(2.0 / 3.0))


def test_batch_mrr_mixed():
    """Mixed ranks: rank 1 (RR=1) and rank 2 (RR=0.5) -> sum = 1.5."""
    output = torch.tensor(
        [
            [0.1, 0.9],  # target=1, score 0.9 is highest -> rank 1
            [0.9, 0.8],  # target=1, score 0.8 is second  -> rank 2
        ]
    )
    target = torch.tensor([1, 1])
    mrr_sum = batch_mrr(output, target)
    torch.testing.assert_close(mrr_sum, torch.tensor(1.5))


def test_batch_mrr_wider_output():
    """4-class output with known ranks, verify exact sum.

    Row 0: scores [0.1, 0.4, 0.9, 0.6], target=2 -> sorted order [2,3,1,0] -> rank 1, RR=1
    Row 1: scores [0.9, 0.1, 0.5, 0.7], target=2 -> sorted order [0,3,2,1] -> rank 3, RR=1/3
    Row 2: scores [0.3, 0.8, 0.5, 0.2], target=3 -> sorted order [1,2,0,3] -> rank 4, RR=1/4
    Sum = 1 + 1/3 + 1/4 = 19/12
    """
    output = torch.tensor(
        [
            [0.1, 0.4, 0.9, 0.6],
            [0.9, 0.1, 0.5, 0.7],
            [0.3, 0.8, 0.5, 0.2],
        ]
    )
    target = torch.tensor([2, 2, 3])
    mrr_sum = batch_mrr(output, target)
    torch.testing.assert_close(mrr_sum, torch.tensor(19.0 / 12.0))
