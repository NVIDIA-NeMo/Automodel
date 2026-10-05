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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("temperature", [1.0, 0.02])
def test_cross_encoder_recipe_temperature_in_train_and_validation(temperature: float) -> None:
    """The top-level temperature scales loss and training gradients."""

    class FixedScores(torch.nn.Module):
        effective_score_temperature = 1.0

        def __init__(self) -> None:
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([[0.5], [0.4]], dtype=torch.bfloat16))

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.scores)

    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": temperature}))
    model = FixedScores()
    recipe.model_parts = [model]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.distributed_config = SimpleNamespace()
    recipe.train_n_passages = recipe.val_n_passages = 2
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe._acc_buffer = []
    recipe._validate_model(recipe.model_parts[0])
    batch = {"input_ids": torch.ones(2, 1, dtype=torch.long), "labels": torch.tensor([0])}
    expected_logits = torch.tensor([[0.5, 0.4]], dtype=torch.bfloat16).float()
    expected_logits = expected_logits / temperature
    expected = torch.nn.functional.cross_entropy(expected_logits, batch["labels"])

    losses = []
    recipe._forward_backward_step(0, batch.copy(), loss_buffer=losses, num_batches=1, is_train=True)
    torch.testing.assert_close(losses[0], expected)
    expected_grad = torch.softmax(expected_logits, dim=-1)
    expected_grad[0, 0] -= 1
    expected_grad = expected_grad / temperature
    torch.testing.assert_close(model.scores.grad, expected_grad.reshape(2, 1).to(torch.bfloat16))

    metrics = recipe._run_validation_epoch([batch.copy()])
    assert metrics.metrics["val_loss"] == pytest.approx(expected.item())


def test_cross_encoder_rejects_two_active_temperatures() -> None:
    """A model temperature and recipe temperature must not both scale scores."""
    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": 0.02}))
    model = SimpleNamespace(effective_score_temperature=0.02)

    with pytest.raises(ValueError, match="configured twice"):
        recipe._validate_model(model)


def test_cross_encoder_allows_model_temperature_when_recipe_temperature_is_unit() -> None:
    """The model may own scoring temperature when the recipe divisor is one."""
    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": 1.0}))
    recipe._validate_model(SimpleNamespace(effective_score_temperature=0.02))


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


def test_cross_encoder_nonfinite_loss_is_detected_before_backward(monkeypatch):
    logits = torch.tensor([[float("inf")], [0.0]], requires_grad=True)
    recipe = TrainCrossEncoderRecipe.__new__(TrainCrossEncoderRecipe)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.distributed_config = SimpleNamespace(autocast_dtype=None, defer_fsdp_grad_sync=True)
    recipe.model_parts = [Mock(return_value=SimpleNamespace(logits=logits))]
    recipe.train_n_passages = 2
    recipe.temperature = 1.0
    recipe._acc_buffer = []
    recipe.step_scheduler = SimpleNamespace(step=7, epoch=0)
    recipe._has_nonfinite_across_dp = lambda local_has_nonfinite: local_has_nonfinite
    monkeypatch.setattr(
        "nemo_automodel.recipes.retrieval.train_cross_encoder.get_sync_ctx", lambda *args, **kwargs: nullcontext()
    )

    loss_buffer = []
    skipped_update = recipe._forward_backward_step(
        0,
        {"input_ids": torch.ones(2, 1, dtype=torch.long), "labels": torch.tensor([0])},
        loss_buffer=loss_buffer,
        num_batches=1,
    )

    assert skipped_update is True
    assert len(loss_buffer) == 1
    assert torch.isnan(loss_buffer[0])
    assert logits.grad is None
