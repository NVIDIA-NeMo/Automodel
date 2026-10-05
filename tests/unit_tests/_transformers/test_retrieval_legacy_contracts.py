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

"""Behavioral regressions for legacy models using the shared retrieval contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaModel

from nemo_automodel._transformers.retrieval import BiEncoderModel, CrossEncoderModel
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalConfig,
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)
from nemo_automodel.components.models.llama_nemotron_vl import model as vl_model
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe


def _text_config(**kwargs) -> LlamaBidirectionalConfig:
    return LlamaBidirectionalConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_dropout=0.0,
        pad_token_id=0,
        **kwargs,
    )


@pytest.mark.parametrize("is_causal", [None, False, True])
def test_nemotron_vl_default_wrapper_and_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, is_causal: bool | None
) -> None:
    """The real composite model loads without downloads and preserves its towers."""
    monkeypatch.setattr(vl_model.AutoProcessor, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    config = vl_model.LlamaNemotronVLConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 8,
            "image_size": 4,
            "patch_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 16,
        },
        llm_config={
            "architectures": ["LlamaBidirectionalModel"],
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
        },
        pooling="avg",
        img_context_token_id=31,
    )
    config.llm_config._attn_implementation = "eager"
    backbone = vl_model.LlamaNemotronVLModel(config).eval()
    encoder = BiEncoderModel(backbone, is_causal=is_causal).eval()
    assert config.get_text_config(decoder=True) is config.llm_config
    assert config.get_text_config() is config.llm_config
    assert config.get_text_config(encoder=True) is config
    assert backbone.get_decoder() is backbone.language_model
    assert encoder.is_causal is (is_causal is True)
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with torch.no_grad():
        expected = encoder(inputs)
    assert torch.isfinite(expected).all()
    backbone.save_pretrained(tmp_path)
    reloaded = BiEncoderModel.build(str(tmp_path), attn_implementation="eager").eval()
    assert reloaded.is_causal is encoder.is_causal
    assert reloaded.model.state_dict().keys() == backbone.state_dict().keys()
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(reloaded.model.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("model_family", ["llama", "nemotron_vl_text"])
def test_legacy_causality_changes_attention_behavior(
    attn_implementation: str, is_causal: bool, model_family: str, tmp_path: Path
) -> None:
    """Future-token influence, including after reload, follows the selected mode."""
    torch.manual_seed(42)
    config = _text_config()
    config._attn_implementation = attn_implementation
    model_class = LlamaBidirectionalModel if model_family == "llama" else vl_model.LlamaBidirectionalModel
    backbone = model_class(config)
    encoder = BiEncoderModel(backbone, pooling="cls", l2_normalize=False, is_causal=is_causal).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}
    changed = {**inputs, "input_ids": torch.tensor([[1, 2, 3, 5, 0]])}
    with torch.no_grad():
        expected = encoder(inputs)
        perturbed = encoder(changed)
    if is_causal:
        torch.testing.assert_close(expected, perturbed, rtol=0, atol=0)
    else:
        assert not torch.allclose(expected, perturbed, atol=1e-6)
    backbone.save_pretrained(tmp_path)
    reloaded = model_class.from_pretrained(tmp_path, attn_implementation=attn_implementation).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs).last_hidden_state[:, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_legacy_llama_forward_and_gradients_match_hf(attn_implementation: str, is_causal: bool) -> None:
    """Both attention modes preserve HF outputs, parameter layout and gradients."""
    torch.manual_seed(7)
    config = _text_config(is_causal=is_causal)
    config._attn_implementation = attn_implementation
    model = LlamaBidirectionalModel(config).eval()
    reference = LlamaModel(config).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    actual = model(**inputs).last_hidden_state
    expected = reference(**inputs).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for name, parameter in model.named_parameters():
        reference_parameter = reference.get_parameter(name)
        assert parameter.grad is not None and reference_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("model_temperature,recipe_temperature", [(0.5, 0.3), (0.5, 1.0), (1.0, 0.3)])
def test_legacy_reranker_temperature_validation_and_loss(model_temperature: float, recipe_temperature: float) -> None:
    """Real wrapper validation prevents applying two non-unit temperatures."""
    torch.manual_seed(42)
    config = _text_config(temperature=model_temperature, num_labels=1)
    config._attn_implementation = "eager"
    encoder = CrossEncoderModel(LlamaBidirectionalForSequenceClassification(config)).eval()
    assert encoder.effective_score_temperature == model_temperature
    recipe = TrainCrossEncoderRecipe(ConfigNode({"temperature": recipe_temperature}))
    if model_temperature != 1.0 and recipe_temperature != 1.0:
        with pytest.raises(ValueError, match="temperature scaling is configured twice"):
            recipe._validate_model(encoder)
        return
    recipe._validate_model(encoder)
    recipe.model_parts = [encoder]
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.distributed_config = None
    recipe.train_n_passages = 2
    recipe._acc_buffer = []
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([0]),
    }
    with torch.no_grad():
        logits = encoder({key: value for key, value in batch.items() if key != "labels"}).logits.reshape(1, 2)
        expected = torch.nn.functional.cross_entropy(logits.float() / recipe_temperature, batch["labels"])
        losses = []
        recipe._forward_backward_step(0, batch, loss_buffer=losses, num_batches=1, is_train=False)
    torch.testing.assert_close(losses[0], expected, rtol=0, atol=0)
