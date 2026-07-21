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

"""Functional tests for retrieval backbone extraction."""

import json
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, Ministral3Config, Mistral3Config
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3ForSequenceClassification,
    Ministral3Model,
)

from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)


def test_llama_nemotron_vl_supported_backbone_for_embedding():
    from nemo_automodel._transformers.retrieval import SUPPORTED_BACKBONES

    assert SUPPORTED_BACKBONES["llama_nemotron_vl"]["embedding"] == "LlamaNemotronVLModel"


def _tiny_mistral3_vlm_config(text_model_type: str) -> Mistral3Config:
    """Build a tiny Mistral3 VLM config with a selectable text backbone."""
    text_config = {
        "model_type": text_model_type,
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }
    if text_model_type in {"mistral", "ministral3"}:
        text_config["head_dim"] = 8

    return Mistral3Config(
        text_config=text_config,
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 4,
            "num_channels": 3,
        },
    )


def _save_tiny_vlm(tmp_path, text_model_type: str):
    """Save a tiny local VLM checkpoint and return its language-model weights."""
    model = AutoModel.from_config(_tiny_mistral3_vlm_config(text_model_type))
    model_dir = tmp_path / f"{text_model_type}_vlm"
    model.save_pretrained(model_dir)
    language_state_dict = {key: tensor.detach().clone() for key, tensor in model.language_model.state_dict().items()}
    return model_dir, language_state_dict


def _save_tiny_ministral_text_model(tmp_path):
    """Save a tiny stock Ministral text checkpoint and return its weights."""
    config = Ministral3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Ministral3Model(config)
    model_dir = tmp_path / "ministral3_text"
    model.save_pretrained(model_dir)
    state_dict = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    return model_dir, state_dict


def _assert_state_dict_equal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    assert set(expected) == set(actual)
    for key, tensor in expected.items():
        assert torch.equal(tensor, actual[key]), f"Weight mismatch for {key}"


def _assert_no_language_model_prefix(model: nn.Module) -> None:
    for key in model.state_dict():
        assert not key.startswith("language_model."), f"VLM prefix in key: {key}"


@pytest.mark.parametrize(("kwargs", "expected_is_final"), [({}, False), ({"is_final_checkpoint": True}, True)])
def test_save_encoder_pretrained_forwards_is_final_checkpoint(tmp_path, kwargs, expected_is_final):
    """Direct retrieval saves default to non-final unless the caller says otherwise."""
    from nemo_automodel._transformers.retrieval import save_encoder_pretrained

    model = nn.Module()
    checkpointer = MagicMock()

    save_encoder_pretrained(model, str(tmp_path), checkpointer=checkpointer, **kwargs)

    checkpointer.save_model.assert_called_once_with(
        model=model,
        weights_path=str(tmp_path),
        peft_config=None,
        tokenizer=None,
        is_final_checkpoint=expected_is_final,
    )


def test_extract_submodel_unsupported_embedding_from_local_vlm(tmp_path):
    """Unsupported extracted text backbones are returned directly for bi-encoder use."""
    from nemo_automodel._transformers import retrieval

    model_dir, language_state_dict = _save_tiny_vlm(tmp_path, "mistral")

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        extract_submodel="language_model",
    )

    assert backbone.__class__.__name__ == "MistralModel"
    assert backbone.config.model_type == "mistral"
    _assert_no_language_model_prefix(backbone)
    _assert_state_dict_equal(language_state_dict, backbone.state_dict())

    save_dir = tmp_path / "mistral_text_backbone"
    backbone.save_pretrained(save_dir)
    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["model_type"] == "mistral"


def test_extract_submodel_llama_embedding_from_local_vlm_converts_to_supported_backbone(tmp_path):
    """A supported extracted Llama text backbone becomes the retrieval Llama encoder."""
    from nemo_automodel._transformers import retrieval

    model_dir, language_state_dict = _save_tiny_vlm(tmp_path, "llama")

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        extract_submodel="language_model",
        pooling="avg",
    )

    assert isinstance(backbone, LlamaBidirectionalModel)
    assert backbone.config.model_type == "llama_bidirec"
    assert backbone.config.pooling == "avg"
    assert all(getattr(layer.self_attn, "is_causal", True) is False for layer in backbone.layers)
    _assert_state_dict_equal(language_state_dict, backbone.state_dict())

    input_ids = torch.randint(0, backbone.config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.last_hidden_state.shape == (2, 8, backbone.config.hidden_size)


def test_ministral_embedding_forwards_hf_kwargs_to_config_and_model(monkeypatch):
    """Ministral config and weights receive the same native loader options."""
    from nemo_automodel._transformers import retrieval

    config = MagicMock()
    config.model_type = "ministral3"
    backbone = MagicMock()
    auto_config_from_pretrained = MagicMock(return_value=config)
    auto_model_from_pretrained = MagicMock(return_value=backbone)
    monkeypatch.setattr(retrieval.AutoConfig, "from_pretrained", auto_config_from_pretrained)
    monkeypatch.setattr(retrieval.AutoModel, "from_pretrained", auto_model_from_pretrained)

    result = retrieval.build_encoder_backbone(
        model_name_or_path="org/model",
        task="embedding",
        trust_remote_code=True,
        revision="revision-a",
        use_auth_token="legacy-token",
        subfolder="encoder",
        token="token",
        cache_dir="/cache",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )

    assert result is backbone
    auto_config_from_pretrained.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="revision-a",
        subfolder="encoder",
        token="token",
        cache_dir="/cache",
        local_files_only=True,
        use_auth_token="legacy-token",
        torch_dtype=torch.bfloat16,
    )
    auto_model_from_pretrained.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="revision-a",
        subfolder="encoder",
        token="token",
        cache_dir="/cache",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        use_auth_token="legacy-token",
    )


def test_standard_ministral_score_uses_sequence_classification_model(tmp_path):
    """Standard Ministral score checkpoints retain the HuggingFace reranker path."""
    from nemo_automodel._transformers import retrieval

    model_dir, source_state_dict = _save_tiny_ministral_text_model(tmp_path)

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="score",
        num_labels=1,
    )

    assert type(backbone) is Ministral3ForSequenceClassification
    assert backbone.config.model_type == "ministral3"
    assert backbone.config.num_labels == 1
    _assert_state_dict_equal(source_state_dict, backbone.model.state_dict())

    input_ids = torch.randint(0, backbone.config.vocab_size, (2, 4))
    attention_mask = torch.ones_like(input_ids)
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape == (2, 1)


def test_ministral_embedding_preserves_hf_config_overrides(tmp_path):
    """Valid HuggingFace config overrides retain native loader behavior."""
    from nemo_automodel._transformers import retrieval

    model_dir, _ = _save_tiny_ministral_text_model(tmp_path)

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        output_attentions=True,
    )

    assert backbone.config.output_attentions is True


def test_ministral_embedding_uses_stock_bidirectional_model(tmp_path):
    """Standard Ministral checkpoints use and save the stock non-causal model."""
    from nemo_automodel._transformers import retrieval

    model_dir, source_state_dict = _save_tiny_ministral_text_model(tmp_path)

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        pooling="avg",
    )

    assert type(backbone) is Ministral3Model
    assert backbone.config.model_type == "ministral3"
    assert backbone.config.is_causal is False
    assert backbone.config.pooling == "avg"
    assert backbone.config.sliding_window == 4
    assert backbone.config._attn_implementation == "sdpa"
    _assert_state_dict_equal(source_state_dict, backbone.state_dict())

    input_ids = torch.randint(0, backbone.config.vocab_size, (1, 4))
    attention_mask = torch.ones_like(input_ids)
    modified_input_ids = input_ids.clone()
    modified_input_ids[0, -1] = (modified_input_ids[0, -1] + 1) % backbone.config.vocab_size
    backbone.eval()
    with torch.no_grad():
        original_output = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        modified_output = backbone(input_ids=modified_input_ids, attention_mask=attention_mask).last_hidden_state
    assert not torch.allclose(original_output[0, 0], modified_output[0, 0])

    save_dir = tmp_path / "saved_stock_ministral"
    backbone.save_pretrained(save_dir)

    assert not list(save_dir.glob("*.py"))
    reloaded = AutoModel.from_pretrained(save_dir)
    assert type(reloaded) is Ministral3Model
    assert reloaded.config.model_type == "ministral3"
    assert reloaded.config.is_causal is False
    assert getattr(reloaded.config, "auto_map", None) is None
    _assert_state_dict_equal(source_state_dict, reloaded.state_dict())


def test_ministral_embedding_uses_bidirectional_flash_attention(tmp_path, monkeypatch):
    """Stock Ministral selects non-causal attention at the FlashAttention kernel boundary."""
    from transformers.integrations import flash_attention

    from nemo_automodel._transformers import retrieval

    model_dir, _ = _save_tiny_ministral_text_model(tmp_path)
    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        pooling="avg",
    )
    assert backbone.config.is_causal is False
    backbone.config._attn_implementation = "flash_attention_2"
    assert all(layer.self_attn.is_causal is True for layer in backbone.layers)

    kernel_calls = []

    def record_flash_attention(query, key, value, attention_mask, **kwargs):
        kernel_calls.append({"is_causal": kwargs.get("is_causal"), "attention_mask": attention_mask})
        return torch.zeros_like(query)

    monkeypatch.setattr(flash_attention, "_flash_attention_forward", record_flash_attention)

    input_ids = torch.randint(0, backbone.config.vocab_size, (1, 4))
    backbone(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert len(kernel_calls) == backbone.config.num_hidden_layers
    assert all(call == {"is_causal": False, "attention_mask": None} for call in kernel_calls)


def test_extract_submodel_ministral_embedding_from_local_vlm_converts_to_supported_backbone(tmp_path):
    """A Ministral3 VLM text backbone becomes a stock non-causal Ministral model."""
    from nemo_automodel._transformers import retrieval

    model_dir, language_state_dict = _save_tiny_vlm(tmp_path, "ministral3")

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        extract_submodel="language_model",
        pooling="avg",
    )

    assert type(backbone) is Ministral3Model
    assert backbone.config.model_type == "ministral3"
    assert backbone.config.is_causal is False
    assert backbone.config.pooling == "avg"
    _assert_state_dict_equal(language_state_dict, backbone.state_dict())

    input_ids = torch.randint(0, backbone.config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.last_hidden_state.shape == (2, 8, backbone.config.hidden_size)


def test_extract_submodel_llama_score_from_local_vlm_converts_to_supported_cross_encoder(tmp_path):
    """A supported extracted Llama text backbone becomes the retrieval reranker."""
    from nemo_automodel._transformers import retrieval

    model_dir, language_state_dict = _save_tiny_vlm(tmp_path, "llama")

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="score",
        extract_submodel="language_model",
        num_labels=1,
        pooling="avg",
        temperature=0.5,
    )

    assert isinstance(backbone, LlamaBidirectionalForSequenceClassification)
    assert backbone.config.model_type == "llama_bidirec"
    assert backbone.config.num_labels == 1
    assert backbone.config.pooling == "avg"
    assert backbone.config.temperature == 0.5
    _assert_state_dict_equal(language_state_dict, backbone.model.state_dict())

    input_ids = torch.randint(0, backbone.config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape == (2, 1)


def test_extract_submodel_ministral_score_from_local_vlm_converts_to_hf_cross_encoder(tmp_path):
    """Reranking still works when no registered score backbone exists for the text model."""
    from nemo_automodel._transformers import retrieval

    model_dir, language_state_dict = _save_tiny_vlm(tmp_path, "ministral3")

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="score",
        extract_submodel="language_model",
        num_labels=1,
    )

    assert backbone.__class__.__name__ == "Ministral3ForSequenceClassification"
    assert backbone.config.model_type == "ministral3"
    assert backbone.config.num_labels == 1
    _assert_state_dict_equal(language_state_dict, backbone.model.state_dict())

    input_ids = torch.randint(0, backbone.config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape == (2, 1)


class _PlainSubmodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


def test_extract_submodel_without_config_raises():
    """The extracted object must carry a config so it can be saved/reloaded."""
    from nemo_automodel._transformers.retrieval import _extract_submodel

    model = nn.Module()
    model.language_model = _PlainSubmodule()

    with pytest.raises(ValueError, match="has no .config attribute"):
        _extract_submodel(model, "language_model")
