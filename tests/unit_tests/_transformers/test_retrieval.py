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

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, Mistral3Config, Qwen2Config, Qwen2Model

from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)
from nemo_automodel.components.models.ministral_bidirectional.model import Ministral3BidirectionalModel


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


def _tiny_qwen2_config() -> Qwen2Config:
    """Build a tiny unsupported decoder-only config for generic backbone tests."""
    return Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )


def _assert_state_dict_equal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    assert set(expected) == set(actual)
    for key, tensor in expected.items():
        assert torch.equal(tensor, actual[key]), f"Weight mismatch for {key}"


def _assert_no_language_model_prefix(model: nn.Module) -> None:
    for key in model.state_dict():
        assert not key.startswith("language_model."), f"VLM prefix in key: {key}"


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


def test_generic_embedding_backbone_persists_noncausal_config_and_is_bidirectional(tmp_path):
    """Generic HF embedding backbones persist non-causal config and run bidirectionally."""
    from nemo_automodel._transformers import retrieval

    torch.manual_seed(1234)
    model_dir = tmp_path / "qwen2"
    Qwen2Model(_tiny_qwen2_config()).eval().save_pretrained(model_dir)

    encoder = retrieval.BiEncoderModel.build(
        model_name_or_path=str(model_dir),
        pooling="avg",
        l2_normalize=False,
        is_causal=False,
        attn_implementation="eager",
    )

    assert "qwen2" not in retrieval.SUPPORTED_BACKBONES
    assert encoder.model.config.is_causal is False
    assert all(getattr(layer.self_attn, "is_causal", True) is False for layer in encoder.model.layers)

    input_ids = torch.tensor([[1, 2, 3, 0]])
    modified_input_ids = torch.tensor([[1, 2, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    encoder.model.eval()
    with torch.no_grad():
        base_outputs = encoder.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        modified_outputs = encoder.model(input_ids=modified_input_ids, attention_mask=attention_mask).last_hidden_state

    assert not torch.allclose(base_outputs[:, 0], modified_outputs[:, 0], atol=1e-7, rtol=1e-7)

    save_dir = tmp_path / "saved_qwen2_encoder"
    encoder.save_pretrained(str(save_dir))
    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["is_causal"] is False


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


def test_extract_submodel_ministral_embedding_from_local_vlm_converts_to_supported_backbone(tmp_path):
    """The real Ministral3 VLM text backbone path becomes the Ministral bi-encoder."""
    from nemo_automodel._transformers import retrieval

    model_dir, language_state_dict = _save_tiny_vlm(tmp_path, "ministral3")

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        extract_submodel="language_model",
    )

    assert isinstance(backbone, Ministral3BidirectionalModel)
    assert backbone.config.model_type == "ministral3_bidirec"
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
