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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModel, Mistral3Config, PretrainedConfig, PreTrainedModel

from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon
from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)
from nemo_automodel.components.models.ministral_bidirectional.model import Ministral3BidirectionalModel


def test_llama_nemotron_vl_supported_backbone_for_embedding():
    from nemo_automodel._transformers.retrieval import SUPPORTED_BACKBONES

    assert SUPPORTED_BACKBONES["llama_nemotron_vl"]["embedding"] == "LlamaNemotronVLModel"


class _RemoteCodeConfig(PretrainedConfig):
    model_type = "remote_code_test"

    def __init__(self, hidden_size: int = 4, **kwargs) -> None:
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class _RemoteCodeModel(PreTrainedModel):
    config_class = _RemoteCodeConfig

    def __init__(self, config: _RemoteCodeConfig) -> None:
        super().__init__(config)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()


def _make_local_remote_code_model(source_dir, name_or_path=None) -> _RemoteCodeModel:
    source_dir.mkdir(parents=True)
    (source_dir / "configuration_remote.py").write_text(
        """
from transformers import PretrainedConfig


class _RemoteCodeConfig(PretrainedConfig):
    model_type = "remote_code_test"

    def __init__(self, hidden_size=4, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
""".lstrip()
    )
    (source_dir / "modeling_remote.py").write_text(
        """
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_remote import _RemoteCodeConfig


class _RemoteCodeModel(PreTrainedModel):
    config_class = _RemoteCodeConfig

    def __init__(self, config):
        super().__init__(config)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def forward(self, hidden_states):
        return self.projection(hidden_states)
""".lstrip()
    )

    config = _RemoteCodeConfig()
    config.name_or_path = str(source_dir) if name_or_path is None else name_or_path
    config.auto_map = {
        "AutoConfig": "configuration_remote._RemoteCodeConfig",
        "AutoModel": "modeling_remote._RemoteCodeModel",
    }
    config.architectures = ["_RemoteCodeModel"]
    return _RemoteCodeModel(config)


def _assert_remote_code_round_trip(model: _RemoteCodeModel, checkpoint_dir) -> None:
    assert (checkpoint_dir / "configuration_remote.py").is_file()
    assert (checkpoint_dir / "modeling_remote.py").is_file()

    reloaded = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True)
    assert type(reloaded).__name__ == "_RemoteCodeModel"
    assert set(reloaded.state_dict()) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor)


def test_direct_encoder_save_preserves_and_reloads_matching_remote_code(tmp_path):
    from nemo_automodel._transformers.retrieval import BiEncoderModel, save_encoder_pretrained

    source_dir = tmp_path / "original_checkpoint"
    output_dir = tmp_path / "direct_export"
    hub_id = "nvidia/test-remote-code"
    model = _make_local_remote_code_model(source_dir, name_or_path=hub_id)

    with (
        patch(
            "nemo_automodel._transformers.retrieval.ModelRegistry.has_retrieval_model",
            return_value=True,
        ),
        patch(
            "nemo_automodel.components.checkpoint.addons.snapshot_download",
            return_value=str(source_dir),
        ) as snapshot_download,
    ):
        encoder = BiEncoderModel(model)
        save_encoder_pretrained(encoder, str(output_dir))

    assert encoder.name_or_path == hub_id
    snapshot_download.assert_called_once_with(hub_id, local_files_only=True)
    _assert_remote_code_round_trip(model, output_dir)


def test_consolidated_addon_preserves_and_reloads_matching_remote_code(tmp_path):
    source_dir = tmp_path / "original_checkpoint"
    metadata_dir = tmp_path / "model" / ".hf_metadata"
    consolidated_dir = tmp_path / "model" / "consolidated"
    metadata_dir.mkdir(parents=True)
    consolidated_dir.mkdir(parents=True)
    model = _make_local_remote_code_model(source_dir)
    save_file(
        {key: tensor.detach().clone().contiguous() for key, tensor in model.state_dict().items()},
        consolidated_dir / "model.safetensors",
    )

    addon = ConsolidatedHFAddon()
    addon.pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(metadata_dir),
        fqn_to_file_index_mapping={},
        original_model_path=str(source_dir),
        tokenizer=None,
    )
    addon.post_save(
        consolidated_path=str(consolidated_dir),
        hf_metadata_path=str(metadata_dir),
    )

    _assert_remote_code_round_trip(model, consolidated_dir)


def test_custom_encoder_exports_local_code_when_remote_class_differs():
    from nemo_automodel._transformers.retrieval import _init_encoder_common

    config = _RemoteCodeConfig()
    config.name_or_path = "/original/checkpoint"
    config.auto_map = {
        "AutoConfig": "configuration_remote.RemoteCodeConfig",
        "AutoModel": "modeling_remote.DifferentModel",
    }
    model = _RemoteCodeModel(config)
    encoder = nn.Module()

    with (
        patch(
            "nemo_automodel._transformers.retrieval.ModelRegistry.has_retrieval_model",
            return_value=True,
        ),
        patch(
            "nemo_automodel._transformers.retrieval.inspect.getfile",
            return_value="/automodel/retrieval_model/model.py",
        ),
    ):
        _init_encoder_common(encoder, model)

    assert encoder.name_or_path == "/automodel/retrieval_model"
    assert config.auto_map == {
        "AutoConfig": "test_retrieval._RemoteCodeConfig",
        "AutoModel": "test_retrieval._RemoteCodeModel",
    }


def test_matching_remote_code_model_can_opt_in_to_export_its_paired_processor():
    from nemo_automodel._transformers.retrieval import _init_encoder_common

    config = _RemoteCodeConfig()
    config.name_or_path = "/original/checkpoint"
    config.auto_map = {
        "AutoConfig": "configuration_remote._RemoteCodeConfig",
        "AutoModel": "modeling_remote._RemoteCodeModel",
        "AutoProcessor": "processing_remote.RemoteProcessor",
    }
    model = _RemoteCodeModel(config)
    model._export_original_processor_with_remote_code = True
    encoder = nn.Module()

    _init_encoder_common(encoder, model)

    assert encoder.name_or_path == "/original/checkpoint"
    assert encoder._export_processor_auto_map == "processing_remote.RemoteProcessor"


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


def test_save_encoder_pretrained_adapts_and_materializes_state_dict(tmp_path):
    """Direct retrieval saves should pass HF-format contiguous tensors to save_pretrained."""
    from nemo_automodel._transformers.retrieval import save_encoder_pretrained

    class Adapter:
        def to_hf(self, state_dict, **kwargs):
            del kwargs
            return {"split.weight": state_dict["model.fused.weight"][:, :1]}

    model = MagicMock()
    model.state_dict.return_value = {"model.fused.weight": torch.arange(4).reshape(2, 2)}
    model.state_dict_adapter = Adapter()

    save_encoder_pretrained(model, str(tmp_path))

    model.model.save_pretrained.assert_called_once()
    saved_state_dict = model.model.save_pretrained.call_args.kwargs["state_dict"]
    assert torch.equal(saved_state_dict["split.weight"], torch.tensor([[0], [2]]))
    assert saved_state_dict["split.weight"].is_contiguous()


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
