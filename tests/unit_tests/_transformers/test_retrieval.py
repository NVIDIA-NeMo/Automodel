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

import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModel,
    LlamaConfig,
    Ministral3Config,
    Mistral3Config,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3ForSequenceClassification,
    Ministral3Model,
)

from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalConfig,
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Ministral3BidirectionalConfig,
    Ministral3BidirectionalModel,
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


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer_backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_max_length=32,
    )


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


def test_bi_encoder_public_api_excludes_export_format_overrides():
    from nemo_automodel._transformers import auto_model, retrieval

    export_only_parameters = {
        "query_prompt",
        "document_prompt",
        "sentence_transformer_max_seq_length",
        "similarity_fn_name",
        "do_lower_case",
    }
    for callable_ in (
        retrieval.BiEncoderModel.__init__,
        retrieval.BiEncoderModel.build,
        auto_model.NeMoAutoModelBiEncoder.from_pretrained,
    ):
        parameters = inspect.signature(callable_).parameters
        assert export_only_parameters.isdisjoint(parameters)
        assert {"pooling", "l2_normalize"} <= parameters.keys()

    for parameter in export_only_parameters:
        with pytest.raises(TypeError, match="derived from effective NeMo settings"):
            retrieval.BiEncoderModel.build("unused", **{parameter: None})


def test_effective_pipeline_prompts_replace_restored_export_defaults():
    from nemo_automodel._transformers import retrieval

    encoder = SimpleNamespace(
        sentence_transformer_export_config=retrieval.SentenceTransformerExportConfig(
            query_prompt="saved query: ",
            document_prompt="saved document: ",
        )
    )

    retrieval.BiEncoderModel.configure_sentence_transformer_prompts(
        encoder,
        query_prompt="current query: ",
        document_prompt="current document: ",
    )

    assert encoder.sentence_transformer_export_config.query_prompt == "current query: "
    assert encoder.sentence_transformer_export_config.document_prompt == "current document: "


def test_direct_standard_export_without_tokenizer_rejects_before_writing(tmp_path):
    from nemo_automodel._transformers import retrieval

    backbone = LlamaBidirectionalModel(
        LlamaBidirectionalConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling="avg", l2_normalize=False)
    save_dir = tmp_path / "missing_tokenizer"

    with pytest.raises(ValueError, match="tokenizer is required"):
        encoder.save_pretrained(save_dir)

    assert not save_dir.exists()


def test_direct_standard_export_uses_general_sequence_capabilities(tmp_path):
    from nemo_automodel._transformers import retrieval

    backbone = LlamaBidirectionalModel(
        LlamaBidirectionalConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling="avg", l2_normalize=True)
    tokenizer = _tiny_tokenizer()
    save_dir = tmp_path / "derived_export"

    encoder.save_pretrained(save_dir, tokenizer=tokenizer)

    metadata = json.loads((save_dir / "sentence_bert_config.json").read_text())
    assert metadata == {"max_seq_length": tokenizer.model_max_length, "do_lower_case": False}


def test_cached_source_model_path_uses_exact_loaded_revision(tmp_path, monkeypatch):
    from nemo_automodel._transformers import retrieval

    local_root = tmp_path / "local"
    (local_root / "encoder").mkdir(parents=True)
    assert retrieval._resolve_cached_source_model_path(
        str(local_root), SimpleNamespace(), {"subfolder": "encoder"}
    ) == str(local_root / "encoder")
    assert retrieval._resolve_cached_source_repository_path(
        str(local_root), str(local_root / "encoder"), {"subfolder": "encoder"}
    ) == str(local_root)

    cached_config = tmp_path / "snapshot" / "encoder" / "config.json"
    cached_config.parent.mkdir(parents=True)
    cached_config.write_text("{}")
    cache_lookup = MagicMock(return_value=str(cached_config))
    monkeypatch.setattr(retrieval, "try_to_load_from_cache", cache_lookup)

    result = retrieval._resolve_cached_source_model_path(
        "org/model",
        SimpleNamespace(_commit_hash="exact-commit"),
        {"cache_dir": "/cache", "revision": "branch", "subfolder": "encoder"},
    )

    assert result == str(cached_config.parent)
    assert retrieval._resolve_cached_source_repository_path("org/model", result, {"subfolder": "encoder"}) == str(
        tmp_path / "snapshot"
    )
    cache_lookup.assert_called_once_with(
        "org/model",
        "encoder/config.json",
        cache_dir="/cache",
        revision="exact-commit",
    )


def test_cache_hub_source_legal_assets_uses_exact_loaded_revision(monkeypatch):
    from nemo_automodel._transformers import retrieval

    snapshot_download = MagicMock(return_value="/cache/snapshot")
    monkeypatch.setattr(retrieval, "snapshot_download", snapshot_download)

    result = retrieval._cache_hub_source_legal_assets(
        "org/model",
        SimpleNamespace(_commit_hash="exact-commit"),
        {
            "cache_dir": "/cache",
            "revision": "branch",
            "use_auth_token": "token",
            "local_files_only": True,
        },
    )

    assert result == "/cache/snapshot"
    snapshot_download.assert_called_once_with(
        repo_id="org/model",
        allow_patterns=retrieval._SOURCE_LEGAL_ASSET_PATTERNS,
        cache_dir="/cache",
        local_files_only=True,
        revision="exact-commit",
        token="token",
    )


def test_direct_standard_export_preserves_cached_source_deployment_limit(tmp_path):
    from nemo_automodel._transformers import retrieval

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "sentence_bert_config.json").write_text('{"max_seq_length": 16}')

    backbone = LlamaBidirectionalModel(
        LlamaBidirectionalConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    )
    encoder = retrieval.BiEncoderModel(
        backbone,
        pooling="avg",
        l2_normalize=True,
    )
    encoder.source_model_path = str(source_dir)
    save_dir = tmp_path / "source_limit"
    encoder.save_pretrained(save_dir, tokenizer=_tiny_tokenizer())

    assert json.loads((save_dir / "sentence_bert_config.json").read_text())["max_seq_length"] == 16


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


@pytest.mark.parametrize("pooling", ["weighted_avg", "colbert", "multi_vector"])
def test_bi_encoder_skips_standard_export_for_unrepresentable_pooling(pooling, tmp_path):
    from nemo_automodel._transformers import retrieval

    backbone = LlamaBidirectionalModel(
        LlamaBidirectionalConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling=pooling, l2_normalize=True)

    assert encoder.sentence_transformer_export_config is None
    encoder.configure_sentence_transformer_prompts(query_prompt="query: ", document_prompt="passage: ")
    save_dir = tmp_path / pooling
    encoder.save_pretrained(save_dir)
    assert (save_dir / "config.json").exists()
    assert not (save_dir / "modules.json").exists()


def test_bi_encoder_skips_standard_export_for_multimodal_backbone():
    from nemo_automodel._transformers import retrieval

    class CompositeBackbone(nn.Module):
        main_input_name = "pixel_values"

        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig()
            self.config.is_composition = True
            self.config.llm_config = PretrainedConfig(hidden_size=16)
            self.config.name_or_path = ""

    encoder = retrieval.BiEncoderModel(CompositeBackbone(), pooling="last", l2_normalize=True)

    assert encoder.sentence_transformer_export_config is None
    encoder.configure_sentence_transformer_prompts(query_prompt="query: ", document_prompt="passage: ")


def test_bi_encoder_export_config_uses_deployable_hf_base_classes():
    from nemo_automodel._transformers import retrieval

    backbone = LlamaBidirectionalModel(
        LlamaBidirectionalConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            pooling="avg",
        )
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling="cls", l2_normalize=True)
    encoder.configure_sentence_transformer_prompts(query_prompt="query: ", document_prompt="passage: ")

    export_config = encoder.get_hf_export_config()

    assert isinstance(export_config, LlamaConfig)
    assert not isinstance(export_config, LlamaBidirectionalConfig)
    assert export_config.model_type == "llama"
    assert export_config.architectures == ["LlamaModel"]
    assert getattr(export_config, "auto_map", None) is None
    assert export_config.is_causal is False
    assert export_config.pooling == "cls"
    assert encoder.config.model_type == "llama_bidirec"


def test_bi_encoder_export_config_uses_class_model_type_when_source_type_is_retained():
    from nemo_automodel._transformers import retrieval

    config = Ministral3BidirectionalConfig.from_dict(
        {
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "pooling": "avg",
        }
    )
    backbone = Ministral3BidirectionalModel(config)
    encoder = retrieval.BiEncoderModel(backbone, pooling="avg", l2_normalize=True)

    assert encoder.config.model_type == "ministral3"
    assert type(encoder.config).model_type == "ministral3_bidirec"

    export_config = encoder.get_hf_export_config()

    assert isinstance(export_config, Ministral3Config)
    assert not isinstance(export_config, Ministral3BidirectionalConfig)
    assert export_config.model_type == "ministral3"
    assert export_config.architectures == ["Ministral3Model"]
    assert getattr(export_config, "auto_map", None) is None
    assert export_config.is_causal is False
    assert export_config.pooling == "avg"


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

    encoder = retrieval.BiEncoderModel(backbone, pooling="avg", l2_normalize=True)
    encoder.configure_sentence_transformer_prompts(query_prompt="query: ", document_prompt="")
    export_config = encoder.get_hf_export_config()

    assert type(export_config) is Ministral3Config
    assert export_config.architectures == ["Ministral3Model"]
    assert export_config.is_causal is False
    assert getattr(export_config, "auto_map", None) is None

    save_dir = tmp_path / "saved_stock_ministral"
    encoder.save_pretrained(save_dir, tokenizer=_tiny_tokenizer())

    assert not list(save_dir.glob("*.py"))
    reloaded = AutoModel.from_pretrained(save_dir)
    assert (save_dir / "tokenizer.json").exists()
    assert json.loads((save_dir / "modules.json").read_text()) == [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ]
    assert json.loads((save_dir / "config_sentence_transformers.json").read_text())["prompts"]["query"] == "query: "

    assert type(reloaded) is Ministral3Model
    assert reloaded.config.model_type == "ministral3"
    assert reloaded.config.architectures == ["Ministral3Model"]
    assert reloaded.config.is_causal is False
    assert getattr(reloaded.config, "auto_map", None) is None
    _assert_state_dict_equal(source_state_dict, reloaded.state_dict())
    assert reloaded.config.sliding_window == 4


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
    assert hasattr(backbone.config, "_attn_implementation")
    backbone.config._attn_implementation = "flash_attention_2"
    assert all(layer.self_attn.is_causal is True for layer in backbone.layers)

    kernel_calls = []

    def record_flash_attention(query, key, value, attention_mask, **kwargs):
        kernel_calls.append(
            {
                "is_causal": kwargs.get("is_causal"),
                "attention_mask": attention_mask,
            }
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(
        flash_attention,
        "_flash_attention_forward",
        record_flash_attention,
    )

    input_ids = torch.randint(0, backbone.config.vocab_size, (1, 4))
    backbone(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert len(kernel_calls) == backbone.config.num_hidden_layers
    for call in kernel_calls:
        assert call["attention_mask"] is None
        assert call["is_causal"] is False


@pytest.mark.parametrize(
    ("pooling", "l2_normalize"),
    [
        ("avg", True),
        ("avg", False),
        ("cls", True),
        ("cls", False),
        ("last", True),
        ("last", False),
    ],
)
def test_sentence_transformers_and_nemo_round_trip_generated_ministral_checkpoint(
    tmp_path,
    monkeypatch,
    pooling,
    l2_normalize,
):
    from sentence_transformers import SentenceTransformer

    from nemo_automodel._transformers import auto_model as auto_model_module
    from nemo_automodel._transformers import retrieval

    model_dir, _ = _save_tiny_ministral_text_model(tmp_path)
    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        pooling=pooling,
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling=pooling, l2_normalize=l2_normalize)
    encoder.configure_sentence_transformer_prompts(query_prompt="query: ", document_prompt="passage: ")
    tokenizer = _tiny_tokenizer()
    raw_texts = ["hello", "hello world"]
    query_texts = [f"query: {text}" for text in raw_texts]
    document_texts = [f"passage: {text}" for text in raw_texts]
    encoder.eval()
    with torch.no_grad():
        expected_query = encoder(tokenizer(query_texts, padding=True, return_tensors="pt"))
        expected_document = encoder(tokenizer(document_texts, padding=True, return_tensors="pt"))
    save_dir = tmp_path / f"sentence_transformers_ministral_{pooling}_{l2_normalize}"
    encoder.save_pretrained(save_dir, tokenizer=tokenizer)

    sentence_transformer = SentenceTransformer(str(save_dir), device="cpu")
    actual_query = sentence_transformer.encode_query(raw_texts, convert_to_tensor=True)
    actual_document = sentence_transformer.encode_document(raw_texts, convert_to_tensor=True)

    setup = SimpleNamespace(
        mesh_context=None,
        strategy_config=None,
        moe_parallel_config=None,
        activation_checkpointing=None,
    )
    monkeypatch.setattr(auto_model_module, "_resolve_distributed_setup", lambda **_: setup)
    monkeypatch.setattr(
        auto_model_module,
        "instantiate_infrastructure",
        lambda **_: (None, None, None, None),
    )
    monkeypatch.setattr(auto_model_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(auto_model_module, "apply_model_infrastructure", lambda model, **_: model)
    nemo_reloaded = auto_model_module.NeMoAutoModelBiEncoder.from_pretrained(
        str(save_dir),
        attn_implementation="eager",
        use_liger_kernel=False,
        use_sdpa_patching=False,
    )
    nemo_reloaded.eval()
    with torch.no_grad():
        nemo_query = nemo_reloaded(tokenizer(query_texts, padding=True, return_tensors="pt"))
        nemo_document = nemo_reloaded(tokenizer(document_texts, padding=True, return_tensors="pt"))

    assert nemo_reloaded.pooling == pooling
    assert nemo_reloaded.l2_normalize is l2_normalize
    assert actual_query.shape == actual_document.shape == nemo_query.shape == nemo_document.shape == (2, 16)
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_document, expected_document)
    torch.testing.assert_close(nemo_query, expected_query)
    torch.testing.assert_close(nemo_document, expected_document)


def test_nemo_bi_encoder_explicit_options_override_sentence_transformer_metadata(tmp_path):
    from nemo_automodel._transformers import retrieval

    model_dir, _ = _save_tiny_ministral_text_model(tmp_path)
    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        pooling="cls",
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling="cls", l2_normalize=False)
    save_dir = tmp_path / "explicit_override"
    encoder.save_pretrained(save_dir, tokenizer=_tiny_tokenizer())

    reloaded = retrieval.BiEncoderModel.build(
        str(save_dir),
        pooling="last",
        l2_normalize=True,
    )

    assert reloaded.pooling == "last"
    assert reloaded.l2_normalize is True


def test_nemo_bi_encoder_saved_prompts_round_trip_through_reexport(tmp_path):
    from nemo_automodel._transformers import retrieval

    model_dir, _ = _save_tiny_ministral_text_model(tmp_path)
    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=str(model_dir),
        task="embedding",
        pooling="avg",
    )
    encoder = retrieval.BiEncoderModel(backbone, pooling="avg", l2_normalize=True)
    encoder.configure_sentence_transformer_prompts(query_prompt="query: ", document_prompt="passage: ")
    tokenizer = _tiny_tokenizer()
    first_export = tmp_path / "first_export"
    encoder.save_pretrained(first_export, tokenizer=tokenizer)

    reloaded = retrieval.BiEncoderModel.build(str(first_export))
    assert reloaded.sentence_transformer_export_config.query_prompt == "query: "
    assert reloaded.sentence_transformer_export_config.document_prompt == "passage: "

    second_export = tmp_path / "second_export"
    reloaded.save_pretrained(second_export, tokenizer=tokenizer)
    metadata = json.loads((second_export / "config_sentence_transformers.json").read_text())
    assert metadata["prompts"] == {"query": "query: ", "document": "passage: "}


def test_nemo_bi_encoder_uses_defaults_without_sentence_transformer_metadata():
    from nemo_automodel._transformers import retrieval

    config = PretrainedConfig()

    assert retrieval._resolve_bi_encoder_options(config, None, None, None) == ("avg", True)


def test_sentence_transformer_source_with_unsupported_module_is_rejected(tmp_path):
    from nemo_automodel._transformers import retrieval

    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
            ]
        )
    )
    (tmp_path / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_mean_tokens": True}))

    with pytest.raises(ValueError, match="exact supported module stack"):
        retrieval._load_sentence_transformer_wrapper_options(str(tmp_path), {})


def test_sentence_transformer_source_excluding_prompts_is_rejected(tmp_path):
    from nemo_automodel._transformers import retrieval

    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ]
        )
    )
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"pooling_mode_mean_tokens": True, "include_prompt": False})
    )

    with pytest.raises(ValueError, match="include_prompt=False"):
        retrieval._load_sentence_transformer_wrapper_options(str(tmp_path), {})


def test_sentence_transformer_source_lowercasing_is_rejected(tmp_path):
    from nemo_automodel._transformers import retrieval

    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ]
        )
    )
    (tmp_path / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_mean_tokens": True}))
    (tmp_path / "sentence_bert_config.json").write_text(json.dumps({"do_lower_case": True}))

    with pytest.raises(ValueError, match="do_lower_case=True"):
        retrieval._load_sentence_transformer_wrapper_options(str(tmp_path), {})


def test_sentence_transformer_metadata_load_fails_closed_on_unexpected_hub_error(monkeypatch):
    from nemo_automodel._transformers import retrieval

    monkeypatch.setattr(retrieval, "hf_hub_download", MagicMock(side_effect=OSError("offline")))

    with pytest.raises(RuntimeError, match="Unable to load Sentence Transformers metadata modules.json"):
        retrieval._load_sentence_transformer_json("org/model", "modules.json", {})


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
