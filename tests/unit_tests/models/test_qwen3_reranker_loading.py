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

"""Real tiny-checkpoint regression coverage for Qwen3 retrieval loading."""

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3ForSequenceClassification,
)

from nemo_automodel._transformers import retrieval
from nemo_automodel.components.models.qwen3_reranker.model import (
    Qwen3RerankerConfig,
    Qwen3RerankerForCausalReranking,
)


@pytest.fixture
def tiny_config():
    return Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        pad_token_id=0,
        num_labels=1,
        attn_implementation="eager",
    )


def _save_tokenizer(path):
    vocab = {f"token_{i}": i for i in range(16) if i not in (0, 1, 5, 7)}
    vocab.update({"[PAD]": 0, "[UNK]": 1, "yes": 5, "no": 7})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.save_pretrained(path)


def _inputs():
    return {"input_ids": torch.tensor([[1, 2, 3], [3, 2, 1]]), "attention_mask": torch.ones(2, 3, dtype=torch.long)}


@pytest.mark.parametrize("extracted", [False, True])
def test_classifier_checkpoint_preserves_trained_head(tmp_path, tiny_config, monkeypatch, extracted):
    torch.manual_seed(0)
    reference = Qwen3ForSequenceClassification(tiny_config).eval()
    reference.save_pretrained(tmp_path)
    if extracted:
        monkeypatch.setattr(
            retrieval.AutoModel, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(language_model=reference)
        )
    loaded = retrieval.CrossEncoderModel.build(
        str(tmp_path), extract_submodel="language_model" if extracted else None, local_files_only=True
    ).eval()

    assert type(loaded.model) is Qwen3ForSequenceClassification
    torch.testing.assert_close(loaded.model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded(_inputs()).logits, reference(**_inputs()).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("saved_tied", [False, True])
@pytest.mark.parametrize("override", ["keyword", "config", "config_path"])
def test_checkpoint_tie_flip_is_rejected(tmp_path, tiny_config, saved_tied, override):
    tiny_config.tie_word_embeddings = saved_tied
    Qwen3ForCausalLM(tiny_config).save_pretrained(tmp_path)
    kwargs = {"tie_word_embeddings": not saved_tied}
    if override != "keyword":
        config = AutoConfig.from_pretrained(tmp_path)
        config.tie_word_embeddings = not saved_tied
        if override == "config_path":
            config.save_pretrained(tmp_path / "override")
            config = str(tmp_path / "override")
        kwargs = {"config": config}

    # No tokenizer exists: reject incompatible weights before tokenizer resolution.
    with pytest.raises(NotImplementedError, match="flipping the flag is not supported"):
        Qwen3RerankerForCausalReranking.from_pretrained(tmp_path, local_files_only=True, **kwargs)


@pytest.mark.parametrize("saved_tied", [False, True])
@pytest.mark.parametrize("extracted", [False, True])
def test_causal_loading_resolves_tokens_and_preserves_scores(tmp_path, tiny_config, monkeypatch, saved_tied, extracted):
    torch.manual_seed(0)
    tiny_config.tie_word_embeddings = saved_tied
    reference = Qwen3ForCausalLM(tiny_config).eval()
    # Exercise tokenizer location options as well as normal weight loading.
    checkpoint_dir = tmp_path / "checkpoint"
    reference.save_pretrained(checkpoint_dir)
    _save_tokenizer(checkpoint_dir)
    if extracted:
        monkeypatch.setattr(
            retrieval.AutoModel, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(language_model=reference)
        )
    loaded = retrieval.CrossEncoderModel.build(
        str(tmp_path),
        extract_submodel="language_model" if extracted else None,
        subfolder="checkpoint",
        local_files_only=True,
        tie_word_embeddings=saved_tied,
    ).eval()

    assert type(loaded.model) is Qwen3RerankerForCausalReranking
    assert loaded.config.yes_token_id == 5
    assert loaded.config.no_token_id == 7
    assert (loaded.model.lm_head.weight is loaded.model.model.embed_tokens.weight) is saved_tied
    torch.testing.assert_close(loaded.model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        logits = reference(**_inputs()).logits[:, -1]
        torch.testing.assert_close(loaded(_inputs()).logits[:, 0], logits[:, 5] - logits[:, 7], rtol=1e-5, atol=1e-6)

    # The loaded scoring head must send the same gradients into the decoder.
    logits = reference(**_inputs()).logits[:, -1]
    (logits[:, 5] - logits[:, 7]).sum().backward()
    loaded(_inputs()).logits.sum().backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in loaded.model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("saved_tied", [False, True])
def test_explicit_config_exports_as_stock_causal_lm_without_mutation(tmp_path, tiny_config, saved_tied):
    tiny_config.tie_word_embeddings = saved_tied
    reference = Qwen3ForCausalLM(tiny_config)
    reference.save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path)
    config._attn_implementation = "eager"
    original_config = config.to_dict()
    loaded = Qwen3RerankerForCausalReranking.from_pretrained(
        tmp_path, config=config, yes_token_id=5, no_token_id=7, local_files_only=True
    ).eval()
    assert isinstance(loaded.config, Qwen3RerankerConfig)
    assert loaded.config._attn_implementation == "eager"
    assert loaded.config.yes_token_id == 5
    assert loaded.config.no_token_id == 7
    assert config.to_dict() == original_config

    encoder = retrieval.CrossEncoderModel(loaded)
    export_path = tmp_path / "export"
    encoder.save_pretrained(export_path)
    exported = json.loads((export_path / "config.json").read_text())
    assert exported["architectures"] == ["Qwen3ForCausalLM"]
    assert "auto_map" not in exported
    reloaded = AutoModelForCausalLM.from_pretrained(export_path, local_files_only=True).eval()
    assert type(reloaded) is Qwen3ForCausalLM
    assert (reloaded.lm_head.weight is reloaded.model.embed_tokens.weight) is saved_tied
    torch.testing.assert_close(reloaded.state_dict(), loaded.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        logits = reloaded(**_inputs()).logits[:, -1]
        torch.testing.assert_close(encoder(_inputs()).logits[:, 0], logits[:, 5] - logits[:, 7])


@pytest.mark.parametrize("extracted", [False, True])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"num_labels": 2}, "num_labels must be 1"),
        ({"pooling": "avg"}, "pooling is not supported"),
        ({"temperature": 0.5}, "temperature on the training recipe"),
    ],
)
def test_causal_loading_rejects_classifier_options(tmp_path, tiny_config, monkeypatch, extracted, options, message):
    reference = Qwen3ForCausalLM(tiny_config).eval()
    reference.save_pretrained(tmp_path)
    _save_tokenizer(tmp_path)
    if extracted:
        monkeypatch.setattr(
            retrieval.AutoModel, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(language_model=reference)
        )
    with pytest.raises(ValueError, match=message):
        retrieval.CrossEncoderModel.build(
            str(tmp_path),
            extract_submodel="language_model" if extracted else None,
            local_files_only=True,
            **options,
        )


def test_causal_checkpoint_default_labels_do_not_change_single_score(tmp_path, tiny_config):
    # Standard causal configs default to two labels, despite having no classifier.
    tiny_config.num_labels = 2
    reference = Qwen3ForCausalLM(tiny_config).eval()
    reference.save_pretrained(tmp_path)
    _save_tokenizer(tmp_path)
    loaded = retrieval.CrossEncoderModel.build(str(tmp_path), num_labels=1, local_files_only=True).eval()
    assert loaded.config.num_labels == 1
    with torch.no_grad():
        logits = reference(**_inputs()).logits[:, -1]
        torch.testing.assert_close(loaded(_inputs()).logits[:, 0], logits[:, 5] - logits[:, 7])


def test_extracted_decoder_keeps_classification_fallback(tmp_path, tiny_config, monkeypatch):
    reference = Qwen3ForCausalLM(tiny_config).eval()
    reference.save_pretrained(tmp_path)
    # The decoder inherits the parent's ForCausalLM config but has no trained LM head.
    monkeypatch.setattr(
        retrieval.AutoModel, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(language_model=reference.model)
    )
    loaded = retrieval.CrossEncoderModel.build(
        str(tmp_path), extract_submodel="language_model", num_labels=1, local_files_only=True
    ).eval()
    assert type(loaded.model) is Qwen3ForSequenceClassification
    torch.testing.assert_close(loaded.model.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        assert torch.isfinite(loaded(_inputs()).logits).all()
