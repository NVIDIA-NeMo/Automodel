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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_automodel._transformers import sentence_transformer_export
from nemo_automodel._transformers.sentence_transformer_export import (
    _resolve_sentence_transformer_max_seq_length,
    _save_generated_sentence_transformer_assets,
)


def test_generated_sentence_transformer_assets_regenerate_semantics_and_preserve_source_limit(tmp_path):
    source_dir = tmp_path / "source"
    metadata_dir = tmp_path / "metadata"
    source_dir.mkdir()
    metadata_dir.mkdir()
    (source_dir / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "prompts": {"query": "source query: ", "document": "source document: "},
                "similarity_fn_name": "cosine",
            }
        )
    )
    (source_dir / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 8192, "do_lower_case": False}))
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=32768),
    )
    export_config = SimpleNamespace(
        query_prompt=None,
        document_prompt=None,
    )
    tokenizer = SimpleNamespace(model_max_length=512)

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        str(source_dir),
        str(metadata_dir),
        tokenizer,
    )

    sentence_config = json.loads((metadata_dir / "config_sentence_transformers.json").read_text())
    assert sentence_config["prompts"] == {"query": "", "document": ""}
    assert sentence_config["similarity_fn_name"] == "cosine"
    assert json.loads((metadata_dir / "sentence_bert_config.json").read_text()) == {
        "max_seq_length": 8192,
        "do_lower_case": False,
    }
    assert json.loads((metadata_dir / "modules.json").read_text())[-1] == {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    }
    pooling_config = json.loads((metadata_dir / "1_Pooling" / "config.json").read_text())
    assert pooling_config["pooling_mode_mean_tokens"] is True


def test_inferred_sentence_transformer_max_seq_length_uses_smallest_finite_capability():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))

    assert (
        _resolve_sentence_transformer_max_seq_length(
            model,
            SimpleNamespace(model_max_length=4096),
        )
        == 512
    )
    assert (
        _resolve_sentence_transformer_max_seq_length(
            model,
            SimpleNamespace(model_max_length=256),
        )
        == 256
    )

    with pytest.raises(ValueError, match="Unable to determine"):
        _resolve_sentence_transformer_max_seq_length(
            SimpleNamespace(config=SimpleNamespace(max_position_embeddings=None)),
            SimpleNamespace(model_max_length=10**30),
        )


def test_generated_sentence_transformer_assets_remove_training_tokenizer_state(tmp_path):
    source_dir = tmp_path / "source"
    metadata_dir = tmp_path / "metadata"
    source_dir.mkdir()
    metadata_dir.mkdir()
    (source_dir / "tokenizer.json").write_text(json.dumps({"truncation": None, "padding": None, "model": {}}))
    (source_dir / "tokenizer_config.json").write_text(json.dumps({"local_files_only": False, "model_max_length": 512}))
    (metadata_dir / "tokenizer.json").write_text(
        json.dumps({"truncation": {"max_length": 32}, "padding": {"length": 32}, "model": {}})
    )
    (metadata_dir / "tokenizer_config.json").write_text(
        json.dumps({"local_files_only": True, "model_max_length": 512, "processor_class": "PixtralProcessor"})
    )
    (metadata_dir / "processor_config.json").write_text('{"processor_class": "PixtralProcessor"}')
    (metadata_dir / "preprocessor_config.json").write_text('{"image_processor_type": "PixtralImageProcessor"}')
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="query: ",
        document_prompt="passage: ",
    )

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        str(source_dir),
        str(metadata_dir),
        SimpleNamespace(model_max_length=512),
    )

    tokenizer_json = json.loads((metadata_dir / "tokenizer.json").read_text())
    tokenizer_config = json.loads((metadata_dir / "tokenizer_config.json").read_text())
    assert tokenizer_json["truncation"] is None
    assert tokenizer_json["padding"] is None
    assert tokenizer_config["local_files_only"] is False
    assert "processor_class" not in tokenizer_config
    assert not (metadata_dir / "processor_config.json").exists()
    assert not (metadata_dir / "preprocessor_config.json").exists()


def test_generated_sentence_transformer_assets_preserve_repository_and_model_root_legal_assets(tmp_path):
    repository_root = tmp_path / "repository"
    model_root = repository_root / "encoder"
    metadata_dir = tmp_path / "metadata"
    (repository_root / "legal").mkdir(parents=True)
    model_root.mkdir()
    metadata_dir.mkdir()
    (repository_root / "LICENSE.md").write_text("repository license")
    (repository_root / "NOTICE.md").write_text("repository notice")
    (repository_root / "THIRD_PARTY_NOTICES.md").write_text("third-party notices")
    (repository_root / "COPYING").write_text("copying")
    (repository_root / "legal" / "LICENSE.txt").write_text("nested license")
    (model_root / "LICENSE.md").write_text("model license")

    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        source_repository_path=str(repository_root),
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
    )

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        str(model_root),
        str(metadata_dir),
        SimpleNamespace(model_max_length=512),
    )

    assert (metadata_dir / "LICENSE.md").read_text() == "model license"
    assert (metadata_dir / "NOTICE.md").read_text() == "repository notice"
    assert (metadata_dir / "THIRD_PARTY_NOTICES.md").read_text() == "third-party notices"
    assert not (metadata_dir / "COPYING").exists()
    assert not (metadata_dir / "LICENSE.txt").exists()


def test_generated_sentence_transformer_assets_reject_unrepresentable_pooling(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    model = SimpleNamespace(
        pooling="weighted_avg",
        l2_normalize=False,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
    )

    with pytest.raises(ValueError, match="cannot be represented"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(metadata_dir),
            tokenizer=None,
        )


@pytest.mark.parametrize(("l2_normalize", "expected_similarity"), [(False, "dot"), (True, "cosine")])
def test_generated_sentence_transformer_assets_derive_interoperability_metadata(
    tmp_path,
    l2_normalize,
    expected_similarity,
):
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=l2_normalize,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(query_prompt="query: ", document_prompt="passage: ")

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        original_model_path=None,
        hf_metadata_dir=str(tmp_path),
        tokenizer=SimpleNamespace(model_max_length=512),
    )

    sentence_config = json.loads((tmp_path / "config_sentence_transformers.json").read_text())
    pooling_config = json.loads((tmp_path / "1_Pooling" / "config.json").read_text())
    transformer_config = json.loads((tmp_path / "sentence_bert_config.json").read_text())
    assert sentence_config["similarity_fn_name"] == expected_similarity
    assert pooling_config["include_prompt"] is True
    assert transformer_config == {"max_seq_length": 512, "do_lower_case": False}


def test_generated_sentence_transformer_assets_require_tokenizer(tmp_path):
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
    )

    with pytest.raises(ValueError, match="tokenizer is required"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(tmp_path),
            tokenizer=None,
        )


def test_cached_source_model_path_uses_exact_loaded_revision(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    (local_root / "encoder").mkdir(parents=True)
    assert sentence_transformer_export._resolve_cached_source_model_path(
        str(local_root), SimpleNamespace(), {"subfolder": "encoder"}
    ) == str(local_root / "encoder")
    assert sentence_transformer_export._resolve_cached_source_repository_path(
        str(local_root), str(local_root / "encoder"), {"subfolder": "encoder"}
    ) == str(local_root)

    cached_config = tmp_path / "snapshot" / "encoder" / "config.json"
    cached_config.parent.mkdir(parents=True)
    cached_config.write_text("{}")
    cache_lookup = MagicMock(return_value=str(cached_config))
    monkeypatch.setattr(sentence_transformer_export, "try_to_load_from_cache", cache_lookup)

    result = sentence_transformer_export._resolve_cached_source_model_path(
        "org/model",
        SimpleNamespace(_commit_hash="exact-commit"),
        {"cache_dir": "/cache", "revision": "branch", "subfolder": "encoder"},
    )

    assert result == str(cached_config.parent)
    assert sentence_transformer_export._resolve_cached_source_repository_path(
        "org/model", result, {"subfolder": "encoder"}
    ) == str(tmp_path / "snapshot")
    cache_lookup.assert_called_once_with(
        "org/model",
        "encoder/config.json",
        cache_dir="/cache",
        revision="exact-commit",
    )


def test_cache_hub_source_legal_assets_uses_exact_loaded_revision(monkeypatch):
    snapshot_download = MagicMock(return_value="/cache/snapshot")
    monkeypatch.setattr(sentence_transformer_export, "snapshot_download", snapshot_download)

    result = sentence_transformer_export._cache_hub_source_legal_assets(
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
        allow_patterns=sentence_transformer_export._SOURCE_LEGAL_ASSET_PATTERNS,
        cache_dir="/cache",
        local_files_only=True,
        revision="exact-commit",
        token="token",
    )


def test_sentence_transformer_source_with_unsupported_module_is_rejected(tmp_path):
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
        sentence_transformer_export._load_sentence_transformer_wrapper_options(str(tmp_path), {})


def test_sentence_transformer_source_excluding_prompts_is_rejected(tmp_path):
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
        sentence_transformer_export._load_sentence_transformer_wrapper_options(str(tmp_path), {})


def test_sentence_transformer_source_lowercasing_is_rejected(tmp_path):
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
        sentence_transformer_export._load_sentence_transformer_wrapper_options(str(tmp_path), {})


def test_sentence_transformer_metadata_load_fails_closed_on_unexpected_hub_error(monkeypatch):
    monkeypatch.setattr(sentence_transformer_export, "hf_hub_download", MagicMock(side_effect=OSError("offline")))

    with pytest.raises(RuntimeError, match="Unable to load Sentence Transformers metadata modules.json"):
        sentence_transformer_export._load_sentence_transformer_json("org/model", "modules.json", {})
