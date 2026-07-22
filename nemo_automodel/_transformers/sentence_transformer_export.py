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

"""Sentence Transformers metadata export for retrieval models."""

import json
import os
import shutil

from torch import nn

_SENTENCE_TRANSFORMER_POOLING_KEYS = {
    "avg": "pooling_mode_mean_tokens",
    "cls": "pooling_mode_cls_token",
    "last": "pooling_mode_lasttoken",
}
_SOURCE_LEGAL_ASSET_PREFIXES = ("license", "notice")
_TEXT_EXPORT_STALE_PROCESSOR_ASSETS = ("processor_config.json", "preprocessor_config.json")


def _write_json(path: str, value) -> None:
    """Write deterministic, human-readable JSON metadata."""
    with open(path, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def _copy_source_legal_assets(
    original_model_path: str | None,
    hf_metadata_dir: str,
    source_repository_path: str | None = None,
) -> None:
    """Preserve repository- and model-root legal notices without inheriting other source semantics."""
    source_roots = []
    for source_root in (source_repository_path, original_model_path):
        if source_root is None or not os.path.isdir(source_root):
            continue
        if any(os.path.samefile(source_root, existing_root) for existing_root in source_roots):
            continue
        source_roots.append(source_root)

    for source_root in source_roots:
        for item in os.scandir(source_root):
            normalized_name = item.name.lower()
            is_legal_asset = (
                normalized_name == ".gitattributes"
                or normalized_name.startswith(_SOURCE_LEGAL_ASSET_PREFIXES)
                or "notice" in normalized_name
            )
            if item.is_file() and is_legal_asset:
                shutil.copy2(item.path, os.path.join(hf_metadata_dir, item.name))


def _remove_stale_text_processor_assets(hf_metadata_dir: str) -> None:
    """Remove source multimodal processor metadata from a text-only export."""
    for asset_name in _TEXT_EXPORT_STALE_PROCESSOR_ASSETS:
        asset_path = os.path.join(hf_metadata_dir, asset_name)
        if os.path.isfile(asset_path):
            os.remove(asset_path)


def _restore_source_tokenizer_serialization_state(
    original_model_path: str | None,
    hf_metadata_dir: str,
) -> None:
    """Remove training-time tokenizer state while retaining tokenizer edits."""
    tokenizer_json_path = os.path.join(hf_metadata_dir, "tokenizer.json")
    source_tokenizer_json_path = (
        os.path.join(original_model_path, "tokenizer.json") if original_model_path is not None else None
    )
    if os.path.isfile(tokenizer_json_path):
        with open(tokenizer_json_path) as f:
            tokenizer_json = json.load(f)
        source_tokenizer_json = {}
        if source_tokenizer_json_path is not None and os.path.isfile(source_tokenizer_json_path):
            with open(source_tokenizer_json_path) as f:
                source_tokenizer_json = json.load(f)
        for key in ("truncation", "padding"):
            tokenizer_json[key] = source_tokenizer_json.get(key)
        _write_json(tokenizer_json_path, tokenizer_json)

    tokenizer_config_path = os.path.join(hf_metadata_dir, "tokenizer_config.json")
    source_tokenizer_config_path = (
        os.path.join(original_model_path, "tokenizer_config.json") if original_model_path is not None else None
    )
    if os.path.isfile(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        source_tokenizer_config = {}
        if source_tokenizer_config_path is not None and os.path.isfile(source_tokenizer_config_path):
            with open(source_tokenizer_config_path) as f:
                source_tokenizer_config = json.load(f)
        if "local_files_only" in source_tokenizer_config:
            tokenizer_config["local_files_only"] = source_tokenizer_config["local_files_only"]
        else:
            tokenizer_config.pop("local_files_only", None)
        tokenizer_config.pop("processor_class", None)
        _write_json(tokenizer_config_path, tokenizer_config)


def _read_source_sentence_transformer_max_seq_length(original_model_path: str | None) -> int | None:
    """Read the source Sentence Transformers deployment limit when present."""
    if original_model_path is None:
        return None
    config_path = os.path.join(original_model_path, "sentence_bert_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        source_config = json.load(f)
    value = source_config.get("max_seq_length")
    if value is None:
        return None
    max_seq_length = int(value)
    if max_seq_length <= 0:
        raise ValueError("Source sentence_bert_config.json max_seq_length must be positive.")
    return max_seq_length


def _resolve_sentence_transformer_max_seq_length(
    model_part: nn.Module,
    tokenizer,
    original_model_path: str | None = None,
) -> int:
    """Resolve deployment sequence length without using training-time truncation."""
    source_max_seq_length = _read_source_sentence_transformer_max_seq_length(original_model_path)
    if source_max_seq_length is not None:
        model_max_seq_length = getattr(getattr(model_part, "config", None), "max_position_embeddings", None)
        if model_max_seq_length is not None:
            model_max_seq_length = int(model_max_seq_length)
            if 0 < model_max_seq_length < 1_000_000_000 and source_max_seq_length > model_max_seq_length:
                raise ValueError("Source Sentence Transformers max_seq_length exceeds max_position_embeddings.")
        return source_max_seq_length

    candidates = [
        getattr(tokenizer, "model_max_length", None),
        getattr(getattr(model_part, "config", None), "max_position_embeddings", None),
    ]
    finite_candidates = []
    for candidate in candidates:
        if candidate is None:
            continue
        max_seq_length = int(candidate)
        if 0 < max_seq_length < 1_000_000_000:
            finite_candidates.append(max_seq_length)
    if finite_candidates:
        return min(finite_candidates)
    raise ValueError("Unable to determine a finite Sentence Transformers deployment max_seq_length.")


def _validate_sentence_transformer_export(
    model_part: nn.Module,
    tokenizer,
    original_model_path: str | None = None,
) -> None:
    """Validate generated metadata inputs before rank-zero-only checkpoint I/O."""
    pooling = getattr(model_part, "pooling", None)
    if pooling not in _SENTENCE_TRANSFORMER_POOLING_KEYS:
        raise ValueError(f"Pooling mode {pooling!r} cannot be represented by standard Sentence Transformers metadata.")

    model_config = getattr(model_part, "config", None)
    embedding_dimension = getattr(model_config, "hidden_size", None)
    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
        raise ValueError("Bi-encoder config must expose a positive hidden_size for Sentence Transformers export.")

    if tokenizer is None:
        raise ValueError("A tokenizer is required to export a loadable Sentence Transformers checkpoint.")

    _resolve_sentence_transformer_max_seq_length(model_part, tokenizer, original_model_path)


def _save_generated_sentence_transformer_assets(
    model_part: nn.Module,
    export_config,
    original_model_path: str | None,
    hf_metadata_dir: str,
    tokenizer,
) -> None:
    """Generate Sentence Transformers metadata from the effective bi-encoder behavior."""
    _validate_sentence_transformer_export(model_part, tokenizer, original_model_path)
    pooling = getattr(model_part, "pooling", None)
    model_config = getattr(model_part, "config", None)
    embedding_dimension = getattr(model_config, "hidden_size", None)

    query_prompt = export_config.query_prompt or ""
    document_prompt = export_config.document_prompt or ""

    normalize = bool(getattr(model_part, "l2_normalize", False))
    similarity_fn_name = "cosine" if normalize else "dot"

    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    if normalize:
        modules.append(
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )

    pooling_config = {
        "word_embedding_dimension": embedding_dimension,
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }
    pooling_config[_SENTENCE_TRANSFORMER_POOLING_KEYS[pooling]] = True

    max_seq_length = _resolve_sentence_transformer_max_seq_length(model_part, tokenizer, original_model_path)

    os.makedirs(os.path.join(hf_metadata_dir, "1_Pooling"), exist_ok=True)
    _write_json(os.path.join(hf_metadata_dir, "modules.json"), modules)
    _write_json(
        os.path.join(hf_metadata_dir, "config_sentence_transformers.json"),
        {
            "prompts": {"query": query_prompt, "document": document_prompt},
            "default_prompt_name": None,
            "similarity_fn_name": similarity_fn_name,
        },
    )
    _write_json(
        os.path.join(hf_metadata_dir, "sentence_bert_config.json"),
        {"max_seq_length": max_seq_length, "do_lower_case": False},
    )
    _write_json(os.path.join(hf_metadata_dir, "1_Pooling", "config.json"), pooling_config)
    _restore_source_tokenizer_serialization_state(original_model_path, hf_metadata_dir)
    _remove_stale_text_processor_assets(hf_metadata_dir)
    _copy_source_legal_assets(
        original_model_path,
        hf_metadata_dir,
        source_repository_path=getattr(model_part, "source_repository_path", None),
    )


class _SentenceTransformerMetadataExporter:
    """Write consolidated Hugging Face and Sentence Transformers metadata for a bi-encoder."""

    def __init__(self, model_part: nn.Module, export_config) -> None:
        self.model_part = model_part
        self.export_config = export_config

    def validate(self, *, tokenizer, original_model_path: str | None) -> None:
        """Validate export inputs on every distributed rank before filesystem writes."""
        _validate_sentence_transformer_export(self.model_part, tokenizer, original_model_path)

    def save(
        self,
        *,
        metadata_reference_path: str | None,
        hf_metadata_dir: str,
        tokenizer,
        original_model_path: str | None,
    ) -> None:
        """Write deployable Hugging Face metadata plus standard Sentence Transformers assets."""
        from nemo_automodel.components.checkpoint.addons import _save_generated_hf_assets

        deploy_config = self.model_part.get_hf_export_config()
        _save_generated_hf_assets(
            self.model_part,
            metadata_reference_path,
            hf_metadata_dir,
            tokenizer,
            v4_compatible=False,
            model_config=deploy_config,
            save_custom_model_code=bool(getattr(deploy_config, "auto_map", None)),
        )
        _save_generated_sentence_transformer_assets(
            self.model_part,
            self.export_config,
            original_model_path,
            hf_metadata_dir,
            tokenizer,
        )
