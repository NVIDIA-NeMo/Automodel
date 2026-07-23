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

"""Sentence Transformers metadata interoperability for retrieval models."""

import json
import os
import shutil
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
from torch import nn
from transformers import PretrainedConfig
from transformers.utils import logging

_SENTENCE_TRANSFORMER_POOLING_KEYS = {
    "avg": "pooling_mode_mean_tokens",
    "cls": "pooling_mode_cls_token",
    "last": "pooling_mode_lasttoken",
}
_HF_HUB_METADATA_KWARGS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "token",
}
_SOURCE_LEGAL_ASSET_PATTERNS = (
    ".gitattributes",
    "LICENSE*",
    "license*",
    "NOTICE*",
    "notice*",
    "*NOTICE*",
    "*notice*",
)
_SOURCE_LEGAL_ASSET_PREFIXES = ("license", "notice")
_TEXT_EXPORT_STALE_PROCESSOR_ASSETS = ("processor_config.json", "preprocessor_config.json")


logger = logging.get_logger(__name__)


@dataclass
class SentenceTransformerExportConfig:
    """Effective Sentence Transformers semantics written with a bi-encoder checkpoint.

    Attributes:
        query_prompt: Exact prompt prepended to queries, or None when no prompt is configured yet.
        document_prompt: Exact prompt prepended to documents, or None when no prompt is configured yet.
    """

    query_prompt: str | None = None
    document_prompt: str | None = None


@dataclass(frozen=True)
class SentenceTransformerWrapperOptions:
    """Bi-encoder wrapper behavior represented by standard Sentence Transformers modules."""

    pooling: str
    l2_normalize: bool
    query_prompt: str | None = None
    document_prompt: str | None = None


def _load_sentence_transformer_json(
    model_name_or_path: str,
    filename: str,
    hf_kwargs: dict,
) -> object | None:
    """Load a standard Sentence Transformers JSON asset from a local path or the Hub."""
    subfolder = hf_kwargs.get("subfolder")
    if os.path.isdir(model_name_or_path):
        root = os.path.join(model_name_or_path, subfolder) if subfolder else model_name_or_path
        asset_path = os.path.join(root, filename)
        if not os.path.isfile(asset_path):
            return None
    else:
        download_kwargs = {key: hf_kwargs[key] for key in _HF_HUB_METADATA_KWARGS if key in hf_kwargs}
        if "token" not in download_kwargs and "use_auth_token" in hf_kwargs:
            download_kwargs["token"] = hf_kwargs["use_auth_token"]
        try:
            asset_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename=filename,
                subfolder=subfolder,
                **download_kwargs,
            )
        except EntryNotFoundError:
            return None
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"Could not establish whether Sentence Transformers metadata {filename} exists locally."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to load Sentence Transformers metadata {filename}.") from exc

    with open(asset_path) as f:
        return json.load(f)


def _load_sentence_transformer_wrapper_options(
    model_name_or_path: str,
    hf_kwargs: dict,
) -> SentenceTransformerWrapperOptions | None:
    """Restore pooling and normalization from the standard Sentence Transformers module stack."""
    modules = _load_sentence_transformer_json(model_name_or_path, "modules.json", hf_kwargs)
    if modules is None:
        return None
    if not isinstance(modules, list):
        raise ValueError("Sentence Transformers modules.json must contain a list of modules.")

    transformer_type = "sentence_transformers.models.Transformer"
    pooling_type = "sentence_transformers.models.Pooling"
    normalize_type = "sentence_transformers.models.Normalize"
    module_types = [module.get("type") if isinstance(module, dict) else None for module in modules]
    if module_types not in ([transformer_type, pooling_type], [transformer_type, pooling_type, normalize_type]):
        raise ValueError(
            "Sentence Transformers metadata must use the exact supported module stack: "
            "Transformer, Pooling, and optional Normalize."
        )
    if modules[0].get("path") != "":
        raise ValueError("Sentence Transformers Transformer metadata must reference the checkpoint root.")

    sentence_bert_config = _load_sentence_transformer_json(
        model_name_or_path,
        "sentence_bert_config.json",
        hf_kwargs,
    )
    if isinstance(sentence_bert_config, dict) and sentence_bert_config.get("do_lower_case"):
        raise ValueError(
            "Sentence Transformers checkpoints with do_lower_case=True are unsupported because "
            "the NeMo inference path does not lowercase text."
        )

    pooling_path = modules[1].get("path")
    if not isinstance(pooling_path, str) or not pooling_path:
        raise ValueError("Sentence Transformers Pooling metadata must reference a module path.")
    pooling_config = _load_sentence_transformer_json(
        model_name_or_path,
        os.path.join(pooling_path, "config.json"),
        hf_kwargs,
    )
    if not isinstance(pooling_config, dict):
        raise ValueError("Sentence Transformers Pooling config.json is missing or invalid.")
    if pooling_config.get("include_prompt", True) is False:
        raise ValueError(
            "Sentence Transformers checkpoints with include_prompt=False are unsupported because "
            "the NeMo pooling path includes prompt tokens."
        )

    active_pooling_keys = {
        key for key, value in pooling_config.items() if key.startswith("pooling_mode_") and bool(value)
    }
    matching_pooling = [
        pooling
        for pooling, metadata_key in _SENTENCE_TRANSFORMER_POOLING_KEYS.items()
        if metadata_key in active_pooling_keys
    ]
    if len(active_pooling_keys) != 1 or len(matching_pooling) != 1:
        raise ValueError(
            "Sentence Transformers pooling metadata cannot be represented by a single NeMo avg, cls, or last mode."
        )

    sentence_transformer_config = _load_sentence_transformer_json(
        model_name_or_path,
        "config_sentence_transformers.json",
        hf_kwargs,
    )
    query_prompt = None
    document_prompt = None
    if sentence_transformer_config is not None:
        if not isinstance(sentence_transformer_config, dict):
            raise ValueError("Sentence Transformers config_sentence_transformers.json is invalid.")
        prompts = sentence_transformer_config.get("prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError("Sentence Transformers prompts metadata must contain a mapping.")
        query_prompt = prompts.get("query")
        document_prompt = prompts.get("document")
        if query_prompt is not None and not isinstance(query_prompt, str):
            raise ValueError("Sentence Transformers query prompt must be a string.")
        if document_prompt is not None and not isinstance(document_prompt, str):
            raise ValueError("Sentence Transformers document prompt must be a string.")

    return SentenceTransformerWrapperOptions(
        pooling=matching_pooling[0],
        l2_normalize=len(modules) == 3,
        query_prompt=query_prompt,
        document_prompt=document_prompt,
    )


def _resolve_cached_source_model_path(model_name_or_path: str, config, hf_kwargs: dict) -> str | None:
    """Resolve the already-downloaded source snapshot without network access."""
    subfolder = hf_kwargs.get("subfolder")
    if os.path.isdir(model_name_or_path):
        source_path = os.path.join(model_name_or_path, subfolder) if subfolder else model_name_or_path
        return source_path if os.path.isdir(source_path) else None
    filename = os.path.join(subfolder, "config.json") if subfolder else "config.json"
    cached_config = try_to_load_from_cache(
        model_name_or_path,
        filename,
        cache_dir=hf_kwargs.get("cache_dir"),
        revision=getattr(config, "_commit_hash", None) or hf_kwargs.get("revision"),
    )
    if isinstance(cached_config, str) and os.path.isfile(cached_config):
        return os.path.dirname(cached_config)
    return None


def _resolve_cached_source_repository_path(
    model_name_or_path: str,
    source_model_path: str | None,
    hf_kwargs: dict,
) -> str | None:
    """Resolve the repository or snapshot root for a model loaded from a subfolder."""
    if source_model_path is None:
        return None
    subfolder = hf_kwargs.get("subfolder")
    if not subfolder:
        return source_model_path
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    repository_path = source_model_path
    for part in os.path.normpath(subfolder).split(os.sep):
        if part not in ("", "."):
            repository_path = os.path.dirname(repository_path)
    return repository_path


def _cache_hub_source_legal_assets(model_name_or_path: str, config, hf_kwargs: dict) -> str | None:
    """Cache source legal assets while the Hub-backed model is being loaded."""
    if os.path.isdir(model_name_or_path):
        return None

    download_kwargs = {
        key: hf_kwargs[key] for key in ("cache_dir", "force_download", "local_files_only", "token") if key in hf_kwargs
    }
    if "token" not in download_kwargs and "use_auth_token" in hf_kwargs:
        download_kwargs["token"] = hf_kwargs["use_auth_token"]
    revision = getattr(config, "_commit_hash", None) or hf_kwargs.get("revision")
    if revision is not None:
        download_kwargs["revision"] = revision
    try:
        return snapshot_download(
            repo_id=model_name_or_path,
            allow_patterns=_SOURCE_LEGAL_ASSET_PATTERNS,
            **download_kwargs,
        )
    except Exception as exc:
        logger.warning("Unable to cache source legal assets for %s: %s", model_name_or_path, exc)
        return None


def _supports_standard_sentence_transformer_export(model: nn.Module, pooling: str) -> bool:
    """Return whether a backbone can be represented by the standard text module stack."""
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    return (
        pooling in _SENTENCE_TRANSFORMER_POOLING_KEYS
        and getattr(model, "main_input_name", "input_ids") == "input_ids"
        and isinstance(config, PretrainedConfig)
        and not bool(getattr(config, "is_composition", False))
        and isinstance(hidden_size, int)
        and hidden_size > 0
    )


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

    def _source_model_path(self, fallback: str | None) -> str | None:
        """Prefer the retrieval model source snapshot over generic checkpoint lookup."""
        source_model_path = getattr(self.model_part, "source_model_path", None)
        return str(source_model_path) if source_model_path is not None else fallback

    def validate(self, *, tokenizer, original_model_path: str | None) -> None:
        """Validate export inputs on every distributed rank before filesystem writes."""
        _validate_sentence_transformer_export(self.model_part, tokenizer, self._source_model_path(original_model_path))

    def save(
        self,
        *,
        hf_metadata_dir: str,
        tokenizer,
        original_model_path: str | None,
    ) -> None:
        """Write deployable Hugging Face metadata plus standard Sentence Transformers assets."""
        from nemo_automodel.components.checkpoint.addons import _save_generated_hf_assets

        deploy_config = self.model_part.get_hf_export_config()
        source_model_path = self._source_model_path(original_model_path)
        _save_generated_hf_assets(
            self.model_part,
            source_model_path,
            hf_metadata_dir,
            tokenizer,
            v4_compatible=False,
            model_config=deploy_config,
            save_custom_model_code=bool(getattr(deploy_config, "auto_map", None)),
        )
        _save_generated_sentence_transformer_assets(
            self.model_part,
            self.export_config,
            source_model_path,
            hf_metadata_dir,
            tokenizer,
        )
