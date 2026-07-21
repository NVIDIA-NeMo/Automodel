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

"""Encoder models for bi-encoder and cross-encoder tasks."""

import inspect
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, MODEL_MAPPING
from transformers.utils import logging

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.loss.intermediate_distill import LayerCapture
from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

logger = logging.get_logger(__name__)


_STANDARD_SENTENCE_TRANSFORMER_POOLING_TYPES = frozenset({"avg", "cls", "last"})
_SENTENCE_TRANSFORMER_POOLING_KEYS = {
    "avg": "pooling_mode_mean_tokens",
    "cls": "pooling_mode_cls_token",
    "last": "pooling_mode_lasttoken",
}
_BI_ENCODER_DEFAULT_POOLING = "avg"
_BI_ENCODER_DEFAULT_L2_NORMALIZE = True
_HF_HUB_METADATA_KWARGS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "token",
}
_EXPORT_ONLY_BI_ENCODER_KWARGS = {
    "query_prompt",
    "document_prompt",
    "sentence_transformer_max_seq_length",
    "similarity_fn_name",
    "do_lower_case",
}


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


def _resolve_bi_encoder_options(
    config: PretrainedConfig,
    saved_options: SentenceTransformerWrapperOptions | None,
    pooling: str | None,
    l2_normalize: bool | None,
) -> tuple[str, bool]:
    """Resolve explicit wrapper arguments, then standard saved metadata, then defaults."""
    if pooling is None:
        if saved_options is not None:
            pooling = saved_options.pooling
        else:
            pooling = getattr(config, "pooling", None) or _BI_ENCODER_DEFAULT_POOLING
    if l2_normalize is None:
        l2_normalize = saved_options.l2_normalize if saved_options is not None else _BI_ENCODER_DEFAULT_L2_NORMALIZE
    return pooling, l2_normalize


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


def _clone_pretrained_config(config: PretrainedConfig) -> PretrainedConfig:
    """Clone a config while retaining private source-revision metadata."""
    cloned_config = config.__class__.from_dict(config.to_dict())
    commit_hash = getattr(config, "_commit_hash", None)
    if commit_hash is not None:
        cloned_config._commit_hash = commit_hash
    return cloned_config


def _extract_submodel(model: nn.Module, extract_submodel: str) -> PreTrainedModel:
    """Extract a nested submodel from a loaded model using a dotted attribute path."""
    extracted_model = model
    for attr in extract_submodel.split("."):
        extracted_model = getattr(extracted_model, attr)
    if not hasattr(extracted_model, "config"):
        raise ValueError(
            f"Extracted submodel at '{extract_submodel}' has no .config attribute. "
            f"The submodel must be a PreTrainedModel for save/reload to work. "
            f"Got {type(extracted_model).__name__}."
        )
    return extracted_model


def _get_supported_backbone_class(model_type: str, task: str) -> type[nn.Module] | None:
    """Return the registered retrieval backbone class for a model type and task."""
    task_map = SUPPORTED_BACKBONES.get(model_type.lower())
    if task_map is None:
        return None

    arch_name = task_map.get(task)
    if arch_name is None:
        raise ValueError(
            f"Unsupported task '{task}' for model type '{model_type}'. Available tasks: {', '.join(task_map)}."
        )

    if arch_name not in ModelRegistry.model_arch_name_to_cls:
        raise ValueError(f"Model class '{arch_name}' not found in ModelRegistry.")

    logger.info(f"Using {arch_name} from registry")
    return ModelRegistry.model_arch_name_to_cls[arch_name]


def _move_to_extracted_dtype(model: nn.Module, extracted_model: nn.Module) -> nn.Module:
    """Move a newly-built model to the dtype used by the extracted model."""
    for parameter in extracted_model.parameters():
        return model.to(dtype=parameter.dtype)
    for buffer in extracted_model.buffers():
        return model.to(dtype=buffer.dtype)
    return model


def _load_from_extracted_state(
    backbone_class: type[PreTrainedModel],
    config,
    extracted_model: PreTrainedModel,
) -> PreTrainedModel:
    """Load a target backbone from an extracted model's in-memory state dict."""
    # Use the base HF loader because some retrieval classes override
    # from_pretrained for path-based checkpoint loading.
    backbone = PreTrainedModel.from_pretrained.__func__(
        backbone_class,
        None,
        config=config,
        state_dict=extracted_model.state_dict(),
    )
    return _move_to_extracted_dtype(backbone, extracted_model)


def _build_backbone_from_extracted_submodel(
    extracted_model: PreTrainedModel,
    task: str,
    pooling: Optional[str],
    num_labels: Optional[int],
    temperature: Optional[float],
) -> PreTrainedModel:
    """Build a task-specific retrieval backbone from an extracted text submodel."""
    text_config = extracted_model.config
    model_type = getattr(text_config, "model_type", "")
    task_map = SUPPORTED_BACKBONES.get(model_type.lower())
    has_supported_target = task_map is not None and task in task_map

    if task_map is not None and not has_supported_target and task != "score":
        raise ValueError(
            f"Unsupported task '{task}' for model type '{model_type}'. Available tasks: {', '.join(task_map)}."
        )

    if model_type.lower() == "ministral3" and task == "embedding":
        config = text_config.__class__.from_dict(text_config.to_dict())
        config.is_causal = False
        if pooling is not None:
            config.pooling = pooling
        if temperature is not None:
            config.temperature = temperature
        try:
            backbone_class = MODEL_MAPPING[type(config)]
        except KeyError as exc:
            raise ValueError(f"No HuggingFace base model found for '{model_type}'.") from exc
    elif task == "score" and not has_supported_target:
        config = text_config.__class__.from_dict(text_config.to_dict())
        try:
            backbone_class = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[type(config)]
        except KeyError as exc:
            raise ValueError(f"No HuggingFace sequence-classification model found for '{model_type}'.") from exc
    elif not has_supported_target:
        return extracted_model
    else:
        backbone_class = _get_supported_backbone_class(model_type, task)
        config_class = getattr(backbone_class, "config_class", None)
        if config_class is None or not hasattr(text_config, "to_dict"):
            return extracted_model

        config_dict = text_config.to_dict()
        config_dict.pop("model_type", None)
        config = config_class(**config_dict)
        if pooling is not None:
            config.pooling = pooling
        if temperature is not None:
            config.temperature = temperature

    attn_implementation = getattr(text_config, "_attn_implementation", None)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    if num_labels is not None:
        config.num_labels = num_labels

    return _load_from_extracted_state(backbone_class, config, extracted_model)


def pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str) -> torch.Tensor:
    """
    Pool hidden states using the specified pooling method.

    Args:
        last_hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
        pool_type: Type of pooling to apply

    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    if pool_type == "mean":
        pool_type = "avg"

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":
        emb = last_hidden.sum(dim=1)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    elif pool_type in {"colbert", "multi_vector"}:
        emb = last_hidden
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def configure_encoder_metadata(model: PreTrainedModel, config) -> None:
    """Configure HuggingFace consolidated checkpoint metadata on a model.

    Sets ``config.architectures`` unconditionally.  For custom retrieval
    architectures registered in :class:`ModelRegistry`, also writes
    ``config.auto_map`` so that the saved checkpoint can be reloaded via
    HuggingFace Auto classes.  Standard HF models already have their own
    auto-resolution and do not need ``auto_map`` entries.

    Args:
        model: The backbone ``PreTrainedModel`` instance.
        config: The model's config object (typically ``model.config``).
    """
    encoder_class_name = model.__class__.__name__
    config.architectures = [encoder_class_name]

    # Only set auto_map for custom retrieval architectures.
    # Standard HF models don't need auto_map pointing to a local model.py.
    if ModelRegistry.has_retrieval_model(encoder_class_name):
        config_class_name = config.__class__.__name__
        config_module = config.__class__.__module__.rsplit(".", 1)[-1]
        model_module = model.__class__.__module__.rsplit(".", 1)[-1]
        config.auto_map = {"AutoConfig": f"{config_module}.{config_class_name}"}
        if "ForSequenceClassification" in encoder_class_name:
            config.auto_map["AutoModelForSequenceClassification"] = f"{model_module}.{encoder_class_name}"
        else:
            config.auto_map["AutoModel"] = f"{model_module}.{encoder_class_name}"


def build_encoder_backbone(
    model_name_or_path: str,
    task: str,
    trust_remote_code: bool = False,
    pooling: Optional[str] = None,
    extract_submodel: Optional[str] = None,
    num_labels: Optional[int] = None,
    temperature: Optional[float] = None,
    loaded_config: PretrainedConfig | None = None,
    **hf_kwargs,
) -> PreTrainedModel:
    """Build an encoder backbone from a pretrained checkpoint.

    When ``extract_submodel`` is set, loads the parent model with HuggingFace
    Auto classes and extracts the dotted path. For supported extracted text
    backbones, it then builds the registered retrieval class for the requested
    task. Extracted Ministral embedding backbones use the stock HuggingFace model
    with ``is_causal=False``. For unsupported extracted text backbones, it returns
    the extracted model for ``"embedding"`` and wraps it with
    ``AutoModelForSequenceClassification`` for ``"score"``.

    Without ``extract_submodel``, standard Ministral embedding checkpoints use
    the stock HuggingFace model with ``is_causal=False``. Model types listed in
    :data:`SUPPORTED_BACKBONES` resolve to custom bidirectional classes from
    :class:`ModelRegistry`; all other model types fall back to HuggingFace Auto
    classes.

    Args:
        model_name_or_path: Path or HuggingFace Hub identifier.
        task: The encoder task (e.g. ``"embedding"``, ``"score"``).
        trust_remote_code: Whether to allow custom remote code.
        pooling: Bi-encoder pooling strategy for registry backbones (e.g. Llama bidirectional)
            that accept it on ``from_pretrained``. Must not be forwarded to standard HF models
            (e.g. Qwen3) loaded via ``AutoModel``; those only receive ``**hf_kwargs``.
        extract_submodel: Dotted attribute path to extract from the loaded model
            (e.g. ``"language_model"`` to extract the text backbone from a VLM).
        num_labels: Number of labels for reranking/classification backbones.
        temperature: Optional retrieval score temperature for custom retrieval backbones.
        loaded_config: A previously loaded config used to keep model and metadata resolution on the same revision.
        **hf_kwargs: Extra keyword arguments forwarded to ``from_pretrained``.

    Returns:
        The constructed ``PreTrainedModel`` backbone.

    Raises:
        ValueError: If the task is unsupported for a known model type, or the
            architecture class is missing from :class:`ModelRegistry`.
    """
    config_load_kwargs = dict(hf_kwargs)
    config = loaded_config
    if config is None:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **config_load_kwargs,
        )
    model_type = getattr(config, "model_type", "")

    if extract_submodel is not None:
        logger.info(f"Loading {model_name_or_path} with HuggingFace Auto classes to extract {extract_submodel}")
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs)
        extracted_model = _extract_submodel(model, extract_submodel)
        return _build_backbone_from_extracted_submodel(
            extracted_model,
            task=task,
            pooling=pooling,
            num_labels=num_labels,
            temperature=temperature,
        )

    if model_type.lower() == "ministral3" and task == "embedding":
        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **hf_kwargs,
        )
        backbone.config.is_causal = False
        if pooling is not None:
            backbone.config.pooling = pooling
        if temperature is not None:
            backbone.config.temperature = temperature
        return backbone

    BidirectionalModelClass = _get_supported_backbone_class(model_type, task)
    if BidirectionalModelClass is not None:
        if pooling is not None:
            hf_kwargs["pooling"] = pooling
        if num_labels is not None:
            hf_kwargs["num_labels"] = num_labels
        if temperature is not None:
            hf_kwargs["temperature"] = temperature
        return BidirectionalModelClass.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
        )

    # Fallback: use HuggingFace Auto classes for model types not in SUPPORTED_BACKBONES
    logger.info(f"Model type '{model_type}' not in SUPPORTED_BACKBONES; falling back to HuggingFace Auto classes")
    if task == "score" and num_labels is not None:
        hf_kwargs["num_labels"] = num_labels
    if task == "score":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
        )
    return AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs)


def _supports_standard_sentence_transformer_export(model: nn.Module, pooling: str) -> bool:
    """Return whether a backbone can be represented by the standard text module stack."""
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    return (
        pooling in _STANDARD_SENTENCE_TRANSFORMER_POOLING_TYPES
        and getattr(model, "main_input_name", "input_ids") == "input_ids"
        and isinstance(config, PretrainedConfig)
        and not bool(getattr(config, "is_composition", False))
        and isinstance(hidden_size, int)
        and hidden_size > 0
    )


def save_encoder_pretrained(model: nn.Module, save_directory: str, **kwargs) -> None:
    """Save an encoder model to an output directory.

    If ``checkpointer`` is present in *kwargs*, delegates to
    ``Checkpointer.save_model`` for distributed/FSDP-safe saving.
    Otherwise saves the inner ``PreTrainedModel`` and generates standard
    Sentence Transformers metadata when the encoder can be represented by that format.

    The inner model is expected to be stored as ``model.model`` (the
    backbone wrapped by the encoder).

    Args:
        model: The encoder ``nn.Module`` (must have a ``.model`` attribute
            that is the ``PreTrainedModel`` backbone).
        save_directory: Filesystem path where the checkpoint is written.
        **kwargs: Optional keys:
            - ``checkpointer``: a Checkpointer instance for distributed saves.
            - ``peft_config``: PEFT configuration (forwarded to checkpointer).
            - ``tokenizer``: tokenizer instance (forwarded to checkpointer).
            - ``is_final_checkpoint``: Whether this is the final scheduled
              training checkpoint. Defaults to ``False`` for direct callers
              that do not have recipe step-scheduler context.
    """
    checkpointer = kwargs.get("checkpointer", None)
    if checkpointer is not None:
        checkpointer.save_model(
            model=model,
            weights_path=save_directory,
            peft_config=kwargs.get("peft_config", None),
            tokenizer=kwargs.get("tokenizer", None),
            is_final_checkpoint=kwargs.get("is_final_checkpoint", False),
        )
        return

    logger.info(f"Saving encoder model to {save_directory}")
    export_config = getattr(model, "sentence_transformer_export_config", None)
    tokenizer = kwargs.get("tokenizer", None)
    if export_config is not None and tokenizer is None:
        raise ValueError("A tokenizer is required to export a loadable Sentence Transformers checkpoint.")
    deploy_config = model.get_hf_export_config() if export_config is not None else None
    original_model_path = getattr(model, "source_model_path", None)
    if original_model_path is None:
        model_reference = getattr(model.model, "name_or_path", None) or getattr(
            model.model.config, "name_or_path", None
        )
        original_model_path = str(model_reference) if model_reference and os.path.isdir(str(model_reference)) else None
    if export_config is not None:
        from nemo_automodel.components.checkpoint.addons import (
            _save_generated_sentence_transformer_assets,
            _validate_sentence_transformer_export,
        )

        _validate_sentence_transformer_export(model, tokenizer, original_model_path)

    model.model.save_pretrained(save_directory)
    if export_config is None:
        return

    deploy_config.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    _save_generated_sentence_transformer_assets(model, export_config, original_model_path, save_directory, tokenizer)


# Model types that require a registered custom retrieval backbone for each task.
_LLAMA_TASKS = {
    "embedding": "LlamaBidirectionalModel",
    "score": "LlamaBidirectionalForSequenceClassification",
}
_MINISTRAL3_BIDIREC_TASKS = {
    "embedding": "Ministral3BidirectionalModel",
}
_LLAMA_NEMOTRON_VL_TASKS = {
    "embedding": "LlamaNemotronVLModel",
}
SUPPORTED_BACKBONES = {
    "llama": _LLAMA_TASKS,
    "llama_bidirec": _LLAMA_TASKS,
    "ministral3_bidirec": _MINISTRAL3_BIDIREC_TASKS,
    "llama_nemotron_vl": _LLAMA_NEMOTRON_VL_TASKS,
}


def _init_encoder_common(encoder: nn.Module, model: PreTrainedModel) -> None:
    """Shared init for BiEncoderModel and CrossEncoderModel."""
    encoder.model = model
    encoder.config = model.config
    if ModelRegistry.has_retrieval_model(model.__class__.__name__):
        encoder.name_or_path = os.path.dirname(inspect.getfile(type(model)))
    else:
        encoder.name_or_path = getattr(model.config, "name_or_path", "")
    encoder.state_dict_adapter = EncoderStateDictAdapter()
    configure_encoder_metadata(model, model.config)


class BiEncoderModel(nn.Module):
    """Bi-encoder model that produces embeddings using a bidirectional backbone."""

    _TASK = "embedding"

    def __init__(
        self,
        model: PreTrainedModel,
        pooling: str = "avg",
        l2_normalize: bool = True,
        do_distributed_inbatch_negative: bool = False,
        detach_distributed_inbatch_negatives: bool = True,
    ):
        super().__init__()
        _init_encoder_common(self, model)
        self.pooling = pooling
        self.l2_normalize = l2_normalize
        self.sentence_transformer_export_config: SentenceTransformerExportConfig | None = None
        if _supports_standard_sentence_transformer_export(model, pooling):
            self.sentence_transformer_export_config = SentenceTransformerExportConfig()
        self.do_distributed_inbatch_negative = do_distributed_inbatch_negative
        self.detach_distributed_inbatch_negatives = detach_distributed_inbatch_negatives

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        task: str = None,
        pooling: str | None = None,
        l2_normalize: bool | None = None,
        do_distributed_inbatch_negative: bool = False,
        detach_distributed_inbatch_negatives: bool = True,
        trust_remote_code: bool = False,
        **hf_kwargs,
    ):
        """Build bi-encoder model from a pretrained backbone."""
        effective_task = cls._TASK if cls._TASK is not None else task
        if effective_task is None:
            raise ValueError("task must be specified when calling build()")
        export_only_kwargs = _EXPORT_ONLY_BI_ENCODER_KWARGS.intersection(hf_kwargs)
        if export_only_kwargs:
            names = ", ".join(sorted(export_only_kwargs))
            raise TypeError(
                f"Sentence Transformers export metadata is derived from effective NeMo settings; remove: {names}."
            )

        logger.info(f"Building BiEncoderModel from {model_name_or_path}")

        config_load_kwargs = {
            key: hf_kwargs[key]
            for key in (
                "cache_dir",
                "code_revision",
                "force_download",
                "local_files_only",
                "proxies",
                "revision",
                "subfolder",
                "token",
                "use_auth_token",
            )
            if key in hf_kwargs
        }
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **config_load_kwargs,
        )
        metadata_kwargs = dict(hf_kwargs)
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash is not None:
            metadata_kwargs["revision"] = commit_hash
        saved_options = _load_sentence_transformer_wrapper_options(model_name_or_path, metadata_kwargs)
        pooling, l2_normalize = _resolve_bi_encoder_options(
            config,
            saved_options,
            pooling,
            l2_normalize,
        )
        backbone = build_encoder_backbone(
            model_name_or_path,
            effective_task,
            trust_remote_code=trust_remote_code,
            pooling=pooling,
            loaded_config=config,
            **hf_kwargs,
        )

        encoder = cls(
            model=backbone,
            pooling=pooling,
            l2_normalize=l2_normalize,
            do_distributed_inbatch_negative=do_distributed_inbatch_negative,
            detach_distributed_inbatch_negatives=detach_distributed_inbatch_negatives,
        )
        if saved_options is not None:
            encoder.configure_sentence_transformer_prompts(
                query_prompt=saved_options.query_prompt or "",
                document_prompt=saved_options.document_prompt or "",
            )
        encoder.source_model_path = _resolve_cached_source_model_path(
            model_name_or_path,
            backbone.config,
            hf_kwargs,
        )
        encoder.source_repository_path = _resolve_cached_source_repository_path(
            model_name_or_path,
            encoder.source_model_path,
            hf_kwargs,
        )
        return encoder

    def configure_sentence_transformer_prompts(self, query_prompt: str, document_prompt: str) -> None:
        """Set the exact prompts used by the current retrieval pipeline."""
        export_config = self.sentence_transformer_export_config
        if export_config is None:
            return
        export_config.query_prompt = query_prompt
        export_config.document_prompt = document_prompt

    def disable_sentence_transformer_export(self) -> None:
        """Disable standard export when runtime behavior cannot be represented faithfully."""
        self.sentence_transformer_export_config = None

    def get_hf_export_config(self) -> PretrainedConfig:
        """Return a deployable Hugging Face config describing the effective bi-encoder."""
        config_dict = self.config.to_dict()
        model_type = getattr(type(self.config), "model_type", "")
        if model_type.endswith("_bidirec"):
            export_config_class = type(self.config).__mro__[1]
            if not issubclass(export_config_class, PretrainedConfig):
                raise TypeError(f"Unable to determine deployable Hugging Face classes for {type(self.model).__name__}.")

            config_dict.pop("model_type", None)
            config_dict.pop("auto_map", None)
            export_config = export_config_class.from_dict(config_dict)
            try:
                export_model_class = MODEL_MAPPING[type(export_config)]
            except KeyError as exc:
                raise TypeError(
                    f"Unable to determine deployable Hugging Face classes for {type(self.model).__name__}."
                ) from exc
            export_config.architectures = [export_model_class.__name__]
            export_config.is_causal = False
        else:
            export_config = self.config.__class__.from_dict(config_dict)

        export_config.pooling = self.pooling
        return export_config

    def save_pretrained(self, save_directory: str, **kwargs):
        save_encoder_pretrained(self, save_directory, **kwargs)

    def encode(self, input_dict: dict) -> Optional[torch.Tensor]:
        """Encode inputs and return pooled embeddings.

        Args:
            input_dict: Tokenized inputs (input_ids, attention_mask, etc.)

        Returns:
            Embeddings [batch_size, hidden_dim], or None if input_dict is empty.
        """
        if not input_dict:
            return None

        forward_args = inspect.getfullargspec(self.model.forward).args
        if "token_type_ids" not in forward_args and "token_type_ids" in input_dict:
            input_dict = {k: v for k, v in input_dict.items() if k != "token_type_ids"}

        model_inputs = {k: v for k, v in input_dict.items() if k not in ["kd_labels", "run_dummy_vision"]}
        if "run_dummy_vision" in forward_args and "run_dummy_vision" in input_dict:
            model_inputs["run_dummy_vision"] = input_dict["run_dummy_vision"]

        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        if hasattr(outputs, "last_hidden_state"):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[-1]

        embeds = pool(
            last_hidden_states=hidden_state,
            attention_mask=input_dict["attention_mask"],
            pool_type=self.pooling,
        )
        if self.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)

        return embeds.contiguous()

    def forward(self, input_dict: dict = None, **kwargs) -> Optional[torch.Tensor]:
        """Forward pass -- going through __call__ ensures FSDP2 unshard hooks fire."""
        return self.encode(input_dict)


class CrossEncoderModel(nn.Module):
    """Cross-encoder model for scoring/classification tasks."""

    _TASK = "score"

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        _init_encoder_common(self, model)

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        **hf_kwargs,
    ):
        """Build cross-encoder model from a pretrained backbone."""
        logger.info(f"Building CrossEncoderModel from {model_name_or_path}")
        backbone = build_encoder_backbone(
            model_name_or_path,
            task=cls._TASK,
            trust_remote_code=trust_remote_code,
            **hf_kwargs,
        )
        return cls(model=backbone)

    def save_pretrained(self, save_directory: str, **kwargs):
        save_encoder_pretrained(self, save_directory, **kwargs)

    def forward(self, input_dict: dict = None, **kwargs) -> Optional[torch.Tensor]:
        inputs = input_dict if input_dict is not None else kwargs
        inputs.setdefault("return_dict", True)
        return self.model(**inputs)


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer block list on a HuggingFace backbone."""
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers") and isinstance(model.model.layers, nn.ModuleList):
        return model.model.layers
    if (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
        and isinstance(model.transformer.h, nn.ModuleList)
    ):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers on model (tried .layers, .model.layers, .transformer.h)")


class RetrieverStudentWithProjection(nn.Module):
    """Bi-encoder student with a trainable linear projection into teacher space."""

    def __init__(
        self,
        student: BiEncoderModel,
        teacher_hidden_size: int,
        capture_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.student = student
        student_hidden = int(self.student.model.config.hidden_size)
        self.student_hidden = student_hidden
        self.teacher_hidden = int(teacher_hidden_size)

        self.projection = nn.Linear(student_hidden, self.teacher_hidden, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection = self.projection.float()
        if torch.cuda.is_available():
            # Keep projection colocated with rank-local activations under distributed training.
            self.projection = self.projection.to(device=torch.cuda.current_device())

        self._capture = LayerCapture(detach=False)
        if capture_layers:
            self.attach_intermediate_capture(capture_layers)

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        teacher_hidden_size: int,
        pooling: str = "avg",
        l2_normalize: bool = False,
        capture_layers: Sequence[int] | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "RetrieverStudentWithProjection":
        from nemo_automodel import NeMoAutoModelBiEncoder

        student = NeMoAutoModelBiEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pooling=pooling,
            l2_normalize=l2_normalize,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(student=student, teacher_hidden_size=teacher_hidden_size, capture_layers=capture_layers)

    def attach_intermediate_capture(self, layer_indices: Iterable[int]) -> "RetrieverStudentWithProjection":
        self._capture.attach(_get_layers(self.student.model), layer_indices)
        return self

    def detach_intermediate_capture(self) -> None:
        self._capture.detach_hooks()

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        # Keep the save format HF-compatible for evaluator tooling: this stores
        # only the student backbone; the projection is saved by the recipe.
        self.student.save_pretrained(save_directory, **kwargs)

    def _encode(self, input_dict: dict) -> torch.Tensor:
        if not input_dict:
            return None
        embeds = self.student(input_dict)
        return embeds.contiguous()

    def forward(
        self,
        input_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        self._capture.reset()
        pooled = self._encode(input_dict)
        pooled_fp32 = pooled.float()

        with torch.amp.autocast(device_type=pooled.device.type, enabled=False):
            projected = self.projection(pooled_fp32)

        return pooled_fp32, projected, dict(self._capture.outputs)


class RetrieverTeacherEmbeddingEncoder(nn.Module):
    """Frozen bi-encoder teacher with optional intermediate-layer capture."""

    def __init__(
        self,
        teacher: BiEncoderModel,
        capture_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.hidden_size = int(self.teacher.model.config.hidden_size)
        self._capture = LayerCapture(detach=True)
        if capture_layers:
            self.attach_intermediate_capture(capture_layers)

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        pooling: str = "avg",
        l2_normalize: bool = False,
        capture_layers: Sequence[int] | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "RetrieverTeacherEmbeddingEncoder":
        from nemo_automodel import NeMoAutoModelBiEncoder

        teacher = NeMoAutoModelBiEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pooling=pooling,
            l2_normalize=l2_normalize,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(teacher=teacher, capture_layers=capture_layers)

    def attach_intermediate_capture(self, layer_indices: Iterable[int]) -> "RetrieverTeacherEmbeddingEncoder":
        self._capture.attach(_get_layers(self.teacher.model), layer_indices)
        return self

    def detach_intermediate_capture(self) -> None:
        self._capture.detach_hooks()

    @torch.no_grad()
    def _encode(self, input_dict: dict) -> torch.Tensor:
        if not input_dict:
            return None
        embeds = self.teacher(input_dict)
        return embeds.contiguous()

    @torch.no_grad()
    def forward(self, input_dict: dict) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        self._capture.reset()
        pooled = self._encode(input_dict)
        return pooled.float(), dict(self._capture.outputs)
