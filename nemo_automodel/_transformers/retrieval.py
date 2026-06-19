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
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from transformers.utils import logging

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

logger = logging.get_logger(__name__)


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


def _get_submodule_parent_and_attr(model: nn.Module, submodule_path: str) -> tuple[nn.Module, str]:
    """Return the parent module and final attribute name for a dotted submodule path."""
    attrs = submodule_path.split(".")
    parent = model
    for attr in attrs[:-1]:
        parent = getattr(parent, attr)
    return parent, attrs[-1]


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

    if task == "score" and not has_supported_target:
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

    attn_implementation = getattr(text_config, "_attn_implementation", None)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    if has_supported_target and pooling is not None:
        config.pooling = pooling
    if num_labels is not None:
        config.num_labels = num_labels
    if has_supported_target and temperature is not None:
        config.temperature = temperature

    return _load_from_extracted_state(backbone_class, config, extracted_model)


def _sync_parent_config_after_submodel_replacement(
    parent_model: PreTrainedModel,
    replace_submodel: str,
    replacement_model: PreTrainedModel,
) -> None:
    """Keep the parent config aligned with a replaced text submodule."""
    parent_config = getattr(parent_model, "config", None)
    replacement_config = getattr(replacement_model, "config", None)
    if parent_config is None or replacement_config is None:
        return

    if replace_submodel.endswith("language_model") and hasattr(parent_config, "text_config"):
        parent_config.text_config = replacement_config


def _replace_submodel_with_retrieval_backbone(
    parent_model: PreTrainedModel,
    replace_submodel: str,
    task: str,
    pooling: Optional[str],
    num_labels: Optional[int],
    temperature: Optional[float],
) -> PreTrainedModel:
    """Replace a nested parent submodule with its retrieval-specific backbone."""
    extracted_model = _extract_submodel(parent_model, replace_submodel)
    replacement_model = _build_backbone_from_extracted_submodel(
        extracted_model,
        task=task,
        pooling=pooling,
        num_labels=num_labels,
        temperature=temperature,
    )
    parent, attr = _get_submodule_parent_and_attr(parent_model, replace_submodel)
    setattr(parent, attr, replacement_model)
    _sync_parent_config_after_submodel_replacement(parent_model, replace_submodel, replacement_model)
    return parent_model


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


def _get_module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Return a real device/dtype pair from a module's parameters or buffers."""
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _sum_nested_tensors(value) -> torch.Tensor | None:
    """Sum tensors from nested model outputs so they can contribute a zero gradient."""
    if isinstance(value, torch.Tensor):
        return value.sum()
    if isinstance(value, (list, tuple)):
        total = None
        for item in value:
            item_sum = _sum_nested_tensors(item)
            if item_sum is not None:
                total = item_sum if total is None else total + item_sum
        return total
    return None


def _dummy_vision_sum(model: nn.Module) -> torch.Tensor | None:
    """Run a one-image dummy vision path for VLMs that expose get_image_features()."""
    if not hasattr(model, "get_image_features") or not hasattr(model, "vision_tower"):
        return None

    vision_tower = model.vision_tower
    device, dtype = _get_module_device_dtype(vision_tower)
    patch_size = getattr(vision_tower, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(vision_tower, "config", None), "patch_size", 16)
    spatial_merge_size = getattr(getattr(model, "config", None), "spatial_merge_size", 1)
    if isinstance(patch_size, (list, tuple)):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_width = patch_size
    if isinstance(spatial_merge_size, (list, tuple)):
        spatial_merge_height, spatial_merge_width = spatial_merge_size
    else:
        spatial_merge_height = spatial_merge_width = spatial_merge_size

    # HF Mistral3 squeezes projected image features; keep more than one merged token
    # so that the token dimension survives before its split-by-image step.
    image_height = int(patch_height) * int(spatial_merge_height) * 2
    image_width = int(patch_width) * int(spatial_merge_width) * 2

    dummy_pixels = torch.zeros(1, 3, image_height, image_width, device=device, dtype=dtype)
    dummy_image_sizes = torch.tensor([[image_height, image_width]], device=device, dtype=torch.long)
    dummy_outputs = model.get_image_features(
        pixel_values=dummy_pixels,
        image_sizes=dummy_image_sizes,
        return_dict=True,
    )
    dummy_sum = _sum_nested_tensors(getattr(dummy_outputs, "pooler_output", None))
    if dummy_sum is None:
        dummy_sum = _sum_nested_tensors(dummy_outputs)
    return dummy_sum


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
    replace_submodel: Optional[str] = None,
    num_labels: Optional[int] = None,
    temperature: Optional[float] = None,
    **hf_kwargs,
) -> PreTrainedModel:
    """Build an encoder backbone from a pretrained checkpoint.

    When ``extract_submodel`` is set, loads the parent model with HuggingFace
    Auto classes and extracts the dotted path. For supported extracted text
    backbones, it then builds the registered retrieval class for the requested
    task (bidirectional base model for ``"embedding"``, sequence-classification
    wrapper for ``"score"``). For unsupported extracted text backbones, it
    returns the extracted model for ``"embedding"`` and wraps it with
    ``AutoModelForSequenceClassification`` for ``"score"``.

    Without ``extract_submodel``, model types listed in
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
        replace_submodel: Dotted attribute path to replace on the loaded parent model
            with a retrieval-specific backbone. The parent model is returned.
        num_labels: Number of labels for reranking/classification backbones.
        temperature: Optional retrieval score temperature for custom retrieval backbones.
        **hf_kwargs: Extra keyword arguments forwarded to ``from_pretrained``.

    Returns:
        The constructed ``PreTrainedModel`` backbone.

    Raises:
        ValueError: If the task is unsupported for a known model type, or the
            architecture class is missing from :class:`ModelRegistry`.
    """
    if extract_submodel is not None and replace_submodel is not None:
        raise ValueError("Only one of extract_submodel and replace_submodel can be set.")

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    if extract_submodel is not None:
        logger.info(f"Loading {model_name_or_path} with HuggingFace Auto classes to extract {extract_submodel}")
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs)
        extracted_model = _extract_submodel(model, extract_submodel)
        extracted_model = _build_backbone_from_extracted_submodel(
            extracted_model,
            task=task,
            pooling=pooling,
            num_labels=num_labels,
            temperature=temperature,
        )
        return extracted_model

    if replace_submodel is not None:
        logger.info(f"Loading {model_name_or_path} with HuggingFace Auto classes to replace {replace_submodel}")
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs)
        return _replace_submodel_with_retrieval_backbone(
            model,
            replace_submodel=replace_submodel,
            task=task,
            pooling=pooling,
            num_labels=num_labels,
            temperature=temperature,
        )

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


def save_encoder_pretrained(model: nn.Module, save_directory: str, **kwargs) -> None:
    """Save an encoder model to an output directory.

    If ``checkpointer`` is present in *kwargs*, delegates to
    ``Checkpointer.save_model`` for distributed/FSDP-safe saving.
    Otherwise falls back to the inner ``PreTrainedModel.save_pretrained``.

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
    model.model.save_pretrained(save_directory)


# HuggingFace model_type -> task -> bidirectional architecture class name in ModelRegistry
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
    "ministral3": _MINISTRAL3_BIDIREC_TASKS,
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
        self.do_distributed_inbatch_negative = do_distributed_inbatch_negative
        self.detach_distributed_inbatch_negatives = detach_distributed_inbatch_negatives

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        task: str = None,
        pooling: str = "avg",
        l2_normalize: bool = True,
        do_distributed_inbatch_negative: bool = False,
        detach_distributed_inbatch_negatives: bool = True,
        trust_remote_code: bool = False,
        **hf_kwargs,
    ):
        """Build bi-encoder model from a pretrained backbone."""
        effective_task = cls._TASK if cls._TASK is not None else task
        if effective_task is None:
            raise ValueError("task must be specified when calling build()")

        logger.info(f"Building BiEncoderModel from {model_name_or_path}")

        backbone = build_encoder_backbone(
            model_name_or_path, effective_task, trust_remote_code=trust_remote_code, pooling=pooling, **hf_kwargs
        )

        return cls(
            model=backbone,
            pooling=pooling,
            l2_normalize=l2_normalize,
            do_distributed_inbatch_negative=do_distributed_inbatch_negative,
            detach_distributed_inbatch_negatives=detach_distributed_inbatch_negatives,
        )

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
        dummy_vision_sum = None
        if "run_dummy_vision" not in forward_args and input_dict.get("run_dummy_vision") is not False:
            if self.training and model_inputs.get("pixel_values") is None:
                dummy_vision_sum = _dummy_vision_sum(self.model)

        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        if hasattr(outputs, "last_hidden_state"):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[-1]
        if dummy_vision_sum is not None:
            hidden_state = hidden_state + dummy_vision_sum.to(hidden_state.dtype) * 0.0

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
