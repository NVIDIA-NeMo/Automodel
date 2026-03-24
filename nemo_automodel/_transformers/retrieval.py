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
from transformers.utils import logging

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

logger = logging.get_logger(__name__)


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
    elif pool_type == "colbert":
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
        config.auto_map = {"AutoConfig": f"model.{config_class_name}"}
        if "ForSequenceClassification" in encoder_class_name:
            config.auto_map["AutoModelForSequenceClassification"] = f"model.{encoder_class_name}"
        else:
            config.auto_map["AutoModel"] = f"model.{encoder_class_name}"


def build_encoder_backbone(
    model_name_or_path: str,
    task: str,
    trust_remote_code: bool = False,
    **hf_kwargs,
) -> PreTrainedModel:
    """Build an encoder backbone from a pretrained checkpoint.

    For model types listed in :data:`SUPPORTED_BACKBONES`, resolves the
    custom bidirectional architecture class from :class:`ModelRegistry`.
    For all other model types, falls back to
    ``AutoModel.from_pretrained`` (or ``AutoModelForSequenceClassification``
    for the ``"score"`` task).

    Args:
        model_name_or_path: Path or HuggingFace Hub identifier.
        task: The encoder task (e.g. ``"embedding"``, ``"score"``).
        trust_remote_code: Whether to allow custom remote code.
        **hf_kwargs: Extra keyword arguments forwarded to ``from_pretrained``.

    Returns:
        The constructed ``PreTrainedModel`` backbone.

    Raises:
        ValueError: If the task is unsupported for a known model type, or the
            architecture class is missing from :class:`ModelRegistry`.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    task_map = SUPPORTED_BACKBONES.get(model_type.lower())
    if task_map is not None:
        arch_name = task_map.get(task)
        if arch_name is None:
            raise ValueError(
                f"Unsupported task '{task}' for model type '{model_type}'. "
                f"Available tasks: {', '.join(task_map)}."
            )

        if arch_name not in ModelRegistry.model_arch_name_to_cls:
            raise ValueError(f"Model class '{arch_name}' not found in ModelRegistry.")

        BidirectionalModelClass = ModelRegistry.model_arch_name_to_cls[arch_name]
        logger.info(f"Using {arch_name} from registry")

        return BidirectionalModelClass.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
        )

    # Fallback: use HuggingFace Auto classes for model types not in SUPPORTED_BACKBONES
    logger.info(
        f"Model type '{model_type}' not in SUPPORTED_BACKBONES; "
        f"falling back to HuggingFace Auto classes"
    )
    if task == "score":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
        )
    return AutoModel.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
    )


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
    """
    checkpointer = kwargs.get("checkpointer", None)
    if checkpointer is not None:
        checkpointer.save_model(
            model=model,
            weights_path=save_directory,
            peft_config=kwargs.get("peft_config", None),
            tokenizer=kwargs.get("tokenizer", None),
        )
        return

    logger.info(f"Saving encoder model to {save_directory}")
    model.model.save_pretrained(save_directory)


# HuggingFace model_type -> task -> bidirectional architecture class name in ModelRegistry
_LLAMA_TASKS = {
    "embedding": "LlamaBidirectionalModel",
    "score": "LlamaBidirectionalForSequenceClassification",
}
SUPPORTED_BACKBONES = {
    "llama": _LLAMA_TASKS,
    "llama_bidirec": _LLAMA_TASKS,
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

    def __init__(self, model: PreTrainedModel, pooling: str = "avg", l2_normalize: bool = True):
        super().__init__()
        _init_encoder_common(self, model)
        self.pooling = pooling
        self.l2_normalize = l2_normalize

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        task: str = None,
        pooling: str = "avg",
        l2_normalize: bool = True,
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

        return cls(model=backbone, pooling=pooling, l2_normalize=l2_normalize)

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

        if "token_type_ids" not in inspect.getfullargspec(self.model.forward).args and "token_type_ids" in input_dict:
            input_dict = {k: v for k, v in input_dict.items() if k != "token_type_ids"}

        outputs = self.model(
            **{k: v for k, v in input_dict.items() if k not in ["kd_labels"]},
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
