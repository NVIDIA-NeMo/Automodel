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
from transformers import AutoConfig, AutoModel, PreTrainedModel
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


# HuggingFace model_type -> task -> bidirectional architecture class name in ModelRegistry
_LLAMA_TASKS = {
    "embedding": "LlamaBidirectionalModel",
    "score": "LlamaBidirectionalForSequenceClassification",
}
SUPPORTED_BACKBONES = {
    "llama": _LLAMA_TASKS,
    "llama_bidirec": _LLAMA_TASKS,
}


def _register_bidirectional_models():
    """Register bidirectional models with HuggingFace Auto classes.

    This is needed so that AutoModel.from_config(LlamaBidirectionalConfig)
    works inside LlamaForSequenceClassification.__init__.
    """
    from nemo_automodel.components.models.llama_bidirectional.model import (
        LlamaBidirectionalConfig,
        LlamaBidirectionalModel,
    )

    try:
        AutoConfig.register(LlamaBidirectionalConfig.model_type, LlamaBidirectionalConfig)
    except ValueError:
        pass  # Already registered
    try:
        AutoModel.register(LlamaBidirectionalConfig, LlamaBidirectionalModel)
    except ValueError:
        pass  # Already registered


_register_bidirectional_models()


class EncoderModel(nn.Module):
    """Base encoder model with shared backbone loading and saving logic."""

    _TASK = None

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.config = self.model.config

        # HuggingFace consolidated checkpoint compatibility
        self.name_or_path = os.path.dirname(inspect.getfile(type(self.model)))
        self.state_dict_adapter = EncoderStateDictAdapter()
        encoder_class_name = self.model.__class__.__name__
        self.config.architectures = [encoder_class_name]
        self.config.auto_map = {
            "AutoModel": f"model.{encoder_class_name}",
            "AutoConfig": f"model.{encoder_class_name.replace('Model', 'Config')}",
        }

    def forward(self, input_dict: dict = None, **kwargs) -> Optional[torch.Tensor]:
        """Forward pass -- subclasses must implement."""
        raise NotImplementedError

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
        """Build encoder model from a pretrained backbone."""
        effective_task = cls._TASK if cls._TASK is not None else task
        if effective_task is None:
            raise ValueError("task must be specified when calling build() on EncoderModel directly")

        # Resolve to correct subclass if called on base
        if cls is EncoderModel:
            _SUBCLASS_MAP = {"embedding": BiEncoderModel, "score": CrossEncoderModel}
            target_cls = _SUBCLASS_MAP.get(effective_task)
            if target_cls is None:
                raise ValueError(f"Unknown task '{effective_task}'. Available: {list(_SUBCLASS_MAP)}")
        else:
            target_cls = cls

        logger.info(f"Building EncoderModel from {model_name_or_path}")

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", "")

        task_map = SUPPORTED_BACKBONES.get(model_type.lower())
        if task_map is None:
            raise ValueError(
                f"Unsupported model type '{model_type}' for encoder. Supported types: {', '.join(SUPPORTED_BACKBONES)}."
            )

        arch_name = task_map.get(effective_task)
        if arch_name is None:
            raise ValueError(
                f"Unsupported task '{effective_task}' for model type '{model_type}'. "
                f"Available tasks: {', '.join(task_map)}."
            )

        if arch_name not in ModelRegistry.model_arch_name_to_cls:
            raise ValueError(f"Model class '{arch_name}' not found in ModelRegistry.")
        BidirectionalModelClass = ModelRegistry.model_arch_name_to_cls[arch_name]
        logger.info(f"Using {arch_name} from registry")

        backbone = BidirectionalModelClass.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
        )

        # Subclass-specific construction
        if issubclass(target_cls, BiEncoderModel):
            return target_cls(model=backbone, pooling=pooling, l2_normalize=l2_normalize)
        return target_cls(model=backbone)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model to output directory.

        If ``checkpointer`` is in kwargs, delegates to Checkpointer.save_model
        for distributed/FSDP-safe saving. Otherwise falls back to HF-native save.
        """
        checkpointer = kwargs.get("checkpointer", None)
        if checkpointer is not None:
            checkpointer.save_model(
                model=self,
                weights_path=save_directory,
                peft_config=kwargs.get("peft_config", None),
                tokenizer=kwargs.get("tokenizer", None),
            )

            return

        logger.info(f"Saving EncoderModel to {save_directory}")

        self.model.save_pretrained(save_directory)


class BiEncoderModel(EncoderModel):
    """Bi-encoder model that produces embeddings using a bidirectional backbone."""

    _TASK = "embedding"

    def __init__(self, model: PreTrainedModel, pooling: str = "avg", l2_normalize: bool = True):
        super().__init__(model)
        self.pooling = pooling
        self.l2_normalize = l2_normalize

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


class CrossEncoderModel(EncoderModel):
    """Cross-encoder model for scoring/classification tasks."""

    _TASK = "score"

    def forward(self, input_dict: dict = None, **kwargs) -> Optional[torch.Tensor]:
        """Forward pass -- going through __call__ ensures FSDP2 unshard hooks fire."""
        inputs = input_dict if input_dict is not None else kwargs
        inputs.setdefault("return_dict", True)
        return self.model(**inputs)
