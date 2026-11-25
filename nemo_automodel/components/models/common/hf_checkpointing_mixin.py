# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""HuggingFace-compatible checkpointing mixin for NeMo Automodel.

This module provides a mixin class that gives models HuggingFace-compatible
save_pretrained() and from_pretrained() methods while using NeMo's checkpointing
infrastructure internally.

Key design principle: We do NOT override state_dict() or load_state_dict().
PyTorch's DCP expects these to behave like standard nn.Module methods.
HF format conversions happen only in save_pretrained() and from_pretrained() via Checkpointer.

Checkpointer is passed explicitly (dependency injection) - no global state.
"""

import logging
from typing import TYPE_CHECKING, Optional, Protocol

from torch import nn

from nemo_automodel.components.models.common.lm_head import LMHead

if TYPE_CHECKING:
    from peft import PeftConfig
    from transformers.tokenization_utils import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ModelSaver(Protocol):
    """The part of a checkpointer that ``save_pretrained`` uses.

    ``nemo_automodel.components.checkpoint.checkpointing.Checkpointer`` provides it; the
    caller passes the checkpointer in, so model code does not import the checkpoint package.
    """

    def save_model(
        self,
        model: nn.Module,
        weights_path: str,
        peft_config: Optional["PeftConfig"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        is_final_checkpoint: bool = False,
    ) -> None:
        """Save ``model``'s weights under ``weights_path``."""
        ...


class HFCheckpointingMixin:
    """Mixin providing HF-compatible API using NeMo's checkpointing infrastructure.

    Provides save_pretrained() and from_pretrained() methods that use Checkpointer
    for unified distributed/async support with HF format conversion.

    Key design: We do NOT override state_dict() or load_state_dict() because
    PyTorch's DCP expects these to behave like standard nn.Module methods.

    For PreTrainedModel subclasses:
    - super().from_pretrained() handles: downloads, quantization config, meta device init
    - Checkpointer.load_base_model() handles: actual weight loading with format conversion

    For nn.Module subclasses (no parent from_pretrained):
    - Falls back to manual config loading + Checkpointer

    For PreTrainedModel subclasses, a vocabulary resize keeps the model's ``LMHead``.
    """

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: int | None = None,
        transposed: bool = False,
        mean_resizing: bool = True,
    ) -> nn.Linear:
        """Resize the output head as Transformers does, keeping an ``LMHead`` and its settings.

        Transformers builds a plain ``nn.Linear`` for the resized vocabulary. An ``LMHead`` takes that module's
        resized parameters instead, so its class, ``logits_dtype`` and softcap survive the resize.

        Args:
            old_lm_head: The output head, with weight of shape [vocab, hidden].
            new_num_tokens: New vocabulary size; None returns the head unchanged.
            transposed: Whether the head's weight is stored as [hidden, vocab].
            mean_resizing: Initialize added rows from the mean and covariance of the existing rows.

        Returns:
            The resized head, with weight of shape [new_num_tokens, hidden].
        """
        new_lm_head = super()._get_resized_lm_head(old_lm_head, new_num_tokens, transposed, mean_resizing)
        if new_lm_head is old_lm_head or not isinstance(old_lm_head, LMHead):
            return new_lm_head
        # Transformers reads the returned head's requires_grad to restore the old one's, so keep it here.
        new_lm_head.requires_grad_(old_lm_head.weight.requires_grad)
        old_lm_head.weight, old_lm_head.bias = new_lm_head.weight, new_lm_head.bias
        old_lm_head.out_features = new_lm_head.out_features
        return old_lm_head

    def save_pretrained(
        self,
        save_directory: str,
        checkpointer: ModelSaver | None = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> None:
        """Save model in HF-compatible format using Checkpointer infrastructure.

        Supports distributed saving, sharding, and async checkpointing.

        Args:
            save_directory: Output path
            checkpointer: Checkpointer instance. Uses self._checkpointer if not provided.
            tokenizer: Optional tokenizer to save alongside model
            **kwargs: Additional arguments, including ``peft_config`` and
                ``is_final_checkpoint``. Direct callers that do not have recipe
                step-scheduler context default ``is_final_checkpoint`` to
                ``False``.
        """
        if checkpointer is None:
            raise ValueError("No checkpointer provided. Please pass the `checkpointer` argument.")

        # Use Checkpointer.save_model() which handles:
        # - ModelState.state_dict()
        # - _maybe_adapt_state_dict_to_hf() conversion
        # - Distributed/sharded saving via DCP
        # - Consolidated HF safetensors output
        # - Async checkpointing if enabled
        checkpointer.save_model(
            model=self,
            weights_path=save_directory,
            peft_config=kwargs.get("peft_config", None),
            tokenizer=tokenizer,
            is_final_checkpoint=kwargs.get("is_final_checkpoint", False),
        )
