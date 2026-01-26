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
state_dict(), load_state_dict(), and save_pretrained() methods while using
NeMo's checkpointing infrastructure internally.

All methods use checkpointing.py for unified distributed/async support:
- state_dict() → ModelState + adapter conversion
- save_pretrained() → Checkpointer.save_model() logic
- from_pretrained() → Checkpointer.load_base_model() logic

Checkpointer is passed explicitly (dependency injection) - no global state.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    _maybe_adapt_state_dict_to_hf,
    _maybe_adapt_state_dict_from_hf,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from transformers.utils import TRANSFORMERS_CACHE

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.tokenization_utils import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class HFCheckpointingMixin:
    """Mixin providing HF-compatible API using NeMo's checkpointing infrastructure.

    All methods use checkpointing.py for unified distributed/async support:
    - state_dict() → ModelState + adapter conversion
    - save_pretrained() → Checkpointer.save_model() logic
    - from_pretrained() → Checkpointer.load_base_model() logic

    Checkpointer is passed explicitly.

    For PreTrainedModel subclasses:
    - super().from_pretrained() handles: downloads, quantization config, meta device init
    - Checkpointer.load_base_model() handles: actual weight loading with format conversion

    For nn.Module subclasses (no parent from_pretrained):
    - Falls back to manual config loading + Checkpointer
    """

    _checkpointer: Optional[Checkpointer] = None  # Set by from_pretrained or user

    def state_dict(self, checkpointer: Optional[Checkpointer] = None) -> dict[str, Any]:
        """Return HF-formatted state dict using ModelState + adapter conversion.

        Uses same logic as Checkpointer.save_model() but returns HF keys for API compatibility.

        Returns:
            dict: State dict with HuggingFace-compatible keys (separate q_proj, k_proj, v_proj)
        """
        if checkpointer is None:
            checkpointer = self._checkpointer
        if checkpointer is None:
            raise ValueError(
                "No checkpointer provided. Please pass the `checkpointer` argument."
            )
        # Use ModelState to get state dict (handles FSDP/DTensor properly)
        model_state = ModelState(self, is_peft=checkpointer.config.is_peft)
        native_state = model_state.state_dict()

        # Convert to HF format using existing adapter from checkpointing.py
        hf_state = _maybe_adapt_state_dict_to_hf(model_state.model[0], native_state, quantization=False, device_mesh=checkpointer.moe_mesh)
        return hf_state

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        """Accept HF-formatted state dict and convert to internal NeMo format.

        Args:
            state_dict: State dict with HuggingFace-compatible keys
        """
        model_state = ModelState(self, is_peft=self._checkpointer.config.is_peft)
        # Convert from HF format using existing adapter from checkpointing.py
        native_state = _maybe_adapt_state_dict_from_hf(model_state.model[0], state_dict, moe_mesh=self._checkpointer.moe_mesh)

        # Load using ModelState (handles FSDP/DTensor properly)
        has_state_dict_adapter = hasattr(model_state.model[0], "state_dict_adapter")
        model_state.load_state_dict(native_state, strict=not (len(model_state.model) > 1 or has_state_dict_adapter))

    def save_pretrained(
        self,
        save_directory: str,
        checkpointer: Optional[Checkpointer] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> None:
        """Save model in HF-compatible format using Checkpointer infrastructure.

        Supports distributed saving, sharding, and async checkpointing.

        Args:
            save_directory: Output path
            checkpointer: Checkpointer instance. Uses self._checkpointer if not provided.
            tokenizer: Optional tokenizer to save alongside model
            **kwargs: Additional arguments
        """
        if checkpointer is None:
            checkpointer = self._checkpointer
        if checkpointer is None:
            raise ValueError(
                "No checkpointer provided. Please pass the `checkpointer` argument."
            )

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
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        checkpointer: Optional[Checkpointer] = None,
        torch_dtype: str = "auto",
        device: Optional[torch.device] = None,
        hf_config: Optional["PretrainedConfig"] = None,
        **kwargs,
    ):
        """Load model using HF infrastructure + Checkpointer for weight loading.

        Args:
            pretrained_model_name_or_path: HF model ID or local path
            checkpointer: Checkpointer instance for weight loading. Created if not provided.
            torch_dtype: Data type for model
            device: Target device for model
            **kwargs: Additional arguments passed to parent from_pretrained

        For PreTrainedModel subclasses (via MRO):
        1. super().from_pretrained() handles: downloads, quantization config, meta device init
        2. Checkpointer.load_base_model() handles: actual weight loading with format conversion

        For nn.Module subclasses (no parent from_pretrained):
        Falls back to manual config loading + Checkpointer
        """
        if checkpointer is None:
            raise ValueError("No checkpointer provided. Please pass the `checkpointer` argument.")

        # Check if parent has from_pretrained (PreTrainedModel does, nn.Module doesn't)
        parent_has_from_pretrained = hasattr(super(), 'from_pretrained')

        if parent_has_from_pretrained:
            # Use HF's from_pretrained for: downloads, quantization setup, config handling
            # But keep model on meta device (weights not loaded yet)
            kwargs['device_map'] = 'meta'  # Prevents HF from loading weights
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        else:
            # nn.Module models: manual config + meta device init
            model = cls(hf_config, *model_args, **kwargs)

        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use Checkpointer to load weights (handles distributed loading + format conversion)
        peft_config = kwargs.get("peft_config", None)
        peft_init_method = peft_config.get("lora_A_init", None) if peft_config else None
        checkpointer.load_base_model(
            model=model,
            device=device,
            root_dir=kwargs.get("cache_dir", TRANSFORMERS_CACHE),
            model_name=pretrained_model_name_or_path,
            peft_init_method=peft_init_method,
            load_base_model=True,
        )

        # Store checkpointer for save_pretrained
        model._checkpointer = checkpointer
        return model
