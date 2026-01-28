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
from typing import TYPE_CHECKING, Optional

import torch

from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
)

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.tokenization_utils import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

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
    """

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
    def from_config(
        cls,
        config: "PretrainedConfig",
        *model_args,
        torch_dtype: str = "auto",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """Create model from config with random weights using Checkpointer for initialization.

        Args:
            config: Model configuration
            torch_dtype: Data type for model
            device: Target device for model
            **kwargs: Additional arguments passed to parent from_config

        For PreTrainedModel subclasses (via MRO):
        - super().from_config() handles: model instantiation with config

        For nn.Module subclasses (no parent from_config):
        - Falls back to direct instantiation with config
        """
        # Check if parent has from_config by looking at MRO after the mixin
        # (hasattr(super(), 'from_config') doesn't work correctly with super objects)
        mro = cls.__mro__
        mixin_idx = mro.index(HFCheckpointingMixin)
        parent_has_from_config = any(
            'from_config' in vars(parent_cls) for parent_cls in mro[mixin_idx + 1:]
        )

        if parent_has_from_config:
            model = super().from_config(
                config,
                *model_args,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        else:
            # nn.Module/PreTrainedModel models: direct instantiation with config only
            # Most model __init__ methods only accept config, not extra kwargs
            model = cls(config, *model_args)

        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use Checkpointer to initialize model on device (no weight loading)
        peft_config = kwargs.get("peft_config", None)
        peft_init_method = peft_config.get("lora_A_init", None) if peft_config else None
        # dummy checkpointer for initialization. We don't load weights so these values don't matter.
        checkpointer = Checkpointer(
            config=CheckpointingConfig(
                enabled=True,
                checkpoint_dir=None,
                model_save_format="safetensors",
                model_cache_dir=None,
                model_repo_id=None,
                save_consolidated=False,
                is_peft=False,
            ),
            dp_rank=0,
            tp_rank=0,
            pp_rank=0,
        )
        checkpointer.load_base_model(
            model=model,
            device=device,
            root_dir=None,
            model_name=None,
            peft_init_method=peft_init_method,
            load_base_model=False,  # Don't load weights, just initialize
        )

        return model
