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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""

    enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    backend: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = enabled
        self.mode = mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.backend = backend
        self.options = options or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "backend": self.backend,
            "options": self.options,
        }


def enable_torch_dynamo_scalar_outputs():
    """Enable torch.dynamo to capture scalar outputs for better Flash Attention + torch.compile compatibility."""
    try:
        import torch._dynamo.config

        torch._dynamo.config.capture_scalar_outputs = True
    except ImportError:
        logger.warning("torch._dynamo not available, skipping scalar output capture configuration")


def patch_prepare_fa2_from_position_ids():
    """
    Apply a simple targeted patch to fix the prepare_fa2_from_position_ids function
    for torch.compile compatibility.

    This is the key function that needs the fix for the max_length computation.
    """
    try:
        import transformers.modeling_flash_attention_utils as fa_utils

        def prepare_fa2_from_position_ids(query, key, value, position_ids):
            """
            This function returns necessary arguments to call `flash_attn_varlen_func`.
            All three query, key, value states will be flattened.
            Cumulative lengths of each examples in the batch will be extracted from position_ids.

            NOTE: ideally cumulative lengths should be prepared at the data collator stage

            This version includes the torch.compile fix for max_length computation.
            """
            query = query.view(-1, query.size(-2), query.size(-1))
            key = key.contiguous().view(-1, key.size(-2), key.size(-1))
            value = value.contiguous().view(-1, value.size(-2), value.size(-1))
            position_ids = position_ids.flatten()
            indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

            cu_seq_lens = torch.cat(
                (
                    indices_q[position_ids == 0],
                    torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
                )
            )

            # KEY FIX: This is the exact fix from the user's request
            # The .item() call ensures we get an integer instead of a FakeTensor during torch.compile
            max_length = position_ids.max().item() + 1

            return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))

        # Apply the patch
        fa_utils.prepare_fa2_from_position_ids = prepare_fa2_from_position_ids

        return True

    except Exception as e:
        logger.warning(f"Failed to patch prepare_fa2_from_position_ids: {e}")
        return False


def apply_flash_attention_compile_fix():
    """
    Apply the Flash Attention + torch.compile compatibility fix.

    This enables scalar output capture and patches the key function that causes issues.
    """
    # Enable scalar output capture
    enable_torch_dynamo_scalar_outputs()

    # Apply the targeted patch
    success = patch_prepare_fa2_from_position_ids()

    if not success:
        logger.warning("Flash Attention + torch.compile compatibility fix failed")

    return success


def compile_model(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile the model with Flash Attention compatibility.

    Args:
        model: The model to compile.
        config: Compile configuration.

    Returns:
        The compiled model.
    """
    if not config.enabled:
        logger.info("torch.compile is disabled")
        return model

    # Apply Flash Attention compatibility fix
    apply_flash_attention_compile_fix()

    # Prepare torch.compile arguments
    options_dict = config.options.to_dict() if hasattr(config.options, "to_dict") else dict(config.options)
    compile_kwargs = {
        "mode": config.mode,
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    if config.backend is not None:
        compile_kwargs["backend"] = config.backend
    compile_kwargs.update(options_dict)

    logger.info(f"Compiling model with backend={config.backend}, mode={config.mode}, dynamic={config.dynamic}")

    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        logger.info("Model compilation successful")
        return compiled_model
    except Exception as e:
        logger.error(f"Model compilation failed: {type(e).__name__}: {e}")
        logger.info("Returning original model")
        return model


def create_compile_config_from_dict(config_dict: Dict[str, Any]) -> CompileConfig:
    """Create a CompileConfig from a dictionary.

    Args:
        config_dict: Dictionary containing compile configuration.

    Returns:
        CompileConfig instance.
    """
    return CompileConfig(
        enabled=config_dict.get("enabled", False),
        mode=config_dict.get("mode", "default"),
        fullgraph=config_dict.get("fullgraph", False),
        dynamic=config_dict.get("dynamic", False),
        backend=config_dict.get("backend", None),
        options=config_dict.get("options", {}),
    )


def build_compile_config(cfg: Optional[Dict[str, Any]]) -> CompileConfig:
    """Build a compile config from configuration.

    Args:
        cfg: Configuration dictionary for compilation.

    Returns:
        CompileConfig instance.
    """
    if cfg is None:
        return CompileConfig(enabled=False)
    else:
        return create_compile_config_from_dict(cfg)


# Apply the fix immediately when this module is imported
_FLASH_ATTENTION_FIX_APPLIED = apply_flash_attention_compile_fix()