# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_param_stats(model: nn.Module) -> tuple[int, int, float]:
    """
    Get the number of trainable parameters and the L2 norm of the model.

    Args:
        model: Model to analyze

    Returns:
        total_params: int
        trainable_params: int
        local_sq_norm: float
    """
    total_params = 0
    trainable_params = 0
    local_sq_norm = 0.0

    for p in model.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
        try:
            local_sq_norm += float(p.detach().float().norm(2).item() ** 2)
        except Exception:
            pass
    return total_params, trainable_params, local_sq_norm


def print_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        trainable_params: int
        total_params: int
    """
    total_params, trainable_params, local_sq_norm = _get_model_param_stats(model)

    try:
        # TODO(@akoumparouli): make this sharding aware.
        local_sq_norm = float(local_sq_norm**0.5)
        trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

        logging.info("Model summary:")
        logging.info("--------------------------------")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters percentage: {trainable_pct:.2f}%")
        logging.info(f"Param L2 norm: {local_sq_norm:.4f}")
        logging.info("--------------------------------")
    except Exception:
        logging.info("Model summary: <unavailable>")

    return trainable_params, total_params


def _freeze_module_by_attribute_and_patterns(model, attribute_name, name_patterns):
    """Helper function to freeze parameters by attribute name and name patterns.

    Args:
        model: The model to apply freezing to.
        attribute_name: Name of the model attribute to freeze (e.g., 'vision_tower').
        name_patterns: List of patterns to match in module names.
    """
    # Freeze by attribute name
    if hasattr(model, attribute_name):
        for param in getattr(model, attribute_name).parameters():
            param.requires_grad = False

    # Freeze by name patterns
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in name_patterns):
            for param in module.parameters():
                param.requires_grad = False


def apply_parameter_freezing(model, freeze_config):
    """Apply parameter freezing based on configuration.

    Args:
        model: The model to apply freezing to.
        freeze_config: Configuration dict specifying what to freeze.

    freeze_config can contain:
        - freeze_embeddings: bool (default True)
        - freeze_vision_tower: bool (default False)
        - freeze_language_model: bool (default False)
    """
    freeze_embeddings = freeze_config.get("freeze_embeddings", True)
    freeze_vision_tower = freeze_config.get("freeze_vision_tower", True)
    freeze_audio_tower = freeze_config.get("freeze_audio_tower", False)
    freeze_language_model = freeze_config.get("freeze_language_model", False)

    # Freeze embeddings
    if freeze_embeddings:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False

    # Freeze vision tower
    if freeze_vision_tower:
        _freeze_module_by_attribute_and_patterns(model, "vision_tower", ["vision", "visual", "image_encoder"])

    # Freeze audio tower
    if freeze_audio_tower:
        _freeze_module_by_attribute_and_patterns(model, "audio_tower", ["audio", "audio_encoder"])

    # Freeze language model backbone
    if freeze_language_model:
        _freeze_module_by_attribute_and_patterns(model, "language_model", ["language", "text", "llm"])
