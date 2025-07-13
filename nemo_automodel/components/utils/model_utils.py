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

from nemo_automodel.components.utils.dist_utils import get_rank_safe

logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if get_rank_safe() == 0:
        print("--------------------------------")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_param:,}")
        print(f"Trainable parameters percentage: {100 * trainable_params / all_param:.2f}%")
        print("--------------------------------")

    return trainable_params, all_param


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
