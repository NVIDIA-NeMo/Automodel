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

from nemo_automodel.utils.dist_utils import get_rank_safe


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
        print(
            f"Trainable parameters percentage: {100 * trainable_params / all_param:.2f}%"
        )
        print("--------------------------------")

    return trainable_params, all_param

def _freeze_with_type(module, module_type):
    """
    Sets requires_grad=None for modules matching the module_type.

    Args:
        module (nn.Module): the module to freeze.
        module_type (nn.Module): the module to match during freeze.
    """
    for m in module.modules():
        if not isinstance(m, module_type):
            continue
        m.requires_grad = False

def _freeze_with_pattern(module, pattern=None):
    """
    Sets requires_grad=None for modules matching the pattern or all modules if pattern is None.

    Args:
        module (nn.Module): The module to freeze.
        pattern (list[str], optional): The pattern of attribute names to match. Defaults to None.

    Returns:
        None: the change happens in-place in input module.
    """
    def matches(name, pattern):
        """
        Checks if name matches any pattern.

        Args:
            name (str): the module name.
            pattern (list[str]): the list of allowed names.

        Returns:
            bool: whether the name matches any pattern
        """
        name = name.lower()
        return any(
            pattern in name
            for pattern in ["vision", "visual", "image_encoder"]
        )

    for name, m in module.named_modules():
        if pattern and not matches(name, pattern):
            continue
        m.requires_grad = False

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
    freeze_language_model = freeze_config.get("freeze_language_model", False)

    # Freeze embeddings
    if freeze_embeddings:
        _freeze_with_type(model, nn.Embedding)

    # Freeze vision tower
    if freeze_vision_tower:
        if hasattr(model, "vision_tower"):
            _freeze_with_pattern(model.vision_tower, None)
        # Alternative patterns for different VLM architectures
        _freeze_with_pattern(model, ["vision", "visual", "image_encoder"])

    # Freeze language model backbone
    if freeze_language_model:
        if hasattr(model, "language_model"):
            _freeze_with_pattern(model.language_model, None)
        # Alternative patterns
        _freeze_with_pattern(model, ["language", "text", "llm"])

