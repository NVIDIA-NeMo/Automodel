# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
LoRA components for diffusion model fine-tuning.

Pure PEFT approach — no diffusers pipeline class dependency.
Identical save/load/train flow for Flux, Wan2.1, and HunyuanVideo15.
"""

from .checkpoint import load_lora_weights_for_inference, save_lora_weights
from .config import (
    FLUX_DEFAULT_TARGET_MODULES,
    HUNYUAN_DEFAULT_TARGET_MODULES,
    MODEL_DEFAULT_TARGET_MODULES,
    WAN_DEFAULT_TARGET_MODULES,
    LoRAConfig,
)
from .setup import inject_lora, register_pre_inject_hook

__all__ = [
    # Config
    "LoRAConfig",
    "MODEL_DEFAULT_TARGET_MODULES",
    "FLUX_DEFAULT_TARGET_MODULES",
    "WAN_DEFAULT_TARGET_MODULES",
    "HUNYUAN_DEFAULT_TARGET_MODULES",
    # Setup
    "inject_lora",
    "register_pre_inject_hook",
    # Checkpoint
    "save_lora_weights",
    "load_lora_weights_for_inference",
]