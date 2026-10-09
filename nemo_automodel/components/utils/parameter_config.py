# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Explicit parameter selection and storage precision before model sharding."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterConfig:
    """Select an exact named update surface; freeze all other parameters.

    Names refer to parameters before activation checkpoint wrapping. Storage
    precision is applied before checkpoint loading so native FP32 source
    tensors never make an intermediate roundtrip through BF16.
    """

    trainable_names: tuple[str, ...]
    trainable_dtype: torch.dtype = torch.float32


def apply_parameter_config(model: nn.Module, config: ParameterConfig) -> None:
    """Select and cast parameter storage before loading or distributed wrapping.

    Args:
        model: Module whose parameters have arbitrary shapes; parameters may be
            on meta or a real device, and must not already be distributed.
        config: Exact local parameter names and selected storage dtype.

    Returns:
        None. Mutates requires_grad and selected storage in place; frozen
        parameter values, storage dtypes and buffer values remain unchanged.
    """
    if len(set(config.trainable_names)) != len(config.trainable_names):
        raise ValueError("ParameterConfig trainable_names must be unique")
    if config.trainable_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Trainable parameter storage must be floating point")
    parameters = dict(model.named_parameters())
    missing = set(config.trainable_names) - parameters.keys()
    if missing:
        raise ValueError(f"Unknown trainable parameters: {sorted(missing)}")
    if not config.trainable_names:
        raise ValueError("ParameterConfig must select at least one trainable parameter")
    selected = set(config.trainable_names)
    for name, parameter in parameters.items():
        trainable = name in selected
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.data = parameter.data.to(config.trainable_dtype)
