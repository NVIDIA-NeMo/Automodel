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

"""MoE parallelizer configuration."""

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class MoEParallelizerConfig:
    """
    Configuration for MoE (Mixture of Experts) model parallelization.

    This config controls how MoE models are parallelized with expert parallelism,
    activation checkpointing, and FSDP settings.

    Attributes:
        activation_checkpointing (bool): Enable activation checkpointing for
            transformer blocks to reduce memory usage. Defaults to False.
        ignore_router_for_ac (bool): If True, uses selective checkpointing that
            saves router outputs during activation checkpointing. This can improve
            training stability for MoE models. Defaults to False.
        reshard_after_forward (bool): If True, reshard parameters after forward
            pass in FSDP. Can reduce memory but may increase communication.
            Defaults to False.
        lm_head_precision (Optional[Union[str, torch.dtype]]): Precision for the
            language model head. If "float32" or torch.float32, uses full precision
            for the lm_head to improve training stability. Defaults to None (uses
            default mixed precision policy).
        wrap_outer_model (bool): If True, wraps the outer model with FSDP when
            the model has nested structure (e.g., CausalLM wrapping base model).
            Defaults to True.
    """

    activation_checkpointing: bool = False
    ignore_router_for_ac: bool = False
    reshard_after_forward: bool = False
    lm_head_precision: Optional[Union[str, torch.dtype]] = None
    wrap_outer_model: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
