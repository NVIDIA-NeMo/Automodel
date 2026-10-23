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

"""PEFT configuration without adapter or distributed runtime dependencies."""

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import torch


class _PeftConfigDict(TypedDict, total=False):
    target_modules: list[str] | str | None
    exclude_modules: list[str] | str | None
    match_all_linear: bool
    dim: int
    alpha: int
    use_dora: bool
    dropout: float
    dropout_position: Literal["pre", "post"]
    lora_A_init: str
    lora_dtype: torch.dtype | str | None
    use_memory_efficient_lora: bool
    use_triton: bool
    moe_rank_scaling: bool
    expert_weight_format: Literal["unquantized", "mxfp4"]


@dataclass
class PeftConfig:
    target_modules: list[str] | str | None = field(default_factory=list)
    exclude_modules: list[str] | str | None = field(default_factory=list)
    match_all_linear: bool = False
    dim: int = 8
    alpha: int = 32
    # Note: we currently support DoRA for nn.Linear only.
    use_dora: bool = False
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "post"
    lora_A_init: str = "xavier"
    lora_dtype: torch.dtype | str | None = None
    use_memory_efficient_lora: bool = True
    use_triton: bool = False
    moe_rank_scaling: bool = False
    # Storage mode, not a torch dtype: "unquantized" uses the ordinary floating-point
    # experts with the model's configured precision (including FP32). "mxfp4"
    # stores frozen expert bases as fp4-e2m1 + e8m0 scales and requires torch_mm.
    expert_weight_format: Literal["unquantized", "mxfp4"] = "unquantized"

    def to_dict(self) -> _PeftConfigDict:
        return {
            "target_modules": self.target_modules,
            "exclude_modules": self.exclude_modules,
            "match_all_linear": self.match_all_linear,
            "dim": self.dim,
            "alpha": self.alpha,
            "use_dora": self.use_dora,
            "dropout": self.dropout,
            "dropout_position": self.dropout_position,
            "lora_A_init": self.lora_A_init,
            "lora_dtype": self.lora_dtype,
            "use_memory_efficient_lora": self.use_memory_efficient_lora,
            "use_triton": self.use_triton,
            "moe_rank_scaling": self.moe_rank_scaling,
            "expert_weight_format": self.expert_weight_format,
        }
