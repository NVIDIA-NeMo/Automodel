# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from typing import Any, Dict, Literal, Optional, Union

import torch
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy

from nemo_automodel.shared.utils import dtype_from_str


@dataclass
class MoEParallelizerConfig:
    """Configuration for MoE model parallelization (EP + FSDP settings)."""

    ignore_router_for_ac: bool = False
    reshard_after_forward: bool = False
    lm_head_precision: Optional[Union[str, torch.dtype]] = None
    wrap_outer_model: bool = True
    mp_policy: Optional[MixedPrecisionPolicy] = None

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(kw_only=True)
class MoEConfig:
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    n_expert_groups: int
    n_limited_groups: int
    train_gate: bool
    gate_bias_update_factor: float
    aux_loss_coeff: float
    score_func: str
    route_scale: float
    dim: int
    inter_dim: int
    moe_inter_dim: int
    norm_topk_prob: bool
    router_bias: bool = False
    expert_bias: bool = False
    expert_activation: Literal["swiglu", "quick_geglu", "relu2"] = "swiglu"
    activation_alpha: float = 1.702
    activation_limit: float = 7.0
    softmax_before_topk: bool = False
    dtype: str | torch.dtype = torch.bfloat16
    shared_expert_gate: bool = False
    shared_expert_inter_dim: int | None = None
    shared_expert_activation: str = "swiglu"  # Activation for shared experts ("swiglu" or "relu2")
    force_e_score_correction_bias: bool = False  # Force creation of e_score_correction_bias buffer
    moe_latent_size: int | None = None

    @property
    def expert_dim(self) -> int:
        """Dimension used for expert projections (latent size when set, otherwise model dim)."""
        return self.moe_latent_size if self.moe_latent_size is not None else self.dim

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = dtype_from_str(self.dtype, default=torch.bfloat16)


@dataclass
class MoEMetricsConfig:
    """Configuration for MoE load balance metrics logging.

    Attributes:
        enabled: Whether to enable load balance metric tracking.
        mode: Logging mode - "brief" for scalar line charts only,
            "detailed" adds per-layer breakdowns.
        detailed_every_steps: How often to log detailed metrics (only used when mode="detailed").
            None means every step.
        top_k_experts: Number of top (highest) and bottom (lowest) utilization experts
            to emit per layer. Reduces wandb key count for models with many experts.
    """

    enabled: bool = False
    mode: str = "brief"
    detailed_every_steps: Optional[int] = None
    top_k_experts: int = 5
