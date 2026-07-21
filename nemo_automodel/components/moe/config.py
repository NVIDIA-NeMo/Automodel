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

"""MoE model configuration."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from nemo_automodel.shared.utils import dtype_from_str


@dataclass(kw_only=True)
class MoEConfig:
    """Configuration for routed and shared MoE expert modules."""

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
    expert_activation: Literal["swiglu", "swigluoai", "quick_geglu", "geglu", "relu2"] = "swiglu"
    activation_alpha: float = 1.702
    activation_limit: float = 7.0
    # When > 0, ``expert_activation="swiglu"`` dispatches to a clamped FP32
    # variant (gate clamped at max=limit, up clamped at +/-limit) matching
    # DeepSeek V4's official ``Expert.forward`` with ``swiglu_limit``.
    # Default 0.0 preserves the existing ``weighted_bias_swiglu_impl`` path.
    swiglu_limit: float = 0.0
    softmax_before_topk: bool = False
    router_topk_sorted: bool = True
    router_weights_fp32: bool = False
    router_weight_uses_score_correction_bias: bool = False
    route_weight_after_down_proj: bool = False
    dtype: str | torch.dtype = torch.bfloat16
    shared_expert_gate: bool = False
    shared_expert_inter_dim: int | None = None
    shared_expert_activation: str = "swiglu"  # Activation for shared experts ("swiglu" or "relu2")
    force_e_score_correction_bias: bool = False  # Force creation of e_score_correction_bias buffer
    moe_latent_size: int | None = None
    # Rollout Routing Replay (R3): when True, each gate records/replays its top-k
    # expert selection so RL training reuses the rollout's routing decisions. See
    # nemo_automodel.components.moe.router_replay.
    enable_routing_replay: bool = False

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
            Set to 0 to disable per-expert utilization logging entirely.
    """

    enabled: bool = False
    mode: str = "brief"
    detailed_every_steps: Optional[int] = None
    top_k_experts: int = 0
