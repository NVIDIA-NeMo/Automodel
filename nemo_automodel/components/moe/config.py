from dataclasses import dataclass
from typing import Literal

import torch

from nemo_automodel.shared.utils import dtype_from_str


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

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = dtype_from_str(self.dtype, default=torch.bfloat16)
