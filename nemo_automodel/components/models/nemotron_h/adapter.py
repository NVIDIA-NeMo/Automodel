"""
Adapter to make HuggingFace NemotronHForCausalLM compatible with
the nemo_automodel MoE parallelizer (Expert Parallelism).

The HF model uses nn.ModuleList[NemotronHMLP] for experts (individual
Linear modules per expert). This adapter converts those into GroupedExperts
with stacked weight tensors of shape [n_experts, ...], wrapped in the
framework's MoE class so that apply_ep / apply_fsdp work.

Usage:
    model = load_and_adapt_nemotron_h("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
"""

import logging

import torch
import torch.nn as nn

from nemo_automodel.components.moe.layers import (
    GroupedExperts,
    MoE,
    MoEConfig,
)
from nemo_automodel.components.moe.utils import BackendConfig

logger = logging.getLogger(__name__)


def _build_moe_config(hf_config) -> MoEConfig:
    """Build framework MoEConfig from HuggingFace NemotronH config."""
    return MoEConfig(
        n_routed_experts=hf_config.n_routed_experts,
        n_shared_experts=1,
        n_activated_experts=hf_config.num_experts_per_tok,
        n_expert_groups=getattr(hf_config, "n_group", 1),
        n_limited_groups=getattr(hf_config, "topk_group", 1),
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=getattr(hf_config, "routed_scaling_factor", 1.0),
        dim=hf_config.hidden_size,
        inter_dim=hf_config.intermediate_size,
        moe_inter_dim=hf_config.moe_intermediate_size,
        norm_topk_prob=getattr(hf_config, "norm_topk_prob", True),
        router_bias=False,
        expert_bias=getattr(hf_config, "mlp_bias", False),
        expert_activation="relu2",
        shared_expert_gate=False,
        shared_expert_inter_dim=getattr(hf_config, "moe_shared_expert_intermediate_size", None),
    )


def _convert_experts_to_grouped(hf_experts: nn.ModuleList, moe_config: MoEConfig) -> GroupedExperts:
    """
    Convert nn.ModuleList[NemotronHMLP] -> GroupedExperts with stacked weights.

    HF: each expert has up_proj.weight [moe_inter_dim, dim] and down_proj.weight [dim, moe_inter_dim]
    Grouped: gate_and_up_projs [n_experts, dim, moe_inter_dim], down_projs [n_experts, moe_inter_dim, dim]
    """
    experts_list = list(hf_experts)

    # Stack and transpose: [moe_inter_dim, dim] -> .t() -> [dim, moe_inter_dim] -> stack -> [n_experts, dim, moe_inter_dim]
    up_weights = torch.stack([e.up_proj.weight.data.t() for e in experts_list], dim=0)
    down_weights = torch.stack([e.down_proj.weight.data.t() for e in experts_list], dim=0)

    grouped = GroupedExperts(moe_config)
    grouped.gate_and_up_projs = nn.Parameter(up_weights)
    grouped.down_projs = nn.Parameter(down_weights)

    # Handle biases if present
    if moe_config.expert_bias:
        up_biases = torch.stack([e.up_proj.bias.data for e in experts_list], dim=0)
        down_biases = torch.stack([e.down_proj.bias.data for e in experts_list], dim=0)
        grouped.gate_up_proj_bias = nn.Parameter(up_biases)
        grouped.down_proj_bias = nn.Parameter(down_biases)

    return grouped


class NemotronGateWrapper(nn.Module):
    """
    Wraps HF NemotronHTopkRouter to match the framework Gate.forward() interface.

    HF router: forward(hidden_states) -> (topk_weights, topk_indices)
    Framework Gate: forward(x, token_mask, cp_mesh) -> (weights, indices, aux_loss)
    """

    def __init__(self, hf_router):
        super().__init__()
        self.router = hf_router
        # Expose weight for compatibility with code that accesses gate.weight
        self.weight = hf_router.weight
        self.e_score_correction_bias = getattr(hf_router, "e_score_correction_bias", None)

    def forward(self, x, token_mask, cp_mesh=None):
        topk_weights, topk_indices = self.router(x)
        return topk_weights, topk_indices, None  # no aux_loss


class NemotronMoEWrapper(MoE):
    """
    Wrapper that replaces NemotronHMOE with a framework-compatible MoE.

    Inherits from MoE so isinstance(block.mixer, MoE) checks work
    in the parallelizer. Bypasses MoE.__init__ and sets up components
    directly from the converted HF modules.
    """

    def __init__(self, hf_moe_module, moe_config: MoEConfig, backend: BackendConfig | None = None):
        # Bypass MoE.__init__ to avoid re-creating submodules
        nn.Module.__init__(self)
        self.backend = backend or BackendConfig(linear="torch")
        self.dim = moe_config.dim
        self.n_routed_experts = moe_config.n_routed_experts
        self.n_activated_experts = moe_config.n_activated_experts

        # Convert HF experts -> GroupedExperts with stacked weights
        self.experts = _convert_experts_to_grouped(hf_moe_module.experts, moe_config)

        # Wrap HF gate to match framework interface
        self.gate = NemotronGateWrapper(hf_moe_module.gate)

        # Keep shared experts as-is (NemotronHMLP, called as shared_experts(x))
        self.shared_experts = hf_moe_module.shared_experts
        self.shared_expert_gate = None


def adapt_nemotron_h_for_ep(model):
    """
    In-place adaptation of a HuggingFace NemotronHForCausalLM model
    to make MoE layers compatible with the framework's EP parallelizer.

    After this call:
      - model.backbone.layers still contains all blocks
      - MoE blocks have block.mixer replaced with NemotronMoEWrapper (isinstance MoE)
      - model.backbone.moe_config is set for the parallelizer
      - Non-MoE blocks (Mamba/Attention/MLP) are untouched
    """
    hf_config = model.config
    moe_config = _build_moe_config(hf_config)

    n_adapted = 0
    for i, block in enumerate(model.backbone.layers):
        if getattr(block, "block_type", None) == "moe":
            wrapper = NemotronMoEWrapper(block.mixer, moe_config)
            # Cast gate router params from float32 -> bfloat16 for FSDP uniformity.
            # The HF router already casts to float32 at forward time, so this is safe.
            wrapper.gate.to(torch.bfloat16)
            block.mixer = wrapper
            n_adapted += 1

    # Attach moe_config so the parallelizer can find it
    model.backbone.moe_config = moe_config

    logger.info(f"Adapted {n_adapted} MoE blocks for expert parallelism")
    return model


def load_and_adapt_nemotron_h(pretrained_model_name_or_path, **kwargs):
    """
    Hydra _target_ entry point: load HF model and adapt MoE layers for EP.

    Args:
        pretrained_model_name_or_path: HuggingFace model ID or path
        **kwargs: Passed to NeMoAutoModelForCausalLM.from_pretrained
    """
    from nemo_automodel import NeMoAutoModelForCausalLM

    model = NeMoAutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
    adapt_nemotron_h_for_ep(model)

    # Ensure all parameters have uniform dtype for FSDP2 compatibility.
    # Some HF modules (e.g. RMSNorm, Conv1d) may remain float32 even when
    # torch_dtype=bfloat16 is passed to from_pretrained.
    model.to(torch.bfloat16)

    return model
