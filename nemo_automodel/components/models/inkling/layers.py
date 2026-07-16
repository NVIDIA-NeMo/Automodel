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

"""Expert-parallel Mixture-of-Experts layer for the Inkling model.

Only the MoE feed-forward is reimplemented here; every other Inkling submodule
(attention, short convolutions, relative-position logits, norms, vision/audio
towers) is reused verbatim from HuggingFace transformers. The routing math and
shared-expert formulation mirror ``transformers.models.inkling.modeling_inkling``
exactly, while the routed experts run through NeMo AutoModel's
:class:`~nemo_automodel.components.moe.experts.GroupedExperts`, which provides
grouped-GEMM compute and expert-parallel (EP) sharding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def build_inkling_moe_config(text_config, backend: BackendConfig) -> MoEConfig:
    """Build the routed-expert :class:`MoEConfig` from an ``InklingTextConfig``.

    Only the fields consumed by :class:`GroupedExperts` matter here (routing and
    shared experts are handled by :class:`InklingGate` / :class:`InklingSharedExperts`),
    so ``n_shared_experts`` is set to 0 and the gate-specific fields are left at
    neutral defaults.

    Args:
        text_config: HuggingFace ``InklingTextConfig``.
        backend: Backend configuration selecting the expert compute/dispatch kernels.

    Returns:
        MoEConfig: Configuration for the routed grouped experts.
    """
    model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=text_config.route_scale,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=model_dtype,
    )


class InklingGate(nn.Module):
    """Top-k router matching ``transformers`` ``InklingTopkRouter``.

    The gate weight bank additionally holds ``n_shared_experts`` rows whose logits
    participate in the softmax normalization; the resulting shared weights
    (``gammas``) scale the always-on shared experts. Selection uses sigmoid scores
    plus a per-expert correction bias, while the returned weights come from a
    log-sigmoid softmax over the selected routed logits concatenated with the
    shared logits, scaled by ``route_scale * global_scale``.
    """

    def __init__(self, config, gate_precision: torch.dtype | None = torch.float32) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_total_experts = self.num_experts + self.n_shared_experts
        self.hidden_dim = config.hidden_size
        self.route_scale = config.route_scale
        self.top_k = config.num_experts_per_tok
        self.gate_precision = gate_precision

        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.weight = nn.Parameter(torch.empty(self.n_total_experts, config.hidden_size, dtype=model_dtype))
        self.global_scale = nn.Parameter(torch.ones(1, dtype=model_dtype))
        # Stored in the model dtype (uniform dtype keeps FSDP2 happy); selection math
        # upcasts to fp32 via ``gate_precision`` in forward.
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts, dtype=model_dtype))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: Input tokens of shape ``[num_tokens, hidden_dim]``.

        Returns:
            tuple: ``(topk_weights, topk_indices, shared_gammas)`` where
            ``topk_weights`` and ``topk_indices`` have shape ``[num_tokens, top_k]``
            and ``shared_gammas`` has shape ``[num_tokens, n_shared_experts]``.
        """
        compute_dtype = self.gate_precision
        flat = hidden_states.reshape(-1, self.hidden_dim)
        weight = self.weight
        if compute_dtype is not None:
            flat = flat.to(compute_dtype)
            weight = weight.to(compute_dtype)
        router_logits = F.linear(flat, weight)

        scores = router_logits.sigmoid()
        routed_scores = scores[..., : -self.n_shared_experts]
        scores_for_choice = routed_scores + self.e_score_correction_bias.to(routed_scores.dtype)
        topk_indices = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)[1]

        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]
        topk_logits = torch.cat([routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
        topk_log_probs = F.logsigmoid(topk_logits)
        topk_weights = torch.exp(topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1, keepdim=True))

        topk_weights = topk_weights * self.route_scale * self.global_scale.to(topk_weights.dtype)

        shared_gammas = topk_weights[..., -self.n_shared_experts :].contiguous()
        topk_weights = topk_weights[..., : self.top_k].contiguous()
        return topk_weights, topk_indices, shared_gammas

    def update_bias(self) -> None:
        """No-op: Inkling's correction bias is a trained parameter, not an aux-loss update."""
        return


class InklingSharedExperts(nn.Module):
    """Always-on shared experts, matching ``transformers`` ``InklingSharedExperts``.

    Weights are stored as stacked 3D tensors, one slice per shared expert, and the
    per-token ``gammas`` (from :class:`InklingGate`) weight each expert's output
    before summation.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n_shared_experts = config.n_shared_experts
        intermediate_dim = config.moe_intermediate_size
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.gate_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, intermediate_dim, config.hidden_size, dtype=model_dtype)
        )
        self.up_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, intermediate_dim, config.hidden_size, dtype=model_dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, config.hidden_size, intermediate_dim, dtype=model_dtype)
        )
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """Apply the shared experts.

        Args:
            hidden_states: Input tokens of shape ``[num_tokens, hidden_dim]``.
            gammas: Per-token shared weights of shape ``[num_tokens, n_shared_experts]``.

        Returns:
            torch.Tensor: Output of shape ``[num_tokens, hidden_dim]``.
        """
        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(1, -1, input_shape[-1]).expand(self.n_shared_experts, -1, -1)
        gammas = gammas.reshape(-1, self.n_shared_experts, 1).transpose(0, 1)

        gate = torch.bmm(hidden_states, self.gate_proj.transpose(1, 2))
        up = torch.bmm(hidden_states, self.up_proj.transpose(1, 2))
        activated = self.act_fn(gate) * up * gammas.to(gate.dtype)
        down = torch.bmm(activated, self.down_proj.transpose(1, 2))

        out = down.float().sum(dim=0).to(hidden_states.dtype)
        return out.view(input_shape)


class InklingMoE(MoE):
    """Inkling MoE feed-forward: custom router + shared experts + grouped routed experts.

    Subclasses :class:`~nemo_automodel.components.moe.layers.MoE` so the distributed
    parallelizer (which discovers MoE blocks via ``isinstance`` and shards
    ``self.experts`` across the expert-parallel mesh) treats it uniformly, but the
    generic ``__init__`` / ``forward`` are replaced to match Inkling's routing.
    """

    def __init__(self, config, backend: BackendConfig, moe_config: MoEConfig | None = None) -> None:
        # Intentionally bypass MoE.__init__ (custom gate + shared experts); the
        # attribute contract (.gate, .experts, .shared_experts, .backend, .dim,
        # .cp_mesh) that the parallelizer and FSDP mixin rely on is preserved.
        nn.Module.__init__(self)
        from nemo_automodel.components.moe.experts import GroupedExperts

        self.backend = backend
        self.moe_config = moe_config or build_inkling_moe_config(config, backend)
        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok

        gate_precision = backend.gate_precision if backend.gate_precision is not None else torch.float32
        self.gate = InklingGate(config, gate_precision=gate_precision)
        self.experts = GroupedExperts(self.moe_config, backend=backend)
        self.shared_experts = InklingSharedExperts(config)

        # Set during context-parallel model parallelization; unused by Inkling for now.
        self.cp_mesh = None
        # Latent projections are not used by Inkling.
        self.fc1_latent_proj = None
        self.fc2_latent_proj = None
        self.shared_expert_gate = None

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Route tokens through the shared and routed experts and recombine.

        Args:
            hidden_states: Input of shape ``[..., hidden_dim]`` (the trailing
                dimension is the model dim; leading dims are flattened to tokens).
            padding_mask: Optional boolean mask of padding positions; padded tokens
                are excluded from routed-expert dispatch.

        Returns:
            torch.Tensor: Output with the same shape as ``hidden_states``.
        """
        shape = hidden_states.size()
        x = hidden_states.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        topk_weights, topk_indices, shared_gammas = self.gate(x)
        routed = self.experts(x, token_mask, topk_weights, topk_indices)
        shared = self.shared_experts(x, shared_gammas)
        return (routed + shared).view(shape)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize MoE parameters in-place (used for from-scratch training)."""
        with buffer_device:
            for p in (self.gate.weight, self.experts.gate_and_up_projs, self.experts.down_projs):
                nn.init.normal_(p, mean=0.0, std=init_std)
            for p in (self.shared_experts.gate_proj, self.shared_experts.up_proj, self.shared_experts.down_proj):
                nn.init.normal_(p, mean=0.0, std=init_std)
            nn.init.ones_(self.gate.global_scale)
            nn.init.zeros_(self.gate.e_score_correction_bias)
