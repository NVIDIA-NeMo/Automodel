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

"""Differentiable Inkling2 primitives and the common MoE integration."""

from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.layers import MoE

from .config import Inkling2TextConfig
from .context_parallel import convolution_history


@dataclass(frozen=True)
class ResidualState:
    """Immutable depth state; every tensor has shape [batch, sequence, hidden].

    Attributes:
        blocks: Completed depth blocks, including the initial embedding block.
        partial: Current incomplete block. Neither field is detached or mutated.
    """

    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor


def residual_mix(state: ResidualState, queries: torch.Tensor, *, normalize_values: bool) -> torch.Tensor:
    """Mix depth sources independently at each token in FP32.

    Args:
        state: Completed blocks and partial, each [batch, sequence, hidden].
        queries: Tensor of shape [queries, hidden], with one or three pseudo-queries.
        normalize_values: Normalize values as well as scoring keys.

    Returns:
        Tensor of shape [queries, batch, sequence, hidden], without mutating state.
    """
    values = torch.stack((*state.blocks, state.partial)).float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    scores = torch.einsum("nbtd,qd->qbtn", normalized, queries.float()) * queries.shape[-1] ** -0.5
    weights = scores.softmax(-1)
    return torch.einsum("qbtn,nbtd->qbtd", weights, normalized if normalize_values else values)


class ResidualQuery(nn.Module):
    """Callable FP32 owner for one set of depth-mixture queries."""

    def __init__(self, hidden_size: int, count: int, *, normalize_values: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(count, hidden_size, dtype=torch.float32))
        self.normalize_values = normalize_values

    def forward(self, state: ResidualState) -> torch.Tensor:
        """Mix immutable depth state.

        Args:
            state: Completed and partial blocks, each [batch, sequence, hidden].

        Returns:
            FP32 tensor of shape [queries, batch, sequence, hidden].
        """
        return residual_mix(state, self.weight, normalize_values=self.normalize_values)


def capped_swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    *,
    gate_softcap: float,
    up_softcap: float,
) -> torch.Tensor:
    """Apply the interleaved capped activation, with routing before the down GEMM.

    Args:
        gate_up: Tensor of shape [tokens, 2 * intermediate], interleaved gate/up.
        weights: Tensor of shape [tokens, 1], containing selected expert weights.
        gate_softcap: Positive gate tanh cap; sigmoid still receives the raw gate.
        up_softcap: Positive up tanh cap.

    Returns:
        Tensor of shape [tokens, intermediate], in gate_up's dtype.
    """
    gate = gate_up[..., 0::2].float()
    up = gate_up[..., 1::2].float()
    result = gate_softcap * torch.tanh(gate / gate_softcap) * torch.sigmoid(gate)
    result = result * (up_softcap * torch.tanh(up / up_softcap)) * weights.float()
    return result.to(gate_up.dtype)


class Inkling2MoE(MoE):
    """Common Gate/GroupedExperts with Inkling2's activation callable."""

    def __init__(self, config: Inkling2TextConfig, backend: BackendConfig, *, dtype: torch.dtype) -> None:
        moe_config = MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=0,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=config.route_scale,
            norm_topk_prob=True,
            router_weights_fp32=True,
            gate_dtype=torch.float32,
            dtype=dtype,
            force_e_score_correction_bias=config.use_gate_bias,
            apply_router_weight_after_down=False,
        )
        super().__init__(moe_config, backend)
        self.moe_config = moe_config
        activation = partial(
            capped_swiglu,
            gate_softcap=config.mlp_gate_softcap,
            up_softcap=config.mlp_up_softcap,
        )
        # The shared factory uses different activation hooks for HybridEP and
        # ordinary grouped experts (including the world-size-one fallback).
        if isinstance(self.experts, GroupedExpertsDeepEP):
            self.experts.expert_activation = activation
        else:
            cast(GroupedExperts, self.experts).expert_activation_grouped = activation


class Inkling2LMHead(nn.Linear):
    """Bias-free output projection with FP32 arithmetic and ordinary FSDP hooks."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project through the module boundary so FSDP gathers its weight.

        Args:
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            FP32 tensor of shape [..., vocab]. Weight storage may be FP32 or BF16.
            The caller applies softcapping and unused-vocabulary masking.
        """
        with torch.autocast(device_type=input.device.type, enabled=False):
            return F.linear(input.float(), self.weight.float(), self.bias.float() if self.bias is not None else None)


class ShortConvolution(nn.Module):
    """Causal width-W convolution with static or token-generated per-head taps."""

    def __init__(
        self,
        channels: int,
        width: int,
        *,
        dtype: torch.dtype,
        dynamic_input_size: int = 0,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.channels, self.width, self.heads = channels, width, heads
        self.cp_mesh: DeviceMesh | None = None
        if channels % heads:
            raise ValueError("Short convolution heads must divide channels")
        if dynamic_input_size:
            self.weight = None
            self.static_w = nn.Parameter(torch.zeros(width, channels, dtype=dtype))
            self.weight_proj = nn.Linear(dynamic_input_size, heads * width, bias=False, dtype=dtype)
        else:
            # Static taps follow conv1d order: oldest first, current position last.
            self.weight = nn.Parameter(torch.zeros(channels, 1, width, dtype=dtype))
            self.static_w = None
            self.weight_proj = None

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize every static and dynamic coefficient without checkpoint data."""
        if self.weight_proj is None:
            assert self.weight is not None
            nn.init.normal_(self.weight, std=init_std)
        else:
            assert self.static_w is not None
            nn.init.normal_(self.static_w, std=init_std)
            nn.init.normal_(self.weight_proj.weight, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        *,
        dynamic_input: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply convolution with exchanged CP history, zero document history and masked padding.

        Args:
            x: Tensor of shape [batch, sequence, channels].
            valid: Boolean tensor of shape [batch, sequence], true for real tokens.
            dynamic_input: Tensor of shape [batch, sequence, hidden] for dynamic taps.
            positions: Logical document positions [batch, sequence], including CP offsets.
                Taps cannot read before the current document start.

        Returns:
            Tensor of shape [batch, sequence, channels], in x's dtype; padding is zero.
        """
        masked = x * valid.unsqueeze(-1)
        history = convolution_history(masked.float(), self.cp_mesh, self.width)
        if self.weight_proj is None:
            assert self.weight is not None
            if dynamic_input is not None:
                raise ValueError("Static convolution does not accept dynamic_input")
            if positions is None:
                filtered = F.conv1d(
                    history.transpose(1, 2),
                    self.weight.float(),
                    groups=self.channels,
                ).transpose(1, 2)
            else:
                # One grouped convolution keeps accumulation and its backward in
                # FP32 while suppressing taps before each document's first token.
                windows = history.unfold(1, self.width, 1)
                lags = torch.arange(self.width - 1, -1, -1, device=x.device)
                windows = windows * (positions[..., None] >= lags)[:, :, None, :]
                filtered = F.conv1d(windows.flatten(0, 1), self.weight.float(), groups=self.channels)
                filtered = filtered.reshape_as(masked)
            scale = (self.width + 1) ** -0.5
        else:
            assert self.static_w is not None
            if dynamic_input is None:
                raise ValueError("Dynamic convolution requires dynamic_input")
            batch, sequence, _ = x.shape
            # Dynamic taps are lag-indexed: current position first.
            generated = self.weight_proj(dynamic_input).float().view(batch, sequence, self.heads, self.width)
            filtered = torch.zeros_like(masked, dtype=torch.float32)
            for lag in range(self.width):
                lagged = history[:, self.width - 1 - lag : self.width - 1 - lag + sequence]
                if positions is not None:
                    lagged = lagged * (positions >= lag).unsqueeze(-1)
                tap = self.static_w[lag].float().view(self.heads, -1)
                weighted = lagged.view(batch, sequence, self.heads, -1) * (tap + generated[..., lag, None])
                filtered = filtered + weighted.reshape_as(filtered)
            scale = (2 * self.width + 1) ** -0.5
        return ((masked.float() + filtered) * scale).to(x.dtype) * valid.unsqueeze(-1)


class L2Clip(nn.Module):
    """Per-head L2 clipping, optionally preceded by learned per-channel scales."""

    def __init__(self, heads: int, width: int, threshold: float, *, learned_scale: bool) -> None:
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(heads, width, dtype=torch.float32)) if learned_scale else None

    def forward(self, x: torch.Tensor, *, tau: torch.Tensor | None = None) -> torch.Tensor:
        """Clip head vectors, then apply optional length scaling.

        Args:
            x: Tensor of shape [batch, sequence, heads, head_dim].
            tau: Optional FP32 tensor of shape [batch, sequence], applied after clipping.

        Returns:
            Tensor of shape [batch, sequence, heads, head_dim], in x's dtype.
        """
        value = x.float() if self.scale is None else x.float() * self.scale.float()
        factor = (self.threshold / (torch.linalg.vector_norm(value, dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
        result = (value * factor).to(x.dtype)
        if tau is not None:
            result = (result.float() * tau[..., None, None]).to(x.dtype)
        return result
