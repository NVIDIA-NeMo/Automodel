# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""Independent Hyper-Connection (iHC) layers for HY V4.

HY V4 keeps ``hc_mult`` independent residual streams.  Before attention and
the MLP, a learned FP32 mixer reduces those streams to one hidden state and
produces post gates.  The sublayer result is then scattered back onto the
unchanged residual streams.  A separate learned head performs the final
reduction before the decoder output norm.

The parameter names in this module intentionally match the Hugging Face
checkpoint (for example ``hc_pre.hc_fn`` rather than a nested Linear weight).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.hy_v4.config import HyV4Config

__all__ = ["HyV4HCHead", "HyV4HCLayer", "HyV4HCPost", "HyV4HCPre"]


def _rms_rsqrt_fp32(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return the FP32 inverse RMS used by the vLLM HY4 reference.

    Args:
        x: Activations shaped ``[..., features]``.
        eps: Scalar stabilizer added to the mean square.

    Returns:
        FP32 inverse RMS shaped ``[..., 1]`` without aliasing ``x``.
    """
    x = x.float()
    return torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _ihc_reduce_fp32(hidden_streams: torch.Tensor, pre_gates: torch.Tensor) -> torch.Tensor:
    """Reduce iHC streams with the vLLM FP32 multiply-and-sum order."""
    return (pre_gates.unsqueeze(-1) * hidden_streams.float()).sum(dim=-2).to(hidden_streams.dtype)


def _ihc_post_fp32(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    post_gates: torch.Tensor,
) -> torch.Tensor:
    """Apply the vLLM iHC post equation with FP32 arithmetic.

    Keeping this as one side-effect-free function lets TorchInductor fuse the
    FP32 casts, multiply, add, and BF16 output store on CUDA.  The eager
    expression materializes several ``[tokens, hc_mult, hidden]`` FP32
    temporaries; for HY4-preview at sequence length 4096 each one is 384 MiB.
    """
    return (post_gates.float().unsqueeze(-1) * sublayer_output.float().unsqueeze(-2) + residual.float()).to(
        sublayer_output.dtype
    )


# vLLM's fused HPC iHC kernels only support SM100/SM103.  Use the existing
# AutoModel TorchInductor pattern as the SM90 training implementation, while
# retaining eager execution on CPU as the numerical oracle.  Compiling the
# reductions prevents their full-size FP32 square and multiply intermediates
# from being materialized at sequence length 4096.
_rms_rsqrt_fp32_compiled = torch.compile(_rms_rsqrt_fp32, dynamic=True)
_ihc_reduce_fp32_compiled = torch.compile(_ihc_reduce_fp32, dynamic=True)
_ihc_post_fp32_compiled = torch.compile(_ihc_post_fp32, dynamic=True)


class HyV4HCPre(nn.Module):
    """Reduce independent residual streams and calculate their post gates."""

    def __init__(self, config: HyV4Config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.magnitude = float(config.hc_magnitude)
        self.hc_eps = float(config.hc_eps)
        self.norm_eps = float(config.rms_norm_eps)

        self.hc_fn = nn.Parameter(torch.empty(2 * self.hc_mult, self.hc_mult * self.hidden_size, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(2 * self.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.hc_fn, mean=0.0, std=init_std)
        nn.init.constant_(self.hc_scale, 0.01)
        initial_pre_bias = -math.log(self.hc_mult - 1.0) if self.hc_mult > 1 else 0.0
        self.hc_base[: self.hc_mult].fill_(initial_pre_bias)
        self.hc_base[self.hc_mult :].zero_()

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reduced hidden states and post gates.

        Args:
            hidden_streams: Independent streams shaped
                ``[..., hc_mult, hidden_size]``.

        Returns:
            A new activation shaped ``[..., hidden_size]`` and FP32 post gates
            shaped ``[..., hc_mult]``. Neither output aliases the input.
        """
        expected = (self.hc_mult, self.hidden_size)
        if tuple(hidden_streams.shape[-2:]) != expected:
            raise ValueError(
                f"HY V4 iHC expected trailing dimensions {expected}, got {tuple(hidden_streams.shape[-2:])}."
            )

        flat = hidden_streams.flatten(start_dim=-2).float()
        # Keep the reference operation order. Although normalizing the input
        # before the projection is algebraically equivalent, it changes FP32
        # rounding and breaks vLLM logits parity.
        rms_fn = _rms_rsqrt_fp32_compiled if hidden_streams.is_cuda else _rms_rsqrt_fp32
        reduce_fn = _ihc_reduce_fp32_compiled if hidden_streams.is_cuda else _ihc_reduce_fp32
        mixes = F.linear(flat, self.hc_fn.float()) * rms_fn(flat, self.norm_eps)
        pre_raw, post_raw = mixes.split(self.hc_mult, dim=-1)
        pre = torch.sigmoid(pre_raw * self.hc_scale[0].float() + self.hc_base[: self.hc_mult].float()) + self.hc_eps
        post = (
            self.magnitude * torch.sigmoid(post_raw * self.hc_scale[1].float() + self.hc_base[self.hc_mult :].float())
            + self.hc_eps
        )
        return reduce_fn(hidden_streams, pre), post


class HyV4HCPost(nn.Module):
    """Scatter one sublayer output onto the independent residual streams."""

    def forward(
        self,
        sublayer_output: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter one sublayer result over the residual streams.

        Args:
            sublayer_output: New activation shaped ``[..., hidden_size]``.
            residual: Independent residual streams shaped
                ``[..., hc_mult, hidden_size]``.
            post_gates: FP32 stream gates shaped ``[..., hc_mult]``.

        Returns:
            New streams shaped ``[..., hc_mult, hidden_size]`` in the sublayer
            dtype. The result aliases none of the inputs.
        """
        if tuple(residual.shape[:-1]) != tuple(post_gates.shape) or sublayer_output.shape != residual.shape[:-2] + (
            residual.shape[-1],
        ):
            raise ValueError(
                "HY V4 iHC post received incompatible shapes: "
                f"output={tuple(sublayer_output.shape)}, residual={tuple(residual.shape)}, "
                f"post={tuple(post_gates.shape)}."
            )
        post_fn = _ihc_post_fp32_compiled if sublayer_output.is_cuda else _ihc_post_fp32
        return post_fn(sublayer_output, residual, post_gates)


class HyV4HCLayer(nn.Module):
    """One iHC pre/post boundary around attention or an MLP."""

    def __init__(self, config: HyV4Config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.enabled = bool(config.enable_ihc)
        if self.enabled:
            self.hc_pre = HyV4HCPre(config)
            self.hc_post = HyV4HCPost()

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Interpret an activation as independent HY4 residual streams.

        Args:
            hidden_states: Tensor ending in ``hidden_size``,
                ``[hc_mult, hidden_size]``, or ``hc_mult * hidden_size``.

        Returns:
            A view shaped ``[..., hc_mult, hidden_size]``. The result aliases
            ``hidden_states`` (and may be an expanded stride-zero view).
        """
        if not self.enabled:
            return hidden_states
        if hidden_states.shape[-1] == self.hidden_size and (
            hidden_states.dim() < 2 or hidden_states.shape[-2] != self.hc_mult
        ):
            return hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], self.hc_mult, self.hidden_size)
        if tuple(hidden_states.shape[-2:]) == (self.hc_mult, self.hidden_size):
            return hidden_states
        if hidden_states.shape[-1] == self.hc_mult * self.hidden_size:
            return hidden_states.unflatten(-1, (self.hc_mult, self.hidden_size))
        raise ValueError(
            "HY V4 iHC input must end in hidden_size, [hc_mult, hidden_size], "
            f"or hc_mult * hidden_size; got {tuple(hidden_states.shape)}."
        )

    def pre(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reduce residual streams before attention or an MLP.

        Args:
            hidden_states: Activation in any layout accepted by
                :meth:`prepare_input`.

        Returns:
            Reduced activation ``[..., hidden_size]``, optional FP32 gates
            ``[..., hc_mult]``, and residual streams
            ``[..., hc_mult, hidden_size]``. The residual aliases the input;
            enabled-mode reduced activations and gates are newly allocated.
        """
        hidden_states = self.prepare_input(hidden_states)
        if not self.enabled:
            return hidden_states, None, hidden_states
        reduced, post_gates = self.hc_pre(hidden_states)
        return reduced, post_gates, hidden_states

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run the parameter-owning pre stage through its FSDP boundary.

        Args:
            hidden_states: Activation in any layout accepted by
                :meth:`prepare_input`.

        Returns:
            Reduced activation, optional post gates, and the input-aliasing
            independent residual streams described by :meth:`pre`.
        """
        return self.pre(hidden_states)

    def post(
        self,
        sublayer_output: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the post gates and residual update.

        Args:
            sublayer_output: Sublayer activation shaped ``[..., hidden_size]``.
            residual: Residual shaped ``[..., hc_mult, hidden_size]`` when iHC
                is enabled, otherwise ``[..., hidden_size]``.
            post_gates: Gates shaped ``[..., hc_mult]`` in enabled mode.

        Returns:
            A newly allocated tensor with the same stream layout as
            ``residual`` and dtype as ``sublayer_output``.
        """
        if not self.enabled:
            return sublayer_output + residual
        if post_gates is None:
            raise ValueError("HY V4 iHC post gates are required when iHC is enabled.")
        return self.hc_post(sublayer_output, residual, post_gates)

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        if self.enabled:
            self.hc_pre.init_weights(init_std)


class HyV4HCHead(nn.Module):
    """Learned final reduction from ``hc_mult`` streams to one hidden state."""

    def __init__(self, config: HyV4Config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.hc_eps = float(config.hc_eps)
        self.norm_eps = float(config.rms_norm_eps)
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * self.hidden_size, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.hc_head_fn, mean=0.0, std=init_std)
        nn.init.constant_(self.hc_head_scale, 0.01)
        initial_bias = -math.log(self.hc_mult - 1.0) if self.hc_mult > 1 else 0.0
        self.hc_head_base.fill_(initial_bias)

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """Reduce the final independent streams before decoder normalization.

        Args:
            hidden_streams: Final streams shaped
                ``[..., hc_mult, hidden_size]``.

        Returns:
            New activation shaped ``[..., hidden_size]`` in the input dtype.
        """
        expected = (self.hc_mult, self.hidden_size)
        if tuple(hidden_streams.shape[-2:]) != expected:
            raise ValueError(
                f"HY V4 iHC head expected trailing dimensions {expected}, got {tuple(hidden_streams.shape[-2:])}."
            )
        flat = hidden_streams.flatten(start_dim=-2).float()
        rms_fn = _rms_rsqrt_fp32_compiled if hidden_streams.is_cuda else _rms_rsqrt_fp32
        reduce_fn = _ihc_reduce_fp32_compiled if hidden_streams.is_cuda else _ihc_reduce_fp32
        mixes = F.linear(flat, self.hc_head_fn.float()) * rms_fn(flat, self.norm_eps)
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.hc_eps
        return reduce_fn(hidden_streams, pre)
