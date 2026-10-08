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

"""Single-pass mHC and modality-aware routing for DeepSeek V4.1.

The coefficient handoff follows section 2.4.1 of the official technical report:
each sublayer consumes the pre-mix produced by the preceding sublayer.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sinkhorn_normalize
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig


# ----------------------------------------------------------------------------------------- fp32 cores
# The mHC coefficient projection, the stream collapse/expand and the fp32 RMSNorm each run several full fp32 passes
# in eager mode, and ``expand`` materialises a ``[batch, sequence, streams, streams, hidden]`` fp32 temporary before
# its stream sum. The arithmetic lives in the plain functions below so ``torch.compile`` can fuse each into one or
# two kernels: ``BackendConfig.compile_hc`` wraps the three mHC cores, ``BackendConfig.compile_norm``
# the RMSNorm core (same lazy once-per-process pattern as Kimi K3's ``compile_norm``). The modules call the
# module-level dispatchers, so instances built before the hook ran still pick up the compiled core. Compiled
# numerics are allclose to eager, not bitwise-identical. The mHC cores are compiled for static shapes only: with
# dynamic shapes inductor keeps the broadcast temporary of ``expand`` and the fusion is lost, so every new sequence
# length compiles them again. The RMSNorm core is compiled per shape as well: the dynamic-shape variant lost 15 % per
# call and shifted the mini benchmark's first-step loss by 9e-4, the static kernels reproduce the eager loss.
def _rms_norm_core(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = hidden_states.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * value).to(hidden_states.dtype)


def _hc_project_core(
    hidden_states: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    streams: int,
    norm_eps: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coefficient projection up to the Sinkhorn input: pre, post and the [streams, streams] combination logits."""
    flat = hidden_states.flatten(2).float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    scales = torch.cat((scale[0].expand(streams), scale[1].expand(streams), scale[2].expand(streams * streams)))
    # The released kernel fuses the affine multiply/add before sigmoid and
    # Sinkhorn. Separate operations round its FP32 coefficients differently;
    # these differences can survive the subsequent BF16 stream collapse. This
    # holds on the eager path; the compiled core decomposes addcmul and is
    # allclose only (see the module note above).
    logits = torch.addcmul(base, mixes, scales)
    pre = torch.sigmoid(logits[..., :streams]) + eps
    post = 2 * torch.sigmoid(logits[..., streams : 2 * streams])
    return pre, post, logits[..., 2 * streams :].unflatten(-1, (streams, streams))


def _hc_collapse_core(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(2).to(hidden_states.dtype)


def _hc_expand_core(
    output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    update = post.unsqueeze(-1) * output.unsqueeze(-2)
    mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)
    return (update + mixed).to(output.dtype)


_rms_norm = _rms_norm_core
_hc_project = _hc_project_core
_hc_collapse = _hc_collapse_core
_hc_expand = _hc_expand_core
_NORM_COMPILED = False
_HC_COMPILED = False


def compile_norm_core() -> None:
    """Wrap the fp32 RMSNorm core with ``torch.compile`` (``BackendConfig.compile_norm``); runs once per process.

    Static shapes: one specialization per norm site (block, attention, compressor and indexer norms) and grad mode.
    The dynamic-shape variant (Kimi K3's setting) was measured 15 % slower per call and moved the first-step loss of
    the 12-layer mini benchmark by 9e-4, while the static kernels reproduced the eager loss to four decimals; no
    ``fullgraph``, so a shape past dynamo's recompile limit falls back to eager instead of raising.
    """
    global _rms_norm, _NORM_COMPILED
    if _NORM_COMPILED:
        return
    _rms_norm = torch.compile(_rms_norm_core, dynamic=False)
    _NORM_COMPILED = True


def compile_hc_cores() -> None:
    """Wrap the mHC projection, collapse and expand cores with ``torch.compile`` (``BackendConfig.compile_hc``).

    Runs once per process, static shapes (``dynamic=False``: the fusion of ``expand`` needs concrete shapes; a new
    sequence length compiles the three cores again). No ``fullgraph``: past dynamo's recompile limit (many distinct
    sequence lengths) a core runs eager instead of raising ``FailOnRecompileLimitHit``; the unit tests assert the
    cores compile without graph breaks. The Sinkhorn normalisation keeps its own backend (TileKernels or torch):
    compiling the 20-iteration torch loop is slower than the fused TileKernels kernel.
    """
    global _hc_project, _hc_collapse, _hc_expand, _HC_COMPILED
    if _HC_COMPILED:
        return
    _hc_project = torch.compile(_hc_project_core, dynamic=False)
    _hc_collapse = torch.compile(_hc_collapse_core, dynamic=False)
    _hc_expand = torch.compile(_hc_expand_core, dynamic=False)
    _HC_COMPILED = True


class DeepseekV41RMSNorm(nn.Module):
    """Normalize in FP32 and multiply the scale before casting the result."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the released reference's RMS normalization.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden], with the input dtype.
        """
        return _rms_norm(hidden_states, self.weight, self.eps)


class DeepseekV41Mix(NamedTuple):
    """FP32 coefficients: pre/post [batch, sequence, streams], comb [batch, sequence, streams, streams]."""

    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class DeepseekV41HyperConnection(nn.Module):
    """Predict one sublayer's residual mixing coefficients in FP32."""

    def __init__(
        self, config: DeepseekV41TextConfig, *, sinkhorn_backend: Literal["torch", "tilelang"] = "torch"
    ) -> None:
        super().__init__()
        if sinkhorn_backend not in ("torch", "tilelang"):
            raise ValueError("DeepSeek V4.1 mHC supports 'torch' or 'tilelang' Sinkhorn backends")
        self.sinkhorn_backend = sinkhorn_backend
        self.streams = config.hc_mult
        self.iterations = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        width = self.streams * (self.streams + 2)
        self.fn = nn.Parameter(torch.empty(width, self.streams * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(width, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.reset_parameters(config.initializer_range)

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02) -> None:
        """Initialize all coefficients before checkpoint-free execution."""
        nn.init.normal_(self.fn, std=std)
        self.base.zero_()
        self.scale.fill_(1)

    def forward(self, hidden_states: torch.Tensor) -> DeepseekV41Mix:
        """Predict coefficients, preserving projection-before-RMS arithmetic.

        Args:
            hidden_states: Tensor of shape [batch, sequence, streams, hidden].

        Returns:
            Coefficients with pre/post tensors of shape [batch, sequence, streams]
            and comb of shape [batch, sequence, streams, streams]. All use FP32.
        """
        pre, post, comb_logits = _hc_project(
            hidden_states, self.fn, self.base, self.scale, self.streams, self.norm_eps, self.eps
        )
        comb = dsv4_sinkhorn_normalize(comb_logits, backend=self.sinkhorn_backend, repeat=self.iterations, eps=self.eps)
        return DeepseekV41Mix(pre, post, comb)

    @staticmethod
    def collapse(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse streams using the preceding sublayer's coefficients.

        Args:
            hidden_states: Tensor of shape [batch, sequence, streams, hidden].
            pre_mix: FP32 tensor of shape [batch, sequence, streams].

        Returns:
            Tensor of shape [batch, sequence, hidden], with the input dtype.
        """
        return _hc_collapse(hidden_states, pre_mix)

    @staticmethod
    def expand(output: torch.Tensor, residual: torch.Tensor, mix: DeepseekV41Mix) -> torch.Tensor:
        """Mix the sublayer output and residual in the source's coefficient orientation.

        Args:
            output: Tensor of shape [batch, sequence, hidden].
            residual: Tensor of shape [batch, sequence, streams, hidden].
            mix: FP32 pre/post tensors of shape [batch, sequence, streams] and
                comb of shape [batch, sequence, input_streams, output_streams].

        Returns:
            Tensor of shape [batch, sequence, streams, hidden], with output's dtype.
        """
        return _hc_expand(output, residual, mix.post, mix.comb)
