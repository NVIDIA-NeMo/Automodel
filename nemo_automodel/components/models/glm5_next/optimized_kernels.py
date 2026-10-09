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

"""Optional model-owned mHC and RMSNorm compilation for GLM-5.3-Flash."""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F

# Preserve the eager BF16 truncation points between fused operations.
_COMPILE_OPTIONS = {
    "emulate_precision_casts": True,
    "eager_numerics.division_rounding": True,
}


def project_core(hidden_streams: torch.Tensor, fn: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the flattened streams in FP32 before the coefficient GEMM.

    Args:
        hidden_streams: Residual streams ``[batch, sequence, streams, hidden]``.
        fn: Projection weight ``[streams * (2 + streams), streams * hidden]``.
        eps: Additive RMS normalization epsilon.

    Returns:
        FP32 raw coefficients ``[batch, sequence, streams * (2 + streams)]``.
    """
    flat = hidden_streams.flatten(start_dim=2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    return F.linear(flat, fn.float())


def collapse_core(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Accumulate FP32 pre-weighted streams, then round the output once.

    Args:
        hidden_streams: Residual streams ``[batch, sequence, streams, hidden]``.
        pre: FP32 coefficients ``[batch, sequence, streams]``.

    Returns:
        Collapsed ``[batch, sequence, hidden]`` in the residual-stream dtype.
    """
    return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)


def expand_core(update: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Keep coefficient, product, BMM output and sum rounding in GLM's order.

    Args:
        update: Sublayer output ``[batch, sequence, hidden]`` in residual dtype.
        residual: Residual streams ``[batch, sequence, streams, hidden]``.
        post: FP32 coefficients ``[batch, sequence, streams]``.
        comb: FP32 combinations ``[batch, sequence, source_stream, destination_stream]``.

    Returns:
        Updated ``[batch, sequence, streams, hidden]`` in residual dtype.
    """
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )


def norm_core(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Round normalized values before multiplying the norm weight, as GLM does.

    Args:
        hidden_states: Input ``[..., hidden]`` with arbitrary leading axes.
        weight: Learned scale ``[hidden]`` in the input dtype.
        eps: Additive RMS normalization epsilon.

    Returns:
        Normalized values with the input shape and dtype.
    """
    states = hidden_states.float()
    states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + eps)
    return weight * states.to(hidden_states.dtype)


ProjectionCore = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
CollapseCore = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ExpandCore = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
NormCore = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


@cache
def hc_cores(compiled: bool) -> tuple[ProjectionCore, CollapseCore, ExpandCore]:
    """Select per-instance mHC functions without changing eager model instances.

    Static shapes match the model-owned DeepSeek V4.1 compilation pattern.
    Past Dynamo's recompile limit, its normal eager fallback remains available.
    Base/scale parameters and their FSDP holder are outside these functions.
    """
    if not compiled:
        return project_core, collapse_core, expand_core
    return (
        torch.compile(project_core, dynamic=False, options=_COMPILE_OPTIONS),
        torch.compile(collapse_core, dynamic=False, options=_COMPILE_OPTIONS),
        torch.compile(expand_core, dynamic=False, options=_COMPILE_OPTIONS),
    )


@cache
def rms_norm_core(compiled: bool) -> NormCore:
    """Select the decoder/final RMSNorm core, retaining GLM's cast order."""
    if not compiled:
        return norm_core
    return torch.compile(norm_core, dynamic=False, options=_COMPILE_OPTIONS)
