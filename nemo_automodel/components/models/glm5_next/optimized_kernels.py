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

import logging
from collections.abc import Callable, Collection
from functools import cache

import torch
import torch._inductor.config
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _division_rounding_option(known_options: Collection[str]) -> str | None:
    """Name Inductor's eager division-rounding control as the given PyTorch spells it.

    Newer PyTorch spells it ``eager_numerics.division_rounding``; older releases
    such as 2.10 only accept the legacy ``emulate_divison_rounding`` key, and
    releases such as 2.8 have neither. ``torch.compile`` rejects unknown keys.

    Args:
        known_options: Option names of the installed Inductor config.

    Returns:
        The division-rounding option name present in ``known_options``, or
        ``None`` when this PyTorch has no such control.
    """
    for name in ("eager_numerics.division_rounding", "emulate_divison_rounding"):
        if name in known_options:
            return name
    return None


def _compile_options_for(known_options: Collection[str]) -> dict[str, bool] | None:
    """Inductor options that preserve the eager BF16 truncation points between fused operations.

    Args:
        known_options: Option names of the installed Inductor config.

    Returns:
        Option mapping for ``torch.compile(options=...)``, or ``None`` when
        ``known_options`` lacks either precision control. Without them a
        compiled core would not reproduce the eager rounding, so callers keep
        the eager core instead.
    """
    division_rounding = _division_rounding_option(known_options)
    if division_rounding is None or "emulate_precision_casts" not in known_options:
        return None
    return {"emulate_precision_casts": True, division_rounding: True}


@cache
def _compile_options() -> dict[str, bool] | None:
    """Precision-preserving options for the installed Inductor, or ``None`` if it cannot honor them."""
    options = _compile_options_for(torch._inductor.config.get_config_copy())
    if options is None:
        logger.warning(
            "PyTorch %s Inductor lacks the eager division-rounding/precision-cast controls; "
            "GLM-5-Next compile_hc/compile_norm run the eager cores instead.",
            torch.__version__,
        )
    return options


def project_core(hidden_streams: torch.Tensor, fn: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the flattened streams in FP32 before the coefficient GEMM.

    Args:
        hidden_streams: Tensor of shape [batch, sequence, streams, hidden],
            where streams is ``config.hc_mult``.
        fn: Tensor of shape [streams * (2 + streams), streams * hidden], the
            projection weight.
        eps: Additive RMS normalization epsilon.

    Returns:
        FP32 tensor of shape [batch, sequence, streams * (2 + streams)],
        ordered as pre (streams), post (streams) and row-major comb
        (streams * streams) raw coefficients.
    """
    flat = hidden_streams.flatten(start_dim=2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    return F.linear(flat, fn.float())


def collapse_core(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Accumulate FP32 pre-weighted streams, then round the output once.

    Args:
        hidden_streams: Tensor of shape [batch, sequence, streams, hidden].
        pre: FP32 tensor of shape [batch, sequence, streams].

    Returns:
        Tensor of shape [batch, sequence, hidden] in the residual-stream dtype.
    """
    return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)


def expand_core(update: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Keep coefficient, product, BMM output and sum rounding in GLM's order.

    Args:
        update: Tensor of shape [batch, sequence, hidden] in the residual
            dtype, the sublayer output.
        residual: Tensor of shape [batch, sequence, streams, hidden].
        post: FP32 tensor of shape [batch, sequence, streams].
        comb: FP32 tensor of shape [batch, sequence, streams, streams],
            indexed [source_stream, destination_stream].

    Returns:
        Tensor of shape [batch, sequence, streams, hidden] in the residual
        dtype. Coefficients are rounded to that dtype before each product.
    """
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )


def _expand_core_four_streams(
    update: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Fuse four-stream BF16 mixing with the expansion's existing cast boundaries.

    Args:
        update: BF16 tensor of shape [batch, sequence, hidden].
        residual: BF16 tensor of shape [batch, sequence, streams, hidden].
        post: FP32 tensor of shape [batch, sequence, streams].
        comb: FP32 tensor of shape [batch, sequence, streams, streams],
            indexed [source_stream, destination_stream].

    Returns:
        Updated streams of shape [batch, sequence, streams, hidden], in the
        residual dtype. Other dtypes or stream counts use the original core.
    """
    if residual.dtype != torch.bfloat16 or residual.shape[-2] != 4:
        return expand_core(update, residual, post, comb)
    mixing = comb.to(residual.dtype).float()
    streams = residual.float()
    mixed = torch.stack(
        tuple((mixing[..., :, destination].unsqueeze(-1) * streams).sum(dim=-2) for destination in range(4)),
        dim=-2,
    ).to(residual.dtype)
    return post.to(residual.dtype).unsqueeze(-1) * update.unsqueeze(-2) + mixed


def norm_core(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Round normalized values before multiplying the norm weight, as GLM does.

    Args:
        hidden_states: Tensor of shape [..., hidden], with arbitrary leading
            dimensions.
        weight: Tensor of shape [hidden] in the input dtype.
        eps: Additive RMS normalization epsilon.

    Returns:
        Tensor with the input shape and dtype. The FP32 normalized values are
        rounded to the input dtype before the weight multiply.
    """
    states = hidden_states.float()
    states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + eps)
    return weight * states.to(hidden_states.dtype)


def gated_norm_core(hidden_states: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize and sigmoid-gate KDA output, preserving eager cast boundaries.

    Args:
        hidden_states: Tensor of shape [batch, sequence, heads, head_dim].
        gate: Tensor of shape [batch, sequence, heads, head_dim].
        weight: Tensor of shape [head_dim] in ``hidden_states``'s dtype.
        eps: Additive RMS normalization epsilon.

    Returns:
        Tensor of shape [batch, sequence, heads, head_dim] in
        ``hidden_states``'s dtype. Normalization runs in FP32 and is rounded
        to that dtype before the weight multiply; the sigmoid gate is
        computed in FP32 and rounded to that dtype before the final product.
    """
    return norm_core(hidden_states, weight, eps) * torch.sigmoid(gate.float()).to(hidden_states.dtype)


GatedNormCore = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]
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
    PyTorch without the precision controls also keeps the eager functions.
    """
    options = _compile_options() if compiled else None
    if options is None:
        return project_core, collapse_core, expand_core
    return (
        torch.compile(project_core, dynamic=False, options=options),
        torch.compile(collapse_core, dynamic=False, options=options),
        torch.compile(_expand_core_four_streams, dynamic=False, options=options),
    )


@cache
def rms_norm_core(compiled: bool) -> NormCore:
    """Select the decoder/final RMSNorm core, retaining GLM's cast order."""
    options = _compile_options() if compiled else None
    if options is None:
        return norm_core
    return torch.compile(norm_core, dynamic=False, options=options)


@cache
def rms_norm_gated_core(compiled: bool) -> GatedNormCore:
    """Select the KDA gated norm core with the same cast policy as decoder norms."""
    options = _compile_options() if compiled else None
    if options is None:
        return gated_norm_core
    return torch.compile(gated_norm_core, dynamic=False, options=options)
