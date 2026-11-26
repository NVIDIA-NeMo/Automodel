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

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F
from torch.utils._python_dispatch import _disable_current_modes

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.lm_head import LMHead
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.layers import MoE

from .config import Inkling2TextConfig
from .context_parallel import convolution_prefix
from .triton_activation import capped_swiglu_gradient as _one_pass_swiglu_gradient
from .triton_convolution import HAVE_TRITON as _HAVE_TRITON
from .triton_convolution import static_convolution as _tiled_static_convolution


@dataclass(frozen=True)
class ResidualState:
    """Immutable depth state; every tensor has shape [batch, sequence, hidden].

    Attributes:
        blocks: Completed depth blocks, including the initial embedding block.
        partial: Current incomplete block. Neither field is detached or mutated.
    """

    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor


def _mix_sources(
    blocks: tuple[torch.Tensor, ...], partial: torch.Tensor, queries: torch.Tensor, normalize_values: bool
) -> list[torch.Tensor]:
    """Mix depth sources one source and one query at a time, the form the compiler fuses best.

    Args:
        blocks: Completed blocks, each [batch, sequence, hidden]; may be empty.
        partial: Current block [batch, sequence, hidden].
        queries: FP32 tensor of shape [queries, hidden].
        normalize_values: Normalize values as well as scoring keys.

    Returns:
        One FP32 mixture [batch, sequence, hidden] per query. A source's RMS scale is one
        number per token, so it leaves both sums over the hidden axis: every score is one
        fused reduction over a stored source and every mixture one fused pass over all of
        them, without an FP32 copy of the sources.
    """
    sources = [source.float() for source in (*blocks, partial)]
    scales = [torch.rsqrt(source.square().mean(-1) + 1e-6) for source in sources]
    rescale = queries.shape[-1] ** -0.5
    mixtures = []
    for query in queries.unbind(0):
        scores = torch.stack([(source * query).sum(-1) * scale for source, scale in zip(sources, scales)])
        weights = (scores * rescale).softmax(0)
        mixed = None
        for index, source in enumerate(sources):
            coefficient = weights[index] * scales[index] if normalize_values else weights[index]
            term = coefficient.unsqueeze(-1) * source
            mixed = term if mixed is None else mixed + term
        mixtures.append(mixed)
    return mixtures


def _mix_core(
    blocks: tuple[torch.Tensor, ...], partial: torch.Tensor, queries: torch.Tensor, normalize_values: bool
) -> tuple[torch.Tensor, ...]:
    """Mix depth sources independently at each token in FP32.

    Args:
        blocks: Completed blocks, each [batch, sequence, hidden]; may be empty.
        partial: Current block [batch, sequence, hidden].
        queries: Tensor of shape [queries, hidden], with one or more pseudo-queries.
        normalize_values: Normalize values as well as scoring keys.

    Returns:
        One FP32 tensor [batch, sequence, hidden] per query. The inputs are unchanged.
    """
    queries = queries.float()
    if torch.compiler.is_compiling():
        return tuple(_mix_sources(blocks, partial, queries, normalize_values))
    values = torch.stack([source.float() for source in (*blocks, partial)])
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    scores = torch.einsum("nbtd,qd->qbtn", normalized, queries) * queries.shape[-1] ** -0.5
    weights = scores.softmax(-1)
    return torch.einsum("qbtn,nbtd->qbtd", weights, normalized if normalize_values else values).unbind(0)


def _mix_normalize_core(
    blocks: tuple[torch.Tensor, ...],
    partial: torch.Tensor,
    queries: torch.Tensor,
    normalize_values: bool,
    norm_weights: tuple[torch.Tensor, ...],
    eps: float,
) -> tuple[torch.Tensor, ...]:
    """Mix depth sources, then apply one RMSNorm per mixture and round to the storage dtype.

    Args:
        blocks: Completed blocks, each [batch, sequence, hidden]; may be empty.
        partial: Current block [batch, sequence, hidden].
        queries: Tensor of shape [queries, hidden].
        normalize_values: Normalize values as well as scoring keys.
        norm_weights: RMSNorm weights, each [hidden], one per mixture.
        eps: RMSNorm epsilon.

    Returns:
        One tensor [batch, sequence, hidden] per query in the norm weights' dtype.
    """
    if torch.compiler.is_compiling():
        mixed = _mix_sources(blocks, partial, queries.float(), normalize_values)
    else:
        mixed = list(_mix_core(blocks, partial, queries, normalize_values))
    width = mixed[0].shape[-1]
    return tuple(
        F.rms_norm(mixture, (width,), weight, eps).to(weight.dtype) for mixture, weight in zip(mixed, norm_weights)
    )


def _control_core(
    destination: torch.Tensor,
    message: torch.Tensor,
    down_weight: torch.Tensor,
    gain_weight: torch.Tensor,
    forget_weight: torch.Tensor,
) -> torch.Tensor:
    """Keep part of the destination and add part of the message, per channel and token.

    Args:
        destination: FP32 layer input of shape [batch, sequence, hidden].
        message: FP32 message of shape [batch, sequence, hidden].
        down_weight: Projection [rank, 2 * hidden] of both normalized inputs.
        gain_weight: Projection [hidden, rank] of the message gate.
        forget_weight: Projection [hidden, rank] of the destination gate.

    Returns:
        FP32 tensor of shape [batch, sequence, hidden]. The inputs are unchanged.
    """
    normalized = torch.cat(
        [value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) for value in (destination, message)],
        dim=-1,
    )
    hidden = F.linear(normalized.to(down_weight.dtype), down_weight).float()
    gain = torch.tanh(F.linear(hidden, gain_weight.float()))
    forget = torch.tanh(F.linear(hidden, forget_weight.float()))
    return (1 - forget) * destination + gain * message


def _normalize_core(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm in FP32 and round to the weight's dtype.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: RMSNorm weight of shape [hidden].
        eps: RMSNorm epsilon.

    Returns:
        Tensor of x's shape in the weight's dtype.
    """
    return F.rms_norm(x.float(), (x.shape[-1],), weight, eps).to(weight.dtype)


def _value_norm_core(value: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize value heads without a learned gain.

    Args:
        value: Tensor of shape [batch, sequence, heads, head_dim].
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of the same shape and dtype; the mean square is taken in FP32.
    """
    wide = value.float()
    return (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)).to(value.dtype)


def _gate_core(context: torch.Tensor, gate: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scale attended heads by their sigmoid gates and zero the padded queries.

    Args:
        context: FP32 tensor of shape [batch, sequence, heads, head_dim].
        gate: Gate logits of shape [batch, sequence, heads * head_dim].
        valid: Boolean tensor of shape [batch, sequence], true for real tokens.

    Returns:
        Tensor of shape [batch, sequence, heads * head_dim] in the gate's dtype.
    """
    return (context.flatten(2) * gate.float().sigmoid() * valid.unsqueeze(-1)).to(gate.dtype)


def _accumulate_core(partial: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Add a sublayer output to the current block in FP32.

    Args:
        partial: Current block [batch, sequence, hidden].
        update: Sublayer output [batch, sequence, hidden].

    Returns:
        FP32 sum of shape [batch, sequence, hidden].
    """
    return partial.float() + update.float()


def _close_core(partial: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Add the last sublayer output and round the block to its storage dtype.

    Args:
        partial: FP32 current block [batch, sequence, hidden].
        update: Sublayer output [batch, sequence, hidden] in the storage dtype.

    Returns:
        Tensor of shape [batch, sequence, hidden] in the update's dtype.
    """
    return (partial + update.float()).to(update.dtype)


def _swiglu_core(gate_up: torch.Tensor, weights: torch.Tensor, gate_softcap: float, up_softcap: float) -> torch.Tensor:
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


def _swiglu_grad_core(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    grad_output: torch.Tensor,
    gate_softcap: float,
    up_softcap: float,
    weights_need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Differentiate the capped activation in FP32 from its inputs alone.

    Args:
        gate_up: Tensor of shape [tokens, 2 * intermediate], interleaved gate/up.
        weights: Tensor of shape [tokens, 1], containing selected expert weights.
        grad_output: Gradient of the activation output, shape [tokens, intermediate].
        gate_softcap: Positive gate tanh cap.
        up_softcap: Positive up tanh cap.
        weights_need_grad: Whether to return the gradient of the weights.

    Returns:
        The gradient of gate_up in its dtype and layout, and the gradient of the weights
        [tokens, 1] in their dtype (None when not needed).
    """
    gate = gate_up[..., 0::2].float()
    up = gate_up[..., 1::2].float()
    grad = grad_output.float()
    # Recompute the forward terms in the forward's order, then apply the chain rule in the order
    # autograd uses for the same expression, so that eager gradients equal the ones autograd
    # derives from the activation itself.
    gate_tanh = torch.tanh(gate / gate_softcap)
    gate_capped = gate_softcap * gate_tanh
    gate_sigmoid = torch.sigmoid(gate)
    gate_value = gate_capped * gate_sigmoid
    up_tanh = torch.tanh(up / up_softcap)
    up_value = up_softcap * up_tanh
    product = gate_value * up_value
    grad_product = grad * weights.float()
    grad_gate_value = grad_product * up_value
    grad_up_value = grad_product * gate_value
    grad_gate = torch.ops.aten.tanh_backward(grad_gate_value * gate_sigmoid * gate_softcap, gate_tanh) / gate_softcap
    grad_gate = grad_gate + torch.ops.aten.sigmoid_backward(grad_gate_value * gate_capped, gate_sigmoid)
    grad_up = torch.ops.aten.tanh_backward(grad_up_value * up_softcap, up_tanh) / up_softcap
    grad_gate_up = torch.stack((grad_gate, grad_up), dim=-1).flatten(-2).to(gate_up.dtype)
    if not weights_need_grad:
        return grad_gate_up, None
    grad_weights = (grad * product).sum(-1, keepdim=True).to(weights.dtype)
    return grad_gate_up, grad_weights


def _static_convolution_core(
    x: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Apply static causal taps that stop at each document's first token.

    Args:
        x: Tensor of shape [batch, sequence, channels].
        valid: Boolean tensor of shape [batch, sequence], true for real tokens.
        positions: Logical document positions [batch, sequence].
        weight: Taps [channels, 1, width] in conv1d order: oldest first, current position last.
        prefix: FP32 history [batch, width - 1, channels] before the first position.
        width: Number of taps.

    Returns:
        Tensor of shape [batch, sequence, channels], in x's dtype; padding is zero.
    """
    masked = x * valid.unsqueeze(-1)
    history = torch.cat((prefix, masked.float()), dim=1)
    taps = weight.float()[:, 0]
    sequence = x.shape[1]
    # Add the taps oldest first in FP32, the order of a grouped convolution. A tap that
    # would read before its document's first token contributes zero.
    filtered = history[:, :sequence] * taps[:, 0]
    filtered = filtered.masked_fill(positions.unsqueeze(-1) < width - 1, 0)
    for slot in range(1, width):
        lag = width - 1 - slot
        term = history[:, slot : slot + sequence] * taps[:, slot]
        if lag:
            term = term.masked_fill(positions.unsqueeze(-1) < lag, 0)
        filtered = filtered + term
    scale = (width + 1) ** -0.5
    return ((masked.float() + filtered) * scale).to(x.dtype) * valid.unsqueeze(-1)


def _dynamic_convolution_core(
    x: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor,
    static_w: torch.Tensor,
    generated: torch.Tensor,
    prefix: torch.Tensor,
    width: int,
    heads: int,
) -> torch.Tensor:
    """Apply static plus token-generated per-head taps, lag-indexed: current position first.

    Args:
        x: Tensor of shape [batch, sequence, channels].
        valid: Boolean tensor of shape [batch, sequence], true for real tokens.
        positions: Logical document positions [batch, sequence].
        static_w: Static taps [width, channels].
        generated: Token-generated taps [batch, sequence, heads * width].
        prefix: FP32 history [batch, width - 1, channels] before the first position.
        width: Number of taps.
        heads: Number of heads sharing one generated tap per lag.

    Returns:
        Tensor of shape [batch, sequence, channels], in x's dtype; padding is zero.
    """
    batch, sequence, _ = x.shape
    masked = x * valid.unsqueeze(-1)
    history = torch.cat((prefix, masked.float()), dim=1)
    generated = generated.float().view(batch, sequence, heads, width)
    filtered = torch.zeros_like(masked, dtype=torch.float32)
    for lag in range(width):
        lagged = history[:, width - 1 - lag : width - 1 - lag + sequence]
        lagged = lagged * (positions >= lag).unsqueeze(-1)
        tap = static_w[lag].float().view(heads, -1)
        weighted = lagged.view(batch, sequence, heads, -1) * (tap + generated[..., lag, None])
        filtered = filtered + weighted.reshape_as(filtered)
    scale = (2 * width + 1) ** -0.5
    return ((masked.float() + filtered) * scale).to(x.dtype) * valid.unsqueeze(-1)


def _clip_core(x: torch.Tensor, threshold: float, tau: torch.Tensor | None) -> torch.Tensor:
    """Clip head vectors to an L2 threshold, then apply optional length scaling.

    Args:
        x: Tensor of shape [batch, sequence, heads, head_dim].
        threshold: Largest allowed L2 norm of a head vector.
        tau: Optional FP32 tensor of shape [batch, sequence], applied after clipping.

    Returns:
        Tensor of shape [batch, sequence, heads, head_dim], in x's dtype.
    """
    value = x.float()
    factor = (threshold / (torch.linalg.vector_norm(value, dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
    result = (value * factor).to(x.dtype)
    if tau is not None:
        result = (result.float() * tau[..., None, None]).to(x.dtype)
    return result


def _scaled_clip_core(x: torch.Tensor, scale: torch.Tensor, threshold: float, tau: torch.Tensor | None) -> torch.Tensor:
    """Apply learned per-channel scales, clip head vectors, then apply optional length scaling.

    Args:
        x: Tensor of shape [batch, sequence, heads, head_dim].
        scale: FP32 scales of shape [heads, head_dim].
        threshold: Largest allowed L2 norm of a scaled head vector.
        tau: Optional FP32 tensor of shape [batch, sequence], applied after clipping.

    Returns:
        Tensor of shape [batch, sequence, heads, head_dim], in x's dtype.
    """
    value = x.float() * scale.float()
    factor = (threshold / (torch.linalg.vector_norm(value, dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
    result = (value * factor).to(x.dtype)
    if tau is not None:
        result = (result.float() * tau[..., None, None]).to(x.dtype)
    return result


# Layers call the tensor chains through these names; compile_cores replaces them once per process.
_mix = _mix_core
_mix_normalize = _mix_normalize_core
_control = _control_core
_normalize = _normalize_core
_value_norm = _value_norm_core
_gate = _gate_core
_accumulate = _accumulate_core
_close = _close_core
_swiglu = _swiglu_core
_dynamic_convolution = _dynamic_convolution_core
_clip = _clip_core
_scaled_clip = _scaled_clip_core
_CORES_COMPILED = False
_ACTIVATION_COMPILED = False
_CORE_VARIANTS = 64


def _compiled(core: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Compile one tensor chain and run it outside the active dispatch modes.

    A compiled kernel benchmarks its launch configurations on first use. Activation
    checkpointing replays a recorded operation stream in its recomputation and must
    see the same operations in both passes, so it sees none of a compiled chain.
    """
    # Static shapes first: a dimension becomes symbolic only once it changes, so the hidden
    # size stays a constant in the generated index arithmetic.
    compiled = torch.compile(core, fullgraph=True)
    threads = torch.get_num_threads()

    def call(*args: torch.Tensor | float | int | bool | None) -> torch.Tensor:
        # The thread count is part of a compiled variant's guards, and the checkpoint
        # recomputation runs in the autograd thread, which has its own count.
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        with _disable_current_modes():
            return compiled(*args)

    return call


def compile_cores() -> None:
    """Compile the FP32 tensor chains, once per process, so that every layer shares them.

    Each chain becomes a few fused kernels instead of one kernel and one FP32 temporary
    per operation. Compilation is lazy (at the first call). Compiled
    numerics are allclose to eager, not bitwise-identical.
    """
    global _mix, _mix_normalize, _control, _normalize, _value_norm, _gate, _accumulate, _close
    global _dynamic_convolution, _clip, _scaled_clip, _CORES_COMPILED
    if _CORES_COMPILED:
        return
    compile_activation()
    _mix = _compiled(_mix_core)
    _mix_normalize = _compiled(_mix_normalize_core)
    _control = _compiled(_control_core)
    _normalize = _compiled(_normalize_core)
    _value_norm = _compiled(_value_norm_core)
    _gate = _compiled(_gate_core)
    _accumulate = _compiled(_accumulate_core)
    _close = _compiled(_close_core)
    _dynamic_convolution = _compiled(_dynamic_convolution_core)
    _clip = _compiled(_clip_core)
    _scaled_clip = _compiled(_scaled_clip_core)
    _CORES_COMPILED = True


def compile_activation() -> None:
    """Compile the expert activation into one fused kernel, once per process."""
    global _swiglu, _ACTIVATION_COMPILED
    if _ACTIVATION_COMPILED:
        return
    _swiglu = _compiled(_swiglu_core)
    _ACTIVATION_COMPILED = True


def compile_variants() -> AbstractContextManager[None]:
    """Allow a compiled chain more variants than dynamo's default recompile limit.

    One chain serves several source counts, query counts, storage dtypes and gradient
    modes. Past the limit dynamo would stop compiling a variant, and activation
    checkpointing needs the forward and its recomputation to run the same kernels.
    """
    if not _CORES_COMPILED:
        return nullcontext()
    limit = max(torch._dynamo.config.recompile_limit, _CORE_VARIANTS)
    return torch._dynamo.config.patch(recompile_limit=limit)


def outside_dispatch_modes() -> AbstractContextManager[None]:
    """Run eager sublayers outside the active dispatch modes, as the compiled chains do.

    Activation checkpointing sees every operation of a block through a Python dispatch
    mode and saves only the MoE routing. Attention and the short convolutions hold
    nothing it saves, so their operations skip that mode in both passes.
    """
    return _disable_current_modes() if _CORES_COMPILED else nullcontext()


def residual_mix(state: ResidualState, queries: torch.Tensor, *, normalize_values: bool) -> tuple[torch.Tensor, ...]:
    """Mix depth sources independently at each token in FP32.

    Args:
        state: Completed blocks and partial, each [batch, sequence, hidden].
        queries: Tensor of shape [queries, hidden], with one or three pseudo-queries.
        normalize_values: Normalize values as well as scoring keys.

    Returns:
        One FP32 tensor [batch, sequence, hidden] per query, without mutating state.
    """
    return _mix(state.blocks, state.partial, queries, normalize_values)


def normalize(x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
    """Apply an RMSNorm module's weight in FP32 and round to its dtype.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        norm: Owner of the weight [hidden] and epsilon.

    Returns:
        Tensor of x's shape in the weight's dtype.
    """
    return _normalize(x, norm.weight, norm.eps)


def value_norm(value: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize value heads [batch, sequence, heads, head_dim] without a learned gain."""
    return _value_norm(value, eps)


def gate_context(context: torch.Tensor, gate: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scale FP32 attended heads [batch, sequence, heads, head_dim] by their sigmoid gates.

    Queries that ``valid`` [batch, sequence] marks as padding give zero rows.
    """
    return _gate(context, gate, valid)


def accumulate(partial: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Return the FP32 sum of the current block and a sublayer output, each [batch, sequence, hidden]."""
    return _accumulate(partial, update)


def close_block(partial: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Add the last sublayer output to the FP32 block and round to the storage dtype."""
    return _close(partial, update)


class ResidualQuery(nn.Module):
    """Callable FP32 owner for one set of depth-mixture queries."""

    def __init__(self, hidden_size: int, count: int, *, normalize_values: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(count, hidden_size, dtype=torch.float32))
        self.normalize_values = normalize_values

    def forward(self, state: ResidualState, norms: tuple[nn.RMSNorm, ...] | None = None) -> tuple[torch.Tensor, ...]:
        """Mix immutable depth state.

        Args:
            state: Completed and partial blocks, each [batch, sequence, hidden].
            norms: Optional RMSNorm owners, one per query.

        Returns:
            One tensor [batch, sequence, hidden] per query: the FP32 mixtures, or with
            norms each mixture after its RMSNorm, rounded to the norm weight's dtype.
        """
        if norms is None:
            return residual_mix(state, self.weight, normalize_values=self.normalize_values)
        weights = tuple(norm.weight for norm in norms)
        return _mix_normalize(state.blocks, state.partial, self.weight, self.normalize_values, weights, norms[0].eps)


class MessageQueries(nn.Module):
    """FP32 owner of the cross-pass message queries, two per loop core layer."""

    def __init__(self, hidden_size: int, core_layers: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2 * core_layers, hidden_size, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Return a copy of the queries.

        Returns:
            FP32 tensor of shape [2 * core_layers, hidden]. Layers read it after FSDP
            has resharded the parameter, so it does not alias the parameter storage.
        """
        return self.weight.clone()


class MessageController(nn.Module):
    """Low-rank gate that mixes a message from the previous pass into one layer input."""

    def __init__(self, hidden_size: int, rank: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(torch.empty(rank, 2 * hidden_size, dtype=dtype))
        self.gain_weight = nn.Parameter(torch.empty(hidden_size, rank, dtype=dtype))
        self.forget_weight = nn.Parameter(torch.empty(hidden_size, rank, dtype=dtype))

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize the three projections without checkpoint data."""
        for weight in (self.down_weight, self.gain_weight, self.forget_weight):
            nn.init.normal_(weight, std=init_std)

    def forward(self, destination: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Keep part of the destination and add part of the message, per channel and token.

        Args:
            destination: FP32 layer input of shape [batch, sequence, hidden].
            message: FP32 message of shape [batch, sequence, hidden].

        Returns:
            FP32 tensor of shape [batch, sequence, hidden]. The inputs are unchanged.
        """
        return _control(destination, message, self.down_weight, self.gain_weight, self.forget_weight)


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
    return _CappedSwiGLU.apply(gate_up, weights, gate_softcap, up_softcap)


class _CappedSwiGLU(torch.autograd.Function):
    """Capped activation that keeps only its inputs for backward and runs one pass per direction."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gate_up: torch.Tensor,
        weights: torch.Tensor,
        gate_softcap: float,
        up_softcap: float,
    ) -> torch.Tensor:
        """Return the activation [tokens, intermediate] of gate_up [tokens, 2 * intermediate]."""
        ctx.save_for_backward(gate_up, weights)
        ctx.caps = (gate_softcap, up_softcap)
        return _swiglu(gate_up, weights, gate_softcap, up_softcap)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, None, None]:
        """Return the gradients of gate_up and of the weights [tokens, 1]."""
        gate_up, weights = ctx.saved_tensors
        gate_softcap, up_softcap = ctx.caps
        # With the compiled chains on a CUDA device the derivative is one Triton pass over the inputs.
        # Experts without a routed token pass one row without the token axis, which the kernel does not take.
        differentiate = _swiglu_grad_core
        if gate_up.is_cuda and gate_up.ndim == 2 and _ACTIVATION_COMPILED and _HAVE_TRITON:
            differentiate = _one_pass_swiglu_gradient
        grad_gate_up, grad_weights = differentiate(
            gate_up, weights, grad_output, gate_softcap, up_softcap, ctx.needs_input_grad[1]
        )
        return grad_gate_up, grad_weights, None, None


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


class Inkling2LMHead(LMHead):
    """Bias-free FP32 output projection that owns the logit softcap and padded-vocabulary mask."""

    def __init__(self, config: Inkling2TextConfig, *, dtype: torch.dtype, logits_dtype: torch.dtype) -> None:
        """Create the [vocab, hidden] head with ``config``'s softcap and usable vocabulary.

        Args:
            config: Text configuration with the hidden and vocabulary sizes, the logit softcap and the usable
                (unpadded) vocabulary size.
            dtype: Dtype of the weight.
            logits_dtype: Dtype of the logits; the projection, softcap and mask run in FP32 before the cast.
        """
        super().__init__(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            logits_dtype=logits_dtype,
            logit_softcapping=config.final_logit_softcapping,
        )
        self.unpadded_vocab_size = config.unpadded_vocab_size

    def project(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project FP32 copies of the hidden states with a supplied vocabulary weight.

        Args:
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions, in any floating dtype.
            weight: Tensor of shape [vocab, hidden] on input's device: the head's own weight, or the weight a
                chunked loss gathered. A column-parallel DTensor uses Shard(0), with input replicated on its mesh.

        Returns:
            Logits of shape [..., vocab] in ``logits_dtype``, with the head's softcap applied and negative infinity
            in the padded vocabulary rows from ``unpadded_vocab_size`` on. Inputs are not mutated.
        """
        logits = super().project(input.float(), weight.float())
        padded = torch.arange(logits.shape[-1], device=logits.device) >= self.unpadded_vocab_size
        return logits.masked_fill(padded, float("-inf"))


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
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply convolution with exchanged CP history, zero document history and masked padding.

        Args:
            x: Tensor of shape [batch, sequence, channels].
            valid: Boolean tensor of shape [batch, sequence], true for real tokens.
            dynamic_input: Tensor of shape [batch, sequence, hidden] for dynamic taps.
            positions: Logical document positions [batch, sequence], including CP offsets.
                Taps cannot read before the current document start.
            prefix: FP32 history [batch, width - 1, channels] before the first position, when the
                caller already exchanged it; None exchanges it here.

        Returns:
            Tensor of shape [batch, sequence, channels], in x's dtype; padding is zero.
        """
        if prefix is None:
            prefix = convolution_prefix(x, valid, self.cp_mesh, self.width)
        if self.weight_proj is None:
            assert self.weight is not None
            if dynamic_input is not None:
                raise ValueError("Static convolution does not accept dynamic_input")
            if positions is None:
                masked = x * valid.unsqueeze(-1)
                history = torch.cat((prefix, masked.float()), dim=1)
                filtered = F.conv1d(
                    history.transpose(1, 2),
                    self.weight.float(),
                    groups=self.channels,
                ).transpose(1, 2)
                scale = (self.width + 1) ** -0.5
                return ((masked.float() + filtered) * scale).to(x.dtype) * valid.unsqueeze(-1)
            if x.is_cuda and _CORES_COMPILED and _HAVE_TRITON:
                # Tiles of positions by channels: the taps of a tile read the same rows.
                return _tiled_static_convolution(x, valid, positions, self.weight, prefix, self.width)
            return _static_convolution_core(x, valid, positions, self.weight, prefix, self.width)
        assert self.static_w is not None
        if dynamic_input is None:
            raise ValueError("Dynamic convolution requires dynamic_input")
        if positions is None:
            # Without document positions no tap is cut off.
            positions = torch.full(valid.shape, self.width - 1, dtype=torch.long, device=x.device)
        generated = self.weight_proj(dynamic_input)
        return _dynamic_convolution(x, valid, positions, self.static_w, generated, prefix, self.width, self.heads)


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
        if self.scale is None:
            return _clip(x, self.threshold, tau)
        return _scaled_clip(x, self.scale, self.threshold, tau)
