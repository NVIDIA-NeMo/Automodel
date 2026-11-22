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

"""ERNIE 5.5 normalization, activation and residual-stream layers.

This module holds the leaf layers of the ERNIE 5.5 decoder:

* ``Ernie5_5RMSNorm``: RMSNorm that rounds its input to the weight dtype before computing FP32 statistics.
* ``ernie_situ``: the SiTU activation on a packed gate/up tensor, with its checkpoint-compatible rounding points.
* ``virtual_head_mix``: the low-rank virtual-head mix ``I + V U^T`` applied to per-head attention output.
* ``Ernie5_5HyperConnection``: the multi-stream residual mixer (mHC) with FP32 stream arithmetic.
* ``Ernie5_5HyperHead``: the learned contraction of the residual streams into one output stream.
* ``Ernie5_5MultiMax``: the learned piecewise correction of vocabulary logits.
* ``Ernie5_5DenseMLP``: the bias-free dense feed-forward block built on ``ernie_situ``.

It also defines ``DENSE_ATTENTION_BACKENDS``, the ``backend.attn`` names that the two attention modules of the
package, ``dsa`` and ``hca``, accept for their dense PyTorch path, and ``shard_start``, the row offset of a
context-parallel shard.

Provenance: the definitions below follow the ERNIE 5.5 implementation in Megatron-Bridge (Apache License 2.0) at
revision ``a38833622b88969564fe8bfa053c5e4e4989c6cd``.

* ``Ernie5_5RMSNorm`` (arithmetic and input conversion):
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/norm.py
* ``ernie_situ`` (operation and rounding order):
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/moe.py
* ``virtual_head_mix`` (mathematical contract of ``vha_postmix``):
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/ops.py
* ``Ernie5_5HyperConnection`` (arithmetic):
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/hyper_connection.py
* ``Ernie5_5MultiMax`` (formula and 512-row accumulation order):
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/output_layer.py
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4HyperConnection

# Values of ``backend.attn`` that select the dense PyTorch attention path of the DSA and HCA layers. The three
# names run the same code: neither layer calls a scaled-dot-product kernel.
DENSE_ATTENTION_BACKENDS: tuple[str, ...] = ("eager", "torch", "sdpa")


def shard_start(cp_group: dist.ProcessGroup | None, sequence: int) -> int:
    """Return the row offset of the first token that this context-parallel rank holds.

    Args:
        cp_group: Context-parallel group whose ranks hold consecutive shards of a row, or None.
        sequence: Number of tokens in every shard.
    """
    return 0 if cp_group is None else dist.get_rank(cp_group) * sequence


class Ernie5_5RMSNorm(nn.Module):
    """Round inputs to the weight dtype, normalize in FP32 and round the result.

    The scale parameter ``weight`` has shape [hidden] and the configured dtype.
    """

    def __init__(self, hidden_size: int, *, eps: float = 1e-6, dtype: torch.dtype = torch.float32) -> None:
        """Create the scale parameter, initialized to ones.

        Args:
            hidden_size: Width of the final input dimension.
            eps: Positive finite epsilon added to the mean square.
            dtype: Floating-point dtype of the scale parameter and output.

        Raises:
            ValueError: If ``hidden_size`` is not positive or ``eps`` is not a positive finite number.
        """
        super().__init__()
        if hidden_size < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("hidden_size and eps must be positive, with finite eps")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize arbitrary leading dimensions over the final hidden axis.

        Args:
            x: Floating tensor of shape [..., hidden], with arbitrary leading
                dimensions and ``hidden`` equal to the scale parameter width.

        Returns:
            Tensor of shape [..., hidden] on the input device, in the scale
            parameter dtype.

        Raises:
            ValueError: If ``x`` has no dimensions, its final dimension differs from the width of the scale
                parameter, or its dtype is not floating point.
        """
        if x.ndim < 1 or x.shape[-1] != self.weight.numel():
            raise ValueError("Input must end with the configured hidden dimension")
        if not x.is_floating_point():
            raise ValueError("Input must have a floating-point dtype")
        rounded = x.to(self.weight.dtype)
        value = rounded.float()
        return (value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(
            self.weight.dtype
        )

    def reset_parameters(self) -> None:
        """Initialize the scale, including after meta-device materialization."""
        nn.init.ones_(self.weight)


def ernie_situ(
    gate_up: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
    routing_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SiTU with rounding before optional routing-weight multiplication.

    Args:
        gate_up: Floating tensor of shape [..., 2 * intermediate], with
            arbitrary leading dimensions and the gate followed by the up
            projection on the final axis.
        beta: Positive finite gate-clamp parameter.
        linear_beta: Optional positive finite up-projection clamp parameter.
        routing_weight: Optional tensor broadcastable with [..., intermediate],
            typically a routing weight of shape [..., 1].

    Returns:
        Activated tensor with the broadcast shape of [..., intermediate] and
        ``routing_weight``, on the input device and in the input dtype.

    Raises:
        ValueError: If the final dimension of ``gate_up`` is odd, ``gate_up`` is not floating point, or ``beta`` or
            ``linear_beta`` is not a positive finite number.
    """
    if gate_up.ndim < 1 or gate_up.shape[-1] % 2:
        raise ValueError("Packed gate/up tensor must have an even final dimension")
    if not gate_up.is_floating_point():
        raise ValueError("Packed gate/up tensor must have a floating-point dtype")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if linear_beta is not None and (not math.isfinite(linear_beta) or linear_beta <= 0):
        raise ValueError("linear_beta must be finite and positive")
    gate, up = gate_up.chunk(2, dim=-1)
    with torch.autocast(device_type=gate_up.device.type, enabled=False):
        gate, up = gate.float(), up.float()
        value = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        activated = (value * up).to(gate_up.dtype)
        if routing_weight is not None:
            activated = (activated.float() * routing_weight.float()).to(gate_up.dtype)
    return activated


def virtual_head_mix(attention: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply the low-rank head mix I + V U^T to attention output.

    Only the activation-dtype path is implemented: ``u`` and ``v`` are converted to the dtype of ``attention`` and
    the mix runs in that dtype with autocast disabled.

    Args:
        attention: Tensor of shape [..., heads, channels], with arbitrary
            leading dimensions. The final two axes are head and head channel.
        u: Input-head factor of shape [heads, rank] on the same device.
        v: Output-head factor of shape [heads, rank] on the same device.

    Returns:
        Newly allocated tensor with the same shape and dtype as attention.
        The head-mixing matrix and output matmul use the activation dtype.

    Raises:
        ValueError: If ``attention`` has fewer than two dimensions, or ``u`` and ``v`` are not equal-shaped
            matrices with one row per attention head.
    """
    if attention.ndim < 2 or u.ndim != 2 or u.shape != v.shape or u.shape[0] != attention.shape[-2]:
        raise ValueError("u and v must have shape [attention heads, rank]")
    heads = attention.shape[-2]
    with torch.autocast(device_type=attention.device.type, enabled=False):
        identity = torch.eye(heads, dtype=attention.dtype, device=attention.device)
        mixing = torch.addmm(identity, v.to(attention.dtype), u.to(attention.dtype).transpose(0, 1))
        return torch.matmul(mixing, attention)


class Ernie5_5HyperConnection(DeepseekV4HyperConnection):
    """Use ERNIE normalization and FP32 stream mixing with eager Sinkhorn.

    The parameters are those of ``DeepseekV4HyperConnection``: ``fn`` of shape [streams * (streams + 2),
    streams * hidden], ``base`` of shape [streams * (streams + 2)] and ``scale`` of shape [3], all FP32. The class
    implements the mixing arithmetic only; checkpoint key mapping, the lifetime of the residual streams, the
    mixed-precision policy, activation recomputation and distributed placement belong to the code that owns the
    module.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
    ) -> None:
        """Build the mixer with the torch Sinkhorn backend and without compilation.

        Args:
            hc_mult: Number of residual streams.
            hidden_size: Width of each residual stream.
            hc_sinkhorn_iters: Number of Sinkhorn normalization iterations.
            hc_eps: Epsilon added to the pre coefficients and used by the Sinkhorn normalization.
            rms_norm_eps: Epsilon added to the RMS of the flattened streams before the coefficient projection.

        Raises:
            ValueError: If ``hc_mult``, ``hidden_size`` or ``hc_sinkhorn_iters`` is not positive.
        """
        if hc_mult < 1 or hidden_size < 1 or hc_sinkhorn_iters < 1:
            raise ValueError("Stream count, hidden size and Sinkhorn iterations must be positive")
        super().__init__(
            hc_mult,
            hidden_size,
            hc_sinkhorn_iters,
            hc_eps,
            rms_norm_eps,
            sinkhorn_backend="torch",
            compile_hc=False,
        )

    @staticmethod
    def _project(
        hidden_streams: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        hc: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the mixing coefficients before Sinkhorn normalization, in FP32 with autocast disabled.

        Args:
            hidden_streams: Tensor of shape [batch, sequence, streams, hidden].
            fn: Projection of shape [mix_components, streams * hidden], where mix_components is
                streams * (streams + 2).
            base: Offsets of shape [mix_components], ordered pre, post, combination.
            scale: Gains of shape [3], ordered pre, post, combination.
            hc: Number of residual streams.
            norm_eps: Epsilon added to the RMS of the flattened streams.
            hc_eps: Epsilon added to the pre coefficients.

        Returns:
            FP32 pre and post coefficients of shape [batch, sequence, streams] and FP32 combination logits of
            shape [batch, sequence, input_streams, output_streams].
        """
        flat = hidden_streams.flatten(start_dim=2).float()
        with torch.autocast(device_type=hidden_streams.device.type, enabled=False):
            proj = torch.matmul(flat, fn.float().T.contiguous())
            inverse_rms = 1.0 / (flat.norm(dim=-1, keepdim=True) / math.sqrt(flat.shape[-1]) + norm_eps)
            mix = inverse_rms * proj
            pre = torch.sigmoid(mix[..., :hc] * scale[0].float() + base[:hc].float()) + hc_eps
            post = 2 * torch.sigmoid(mix[..., hc : 2 * hc] * scale[1].float() + base[hc : 2 * hc].float())
            comb = mix[..., 2 * hc :].reshape(*mix.shape[:-1], hc, hc) * scale[2].float()
            comb = comb + base[2 * hc :].reshape(hc, hc).float()
        return pre, post, comb

    @staticmethod
    def collapse(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Sum streams in FP32; the following layer owns any downcast.

        Args:
            hidden_streams: Tensor of shape [batch, sequence, streams, hidden].
            pre: Coefficients of shape [batch, sequence, streams].

        Returns:
            FP32 tensor of shape [batch, sequence, hidden]. Unlike the parent class, the result is not converted
            back to the dtype of ``hidden_streams``.
        """
        with torch.autocast(device_type=hidden_streams.device.type, enabled=False):
            return (pre.float().unsqueeze(-1) * hidden_streams.float()).sum(dim=-2)

    @staticmethod
    def expand(output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """Mix the residual and layer output in FP32, rounding once.

        Args:
            output: Tensor of shape [batch, sequence, hidden].
            residual: Tensor of shape [batch, sequence, streams, hidden].
            post: Coefficients of shape [batch, sequence, streams].
            comb: Coefficients of shape [batch, sequence, input_streams, output_streams].

        Returns:
            Tensor of shape [batch, sequence, streams, hidden] in the residual dtype.
        """
        with torch.autocast(device_type=residual.device.type, enabled=False):
            mixed = torch.matmul(comb.float().transpose(-1, -2), residual.float())
            result = mixed + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)
        return result.to(residual.dtype)


class Ernie5_5HyperHead(nn.Module):
    """Collapse multiple residual streams using FP32 learned coefficients.

    The parameters ``hc_fn`` of shape [streams, streams * hidden], ``hc_base`` of shape [streams] and ``hc_scale``
    of shape [1] are created in FP32, and ``forward`` raises if any of them has another dtype. Full-model and FSDP
    precision policies remain the caller's responsibility.
    """

    def __init__(self, streams: int, hidden: int, *, eps: float = 1e-6) -> None:
        """Create the FP32 contraction parameters.

        Args:
            streams: Number of residual streams.
            hidden: Width of each residual stream.
            eps: Epsilon added to the mean square of the flattened streams and to the stream weights.

        Raises:
            ValueError: If ``streams``, ``hidden`` or ``eps`` is not positive.
        """
        super().__init__()
        if streams < 1 or hidden < 1 or eps <= 0:
            raise ValueError("streams, hidden and eps must be positive")
        self.streams, self.hidden, self.eps = streams, hidden, eps
        self.hc_fn = nn.Parameter(torch.empty(streams, streams * hidden, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.zeros(streams, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        nn.init.normal_(self.hc_fn, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract [batch, sequence, streams, hidden] into [batch, sequence, hidden].

        Args:
            x: Tensor of shape [batch, sequence, streams, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden] in the input dtype. The stream weights are computed and
            applied in FP32 with autocast disabled.

        Raises:
            ValueError: If ``x`` does not have shape [batch, sequence, streams, hidden].
            RuntimeError: If ``hc_fn``, ``hc_base`` or ``hc_scale`` is not FP32.
        """
        if x.ndim != 4 or x.shape[-2:] != (self.streams, self.hidden):
            raise ValueError("Input must be [batch, sequence, streams, hidden]")
        if any(parameter.dtype != torch.float32 for parameter in (self.hc_fn, self.hc_base, self.hc_scale)):
            raise RuntimeError("ERNIE hyper-head parameters must remain FP32")
        with torch.autocast(device_type=x.device.type, enabled=False):
            # One shared cast makes the two input-gradient paths accumulate in
            # FP32 before conversion back to the input dtype.
            streams = x.float()
            flat = streams.flatten(start_dim=2)
            inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.eps)
            logits = F.linear(flat, self.hc_fn) * inverse_rms
            weights = torch.sigmoid(logits * self.hc_scale + self.hc_base) + self.eps
            result = (weights.unsqueeze(-1) * streams).sum(dim=-2)
        return result.to(x.dtype)

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize parameters after meta materialization.

        Args:
            init_std: Standard deviation of the normal initialization of ``hc_fn``.

        Raises:
            ValueError: If ``init_std`` is not positive.
        """
        if init_std <= 0:
            raise ValueError("init_std must be positive")
        nn.init.normal_(self.hc_fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)


_CHUNK_ROWS = 512


def _correction(logits: torch.Tensor, ranges: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Compute piecewise correction in FP32.

    Args:
        logits: FP32 tensor of shape [rows, vocab].
        ranges: FP32 threshold tensor of shape [4].
        scales: FP32 scale tensor of shape [4].

    Returns:
        FP32 corrected logits of shape [rows, vocab].
    """
    r0, r1, r2, r3 = ranges.unbind()
    t0, t1, t2, t3 = scales.unbind()
    return (
        logits
        + t0 * torch.relu(r0 - logits)
        + t1 * torch.relu(logits - r1)
        + t2 * torch.relu(r2 - logits).square()
        + t3 * torch.relu(logits - r3).square()
    )


class _MultiMaxFunction(torch.autograd.Function):
    """Chunk temporary logits and accumulate parameter gradients in FP32."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, logits: torch.Tensor, ranges: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        """Correct logits using learned thresholds and scales.

        Args:
            ctx: Autograd context retaining inputs for first-order gradients.
            logits: Floating tensor of shape [..., vocab], with arbitrary leading dimensions.
            ranges: Floating threshold tensor of shape [4], on the logits device.
            scales: Floating scale tensor of shape [4], on the logits device.

        Returns:
            Corrected tensor of shape [..., vocab], with logits dtype and device.
        """
        ctx.save_for_backward(logits, ranges, scales)
        rows = logits.reshape(-1, logits.shape[-1])
        result = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
        result_rows = result.view_as(rows)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            ranges_fp32, scales_fp32 = ranges.float(), scales.float()
            for start in range(0, rows.shape[0], _CHUNK_ROWS):
                stop = start + _CHUNK_ROWS
                result_rows[start:stop] = _correction(rows[start:stop].float(), ranges_fp32, scales_fp32)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute first-order gradients with FP32 parameter accumulation.

        Args:
            ctx: Autograd context with the forward inputs.
            grad_output: Incoming gradient of shape [..., vocab], matching forward logits.

        Returns:
            Logits gradient of shape [..., vocab], threshold gradient [4] and scale
            gradient [4], each with the corresponding forward input dtype and device.
        """
        logits, ranges, scales = ctx.saved_tensors
        rows = logits.reshape(-1, logits.shape[-1])
        incoming = grad_output.reshape_as(rows)
        grad_logits = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
        grad_rows = grad_logits.view_as(rows)
        range_leaf = ranges.detach().float().requires_grad_()
        scale_leaf = scales.detach().float().requires_grad_()
        grad_ranges = torch.zeros_like(range_leaf)
        grad_scales = torch.zeros_like(scale_leaf)
        with torch.enable_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
            for start in range(0, rows.shape[0], _CHUNK_ROWS):
                stop = start + _CHUNK_ROWS
                chunk = rows[start:stop].detach().requires_grad_()
                output = _correction(chunk.float(), range_leaf, scale_leaf).to(logits.dtype)
                dx, dr, dt = torch.autograd.grad(output, (chunk, range_leaf, scale_leaf), incoming[start:stop])
                grad_rows[start:stop] = dx
                grad_ranges += dr
                grad_scales += dt
        return grad_logits, grad_ranges.to(ranges.dtype), grad_scales.to(scales.dtype)


class Ernie5_5MultiMax(nn.Module):
    """Apply learned MultiMax correction to materialized vocabulary logits.

    The thresholds ``multimax_ranges`` and the scales ``multimax_ts`` each have shape [4].
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        """Create zero thresholds and scales, with which the correction is the identity.

        Args:
            dtype: Floating dtype of the learned thresholds and scales.
        """
        super().__init__()
        self.multimax_ranges = nn.Parameter(torch.zeros(4, dtype=dtype))
        self.multimax_ts = nn.Parameter(torch.zeros(4, dtype=dtype))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Correct logits while retaining their shape and dtype.

        Args:
            logits: Floating tensor of shape [..., vocab], with arbitrary
                leading dimensions and nonempty ``vocab``.

        Returns:
            Corrected tensor of shape [..., vocab], with the input dtype and
            device.

        Raises:
            ValueError: If ``logits`` has no dimensions, an empty final dimension, or a dtype that is not floating
                point.
        """
        if logits.ndim < 1 or logits.shape[-1] == 0:
            raise ValueError("MultiMax expects logits with a nonempty final vocabulary dimension")
        if not logits.is_floating_point():
            raise ValueError("MultiMax logits must have a floating-point dtype")
        return _MultiMaxFunction.apply(logits, self.multimax_ranges, self.multimax_ts)


class Ernie5_5DenseMLP(nn.Module):
    """Bias-free gate/up/down projections and ERNIE SiTU."""

    def __init__(
        self, hidden: int, intermediate: int, *, dtype: torch.dtype, beta: float = 4.0, linear_beta: float = 25.0
    ) -> None:
        """Create the three bias-free projections.

        Args:
            hidden: Input and output width.
            intermediate: Width of the gate and up projections.
            dtype: Dtype of the projection weights.
            beta: Gate-clamp parameter of SiTU.
            linear_beta: Up-projection clamp parameter of SiTU.

        Raises:
            ValueError: If ``hidden`` or ``intermediate`` is not positive.
        """
        super().__init__()
        if hidden < 1 or intermediate < 1:
            raise ValueError("hidden and intermediate dimensions must be positive")
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.beta, self.linear_beta = beta, linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the dense FFN on normalized activations.

        Args:
            x: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden].
        """
        packed = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(ernie_situ(packed, beta=self.beta, linear_beta=self.linear_beta))

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize projection weights after meta materialization.

        Args:
            init_std: Standard deviation of the normal initialization of the three projection weights.

        Raises:
            ValueError: If ``init_std`` is not positive.
        """
        if init_std <= 0:
            raise ValueError("init_std must be positive")
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(module.weight, mean=0, std=init_std)
