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
#
# Code provenance: ``_scan_backward`` and ``_gated_rmsnorm_backward`` are adapted from
# ``MambaSplitConv1dScanCombinedFn.backward`` in ``mamba_ssm/ops/triton/ssd_combined.py`` (Copyright (c) 2024,
# Tri Dao, Albert Gu) and from ``_layer_norm_bwd`` in ``mamba_ssm/ops/triton/layernorm_gated.py`` (Copyright (c)
# 2024, Tri Dao) of mamba-ssm 2.3.0, https://github.com/akoumpa/cute-mamba at revision
# a8b8882fa03668613f7b06831950d504a03b647a (the revision this repository pins), Apache License 2.0. They have to
# issue the same kernel calls on the tensors that library's forward saves, so the call sequence is kept; the
# launch grid of the gated-RMSNorm backward kernel is chosen here.

"""Fused Mamba2 training scan with a configurable launch grid for the gated-RMSNorm backward kernel.

``split_conv1d_scan`` computes what ``mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined``
computes for the Nemotron V3 mixer: causal conv1d, chunked SSD scan, gated RMSNorm and, when a weight is
passed, the output projection. The forward is that library's forward, so the output and the saved tensors
are the same. The backward issues the same kernel calls in the same order, with one difference: the
gated-RMSNorm backward kernel is launched from here.

mamba_ssm launches that kernel with about one program per streaming multiprocessor, each looping over
thousands of rows (152 programs of 3,450 rows for 8 sequences of 8192 tokens on a GB300).
``norm_backward_grid_scale`` multiplies the number of programs, which shortens the per-program row loop and
fills each multiprocessor with several programs. The kernels and their inputs are otherwise the same, so the
output and the gradients agree with mamba_ssm's as closely as two runs of mamba_ssm agree with each other
(it accumulates the conv, ``dt_bias`` and ``A`` gradients with atomic adds). The one exception is the gradient
of the norm weight: it is the sum of per-program partial sums, so its float32 summation order follows the grid.

``checkpoint_exempt=True`` builds the scan from two autograd nodes instead of one. The inputs node saves what
the scan reads (the projected input and the parameters); the kernel node runs the forward and keeps the scan
output and the norm statistics outside autograd's saved tensors. Under a non-reentrant activation checkpoint
the saved tensors of the inputs node are the last ones the checkpoint tracks, so its recompute ends before
the scan kernels run again, while the kernel node's state stays alive from the forward. The backward is the
same computation, issued by the inputs node once the recompute has rebuilt the projected input.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _gated_rmsnorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rstd: torch.Tensor,
    z: torch.Tensor,
    *,
    group_size: int,
    dz: torch.Tensor,
    recomputed_out: torch.Tensor | None,
    grid_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``y = rmsnorm(x * silu(z)) * weight``, normalized over groups of ``group_size`` columns.

    Launches mamba_ssm's ``_layer_norm_bwd_kernel`` with ``grid_scale`` times the row blocks mamba_ssm uses.
    The kernel reads every row as ``width`` consecutive elements. A ``dy`` laid out otherwise is copied first:
    the gradient of a reduction such as ``out.sum()`` arrives as one element expanded to the full shape.

    Args:
        dy: Gradient of the norm output, ``[rows, width]``, any strides.
        x: Scan output before the gate, ``[rows, width]``, consecutive elements within a row.
        weight: Norm weight, ``[width]``.
        eps: Norm epsilon.
        rstd: Reciprocal RMS saved by the forward, ``[(width // group_size) * rows]``: the values of one
            group for all rows are consecutive.
        z: Gate input, ``[rows, width]``, consecutive elements within a row.
        group_size: Number of columns per normalization group.
        dz: Receives the gradient of ``z``, ``[rows, width]``.
        recomputed_out: When given, receives the forward output of the norm, ``[rows, width]``.
        grid_scale: Multiplier on mamba_ssm's number of row blocks; at least 1.

    Returns:
        The gradient of ``x``, ``[rows, width]``, and the gradient of ``weight``, ``[width]``.

    Raises:
        ValueError: If a group is wider than the kernel supports, or ``x`` or ``z`` do not hold the elements
            of a row consecutively.
    """
    from mamba_ssm.ops.triton.layernorm_gated import _layer_norm_bwd_kernel

    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    if x.stride(-1) != 1 or z.stride(-1) != 1:
        raise ValueError(
            "gated RMSNorm backward needs the elements of a row to be consecutive in x and z, got strides "
            f"{tuple(x.stride())} and {tuple(z.stride())}"
        )
    rows, width = x.shape
    ngroups = width // group_size
    block_n = 1 << (group_size - 1).bit_length()
    if block_n > 65536 // x.element_size():
        raise ValueError(f"gated RMSNorm backward supports groups of up to 64 KB, got group_size={group_size}")
    num_warps = min(max(block_n // 256, 1), 8)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    # One program handles one group of `rows_per_program` consecutive rows and keeps its partial weight
    # gradient in registers; the partial sums are reduced below.
    row_blocks = min(math.ceil(sm_count * grid_scale * math.ceil(4 / num_warps) / ngroups), rows)
    rows_per_program = math.ceil(rows / row_blocks)
    dx = torch.empty_like(x)
    partial_dweight = torch.empty((row_blocks, width), dtype=torch.float32, device=weight.device)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[(row_blocks, ngroups)](
            x,
            weight,
            None,
            z,
            recomputed_out,
            dy,
            dx,
            partial_dweight,
            None,
            dz,
            None,
            rstd,
            x.stride(0),
            z.stride(0),
            0 if recomputed_out is None else recomputed_out.stride(0),
            dy.stride(0),
            dx.stride(0),
            dz.stride(0),
            partial_dweight.stride(0),
            0,
            rows,
            group_size,
            eps,
            rows_per_program,
            BLOCK_N=block_n,
            NORM_BEFORE_GATE=False,
            IS_RMS_NORM=True,
            num_warps=num_warps,
        )
    return dx, partial_dweight.sum(0).to(weight.dtype)


def _scan_backward(
    dout: torch.Tensor,
    zxbcdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    seq_idx: torch.Tensor | None,
    rmsnorm_weight: torch.Tensor,
    outproj_weight: torch.Tensor | None,
    outproj_bias: torch.Tensor | None,
    scan_out: torch.Tensor,
    rstd: torch.Tensor,
    settings: "_ScanSettings",
    norm_backward_grid_scale: int,
) -> tuple[torch.Tensor | None, ...]:
    """Backward of the fused scan from the tensors mamba_ssm's forward saves.

    ``dim = nheads * headdim`` and ``conv_dim = dim + 2 * ngroups * dstate`` below.

    Args:
        dout: Gradient of the scan's output: ``[batch, seqlen, dim]``, or ``[batch, seqlen, out_dim]`` with an
            output projection. Any strides.
        zxbcdt: Projected input, ``[batch, seqlen, dim + conv_dim + nheads]``.
        conv1d_weight: ``[conv_dim, kernel_size]``.
        conv1d_bias: ``[conv_dim]`` or None.
        dt_bias: ``[nheads]``.
        A: ``[nheads]``.
        D: ``[nheads]``.
        seq_idx: ``[batch, seqlen]`` int32 sub-sequence index, or None.
        rmsnorm_weight: Weight of the gated RMSNorm, ``[dim]``.
        outproj_weight: ``[out_dim, dim]`` when the output projection ran inside the scan, else None.
        outproj_bias: ``[out_dim]`` or None.
        scan_out: Scan output before the gate, ``[batch, seqlen, nheads, headdim]``.
        rstd: Reciprocal RMS saved by the norm's forward, ``[ngroups * batch * seqlen]``.
        settings: Scalar settings of the forward.
        norm_backward_grid_scale: Multiplier on mamba_ssm's number of row blocks for the norm backward kernel.

    Returns:
        Gradients of ``zxbcdt``, ``conv1d_weight``, ``conv1d_bias``, ``dt_bias``, ``A``, ``D``,
        ``rmsnorm_weight``, ``outproj_weight`` and ``outproj_bias``, each in the shape of its tensor; None for
        an absent ``conv1d_bias``, ``outproj_weight`` or ``outproj_bias``.
    """
    from mamba_ssm.ops.triton.ssd_combined import (
        _mamba_chunk_scan_combined_bwd,
        causal_conv1d_bwd_function,
        causal_conv1d_fwd_function,
        ensure_stride,
    )

    batch, seqlen, _ = zxbcdt.shape
    headdim, ngroups = settings.headdim, settings.ngroups
    nheads = D.shape[0]
    dim = nheads * headdim
    bc_dim = (conv1d_weight.shape[0] - dim) // 2
    dstate = bc_dim // ngroups
    conv_activation = settings.activation in ("silu", "swish")
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * bc_dim, nheads], dim=-1)

    # The conv output is not saved: rebuild x, B and C.
    xBC_conv = causal_conv1d_fwd_function(
        ensure_stride(xBC).permute(0, 2, 1), conv1d_weight, conv1d_bias, seq_idx, None, None, conv_activation
    ).permute(0, 2, 1)
    x, B, C = torch.split(xBC_conv, [dim, bc_dim, bc_dim], dim=-1)
    x = x.reshape(batch, seqlen, nheads, headdim)
    B = B.reshape(batch, seqlen, ngroups, dstate)
    C = C.reshape(batch, seqlen, ngroups, dstate)

    # Gradient buffers; the kernels below write into views of them.
    dzxbcdt = torch.empty_like(zxbcdt)
    dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [dim, dim + 2 * bc_dim, nheads], dim=-1)
    dxBC = torch.empty_like(xBC)
    dx, dB, dC = torch.split(dxBC, [dim, bc_dim, bc_dim], dim=-1)
    dx = dx.view(batch, seqlen, nheads, headdim)
    dB = dB.view(batch, seqlen, ngroups, dstate)
    dC = dC.view(batch, seqlen, ngroups, dstate)

    recomputed_out = None
    if outproj_weight is not None:
        dout_projected = dout
        dout = F.linear(dout, outproj_weight.t())
        recomputed_out = torch.empty(batch, seqlen, dim, device=scan_out.device, dtype=scan_out.dtype)

    dscan_out, drmsnorm_weight = _gated_rmsnorm_backward(
        dout.reshape(batch * seqlen, dim),
        scan_out.reshape(batch * seqlen, dim),
        rmsnorm_weight,
        settings.rmsnorm_eps,
        rstd,
        z.reshape(batch * seqlen, dim),
        group_size=dim // ngroups,
        dz=dz.view(batch * seqlen, dim),
        recomputed_out=None if recomputed_out is None else recomputed_out.view(batch * seqlen, dim),
        grid_scale=norm_backward_grid_scale,
    )
    _, _, dA, _, _, dD, _, ddt_bias, _ = _mamba_chunk_scan_combined_bwd(
        dscan_out.view(batch, seqlen, nheads, headdim),
        x,
        dt,
        A,
        B,
        C,
        scan_out,
        settings.chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        dfinal_states=None,
        seq_idx=seq_idx,
        dt_softplus=True,
        dt_limit=settings.dt_limit,
        dx=dx,
        ddt=ddt_given,
        dB=dB,
        dC=dC,
    )

    doutproj_weight, doutproj_bias = None, None
    if outproj_weight is not None:
        doutproj_weight = torch.einsum("bso,bsd->od", dout_projected, recomputed_out)
        doutproj_bias = dout_projected.sum(dim=(0, 1)) if outproj_bias is not None else None

    dxBC_given_update, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
        ensure_stride(xBC).permute(0, 2, 1),
        conv1d_weight,
        conv1d_bias,
        ensure_stride(dxBC).permute(0, 2, 1),
        seq_idx,
        None,
        None,
        ensure_stride(dxBC_given).permute(0, 2, 1),
        False,
        conv_activation,
    )
    dxBC_given_update = dxBC_given_update.permute(0, 2, 1)
    if dxBC_given.stride() != dxBC_given_update.stride():
        dxBC_given.copy_(dxBC_given_update)
    return (
        dzxbcdt,
        dconv1d_weight,
        dconv1d_bias,
        ddt_bias,
        dA,
        dD,
        drmsnorm_weight,
        doutproj_weight,
        doutproj_bias,
    )


@dataclass(frozen=True)
class _ScanSettings:
    """Scalar settings of one scan call."""

    chunk_size: int
    dt_limit: tuple[float, float]
    activation: str | None
    rmsnorm_eps: float
    headdim: int
    ngroups: int


def _library_forward(
    ctx: "torch.autograd.function.FunctionCtx | _KernelState",
    zxbcdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    seq_idx: torch.Tensor | None,
    rmsnorm_weight: torch.Tensor,
    outproj_weight: torch.Tensor | None,
    outproj_bias: torch.Tensor | None,
    settings: _ScanSettings,
) -> torch.Tensor:
    """mamba_ssm's fused forward.

    It hands ``ctx.save_for_backward`` the tuple (zxbcdt, conv1d_weight, conv1d_bias, scan output, A, D,
    dt_bias, None, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias).

    Args:
        ctx: Receives the tensors the backward needs through ``save_for_backward``.
        zxbcdt: Projected input, ``[batch, seqlen, dim + conv_dim + nheads]`` with ``dim = nheads * headdim``
            and ``conv_dim = dim + 2 * ngroups * dstate``.
        conv1d_weight: ``[conv_dim, kernel_size]``.
        conv1d_bias: ``[conv_dim]`` or None.
        dt_bias: ``[nheads]``.
        A: ``[nheads]``.
        D: ``[nheads]``.
        seq_idx: ``[batch, seqlen]`` int32 sub-sequence index, or None.
        rmsnorm_weight: Weight of the gated RMSNorm, ``[dim]``.
        outproj_weight: ``[out_dim, dim]`` to apply the output projection inside, or None.
        outproj_bias: ``[out_dim]`` or None.
        settings: Scalar settings of the scan.

    Returns:
        ``[batch, seqlen, dim]``, or ``[batch, seqlen, out_dim]`` with an output projection.
    """
    from mamba_ssm.ops.triton.ssd_combined import MambaSplitConv1dScanCombinedFn

    return MambaSplitConv1dScanCombinedFn.forward(
        ctx,
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        settings.chunk_size,
        initial_states=None,
        seq_idx=seq_idx,
        dt_limit=settings.dt_limit,
        return_final_states=False,
        activation=settings.activation,
        rmsnorm_weight=rmsnorm_weight,
        rmsnorm_eps=settings.rmsnorm_eps,
        outproj_weight=outproj_weight,
        outproj_bias=outproj_bias,
        headdim=settings.headdim,
        ngroups=settings.ngroups,
        norm_before_gate=False,
    )


class _SplitConv1dScanFn(torch.autograd.Function):
    """mamba_ssm's fused conv1d + SSD scan + gated RMSNorm (+ output projection) with the norm backward launched here."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        zxbcdt: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        dt_bias: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        seq_idx: torch.Tensor | None,
        rmsnorm_weight: torch.Tensor,
        outproj_weight: torch.Tensor | None,
        outproj_bias: torch.Tensor | None,
        settings: _ScanSettings,
        norm_backward_grid_scale: int,
    ) -> torch.Tensor:
        """Run mamba_ssm's fused forward and keep the settings for the backward.

        Args:
            ctx: Autograd context; mamba_ssm's forward saves its tensors on it.
            zxbcdt: Projected input, ``[batch, seqlen, dim + conv_dim + nheads]``.
            conv1d_weight: ``[conv_dim, kernel_size]``.
            conv1d_bias: ``[conv_dim]`` or None.
            dt_bias: ``[nheads]``.
            A: ``[nheads]``.
            D: ``[nheads]``.
            seq_idx: ``[batch, seqlen]`` int32, or None.
            rmsnorm_weight: ``[dim]``.
            outproj_weight: ``[out_dim, dim]`` or None.
            outproj_bias: ``[out_dim]`` or None.
            settings: Scalar settings of the scan.
            norm_backward_grid_scale: Multiplier on the row blocks of the norm backward kernel.

        Returns:
            ``[batch, seqlen, dim]``, or ``[batch, seqlen, out_dim]`` with an output projection.
        """
        ctx.settings = settings
        ctx.norm_backward_grid_scale = norm_backward_grid_scale
        return _library_forward(
            ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, seq_idx, rmsnorm_weight, outproj_weight,
            outproj_bias, settings,
        )  # fmt: skip

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: torch.autograd.function.FunctionCtx, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Compute every gradient with :func:`_scan_backward`.

        Args:
            ctx: Autograd context of the forward.
            dout: Gradient of the forward's output, in its shape (``[batch, seqlen, dim]`` or
                ``[batch, seqlen, out_dim]``), any strides.

        Returns:
            One entry per forward argument: the gradients of the nine differentiable tensors in the shape of
            their tensor, None for ``seq_idx`` and the two settings.
        """
        (
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            scan_out,
            A,
            D,
            dt_bias,
            _,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        ) = ctx.saved_tensors
        dzxbcdt, dconv1d_weight, dconv1d_bias, ddt_bias, dA, dD, drmsnorm_weight, doutproj_weight, doutproj_bias = (
            _scan_backward(
                dout,
                zxbcdt,
                conv1d_weight,
                conv1d_bias,
                dt_bias,
                A,
                D,
                seq_idx,
                rmsnorm_weight,
                outproj_weight,
                outproj_bias,
                scan_out,
                rstd,
                ctx.settings,
                ctx.norm_backward_grid_scale,
            )  # fmt: skip
        )
        return (
            dzxbcdt,
            dconv1d_weight,
            dconv1d_bias,
            ddt_bias,
            dA,
            dD,
            None,
            drmsnorm_weight,
            doutproj_weight,
            doutproj_bias,
            None,
            None,
        )


class _KernelState:
    """What the kernel node of a split scan keeps from its forward for the backward of the inputs node.

    It stands in for the autograd context in mamba_ssm's forward, so the tensors that forward would save are
    received here instead of entering autograd's saved tensors.
    """

    def __init__(self) -> None:
        self.scan_out: torch.Tensor | None = None
        self.rstd: torch.Tensor | None = None
        self.dout: torch.Tensor | None = None
        # The autocast state of the forward; the backward of the inputs node runs in it, as the backward of
        # the one-node scan does.
        self.used_autocast = False
        self.autocast_dtype = torch.float16

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None:
        """Keep the scan output and the norm statistics; the inputs node saves the rest."""
        self.scan_out, self.rstd = tensors[3], tensors[10]


class _ScanInputsFn(torch.autograd.Function):
    """Inputs node of a split scan: saves what the scan reads and computes every gradient in its backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        zxbcdt: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        dt_bias: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        seq_idx: torch.Tensor | None,
        rmsnorm_weight: torch.Tensor,
        outproj_weight: torch.Tensor | None,
        outproj_bias: torch.Tensor | None,
        settings: _ScanSettings,
        norm_backward_grid_scale: int,
        state: _KernelState,
    ) -> torch.Tensor:
        """Save the tensors the scan reads and pass the projected input on.

        Args:
            ctx: Autograd context; receives the ten tensors below through ``save_for_backward``.
            zxbcdt: Projected input, ``[batch, seqlen, dim + conv_dim + nheads]``.
            conv1d_weight: ``[conv_dim, kernel_size]``.
            conv1d_bias: ``[conv_dim]`` or None.
            dt_bias: ``[nheads]``.
            A: ``[nheads]``.
            D: ``[nheads]``.
            seq_idx: ``[batch, seqlen]`` int32, or None.
            rmsnorm_weight: ``[dim]``.
            outproj_weight: ``[out_dim, dim]`` or None.
            outproj_bias: ``[out_dim]`` or None.
            settings: Scalar settings of the scan.
            norm_backward_grid_scale: Multiplier on the row blocks of the norm backward kernel.
            state: Shared with the kernel node, which fills it.

        Returns:
            A view of ``zxbcdt``, same shape, for the kernel node to read.
        """
        ctx.save_for_backward(
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, seq_idx, rmsnorm_weight, outproj_weight, outproj_bias
        )
        ctx.settings = settings
        ctx.norm_backward_grid_scale = norm_backward_grid_scale
        ctx.state = state
        return zxbcdt.view_as(zxbcdt)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, placeholder: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Compute every gradient from the output gradient the kernel node left in the shared state.

        Args:
            ctx: Autograd context of the forward.
            placeholder: The zero the kernel node returns for the projected input, ``[batch, seqlen,
                dim + conv_dim + nheads]`` expanded from one element; its values are not read.

        Returns:
            One entry per forward argument: the gradients of the nine differentiable tensors in the shape of
            their tensor, None for ``seq_idx``, the two settings and the state.
        """
        del placeholder
        state = ctx.state
        if state.dout is None or state.scan_out is None or state.rstd is None:
            raise RuntimeError("the kernel node of a split Mamba scan did not run before its inputs node")
        (
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D,
            seq_idx,
            rmsnorm_weight,
            outproj_weight,
            outproj_bias,
        ) = ctx.saved_tensors
        with torch.autocast("cuda", enabled=state.used_autocast, dtype=state.autocast_dtype):
            dzxbcdt, dconv1d_weight, dconv1d_bias, ddt_bias, dA, dD, drmsnorm_weight, doutproj_weight, doutproj_bias = (
                _scan_backward(
                    state.dout,
                    zxbcdt,
                    conv1d_weight,
                    conv1d_bias,
                    dt_bias,
                    A,
                    D,
                    seq_idx,
                    rmsnorm_weight,
                    outproj_weight,
                    outproj_bias,
                    state.scan_out,
                    state.rstd,
                    ctx.settings,
                    ctx.norm_backward_grid_scale,
                )  # fmt: skip
            )
        state.dout = None
        return (
            dzxbcdt,
            dconv1d_weight,
            dconv1d_bias,
            ddt_bias,
            dA,
            dD,
            None,
            drmsnorm_weight,
            doutproj_weight,
            doutproj_bias,
            None,
            None,
            None,
        )


class _ScanKernelFn(torch.autograd.Function):
    """Kernel node of a split scan: runs the forward and hands the output gradient to the inputs node."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        zxbcdt: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        dt_bias: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        seq_idx: torch.Tensor | None,
        rmsnorm_weight: torch.Tensor,
        outproj_weight: torch.Tensor | None,
        outproj_bias: torch.Tensor | None,
        settings: _ScanSettings,
        state: _KernelState,
    ) -> torch.Tensor:
        """Run mamba_ssm's fused forward; ``state`` receives what it would save for the backward.

        Args:
            ctx: Autograd context; keeps ``state`` and the shape of ``zxbcdt``.
            zxbcdt: Output of the inputs node, ``[batch, seqlen, dim + conv_dim + nheads]``.
            conv1d_weight: ``[conv_dim, kernel_size]``, detached.
            conv1d_bias: ``[conv_dim]`` or None, detached.
            dt_bias: ``[nheads]``, detached.
            A: ``[nheads]``, detached.
            D: ``[nheads]``, detached.
            seq_idx: ``[batch, seqlen]`` int32, or None.
            rmsnorm_weight: ``[dim]``, detached.
            outproj_weight: ``[out_dim, dim]`` or None, detached.
            outproj_bias: ``[out_dim]`` or None, detached.
            settings: Scalar settings of the scan.
            state: Receives the scan output, the norm statistics and the autocast state.

        Returns:
            ``[batch, seqlen, dim]``, or ``[batch, seqlen, out_dim]`` with an output projection.
        """
        state.used_autocast = torch.is_autocast_enabled("cuda")
        state.autocast_dtype = torch.get_autocast_dtype("cuda")
        ctx.state = state
        ctx.input_shape = zxbcdt.shape
        return _library_forward(
            state, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, seq_idx, rmsnorm_weight, outproj_weight,
            outproj_bias, settings,
        )  # fmt: skip

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Hand the output gradient to the inputs node.

        Args:
            ctx: Autograd context of the forward.
            dout: Gradient of the forward's output, in its shape, any strides.

        Returns:
            A zero in the shape of ``zxbcdt`` for the inputs node and None for the eleven other arguments.
        """
        ctx.state.dout = dout
        # The inputs node computes every gradient. It runs only if this node passes it a gradient, so pass a
        # zero of the input's shape that occupies one element.
        return (dout.new_zeros(()).expand(ctx.input_shape), *([None] * 11))


def split_conv1d_scan(
    zxbcdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    *,
    chunk_size: int,
    seq_idx: torch.Tensor | None,
    dt_limit: tuple[float, float],
    activation: str | None,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    outproj_weight: torch.Tensor | None,
    outproj_bias: torch.Tensor | None,
    headdim: int,
    ngroups: int,
    norm_backward_grid_scale: int,
    checkpoint_exempt: bool,
) -> torch.Tensor:
    """Fused Mamba2 training scan: causal conv1d, SSD scan, ``rmsnorm(x * silu(z))`` and an optional projection.

    Args:
        zxbcdt: Input projection output, ``[batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads]`` with
            ``dim = nheads * headdim``.
        conv1d_weight: ``[dim + 2 * ngroups * dstate, kernel_size]``.
        conv1d_bias: ``[dim + 2 * ngroups * dstate]`` or None.
        dt_bias: ``[nheads]``.
        A: ``[nheads]``.
        D: ``[nheads]``.
        chunk_size: Chunk length of the SSD scan.
        seq_idx: ``[batch, seqlen]`` int32 sub-sequence index, or None.
        dt_limit: Clamp range of the time step.
        activation: Conv activation, ``"silu"``, ``"swish"`` or None.
        rmsnorm_weight: Weight of the gated RMSNorm, ``[dim]``.
        rmsnorm_eps: Epsilon of the gated RMSNorm.
        outproj_weight: ``[out_dim, dim]`` to apply the output projection inside, or None.
        outproj_bias: ``[out_dim]`` or None; requires ``outproj_weight``.
        headdim: Head dimension.
        ngroups: Number of B / C groups, which is also the number of normalization groups.
        norm_backward_grid_scale: Multiplier on mamba_ssm's number of row blocks for the gated-RMSNorm
            backward kernel; at least 1.
        checkpoint_exempt: Build the scan from an inputs node and a kernel node, so that an enclosing
            non-reentrant activation checkpoint recomputes the projected input only and the scan output
            and norm statistics are kept from the forward.

    Returns:
        ``[batch, seqlen, dim]``, or ``[batch, seqlen, out_dim]`` with an output projection.
    """
    if norm_backward_grid_scale < 1:
        raise ValueError(f"norm_backward_grid_scale must be at least 1, got {norm_backward_grid_scale}")
    settings = _ScanSettings(
        chunk_size=chunk_size,
        dt_limit=dt_limit,
        activation=activation,
        rmsnorm_eps=rmsnorm_eps,
        headdim=headdim,
        ngroups=ngroups,
    )
    tensors = (zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, seq_idx, rmsnorm_weight, outproj_weight, outproj_bias)
    if not checkpoint_exempt:
        return _SplitConv1dScanFn.apply(*tensors, settings, norm_backward_grid_scale)
    state = _KernelState()
    tracked = _ScanInputsFn.apply(*tensors, settings, norm_backward_grid_scale, state)
    # The parameters reach autograd through the inputs node; the kernel node reads their values only.
    return _ScanKernelFn.apply(tracked, *(None if t is None else t.detach() for t in tensors[1:]), settings, state)
