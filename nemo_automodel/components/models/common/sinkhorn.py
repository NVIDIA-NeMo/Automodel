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

"""Sinkhorn normalization of hyper-connection (mHC) mixing logits.

The torch implementation is the numerical reference. The optional CUDA path
uses DeepSeek TileKernels ``tile_kernels.modeling.mhc.ops.sinkhorn_normalize``
and its low-level kernels; no TileKernels source is vendored. Upstream source:
https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/modeling/mhc/ops/sinkhorn.py
Upstream license: MIT, copyright 2026 DeepSeek.
"""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import_from

_HAS_TILE_KERNELS_SINKHORN, _tile_kernels_sinkhorn = safe_import_from(
    "tile_kernels.modeling.mhc.ops",
    "sinkhorn_normalize",
    msg="TileKernels sinkhorn is unavailable. Install tile_kernels and tilelang to use the TileKernels Sinkhorn.",
)
_HAS_TILE_KERNELS_SINKHORN_FWD, _tile_kernels_sinkhorn_fwd = safe_import_from(
    "tile_kernels.mhc.sinkhorn_kernel",
    "_mhc_sinkhorn_fwd",
    msg="TileKernels low-level sinkhorn forward kernel is unavailable.",
)
_HAS_TILE_KERNELS_SINKHORN_BWD, _tile_kernels_sinkhorn_bwd = safe_import_from(
    "tile_kernels.mhc.sinkhorn_kernel",
    "_mhc_sinkhorn_bwd",
    msg="TileKernels low-level sinkhorn backward kernel is unavailable.",
)


def is_tile_kernels_sinkhorn_available() -> bool:
    """Return whether the TileKernels Sinkhorn wrapper and its forward/backward kernels are importable."""
    return _HAS_TILE_KERNELS_SINKHORN and _HAS_TILE_KERNELS_SINKHORN_FWD and _HAS_TILE_KERNELS_SINKHORN_BWD


def sinkhorn_normalize_torch(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Normalize mixing logits into an approximately doubly stochastic matrix.

    Args:
        x: Logits of shape [..., streams, streams], with arbitrary leading dimensions.
        repeat: Number of row/column normalization rounds; the first round starts from a softmax.
        eps: Additive stabilizer applied to every normalization denominator.

    Returns:
        Tensor with the shape and dtype of ``x``.
    """
    x = x.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


class _TileKernelsSinkhorn(torch.autograd.Function):
    """TileKernels Sinkhorn wrapper that accepts non-contiguous backward gradients.

    The upstream high-level wrapper launches the backward kernel with
    ``grad_output`` as-is. Hyper-connection combinations are consumed through
    transposed matmul sites, so autograd can provide a transposed gradient
    layout. The low-level TileKernels backward kernel requires contiguous
    row-major inputs.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        repeat: int,
        eps: float,
    ) -> torch.Tensor:
        """Run the TileKernels forward kernel.

        Args:
            ctx: Autograd context.
            x: CUDA logits of shape [..., streams, streams], with arbitrary leading dimensions.
            repeat: Number of normalization rounds.
            eps: Additive stabilizer.

        Returns:
            Tensor with the shape of ``x``, independent of its storage.
        """
        flat_x = x.contiguous().view(-1, *x.shape[-2:])
        hidden_size = flat_x.shape[1]
        flat_output = torch.empty_like(flat_x)
        fwd_kernel = _tile_kernels_sinkhorn_fwd(hidden_size, 1, repeat, eps)
        bwd_kernel = _tile_kernels_sinkhorn_bwd(hidden_size, 32, repeat, eps)
        fwd_kernel(flat_x, flat_output)
        ctx.save_for_backward(flat_x)
        ctx.bwd_kernel = bwd_kernel
        ctx.input_shape = x.shape
        return flat_output.view_as(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Run the TileKernels backward kernel on a contiguous copy of the gradient.

        Args:
            ctx: Autograd context.
            grad_output: Gradient of shape [..., streams, streams], any strides.

        Returns:
            Input gradient with the forward input's shape, then ``None`` for ``repeat`` and ``eps``.
        """
        (flat_x,) = ctx.saved_tensors
        flat_grad_output = grad_output.contiguous().view_as(flat_x)
        flat_grad_input = torch.empty_like(flat_x)
        ctx.bwd_kernel(flat_grad_output, flat_x, flat_grad_input)
        return flat_grad_input.view(ctx.input_shape), None, None


def sinkhorn_normalize_tile_kernels(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Run the TileKernels Sinkhorn; never falls back to the torch path.

    Args:
        x: CUDA logits of shape [..., streams, streams], with arbitrary leading dimensions.
        repeat: Number of normalization rounds.
        eps: Additive stabilizer.

    Returns:
        Tensor with the shape of ``x``.

    Raises:
        RuntimeError: If TileKernels is not importable or ``x`` is not a CUDA tensor.
    """
    if not is_tile_kernels_sinkhorn_available() or not x.is_cuda:
        raise RuntimeError(
            "TileKernels Sinkhorn was requested, but the optional kernel is unavailable or the input is not on CUDA."
        )
    return _TileKernelsSinkhorn.apply(x, repeat, eps)
