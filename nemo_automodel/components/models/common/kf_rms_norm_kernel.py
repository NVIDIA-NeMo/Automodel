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

"""Generated-kernel slot for the ``moonlight_v4_rmsnorm_h2048`` Kernel Factory campaign.

This module holds one thing: the forward RMSNorm kernel for
``examples/scalable_ai/kernel_factory/moonlight-rmsnorm``. It is the only file that
the campaign's output replaces, and it is deliberately kept apart from
``kf_triton_rms_norm.py``, which owns the autograd, DTensor and checkpoint
contracts and must not be regenerated.

To install a campaign winner::

    kf campaign results moonlight-v4-rmsnorm --output-dir ./winner
    # copy ./winner/<solution>/kernel.py over the `run` implementation below

The kernel currently in this slot is a straightforward one-row-per-program Triton
RMSNorm written by hand. It exists so the backend is runnable and testable before
the campaign returns; it is a placeholder for the search result, not a tuned
kernel, and it is not expected to beat ``rms_norm="torch_fp32"``.

Entry-point contract, fixed by ``definition.json`` (destination-passing style):

    run(x, weight, eps, y) -> None

``x`` and ``y`` are ``[tokens, 2048]`` bf16, row-major with unit stride on the
hidden axis; ``y`` does not alias ``x``. ``weight`` is ``[2048]`` bf16. The mean
square, the reciprocal square root and the weight multiply happen in fp32, with a
single rounding to bf16 on the store.
"""

import torch

from nemo_automodel.shared.import_utils import safe_import

_MISSING_TRITON = (
    "rms_norm='kf_triton_h2048' requires Triton, which ships with the CUDA builds of PyTorch. "
    "Install nemo-automodel[cuda] or use rms_norm='torch_fp32'."
)

HAVE_TRITON, triton = safe_import("triton", msg=_MISSING_TRITON)
_, tl = safe_import("triton.language", msg=_MISSING_TRITON)

if HAVE_TRITON:

    @triton.jit
    def _kf_rms_norm_fwd_kernel(
        x_ptr,
        weight_ptr,
        y_ptr,
        x_row_stride,
        y_row_stride,
        hidden,
        eps,
        BLOCK: tl.constexpr,
    ):
        """Normalize one row per program.

        Args:
            x_ptr: Pointer to the bf16 input of shape [tokens, hidden], unit stride on hidden.
            weight_ptr: Pointer to the bf16 gain of shape [hidden], unit stride.
            y_ptr: Pointer to the bf16 output of shape [tokens, hidden], unit stride on hidden.
            x_row_stride: Element stride between consecutive rows of x.
            y_row_stride: Element stride between consecutive rows of y.
            hidden: Length of the normalized axis.
            eps: Epsilon added to the mean square.
            BLOCK: Power-of-two tile covering the whole hidden axis.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < hidden
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / hidden
        rstd = 1.0 / tl.sqrt(mean_square + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * weight
        tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, weight: torch.Tensor, eps: float, y: torch.Tensor) -> None:
    """Write the RMSNorm of ``x`` into ``y``.

    Args:
        x: Tensor of shape [tokens, hidden] in bf16, row-major with unit stride on hidden.
        weight: Tensor of shape [hidden] in bf16, contiguous.
        eps: Epsilon added to the mean square.
        y: Output tensor of shape [tokens, hidden] in bf16, row-major with unit stride on
            hidden. Written in place and must not alias ``x``.

    Raises:
        ImportError: If Triton is not installed.
    """
    if not HAVE_TRITON:
        raise ImportError(_MISSING_TRITON)
    tokens, hidden = x.shape
    if tokens == 0:
        return
    block = triton.next_power_of_2(hidden)
    _kf_rms_norm_fwd_kernel[(tokens,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        hidden,
        eps,
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
    )
