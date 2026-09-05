# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-tensor gradient-norm reductions.

Gradient clipping over a sharded MoE model reduces thousands of separate
gradient tensors. Doing that with per-tensor PyTorch ops costs a handful of
kernel launches per parameter -- on DiffusionGemma-26B-A4B that measured 5,434
launches and 74 ms per step, more launches than the entire MoE GEMM path -- so
the reduction is launch-bound rather than bandwidth-bound.

These kernels reduce every tensor in one launch per dtype, and are explicit
about the two things that are easy to get wrong in a norm reduction:

**Precision.** Values are widened to fp32 on load, so a BF16 gradient is never
squared in BF16 (|g| > 256 overflows BF16's range once squared, and the
mantissa loses most of its bits well before that). Within a tile the reduction
is a ``tl.sum`` tree reduction carried out **in fp64** (reducing the tile in
fp32 and widening only the tile total leaves fp32-level error, ~1e-7 relative,
which is what this kernel exists to avoid); across tiles each program
accumulates into an fp64 scalar; across programs the fp64 partials are summed
by PyTorch. Accumulation error therefore stays at fp64 level regardless of how
many elements a parameter has. ``max|x|`` accumulates nothing and stays in
fp32, where it is already exact.

**Determinism.** The decomposition into chunks is computed on the host and is a
pure function of the tensor sizes, and no atomics are used, so the result is
bit-identical run to run for a given parameter layout. An atomic-accumulate
version would be faster to write but would make the gradient norm -- and hence
every subsequent optimizer step -- non-reproducible.

Widening on load also makes the usual "divide by the max first, then square"
overflow dance unnecessary, so the 2-norm needs a single pass over the
gradients instead of two.
"""

from typing import List, Sequence

import torch

from nemo_automodel.shared.import_utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TRITON = False
    triton = None

    class _TritonStub:
        jit = staticmethod(null_decorator)

        def __getattr__(self, _name):
            raise RuntimeError("Triton is not available")

    tl = _TritonStub()


# Elements handled by one program. Large enough that launch overhead is
# amortized, small enough that a big parameter is split across programs (which
# also bounds the fp32-within-tile error before the fp64 accumulate).
_CHUNK = 8192
_BLOCK = 1024

_REDUCE_SUMSQ = 0
_REDUCE_ABSMAX = 1


@triton.jit
def _multi_tensor_reduce_kernel(
    ptrs_ptr,  # int64[num_tensors]  -- data_ptr() of each tensor
    numel_ptr,  # int64[num_tensors]
    chunk_tensor_ptr,  # int32[num_chunks] -- which tensor this program reduces
    chunk_start_ptr,  # int64[num_chunks] -- element offset within that tensor
    partial_ptr,  # float64[num_chunks] -- one partial per program
    REDUCE_OP: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reduce one chunk of one tensor into a single fp64 partial."""
    pid = tl.program_id(0)

    tensor_idx = tl.load(chunk_tensor_ptr + pid)
    start = tl.load(chunk_start_ptr + pid)
    numel = tl.load(numel_ptr + tensor_idx)
    base_addr = tl.load(ptrs_ptr + tensor_idx)

    # Re-materialize the typed pointer from the raw address. Specializing on
    # DTYPE_ID keeps this a compile-time choice.
    if DTYPE_ID == 0:
        base = base_addr.to(tl.pointer_type(tl.bfloat16))
    elif DTYPE_ID == 1:
        base = base_addr.to(tl.pointer_type(tl.float16))
    elif DTYPE_ID == 2:
        base = base_addr.to(tl.pointer_type(tl.float32))
    else:
        base = base_addr.to(tl.pointer_type(tl.float64))

    end = tl.minimum(start + CHUNK, numel)

    acc = tl.zeros([], dtype=tl.float64)
    nan_count = tl.zeros([], dtype=tl.float64)
    for tile in range(0, CHUNK, BLOCK):
        offs = start + tile + tl.arange(0, BLOCK)
        mask = offs < end
        # Widen the BLOCK-sized tile *in registers* -- never a whole tensor.
        # Squaring in the storage dtype would overflow BF16 above |g| ~ 256 and
        # lose most of the mantissa well before that, but materializing a
        # widened copy of the gradient would cost more memory than the
        # reduction saves in launches.
        # Literals, not the module constants: a @triton.jit body may only read
        # globals declared as tl.constexpr. 0 == _REDUCE_SUMSQ, 1 == _REDUCE_ABSMAX.
        if REDUCE_OP == 0:
            # fp64 for the square and the in-tile tree reduction, not just for
            # the tile total: reducing a 1024-element tile in fp32 and widening
            # afterwards leaves fp32-level error (~1e-7 relative), which defeats
            # the point of accumulating across tiles in fp64.
            # NaN/Inf need no special handling here: NaN*NaN and Inf*Inf
            # propagate through fp64 addition, so a non-finite gradient reaches
            # the caller's error_if_nonfinite check intact.
            x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float64)
            acc += tl.sum(x * x)
        else:
            # max|x| accumulates nothing, so fp32 is already exact here.
            x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.max(tl.abs(x)).to(tl.float64))
            # tl.maximum follows IEEE maxNum, which *ignores* NaN and returns
            # the other operand -- so an inf-norm over NaN gradients would come
            # back finite and silently defeat error_if_nonfinite. Count NaNs
            # explicitly (x != x) and poison the partial below.
            nan_count += tl.sum((x != x).to(tl.float64))

    if REDUCE_OP == 1:
        acc = tl.where(nan_count > 0.0, float("nan"), acc)
    tl.store(partial_ptr + pid, acc)


_DTYPE_IDS = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


def _build_chunk_table(tensors: Sequence[torch.Tensor], device: torch.device):
    """Host-side, size-only decomposition -- deterministic for a given layout."""
    tensor_ids: List[int] = []
    starts: List[int] = []
    for i, t in enumerate(tensors):
        n = t.numel()
        for s in range(0, n, _CHUNK):
            tensor_ids.append(i)
            starts.append(s)
    return (
        torch.tensor(tensor_ids, dtype=torch.int32, device=device),
        torch.tensor(starts, dtype=torch.int64, device=device),
    )


def _kernel_eligible(t: torch.Tensor) -> bool:
    """Whether the kernel can read ``t`` directly.

    Excluded, each routed to the reference path rather than converted:
    * non-CUDA -- the kernel is CUDA-only;
    * non-contiguous -- the flat element offset would be wrong, and
      ``.contiguous()`` would allocate a full-size copy of the gradient;
    * dtypes outside ``_DTYPE_IDS`` (notably the FP8 formats, whose reductions
      belong in a scaled path rather than a raw square anyway).
    """
    return t.is_cuda and t.is_contiguous() and t.dtype in _DTYPE_IDS


def _reduce_one_dtype(tensors: Sequence[torch.Tensor], reduce_op: int, device, dtype) -> torch.Tensor:
    """Launch the kernel once for a set of same-dtype, kernel-eligible tensors."""
    ptrs = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=device)
    numels = torch.tensor([t.numel() for t in tensors], dtype=torch.int64, device=device)
    chunk_tensor, chunk_start = _build_chunk_table(tensors, device)
    num_chunks = chunk_tensor.numel()

    partials = torch.empty(num_chunks, dtype=torch.float64, device=device)
    _multi_tensor_reduce_kernel[(num_chunks,)](
        ptrs,
        numels,
        chunk_tensor,
        chunk_start,
        partials,
        REDUCE_OP=reduce_op,
        DTYPE_ID=_DTYPE_IDS[dtype],
        CHUNK=_CHUNK,
        BLOCK=_BLOCK,
    )
    # fp64 partials, tree-reduced by PyTorch.
    return partials.sum() if reduce_op == _REDUCE_SUMSQ else partials.max()


def _reduce(tensors: Sequence[torch.Tensor], reduce_op: int) -> torch.Tensor:
    """Reduce ``tensors`` to an fp64 scalar.

    ``tensors`` are plain local tensors: DTensor gradients must already be
    unwrapped by the caller (``to_local()``/``full_tensor()``), which also owns
    the cross-rank reduction of the scalar this returns.
    """
    tensors = [t for t in tensors if t.numel() > 0]
    if not tensors:
        return torch.zeros((), dtype=torch.float64, device="cpu")

    device = next((t.device for t in tensors if t.is_cuda), tensors[0].device)

    # One launch per dtype: the kernel reinterprets a raw address with a single
    # compile-time element type, so a mixed-dtype batch would read (say) BF16
    # storage as FP32.
    by_dtype: dict = {}
    ineligible: List[torch.Tensor] = []
    for t in tensors:
        if _kernel_eligible(t):
            by_dtype.setdefault(t.dtype, []).append(t)
        else:
            ineligible.append(t)

    combine = torch.add if reduce_op == _REDUCE_SUMSQ else torch.maximum
    result = torch.zeros((), dtype=torch.float64, device=device)

    for dtype, group in by_dtype.items():
        result = combine(result, _reduce_one_dtype(group, reduce_op, device, dtype))

    if ineligible:
        extra = sumsq_reference(ineligible) if reduce_op == _REDUCE_SUMSQ else absmax_reference(ineligible)
        result = combine(result, extra.to(device))
    return result


def multi_tensor_sumsq(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum of squares of every element, accumulated in fp64."""
    return _reduce(tensors, _REDUCE_SUMSQ)


def multi_tensor_absmax(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Largest absolute value across every element, in fp64."""
    return _reduce(tensors, _REDUCE_ABSMAX)


# Elements per slice in the fallback path. Bounds the widened temporary to a
# few MiB instead of allocating an fp32 copy of an entire gradient.
_FALLBACK_SLICE = 1 << 20


def sumsq_reference(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pure-PyTorch equivalent of :func:`multi_tensor_sumsq` (fallback / tests).

    Widens a slice at a time rather than the whole gradient: ``t.to(float64)``
    on a full tensor would allocate a copy 2-8x the size of the gradient for
    every parameter, which on a model large enough to need this kernel is a
    materially worse problem than the launch overhead it set out to fix. One
    slice is 1Mi elements, so the widened temporary stays at 8 MiB.
    """
    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for t in tensors:
        if not t.numel():
            continue
        flat = t.detach().reshape(-1)
        for start in range(0, flat.numel(), _FALLBACK_SLICE):
            # fp64 for the square as well as the sum: squaring in fp32 rounds
            # every term (~1e-7 relative), which would make this a weaker
            # reference than the kernel it is meant to check.
            chunk = flat[start : start + _FALLBACK_SLICE].to(torch.float64)
            total = total + chunk.square().sum().cpu()
    return total


def absmax_reference(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pure-PyTorch equivalent of :func:`multi_tensor_absmax` (fallback / tests).

    ``max(|x|)`` needs no widening at all -- it is a pure reduction, exact in
    any float dtype -- and is taken as ``max(|max|, |min|)`` so that no ``abs()``
    temporary is materialized either.
    """
    best = torch.zeros((), dtype=torch.float64, device="cpu")
    for t in tensors:
        if not t.numel():
            continue
        d = t.detach()
        hi = d.max().to(torch.float64).cpu()
        lo = d.min().to(torch.float64).cpu()
        best = torch.maximum(best, torch.maximum(hi.abs(), lo.abs()))
    return best
