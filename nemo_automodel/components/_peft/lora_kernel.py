# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import TypedDict
from unittest.mock import MagicMock

import torch
from packaging import version

from nemo_automodel.shared.import_utils import MISSING_TRITON_MSG, null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("2.0.0") or (
        version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available()
    ):
        raise ImportError("Installed Triton version is unsupported on this device")
    # Python launchers validate availability; JIT kernels must not read this runtime bool.
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


class _GridMeta(TypedDict):
    """Integer block sizes read from the full Triton launch-argument mapping."""

    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int


def forward_autotune_configs() -> list[triton.Config]:
    """
    Method for generating Triton configs for lora_forward_kernel.
    """
    out = list()
    for blk_m in [16, 32, 64]:
        for blk_k in [128, 256, 512]:
            for blk_l in [128, 256, 512]:
                out.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": blk_m, "BLOCK_SIZE_K": blk_k, "BLOCK_SIZE_L": blk_l, "GROUP_SIZE_M": 8},
                        num_stages=4,
                        num_warps=4,
                    )
                )
    return out


@triton.jit
def get_pid_coords(
    M: tl.tensor, N: tl.tensor, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr
) -> tuple[tl.tensor, tl.tensor]:
    """
    Converts one-dimensional triton pids into two dimensions.
    """
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def inner_kernel(
    pid_m: tl.tensor,
    pid_n: tl.tensor,
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    M: tl.tensor,
    K: tl.tensor,
    N: tl.tensor,
    stride_am: tl.tensor,
    stride_ak: tl.tensor,
    stride_bk: tl.tensor,
    stride_bn: tl.tensor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    scale: tl.tensor,
) -> tl.tensor:
    """
    Performs the matrix multiplication AB.

    A is an M x K matrix and B is an N x K matrix.
    The result is returned to be stored by the calling method.
    """
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        ab += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    return scale * ab


@triton.jit
def block_vector_mul(
    pid_m: tl.tensor,
    pid_n: tl.tensor,
    ab_result: tl.tensor,
    c_ptr: tl.tensor,
    d_ptr: tl.tensor,
    M: tl.tensor,
    N: tl.tensor,
    L: tl.tensor,
    stride_cn: tl.tensor,
    stride_cl: tl.tensor,
    stride_dm: tl.tensor,
    stride_dl: tl.tensor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
) -> None:
    """
    Multiplies an M x N vector AB and and N x L vector C and adds the result to the output vector D.

    N is assumed to be smaller than BLOCK_SIZE_N.
    """
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_dm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    c_ptrs = c_ptr + (offs_cn[:, None] * stride_cn + offs_l[None, :] * stride_cl)
    d_ptrs = d_ptr + stride_dm * offs_dm[:, None] + stride_dl * offs_l[None, :]
    d_mask = (offs_dm[:, None] < M) & (offs_l[None, :] < L)
    c_mask = (offs_cn[:, None] < N) & (offs_l[None, :] < L)

    for lx in tl.range(0, tl.cdiv(L, BLOCK_SIZE_L)):
        d_mask = (offs_dm[:, None] < M) & (offs_l[None, :] < L - lx * BLOCK_SIZE_L)
        c_mask = (offs_cn[:, None] < N) & (offs_l[None, :] < L - lx * BLOCK_SIZE_L)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        abc = tl.dot(ab_result, c)
        tl.store(d_ptrs, abc, mask=d_mask)
        c_ptrs += BLOCK_SIZE_L * stride_cl
        d_ptrs += BLOCK_SIZE_L * stride_dl


@triton.autotune(
    configs=forward_autotune_configs(),
    key=["N", "K", "L"],
)
# This optimization exploits that N is the LoRA dimension and thus we only need one block.
@triton.heuristics(values={"BLOCK_SIZE_N": lambda args: max(triton.next_power_of_2(args["N"]), 16)})
@triton.jit
def lora_forward_kernel(
    x_ptr: tl.tensor,
    la_ptr: tl.tensor,
    lb_ptr: tl.tensor,
    res_ptr: tl.tensor,
    M: tl.tensor,
    N: tl.tensor,
    K: tl.tensor,
    L: tl.tensor,
    stride_x_m: tl.tensor,
    stride_x_k: tl.tensor,
    stride_la_k: tl.tensor,
    stride_la_n: tl.tensor,
    stride_lb_n: tl.tensor,
    stride_lb_l: tl.tensor,
    stride_res_m: tl.tensor,
    stride_res_l: tl.tensor,
    # scale factor
    scale: tl.tensor,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    BLOCK_SIZE_L: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
) -> None:
    """
    Kernel for computing the matmul D = A x B x C.

    A has shape (M, K), B has shape (K, N), C has shape (N, L), and D has shape (M, L)
    N, the LoRA dimension must be less than or equal to than BLOCK_SIZE_N.
    """
    pid_m, pid_n = get_pid_coords(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    ab_result = inner_kernel(
        pid_m,
        pid_n,
        x_ptr,
        la_ptr,
        M,
        K,
        N,
        stride_x_m,
        stride_x_k,
        stride_la_k,
        stride_la_n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        scale,
    )
    ab_result = ab_result.to(lb_ptr.dtype.element_ty)

    block_vector_mul(
        pid_m,
        pid_n,
        ab_result,
        lb_ptr,
        res_ptr,
        M,
        N,
        L,
        stride_lb_n,
        stride_lb_l,
        stride_res_m,
        stride_res_l,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_L,
    )


def lora_forward_wrapper(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    res: torch.Tensor | None,
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Computes LoRA forward pass.

    Args:
        x: Tensor of shape [tokens, input_features].
        lora_A: Tensor of shape [input_features, rank].
        lora_B: Tensor of shape [rank, output_features].
        res: Tensor of shape [tokens, output_features], overwritten in place, or None to allocate.
        scale: LoRA scale factor (scalar)
        dtype: dtype for output

    Returns:
        Tensor of shape [tokens, output_features], aliasing res when supplied.
    """
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)

    assert x.shape[1] == lora_A.shape[0], "Incompatible X and LoRA A dimensions"
    assert lora_A.shape[1] == lora_B.shape[0], "Incompatible LoRA dimensions"
    if res is not None:
        assert x.shape[0] == res.shape[0], "Incompatible X and output dimensions"
        assert lora_B.shape[1] == res.shape[1], "Incompatible LoRA B and output dimensions"

    M, K = x.shape
    K, N = lora_A.shape
    N, L = lora_B.shape

    if res is None:
        res = torch.empty((M, L), device=x.device, dtype=dtype)

    def grid(meta: _GridMeta) -> tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    lora_forward_kernel[grid](
        x,
        lora_A,
        lora_B,
        res,
        M,
        N,
        K,
        L,
        x.stride(0),
        x.stride(1),
        lora_A.stride(0),
        lora_A.stride(1),
        lora_B.stride(0),
        lora_B.stride(1),
        res.stride(0),
        res.stride(1),
        scale,
    )

    return res


def da_dx_autotune_configs() -> list[triton.Config]:
    """
    Method for generating Triton configs for lora_da_dx_kernel.
    """
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)

    out = list()
    for blk_k in [64, 128]:
        for blk_l in [64, 128, 256]:
            for blk_m in [64, 128]:
                out.append(
                    triton.Config(
                        {"BLOCK_SIZE_K": blk_k, "BLOCK_SIZE_L": blk_l, "BLOCK_SIZE_M": blk_m, "GROUP_SIZE_M": 8},
                        num_stages=4,
                        num_warps=4,
                    )
                )
    return out


@triton.autotune(
    configs=da_dx_autotune_configs() if HAVE_TRITON else list(),
    key=["N", "K", "L"],
)
@triton.heuristics(values={"BLOCK_SIZE_N": lambda args: max(triton.next_power_of_2(args["N"]), 16)})
@triton.jit
def lora_da_dx_kernel(
    dy_ptr: tl.tensor,
    b_ptr: tl.tensor,
    a_ptr: tl.tensor,
    dx_ptr: tl.tensor,
    dyb_ptr: tl.tensor,
    M: tl.tensor,
    K: tl.tensor,
    N: tl.tensor,
    L: tl.tensor,
    stride_dy_m: tl.tensor,
    stride_dy_k: tl.tensor,
    stride_lorab_k: tl.tensor,
    stride_lorab_n: tl.tensor,
    stride_loraa_n: tl.tensor,
    stride_loraa_l: tl.tensor,
    stride_dx_m: tl.tensor,
    stride_dx_l: tl.tensor,
    stride_dyb_m: tl.tensor,
    stride_dyb_n: tl.tensor,
    scale: tl.tensor,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
) -> None:
    """
    Kernel for computing the matmul DYB = DY x B and DX = DY * B * A.

    XT has shape (S, M), DY has shape (M, K), B has shape (K, N), and A has shape (N, L)
    N, the LoRA dimension must be less than or equal to than BLOCK_SIZE_N.
    The result returned by this kernel is reduced in the wrapper.
    """
    pid_m, pid_n = get_pid_coords(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    dyb = inner_kernel(
        pid_m,
        pid_n,
        dy_ptr,
        b_ptr,
        M,
        K,
        N,
        stride_dy_m,
        stride_dy_k,
        stride_lorab_k,
        stride_lorab_n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        scale,
    )
    dyb = dyb.to(a_ptr.dtype.element_ty)

    offs_la_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_dx_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_l = tl.arange(0, BLOCK_SIZE_L)

    dx_ptrs = dx_ptr + stride_dx_m * offs_dx_m[:, None] + stride_dx_l * offs_l[None, :]
    la_ptrs = a_ptr + stride_loraa_n * offs_la_n[:, None] + stride_loraa_l * offs_l[None, :]

    for lx in tl.range(0, tl.cdiv(L, BLOCK_SIZE_L)):
        dx_mask = (offs_dx_m[:, None] < M) & (offs_l[None, :] < L - lx * BLOCK_SIZE_L)
        la_mask = (offs_la_n[:, None] < N) & (offs_l[None, :] < L - lx * BLOCK_SIZE_L)

        lora_a = tl.load(la_ptrs, mask=la_mask, other=0.0)
        dx = tl.dot(dyb, lora_a)
        dx = dx.to(a_ptr.dtype.element_ty)
        tl.store(dx_ptrs, dx, mask=dx_mask)

        la_ptrs += BLOCK_SIZE_L * stride_loraa_l
        dx_ptrs += BLOCK_SIZE_L * stride_dx_l

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dyb_ptrs = dyb_ptr + stride_dyb_m * offs_cm[:, None] + stride_dyb_n * offs_cn[None, :]
    dyb_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(dyb_ptrs, dyb, mask=dyb_mask)


def lora_da_dx_update_wrapper(
    xt: torch.Tensor,
    dy: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes dlora_A and dx.

    Args:
        xt: Tensor of shape [input_features, tokens], transposed input activations.
        dy: Tensor of shape [tokens, output_features].
        lora_B: Tensor of shape [output_features, rank].
        lora_A: Tensor of shape [rank, input_features].
        scale: LoRA scale factor.
        dtype: Output dtype.

    Returns:
        LoRA A gradient of shape [input_features, rank] and input gradient of
        shape [tokens, input_features].
    """
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)

    assert xt.shape[1] == dy.shape[0], "Incompatible X and dY dimensions"
    assert dy.shape[1] == lora_B.shape[0], "Incompatible dY and B dimensions"
    assert lora_B.shape[1] == lora_A.shape[0], "LoRA dimensions must match"

    _, M = xt.shape
    M, K = dy.shape
    K, N = lora_B.shape
    N, L = lora_A.shape

    dx = torch.empty((M, L), device=xt.device, dtype=dtype)
    dyb = torch.empty((M, N), device=xt.device, dtype=dtype)

    def grid(meta: _GridMeta) -> tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    lora_da_dx_kernel[grid](
        dy,
        lora_B,
        lora_A,
        dx,
        dyb,
        M,
        K,
        N,
        L,
        dy.stride(0),
        dy.stride(1),  #
        lora_B.stride(0),
        lora_B.stride(1),
        lora_A.stride(0),
        lora_A.stride(1),
        dx.stride(0),
        dx.stride(1),
        dyb.stride(0),
        dyb.stride(1),
        scale,
    )
    dlora_A = torch.matmul(xt, dyb)
    return dlora_A, dx


def db_autotune_configs() -> list[triton.Config]:
    """
    Method for generating Triton configs for lora_db_kernel.
    """
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)

    out = list()
    for blk_n in [32, 64, 128]:
        for blk_k in [32, 64, 128]:
            for blk_m in [64, 128]:
                out.append(
                    triton.Config(
                        {"BLOCK_SIZE_N": blk_n, "BLOCK_SIZE_K": blk_k, "BLOCK_SIZE_M": blk_m, "GROUP_SIZE_M": 8},
                        num_stages=4,
                        num_warps=4,
                    )
                )
    return out


@triton.autotune(
    configs=db_autotune_configs() if HAVE_TRITON else list(),
    key=["M", "K"],
)
@triton.jit
def lora_db_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    M: tl.tensor,
    K: tl.tensor,
    N: tl.tensor,
    stride_am: tl.tensor,
    stride_ak: tl.tensor,
    stride_bk: tl.tensor,
    stride_bn: tl.tensor,
    stride_cm: tl.tensor,
    stride_cn: tl.tensor,
    scale: tl.tensor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    """
    Kernel for computing the matmul AXT = A x X^T.

    A has shape (M, K), X has shape (N, K).
    """
    pid_m, pid_n = get_pid_coords(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    ab = inner_kernel(
        pid_m,
        pid_n,
        a_ptr,
        b_ptr,
        M,
        K,
        N,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        scale,
    )

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, ab, mask=c_mask)


def lora_db_update_wrapper(
    lora_A: torch.Tensor, xt: torch.Tensor, dy: torch.Tensor, scale: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Computes d_lora_B.

    Args:
        lora_A: Tensor of shape [rank, input_features].
        xt: Tensor of shape [input_features, tokens], transposed input activations.
        dy: Tensor of shape [tokens, output_features].
        scale: LoRA scale factor.
        dtype: Output dtype.

    Returns:
        LoRA B gradient of shape [output_features, rank].
    """
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)

    assert xt.shape[1] == dy.shape[0], "Incompatible X and dY dimensions"
    assert lora_A.shape[1] == xt.shape[0], "Incompatible X and A dimensions"

    M, K = lora_A.shape
    K, N = xt.shape
    N, _ = dy.shape

    axt = torch.empty((M, N), device=dy.device, dtype=dtype)

    def grid(meta: _GridMeta) -> tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    lora_db_kernel[grid](
        lora_A,
        xt,
        axt,
        M,
        K,
        N,
        lora_A.stride(0),
        lora_A.stride(1),
        xt.stride(0),
        xt.stride(1),
        axt.stride(0),
        axt.stride(1),
        scale,
    )

    return torch.matmul(axt, dy).t()
