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


import torch
import triton
import triton.language as tl


def forward_autotune_configs():
    """
    Method for generating Triton configs for lora_forward_kernel.
    """
    out = list()
    for blk_m in [16, 32, 64]:
        for blk_k in [128, 256, 512]:
            for blk_l in [128, 256, 512]:
                out.append(
                    triton.Config(
                        {'BLOCK_SIZE_M': blk_m, 'BLOCK_SIZE_K': blk_k, 'BLOCK_SIZE_L': blk_l, 'GROUP_SIZE_M': 8},
                        num_stages=4,
                        num_warps=4,
                    ))
    return out


@triton.jit
def get_pid_coords(M, N,
                   BLOCK_SIZE_M: tl.constexpr,
                   BLOCK_SIZE_N: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr):
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
def inner_kernel(pid_m, pid_n,
                 a_ptr, b_ptr,
                 M, K, N,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 BLOCK_SIZE_M: tl.constexpr,
                 BLOCK_SIZE_K: tl.constexpr,
                 BLOCK_SIZE_N: tl.constexpr,
                 scale):
    """
    Performs the matrix multiplication AB.

    A is an M x K matrix and B is an N x K matrix.
    The result is returned to be stored by the calling method.
    """
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
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
def block_vector_mul(pid_m, pid_n,
           ab_result, c_ptr, d_ptr,
           M, N, L,
           stride_cn, stride_cl, stride_dm, stride_dl,
           BLOCK_SIZE_M: tl.constexpr,
           BLOCK_SIZE_N: tl.constexpr,
           BLOCK_SIZE_L: tl.constexpr):
    """
    Multiplies an M x N vector AB and and N x L vector C and adds the result to the output vector D.

    N is assumed to be smaller than BLOCK_SIZE_N.
    """
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_dm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))

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
    key=['N', 'K', 'L'],
)
# This optimization exploits that N is the LoRA dimension and thus we only need one block.
@triton.heuristics(values={'BLOCK_SIZE_N': lambda args: max(triton.next_power_of_2(args['N']), 16)})
@triton.jit
def lora_forward_kernel(
    x_ptr, la_ptr, lb_ptr, res_ptr,
    M, N, K, L,
    stride_x_m, stride_x_k,
    stride_la_k, stride_la_n,
    stride_lb_n, stride_lb_l,
    stride_res_m, stride_res_l,
    # scale factor
    scale,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    BLOCK_SIZE_L: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
):
    """
    Kernel for computing the matmul D = A x B x C.

    A has shape (M, K), B has shape (K, N), C has shape (N, L), and D has shape (M, L)
    N, the LoRA dimension must be less than or equal to than BLOCK_SIZE_N.
    """
    pid_m, pid_n = get_pid_coords(M, N,
                                  BLOCK_SIZE_M,
                                  BLOCK_SIZE_N,
                                  GROUP_SIZE_M)

    ab_result = inner_kernel(pid_m, pid_n,
                 x_ptr, la_ptr,
                 M, K, N,
                 stride_x_m, stride_x_k,
                 stride_la_k, stride_la_n,
                 BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                 scale)
    ab_result = ab_result.to(lb_ptr.dtype.element_ty)

    block_vector_mul(pid_m, pid_n,
                ab_result, lb_ptr, res_ptr,
                M, N, L,
                stride_lb_n, stride_lb_l, stride_res_m, stride_res_l,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_L)



def lora_forward_wrapper(x, lora_A, lora_B, res, scale, dtype=torch.float32):
    """
    Computes LoRA forward pass.

    Args:
        x: input activations,  (M x K)
        lora_A: LoRA A weights (K x N)
        lora_B: LoRA B weights (N x L)
        res (optional(torch.Tensor)): output tensor
        scale: LoRA scale factor (scalar)
        dtype: dtype for output
    """
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

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)  # noqa: E731
    lora_forward_kernel[grid](
        x, lora_A, lora_B, res,
        M, N, K, L,
        x.stride(0), x.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        lora_B.stride(0), lora_B.stride(1),
        res.stride(0), res.stride(1),
        scale,
    )

    return res


def da_dx_autotune_configs():
    """
    Method for generating Triton configs for lora_da_dx_kernel.
    """
    out = list()
    for blk_k in [64, 128]:
        for blk_l in [64, 128, 256]:
            for blk_m in [64, 128]:
                out.append(
                    triton.Config(
                        {'BLOCK_SIZE_K': blk_k,  'BLOCK_SIZE_L': blk_l,
                        'BLOCK_SIZE_M': blk_m, 'GROUP_SIZE_M': 8},
                        num_stages=4,
                        num_warps=4,
                    ))
    return out


@triton.autotune(
    configs=da_dx_autotune_configs(),
    key=['N', 'K', 'L'],
)
@triton.heuristics(values={'BLOCK_SIZE_N': lambda args: max(triton.next_power_of_2(args['N']), 16)})
@triton.jit
def lora_da_dx_kernel(dy_ptr, b_ptr, a_ptr, dx_ptr, dyb_ptr,
                    M, K, N, L,
                    stride_dy_m, stride_dy_k,
                    stride_lorab_k, stride_lorab_n,
                    stride_loraa_n, stride_loraa_l,
                    stride_dx_m, stride_dx_l,
                    stride_dyb_m, stride_dyb_n,
                    scale,
                    BLOCK_SIZE_M: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    BLOCK_SIZE_L: tl.constexpr,
):
    """
    Kernel for computing the matmul DYB = DY x B and DX = DY * B * A.

    XT has shape (S, M), DY has shape (M, K), B has shape (K, N), and A has shape (N, L)
    N, the LoRA dimension must be less than or equal to than BLOCK_SIZE_N.
    The result returned by this kernel is reduced in the wrapper.
    """
    pid_m, pid_n = get_pid_coords(M, N,
                                  BLOCK_SIZE_M,
                                  BLOCK_SIZE_N,
                                  GROUP_SIZE_M)
    dyb = inner_kernel(pid_m, pid_n,
                    dy_ptr, b_ptr,
                    M, K, N,
                    stride_dy_m, stride_dy_k,
                    stride_lorab_k, stride_lorab_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                    scale)
    dyb = dyb.to(a_ptr.dtype.element_ty)

    offs_la_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_dx_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    dx_ptrs = dx_ptr + stride_dx_m * offs_dx_m[:, None] + stride_dx_l * offs_l[None, :]
    dx_mask = (offs_dx_m[:, None] < M) & (offs_l[None, :] < L)

    la_ptrs = a_ptr + stride_loraa_n * offs_la_n[:, None] + stride_loraa_l * offs_l[None, :]
    la_mask = (offs_la_n[:, None] < N) & (offs_l[None, :] < L)

    for _ in tl.range(0, tl.cdiv(L, BLOCK_SIZE_L)):
        lora_a = tl.load(la_ptrs, mask=la_mask, other=0.0)
        dx = tl.dot(dyb, lora_a)
        dx = dx.to(a_ptr.dtype.element_ty)
        tl.store(dx_ptrs, dx, mask=dx_mask)

        la_ptrs += BLOCK_SIZE_L * stride_loraa_l
        dx_ptrs += BLOCK_SIZE_L * stride_dx_l

    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    dyb_ptrs = dyb_ptr + stride_dyb_m * offs_cm[:, None] + stride_dyb_n * offs_cn[None, :]
    dyb_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(dyb_ptrs, dyb, mask=dyb_mask)


def lora_da_dx_update_wrapper(xt, dy, lora_B, lora_A, scale, dtype=torch.float32):
    """Computes dlora_A and dx.

    xt: input activation weights, transposed (S x M)
    dy: gradients (M x K)
    lora_B: LoRA B weights (K x N)
    lora_A: LoRA A weights (N x L)
    scale: LoRA scale factor (scalar)
    dtype: dtype for output
    """
    assert xt.shape[1] == dy.shape[0], "Incompatible X and dY dimensions"
    assert dy.shape[1] == lora_B.shape[0], "Incompatible dY and B dimensions"
    assert lora_B.shape[1] == lora_A.shape[0], "LoRA dimensions must match"

    _, M = xt.shape
    M, K = dy.shape
    K, N = lora_B.shape
    N, L = lora_A.shape

    dx = torch.empty((M, L), device=xt.device, dtype=dtype)
    dyb = torch.empty((M, N), device=xt.device, dtype=dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),) # noqa: E731
    lora_da_dx_kernel[grid](dy, lora_B, lora_A, dx, dyb,
                    M, K, N, L,
                    dy.stride(0), dy.stride(1),  #
                    lora_B.stride(0), lora_B.stride(1),
                    lora_A.stride(0), lora_A.stride(1),
                    dx.stride(0), dx.stride(1),
                    dyb.stride(0), dyb.stride(1),
                    scale
                    )
    dlora_A = torch.matmul(xt, dyb)
    return dlora_A, dx


def db_autotune_configs():
    """
    Method for generating Triton configs for lora_db_kernel.
    """
    out = list()
    for blk_n in [32, 64, 128]:
        for blk_k in [32, 64, 128]:
            for blk_m in [64, 128]:
                out.append(
                    triton.Config(
                        {'BLOCK_SIZE_N': blk_n, 'BLOCK_SIZE_K': blk_k,
                         'BLOCK_SIZE_M': blk_m, 'GROUP_SIZE_M': 8},
                        num_stages=4,
                        num_warps=4
                        )
                    )
    return out

@triton.autotune(
    configs=db_autotune_configs(),
    key=['S', 'M', 'K'],
    )
@triton.jit
def lora_db_kernel(a_ptr, b_ptr, c_ptr,
                    M, K, N,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    scale,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr
                    ):
    """
    Kernel for computing the matmul AXT = A x X^T.

    A has shape (M, K), X has shape (N, K).
    """
    pid_m, pid_n = get_pid_coords(M, N,
                                BLOCK_SIZE_M,
                                BLOCK_SIZE_N,
                                GROUP_SIZE_M)
    ab = inner_kernel(pid_m, pid_n,
                    a_ptr, b_ptr,
                    M, K, N,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                    scale)

    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, ab, mask=c_mask)


def lora_db_update_wrapper(lora_A, xt, dy, scale, dtype=torch.float32):
    """Computes d_lora_B.

    lora_A: LoRA A weights (M x K)
    xt: input activation weights, transposed (K x N)
    dy: gradients (N x S)
    scale: LoRA scale factor (scalar)
    dtype: dtype for output
    """
    assert xt.shape[1] == dy.shape[0], "Incompatible X and dY dimensions"
    assert lora_A.shape[1] == xt.shape[0], "Incompatible X and A dimensions"

    M, K = lora_A.shape
    K, N = xt.shape
    N, _ = dy.shape

    axt = torch.empty((M, N), device=dy.device, dtype=dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    lora_db_kernel[grid](lora_A, xt, axt,
                         M, K, N,
                         lora_A.stride(0), lora_A.stride(1),
                         xt.stride(0), xt.stride(1),
                         axt.stride(0), axt.stride(1),
                         scale)

    return torch.matmul(axt, dy).t()
