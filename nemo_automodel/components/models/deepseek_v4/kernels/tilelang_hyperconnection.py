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

"""vLLM-compatible DeepSeek-V4 HyperConnection forward kernels.

The inference source of truth uses three precision-sensitive stages:

* DeepGEMM ``tf32_hc_prenorm_gemm`` computes the HC projection and squared
  norm with a split-K reduction;
* a TileLang kernel reduces the splits, creates the pre/post/Sinkhorn mixes,
  and serially collapses the four residual streams;
* a second TileLang kernel expands a layer output and serially accumulates
  the four residual streams.

Using eager ``F.linear`` and ``torch.matmul`` changes BF16 rounding at both
boundaries.  Those small differences are amplified by the following FP8 MoE.
The kernels below preserve vLLM's exact forward order without importing vLLM.

The forward kernels are adapted from vLLM's Apache-2.0
``vllm/model_executor/layers/mhc.py``.  Training backward recomputes the
mathematically equivalent PyTorch graph in FP32, because DeepGEMM's HC kernel
is inference-forward-only.
"""

# ruff: noqa

from __future__ import annotations

import importlib.util
import math

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


HAS_DEEP_GEMM = _module_available("deep_gemm") or _module_available("vllm.third_party.deep_gemm")
_HC_MULT = 4
_HC_MULT3 = _HC_MULT * (_HC_MULT + 2)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _compute_num_splits(hidden_size: int, num_tokens: int) -> int:
    """Match vLLM/DeepGEMM's split-K heuristic for standalone ``mhc_pre``."""
    block_k = 64
    block_m = 64
    grid_size = _ceil_div(num_tokens, block_m)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    split_k = num_sms // grid_size
    num_block_k = _ceil_div(_HC_MULT * hidden_size, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    hc_mult: int = _HC_MULT,
):
    """Reduce DeepGEMM splits and create all HC-pre outputs."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as token:
        T.pdl_sync()
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[split, token]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for index in T.Parallel(hc_mult3):
            mixes[index] = 0
            for split in T.serial(n_splits):
                mixes[index] += gemm_out_mul[split, token, index]
            mixes[index] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            combination = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for index in T.Parallel(hc_mult):
                post_mix[token, index] = (
                    T.sigmoid(mixes_shared[index + hc_mult] * hc_scale[1] + hc_base[index + hc_mult])
                    * hc_post_mult_value
                )
            for row, column in T.Parallel(hc_mult, hc_mult):
                combination[row, column] = (
                    mixes_shared[row * hc_mult + column + hc_mult * 2] * hc_scale[2]
                    + hc_base[row * hc_mult + column + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            column_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(combination, row_max, dim=1)
            for row, column in T.Parallel(hc_mult, hc_mult):
                combination[row, column] = T.exp(combination[row, column] - row_max[row])
            T.reduce_sum(combination, row_sum, dim=1)
            for row, column in T.Parallel(hc_mult, hc_mult):
                combination[row, column] = combination[row, column] / row_sum[row] + hc_sinkhorn_eps

            T.reduce_sum(combination, column_sum, dim=0)
            for row, column in T.Parallel(hc_mult, hc_mult):
                combination[row, column] = combination[row, column] / (column_sum[column] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(combination, row_sum, dim=1)
                for row, column in T.Parallel(hc_mult, hc_mult):
                    combination[row, column] = combination[row, column] / (row_sum[row] + hc_sinkhorn_eps)
                T.reduce_sum(combination, column_sum, dim=0)
                for row, column in T.Parallel(hc_mult, hc_mult):
                    combination[row, column] = combination[row, column] / (column_sum[column] + hc_sinkhorn_eps)

            for row, column in T.Parallel(hc_mult, hc_mult):
                comb_mix[token, row * hc_mult + column] = combination[row, column]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for index in T.Parallel(hc_mult):
                pre_mix_shared[index] = T.sigmoid(mixes_shared[index] * hc_scale[0] + hc_base[index]) + hc_pre_eps

            for block in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                residual_shared = T.alloc_shared((hc_mult, hidden_block), T.float32)
                residual_local = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[token, 0, block * hidden_block], residual_shared)
                T.copy(residual_shared, residual_local)

                output_local = T.alloc_fragment(hidden_block, T.float32)
                T.clear(output_local)
                for stream in T.serial(hc_mult):
                    pre_value = pre_mix_shared[stream]
                    for offset in T.Parallel(hidden_block):
                        output_local[offset] += pre_value * residual_local[stream, offset]
                T.copy(output_local, layer_input[token, block * hidden_block])
        T.pdl_trigger()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    comb_mix,
    residual,
    post_mix,
    layer_output,
    output,
    hc_mult: int,
    hidden_size: int,
    threads: int = 128,
    hidden_block: int = 1024,
):
    """Serial FP32 HC-post accumulation with a final BF16 store."""
    num_tokens = T.dynamic("num_tokens")
    hidden_block = math.gcd(hidden_size, hidden_block)

    comb_mix: T.Tensor((num_tokens, hc_mult, hc_mult), T.float32)
    residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16)
    post_mix: T.Tensor((num_tokens, hc_mult), T.float32)
    layer_output: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    output: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16)

    with T.Kernel(num_tokens, threads=threads) as token:
        output_shared = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
        residual_shared = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
        layer_output_shared = T.alloc_shared(hidden_block, T.bfloat16)
        output_local = T.alloc_fragment((hc_mult, hidden_block), T.float32)
        residual_local = T.alloc_fragment((hc_mult, hidden_block), T.float32)
        layer_output_local = T.alloc_fragment(hidden_block, T.float32)
        comb_local = T.alloc_fragment((hc_mult, hc_mult), T.float32)
        post_local = T.alloc_fragment(hc_mult, T.float32)

        T.pdl_sync()
        T.copy(comb_mix[token, 0, 0], comb_local)
        T.copy(post_mix[token, 0], post_local)
        for block in T.Pipelined(T.ceildiv(hidden_size, hidden_block), num_stages=2):
            T.copy(residual[token, 0, block * hidden_block], residual_shared)
            T.copy(layer_output[token, block * hidden_block], layer_output_shared)
            T.copy(residual_shared, residual_local)
            T.copy(layer_output_shared, layer_output_local)

            for output_stream, offset in T.Parallel(hc_mult, hidden_block):
                output_local[output_stream, offset] = post_local[output_stream] * layer_output_local[offset]
                for input_stream in T.serial(hc_mult):
                    output_local[output_stream, offset] += (
                        comb_local[input_stream, output_stream] * residual_local[input_stream, offset]
                    )
            T.copy(output_local, output_shared)
            T.copy(output_shared, output[token, 0, block * hidden_block])
        T.pdl_trigger()


def _torch_sinkhorn(logits: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    combination = logits.softmax(dim=-1) + eps
    combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        combination = combination / (combination.sum(dim=-1, keepdim=True) + eps)
        combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    return combination


def _torch_prepare(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable mathematical reference used only to compute backward."""
    outer_shape = residual.shape[:-2]
    flat = residual.flatten(start_dim=-2).float()
    mixes = F.linear(F.rms_norm(flat, (flat.shape[-1],), eps=norm_eps), fn.float())
    pre = torch.sigmoid(mixes[..., :_HC_MULT] * scale[0] + base[:_HC_MULT]) + pre_eps
    post = (
        torch.sigmoid(mixes[..., _HC_MULT : 2 * _HC_MULT] * scale[1] + base[_HC_MULT : 2 * _HC_MULT]) * post_mult_value
    )
    combination_logits = mixes[..., 2 * _HC_MULT :].view(*outer_shape, _HC_MULT, _HC_MULT) * scale[2] + base[
        2 * _HC_MULT :
    ].view(_HC_MULT, _HC_MULT)
    combination = _torch_sinkhorn(combination_logits, sinkhorn_repeat, sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=-2).to(residual.dtype)
    return layer_input, post, combination


def _deep_gemm_prepare_forward(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not HAS_DEEP_GEMM:
        raise RuntimeError(
            "The exact DSV4 HyperConnection TileLang path requires DeepGEMM with tf32_hc_prenorm_gemm support."
        )
    try:
        import deep_gemm
    except ImportError:
        # vLLM wheels vendor the same upstream DeepGEMM package. This fallback
        # keeps MOLT's shared AM/vLLM container usable while a standalone AM
        # environment can install ``deep_gemm`` directly.
        from vllm.third_party import deep_gemm

    kernel = getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)
    if kernel is None:
        raise RuntimeError(
            "Installed DeepGEMM does not expose tf32_hc_prenorm_gemm; install a DeepSeek-V4-capable DeepGEMM build."
        )

    outer_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.contiguous().view(-1, _HC_MULT, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            torch.empty(*outer_shape, hidden_size, dtype=residual.dtype, device=residual.device),
            torch.empty(*outer_shape, _HC_MULT, dtype=torch.float32, device=residual.device),
            torch.empty(
                *outer_shape,
                _HC_MULT,
                _HC_MULT,
                dtype=torch.float32,
                device=residual.device,
            ),
        )

    n_splits = _compute_num_splits(hidden_size, num_tokens)
    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        _HC_MULT3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    post_mix = torch.empty(
        num_tokens,
        _HC_MULT,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        _HC_MULT * _HC_MULT,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=residual.dtype,
        device=residual.device,
    )

    kernel(
        residual_flat.view(num_tokens, _HC_MULT * hidden_size),
        fn.contiguous(),
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        scale,
        base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        norm_eps,
        pre_eps,
        sinkhorn_eps,
        post_mult_value,
        sinkhorn_repeat,
        n_splits,
        _HC_MULT,
    )
    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, _HC_MULT),
        comb_mix.view(*outer_shape, _HC_MULT, _HC_MULT),
    )


class _ExactMhcPrepare(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult_value: float,
        sinkhorn_repeat: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = _deep_gemm_prepare_forward(
            residual,
            fn,
            scale,
            base,
            norm_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult_value,
            sinkhorn_repeat,
        )
        ctx.save_for_backward(residual, fn, scale, base)
        ctx.options = (
            norm_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult_value,
            sinkhorn_repeat,
        )
        return outputs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_layer_input: torch.Tensor | None,
        grad_post: torch.Tensor | None,
        grad_comb: torch.Tensor | None,
    ):
        residual, fn, scale, base = ctx.saved_tensors
        tensors = (residual, fn, scale, base)
        detached = tuple(
            tensor.detach().requires_grad_(needs_grad)
            for tensor, needs_grad in zip(tensors, ctx.needs_input_grad[:4], strict=True)
        )
        grad_inputs = [tensor for tensor in detached if tensor.requires_grad]
        if not grad_inputs:
            return (None,) * 9

        with torch.enable_grad():
            outputs = _torch_prepare(*detached, *ctx.options)
            selected_outputs = []
            selected_grads = []
            for output, grad_output in zip(
                outputs,
                (grad_layer_input, grad_post, grad_comb),
                strict=True,
            ):
                if grad_output is not None:
                    selected_outputs.append(output)
                    selected_grads.append(grad_output)
            calculated = torch.autograd.grad(
                selected_outputs,
                grad_inputs,
                grad_outputs=selected_grads,
                allow_unused=True,
            )

        calculated_iter = iter(calculated)
        result = []
        for tensor in detached:
            result.append(next(calculated_iter) if tensor.requires_grad else None)
        return (*result, None, None, None, None, None)


class _ExactMhcPost(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        residual: torch.Tensor,
        layer_output: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        outer_shape = residual.shape[:-2]
        hidden_size = residual.shape[-1]
        residual_flat = residual.contiguous().view(-1, _HC_MULT, hidden_size)
        layer_output_flat = layer_output.contiguous().view(-1, hidden_size)
        post_flat = post.contiguous().view(-1, _HC_MULT)
        comb_flat = comb.contiguous().view(-1, _HC_MULT, _HC_MULT)
        output = torch.empty_like(residual_flat)
        if residual_flat.shape[0] > 0:
            mhc_post_tilelang(
                comb_flat,
                residual_flat,
                post_flat,
                layer_output_flat,
                output,
                _HC_MULT,
                hidden_size,
            )
        ctx.save_for_backward(residual, layer_output, post, comb)
        return output.view(*outer_shape, _HC_MULT, hidden_size)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        residual, layer_output, post, comb = ctx.saved_tensors
        grad = grad_output.float()
        grad_residual = torch.matmul(comb.float(), grad).to(residual.dtype)
        grad_layer_output = (grad * post.float().unsqueeze(-1)).sum(dim=-2).to(layer_output.dtype)
        grad_post = (grad * layer_output.float().unsqueeze(-2)).sum(dim=-1).to(post.dtype)
        grad_comb = torch.matmul(
            residual.float(),
            grad.transpose(-1, -2),
        ).to(comb.dtype)
        return grad_residual, grad_layer_output, grad_post, grad_comb


def exact_mhc_prepare(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(layer_input, post, comb)`` with vLLM-exact forward semantics."""
    if not residual.is_cuda or residual.dtype != torch.bfloat16:
        raise RuntimeError("Exact DSV4 HyperConnection requires a CUDA BF16 residual")
    if residual.shape[-2] != _HC_MULT or fn.shape != (
        _HC_MULT3,
        _HC_MULT * residual.shape[-1],
    ):
        raise ValueError(
            "Exact DSV4 HyperConnection requires hc_mult=4 and fn [24,4*hidden]; "
            f"got residual={tuple(residual.shape)}, fn={tuple(fn.shape)}"
        )
    if fn.dtype != torch.float32 or scale.dtype != torch.float32 or base.dtype != torch.float32:
        raise TypeError("Exact DSV4 HyperConnection fn/scale/base must remain FP32")
    return _ExactMhcPrepare.apply(
        residual,
        fn,
        scale,
        base,
        norm_eps,
        pre_eps,
        sinkhorn_eps,
        post_mult_value,
        sinkhorn_repeat,
    )


def exact_mhc_post(
    residual: torch.Tensor,
    layer_output: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Apply vLLM's serial FP32 HC-post accumulation."""
    if not (
        residual.is_cuda
        and layer_output.is_cuda
        and residual.dtype == torch.bfloat16
        and layer_output.dtype == torch.bfloat16
    ):
        raise RuntimeError("Exact DSV4 HC post requires CUDA BF16 residual and layer output")
    if residual.shape[:-2] != layer_output.shape[:-1] or residual.shape[-2] != _HC_MULT:
        raise ValueError(
            f"DSV4 HC post shape mismatch: residual={tuple(residual.shape)}, layer_output={tuple(layer_output.shape)}"
        )
    return _ExactMhcPost.apply(residual, layer_output, post, comb)
