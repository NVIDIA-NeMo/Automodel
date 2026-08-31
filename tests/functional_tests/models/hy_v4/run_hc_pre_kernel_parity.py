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

"""Compare compiled HY4 iHC pre reductions with the vLLM eager equations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

from nemo_automodel.components.models.hy_v4.hc import (
    _ihc_reduce_fp32,
    _ihc_reduce_fp32_compiled,
    _rms_rsqrt_fp32,
    _rms_rsqrt_fp32_compiled,
)


def _run(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    output_gradient: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
    cloned_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn(*cloned_inputs)
    gradients = torch.autograd.grad(output, cloned_inputs, output_gradient)
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline
    return output.detach().cpu(), tuple(gradient.detach().cpu() for gradient in gradients), incremental_peak


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape:
        raise AssertionError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    max_abs = 0.0
    sum_squared_difference = 0.0
    sum_squared_left = 0.0
    sum_squared_right = 0.0
    dot = 0.0
    chunk_elements = 1 << 20
    for start in range(0, left_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, left_flat.numel())
        left_chunk = left_flat[start:end].double()
        right_chunk = right_flat[start:end].double()
        difference = left_chunk - right_chunk
        max_abs = max(max_abs, float(difference.abs().max().item()))
        sum_squared_difference += float(difference.square().sum().item())
        sum_squared_left += float(left_chunk.square().sum().item())
        sum_squared_right += float(right_chunk.square().sum().item())
        dot += float((left_chunk * right_chunk).sum().item())
    return {
        "max_abs": max_abs,
        "relative_l2": (sum_squared_difference / max(sum_squared_right, 1e-60)) ** 0.5,
        "cosine": dot / max((sum_squared_left * sum_squared_right) ** 0.5, 1e-60),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("HY4 iHC pre kernel parity requires CUDA")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    hidden_streams = torch.randn(
        args.tokens,
        args.hc_mult,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    flat = hidden_streams.flatten(start_dim=-2).float()
    pre_gates = torch.randn(args.tokens, args.hc_mult, device=device, dtype=torch.float32).sigmoid()
    rms_output_gradient = torch.randn(args.tokens, 1, device=device, dtype=torch.float32)
    reduce_output_gradient = torch.randn(args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16)

    rms_reference, rms_reference_grads, rms_reference_peak = _run(
        lambda value: _rms_rsqrt_fp32(value, args.eps),
        (flat,),
        rms_output_gradient,
    )
    reduce_reference, reduce_reference_grads, reduce_reference_peak = _run(
        _ihc_reduce_fp32,
        (hidden_streams, pre_gates),
        reduce_output_gradient,
    )

    # Warm both dynamic compiled paths before measuring steady-state memory.
    _run(
        lambda value: _rms_rsqrt_fp32_compiled(value, args.eps),
        (flat[:32],),
        rms_output_gradient[:32],
    )
    _run(
        _ihc_reduce_fp32_compiled,
        (hidden_streams[:32], pre_gates[:32]),
        reduce_output_gradient[:32],
    )
    torch.cuda.empty_cache()

    rms_compiled, rms_compiled_grads, rms_compiled_peak = _run(
        lambda value: _rms_rsqrt_fp32_compiled(value, args.eps),
        (flat,),
        rms_output_gradient,
    )
    reduce_compiled, reduce_compiled_grads, reduce_compiled_peak = _run(
        _ihc_reduce_fp32_compiled,
        (hidden_streams, pre_gates),
        reduce_output_gradient,
    )

    report = {
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "hc_mult": args.hc_mult,
        "rms": {
            "output": _metrics(rms_compiled, rms_reference),
            "grad_input": _metrics(rms_compiled_grads[0], rms_reference_grads[0]),
            "reference_incremental_peak_bytes": rms_reference_peak,
            "compiled_incremental_peak_bytes": rms_compiled_peak,
        },
        "reduce": {
            "output": _metrics(reduce_compiled, reduce_reference),
            "grad_hidden_streams": _metrics(reduce_compiled_grads[0], reduce_reference_grads[0]),
            "grad_pre_gates": _metrics(reduce_compiled_grads[1], reduce_reference_grads[1]),
            "reference_incremental_peak_bytes": reduce_reference_peak,
            "compiled_incremental_peak_bytes": reduce_compiled_peak,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")

    for kernel_name in ("rms", "reduce"):
        kernel = report[kernel_name]
        for metric_name, metrics in kernel.items():
            if not isinstance(metrics, dict):
                continue
            if metrics["relative_l2"] >= 1.0e-3 or metrics["cosine"] <= 0.999999:
                raise AssertionError(f"compiled HY4 iHC {kernel_name} {metric_name} parity failed: {metrics}")
        if kernel["compiled_incremental_peak_bytes"] >= kernel["reference_incremental_peak_bytes"]:
            raise AssertionError(f"compiled HY4 iHC {kernel_name} did not reduce peak activation memory: {kernel}")


if __name__ == "__main__":
    main()
