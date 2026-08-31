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

"""Compare the fused CUDA HY4 iHC post path with the vLLM eager equation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

from nemo_automodel.components.models.hy_v4.hc import _ihc_post_fp32, _ihc_post_fp32_compiled


def _run(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    post_gates: torch.Tensor,
    output_gradient: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
    inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (sublayer_output, residual, post_gates))
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn(*inputs)
    gradients = torch.autograd.grad(output, inputs, output_gradient)
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline
    return output.detach().cpu(), tuple(gradient.detach().cpu() for gradient in gradients), incremental_peak


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Accumulate parity metrics in bounded-memory CPU chunks."""
    if left.shape != right.shape:
        raise AssertionError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    max_abs = 0.0
    sum_abs = 0.0
    sum_squared_difference = 0.0
    sum_squared_left = 0.0
    sum_squared_right = 0.0
    dot = 0.0
    mismatch_count = 0
    chunk_elements = 1 << 20
    for start in range(0, left_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, left_flat.numel())
        left_chunk = left_flat[start:end].double()
        right_chunk = right_flat[start:end].double()
        difference = left_chunk - right_chunk
        max_abs = max(max_abs, float(difference.abs().max().item()))
        sum_abs += float(difference.abs().sum().item())
        sum_squared_difference += float(difference.square().sum().item())
        sum_squared_left += float(left_chunk.square().sum().item())
        sum_squared_right += float(right_chunk.square().sum().item())
        dot += float((left_chunk * right_chunk).sum().item())
        mismatch_count += int(torch.count_nonzero(left_chunk != right_chunk).item())
    return {
        "max_abs": max_abs,
        "mean_abs": sum_abs / left_flat.numel(),
        "relative_l2": (sum_squared_difference / max(sum_squared_right, 1e-60)) ** 0.5,
        "cosine": dot / max((sum_squared_left * sum_squared_right) ** 0.5, 1e-60),
        "element_mismatch_fraction": mismatch_count / left_flat.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("HY4 iHC post kernel parity requires CUDA")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    sublayer_output = torch.randn(args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(args.tokens, args.hc_mult, args.hidden_size, device=device, dtype=torch.bfloat16)
    post_gates = torch.randn(args.tokens, args.hc_mult, device=device, dtype=torch.float32)
    output_gradient = torch.randn_like(residual)

    reference_output, reference_grads, reference_peak = _run(
        _ihc_post_fp32,
        sublayer_output,
        residual,
        post_gates,
        output_gradient,
    )

    # Compile once before measuring the steady-state fused path.
    warmup_output, warmup_grads, _ = _run(
        _ihc_post_fp32_compiled,
        sublayer_output[:32],
        residual[:32],
        post_gates[:32],
        output_gradient[:32],
    )
    del warmup_output, warmup_grads
    torch.cuda.empty_cache()
    compiled_output, compiled_grads, compiled_peak = _run(
        _ihc_post_fp32_compiled,
        sublayer_output,
        residual,
        post_gates,
        output_gradient,
    )

    output_metrics = _metrics(compiled_output, reference_output)
    gradient_metrics = [
        _metrics(compiled_gradient, reference_gradient)
        for compiled_gradient, reference_gradient in zip(compiled_grads, reference_grads)
    ]
    report = {
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "hc_mult": args.hc_mult,
        "dtype": str(sublayer_output.dtype),
        "output": output_metrics,
        "grad_sublayer_output": gradient_metrics[0],
        "grad_residual": gradient_metrics[1],
        "grad_post_gates": gradient_metrics[2],
        "reference_incremental_peak_bytes": reference_peak,
        "compiled_incremental_peak_bytes": compiled_peak,
        "peak_memory_reduction_bytes": reference_peak - compiled_peak,
        "peak_memory_ratio": compiled_peak / reference_peak,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")

    parity_metrics = {
        "output": output_metrics,
        "grad_sublayer_output": gradient_metrics[0],
        "grad_residual": gradient_metrics[1],
        "grad_post_gates": gradient_metrics[2],
    }
    max_abs_limits = {
        "output": 0.03125,
        "grad_sublayer_output": 0.03125,
        "grad_residual": 0.0,
        "grad_post_gates": 1.0e-4,
    }
    for name, metrics in parity_metrics.items():
        if metrics["relative_l2"] >= 1.0e-3 or metrics["cosine"] <= 0.999999:
            raise AssertionError(f"compiled HY4 iHC post {name} parity failed: {metrics}")
        if metrics["max_abs"] > max_abs_limits[name]:
            raise AssertionError(f"compiled HY4 iHC post {name} max-abs error is too high: {metrics}")
    if compiled_peak >= reference_peak / 2:
        raise AssertionError(
            f"compiled HY4 iHC post peak {compiled_peak} did not halve eager reference peak {reference_peak}"
        )


if __name__ == "__main__":
    main()
