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

"""Actual CUDA norm fusion at local training shapes and checkpoint boundaries."""

import json
from collections.abc import Callable

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.norm import Step5RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    error = actual.float() - expected.float()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return {
        "relative_l2": (error.square().mean().sqrt() / scale).item(),
        "max_over_rms": (error.abs().max() / scale).item(),
        "max_abs": error.abs().max().item(),
    }


def _time_cuda(call: Callable[[], object]) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(20):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 20


@pytest.mark.parametrize("sequence", [4096, 8192])
@pytest.mark.parametrize("kind", ["residual", "residual_cast", "query", "key", "final"])
def test_norm_cuda_forward_backward_recompute(sequence: int, kind: str, record_property) -> None:
    torch.manual_seed(983)
    shape = (1, sequence, 64, 192) if kind == "query" else (1, sequence, 4096)
    if kind == "key":
        shape = (1, sequence, 4, 192)
    dtype = torch.float32 if kind in ("residual", "residual_cast") else torch.bfloat16
    output_dtype = torch.bfloat16 if kind == "residual_cast" else dtype
    reference = Step3p5RMSNorm(shape[-1]).cuda()
    candidate = Step5RMSNorm(shape[-1], compile_norm=True).cuda()
    with torch.no_grad():
        reference.weight.copy_(torch.linspace(-0.7, 0.7, shape[-1], device="cuda") + 0.00123)
        candidate.weight.copy_(reference.weight)
    original = torch.randn(shape, device="cuda", dtype=dtype)
    if kind == "key":
        # Actual key projection is a strided slice of interleaved GQA QKV.
        original = torch.randn(1, sequence, 4, 18, 192, device="cuda", dtype=dtype)[..., 16, :]
    upstream = torch.randn_like(original, dtype=output_dtype)
    expected_input = original.detach().requires_grad_()
    actual_input = original.detach().requires_grad_()
    expected = reference(expected_input).to(output_dtype)
    actual = candidate(actual_input, output_dtype=output_dtype)
    expected_grads = torch.autograd.grad(expected, (expected_input, reference.weight), upstream)
    actual_grads = torch.autograd.grad(actual, (actual_input, candidate.weight), upstream)
    rows = {}
    for name, observed, control in zip(
        ("output", "input_gradient", "weight_gradient"), (actual, *actual_grads), (expected, *expected_grads)
    ):
        assert torch.isfinite(observed).all()
        rows[name] = _metrics(observed, control)
        assert rows[name]["relative_l2"] < 0.002, (name, rows[name])
        assert rows[name]["max_over_rms"] < 0.05, (name, rows[name])
    with torch.no_grad():
        inference = candidate(original, output_dtype=output_dtype)
    rows["grad_no_grad"] = _metrics(inference, actual.detach())
    assert torch.equal(inference, actual.detach()), rows["grad_no_grad"]
    replay_input = original.detach().requires_grad_()
    replay = checkpoint(candidate, replay_input, use_reentrant=False, output_dtype=output_dtype)
    replay_grads = torch.autograd.grad(replay, (replay_input, candidate.weight), upstream)
    for observed, control in zip((replay, *replay_grads), (actual, *actual_grads)):
        torch.testing.assert_close(observed, control, rtol=0, atol=0)
    record_property("metrics", json.dumps(rows))

    # Bracket the optimized call with the same eager control; never assert timing.
    def baseline_step() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            reference(expected_input).to(output_dtype), (expected_input, reference.weight), upstream
        )

    def candidate_step() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            candidate(actual_input, output_dtype=output_dtype), (actual_input, candidate.weight), upstream
        )

    record_property(
        "cuda_fwb_ms",
        json.dumps(
            {
                "baseline_before": _time_cuda(baseline_step),
                "compiled": _time_cuda(candidate_step),
                "baseline_after": _time_cuda(baseline_step),
            }
        ),
    )
    peaks = {}
    for name, call in [("baseline", baseline_step), ("compiled", candidate_step)]:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - before
    record_property("incremental_fwb_peak_bytes", json.dumps(peaks))
