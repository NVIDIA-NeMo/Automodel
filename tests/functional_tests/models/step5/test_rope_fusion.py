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

"""Actual CUDA partial RoPE fusion against the original unfused arithmetic."""

import json
from collections.abc import Callable

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
@pytest.mark.parametrize("span", [32, 64, 192])
def test_rope_cuda_long_positions_packed_backward_recompute(sequence: int, span: int, record_property) -> None:
    torch.manual_seed(985)
    config = Step5Config()
    config.rope_theta = 10000.0 if span == 192 else 10000000.0
    heads, dim = (16, 256) if span == 32 else (64, 192)
    rope = Step5RotaryEmbedding(config, 0, span=span).cuda()
    positions = torch.arange(sequence, device="cuda").unsqueeze(0) + 131072 - sequence
    positions[:, sequence // 2 :] = torch.arange(sequence - sequence // 2, device="cuda")
    hidden = torch.randn(1, sequence, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    half = span // 2
    # Original inference arithmetic and explicit BF16 table, kept outside candidate helpers.
    inverse = 1.0 / config.rope_theta ** (torch.arange(0, span, 2, device="cuda", dtype=torch.float32) / span)
    phases = positions.float().unsqueeze(-1) * inverse
    cos = phases.cos().bfloat16().float().unsqueeze(-2)
    sin = phases.sin().bfloat16().float().unsqueeze(-2)
    left, right = hidden[..., :half].float(), hidden[..., half:span].float()
    expected = torch.cat(
        ((left * cos - right * sin).bfloat16(), (left * sin + right * cos).bfloat16(), hidden[..., span:]), -1
    )
    actual = rope(hidden, positions)
    upstream = torch.randn_like(hidden)
    expected_grad = torch.autograd.grad(expected, hidden, upstream)[0]
    actual_grad = torch.autograd.grad(actual, hidden, upstream)[0]
    rows = {}
    for name, observed, reference in [("output", actual, expected), ("input_gradient", actual_grad, expected_grad)]:
        delta = observed.float() - reference.float()
        rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
        rows[name] = {
            "relative_l2": (delta.square().mean().sqrt() / rms).item(),
            "max_over_rms": (delta.abs().max() / rms).item(),
            "max_abs": delta.abs().max().item(),
        }
        assert torch.isfinite(observed).all()
        assert rows[name]["relative_l2"] < 0.002, rows[name]
        assert rows[name]["max_over_rms"] < 0.05, rows[name]
    with torch.no_grad():
        no_grad = rope(hidden, positions)
    assert torch.equal(actual.detach(), no_grad)
    replay = checkpoint(rope, hidden, positions, use_reentrant=False)
    replay_grad = torch.autograd.grad(replay, hidden, upstream)[0]
    torch.testing.assert_close(replay, actual, rtol=0, atol=0)
    torch.testing.assert_close(replay_grad, actual_grad, rtol=0, atol=0)
    assert torch.equal(actual[..., span:], hidden[..., span:])
    record_property("metrics", json.dumps(rows))
    # Application-only bracket: table generation is unchanged and excluded on both sides.
    from nemo_automodel.components.models.step5.rope import _apply_rotary_compiled

    def baseline_step() -> torch.Tensor:
        a, b = hidden[..., :half].float(), hidden[..., half:span].float()
        output = torch.cat(((a * cos - b * sin).bfloat16(), (a * sin + b * cos).bfloat16(), hidden[..., span:]), -1)
        return torch.autograd.grad(output, hidden, upstream)[0]

    def candidate_step() -> torch.Tensor:
        output = _apply_rotary_compiled(hidden, cos, sin, span)
        return torch.autograd.grad(output, hidden, upstream)[0]

    record_property(
        "cuda_apply_fwb_ms",
        json.dumps(
            {
                "baseline_before": _time_cuda(baseline_step),
                "compiled": _time_cuda(candidate_step),
                "baseline_after": _time_cuda(baseline_step),
            }
        ),
    )
