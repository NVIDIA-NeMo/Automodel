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

"""CUDA gate for paired Q/K table reuse, including original QKV view gradients."""

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


@pytest.mark.parametrize("span", [64, 192])
@pytest.mark.parametrize("sequence", [0, 1, 4096, 8192])
def test_paired_rope_cuda_output_qkv_gradient_and_recompute(sequence: int, span: int, record_property) -> None:
    torch.manual_seed(988)
    config = Step5Config()
    config.rope_theta = 10000000.0 if span == 64 else 10000.0
    rope = Step5RotaryEmbedding(config, 3 if span == 64 else 0, span=span).cuda()
    source = torch.randn(1, sequence, 4, 18, 192, device="cuda", dtype=torch.bfloat16)
    expected_input = source.clone().requires_grad_()
    actual_input = source.clone().requires_grad_()
    positions = torch.arange(sequence, device="cuda").unsqueeze(0) + 131072 - sequence
    positions[:, sequence // 2 :] = torch.arange(sequence - sequence // 2, device="cuda")

    def qk(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Original interleaved QKV source: key is a strided view.
        return value[..., :16, :].reshape(1, sequence, 64, 192), value[..., 16, :]

    expected_q, expected_k = qk(expected_input)
    expected = rope(expected_q, positions), rope(expected_k, positions)
    actual = rope.forward_pair(*qk(actual_input), positions)
    upstream = tuple(torch.randn_like(value) for value in expected)
    expected_grad = torch.autograd.grad(expected, expected_input, upstream)[0]
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    for observed, reference in zip((*actual, actual_grad), (*expected, expected_grad)):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    with torch.no_grad():
        no_grad = rope.forward_pair(*qk(source), positions)
    for observed, reference in zip(no_grad, actual):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    replay_input = source.clone().requires_grad_()
    replay = checkpoint(rope.forward_pair, *qk(replay_input), positions, use_reentrant=False)
    replay_grad = torch.autograd.grad(replay, replay_input, upstream)[0]
    for observed, reference in zip((*replay, replay_grad), (*actual, actual_grad)):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    if sequence < 4096:
        return

    def baseline_step() -> torch.Tensor:
        query, key = qk(expected_input)
        result = rope(query, positions), rope(key, positions)
        return torch.autograd.grad(result, expected_input, upstream)[0]

    def paired_step() -> torch.Tensor:
        result = rope.forward_pair(*qk(actual_input), positions)
        return torch.autograd.grad(result, actual_input, upstream)[0]

    record_property(
        "cuda_pair_fwb_ms",
        json.dumps(
            {
                "baseline_before": _time_cuda(baseline_step),
                "paired": _time_cuda(paired_step),
                "baseline_after": _time_cuda(baseline_step),
            }
        ),
    )
    peaks = {}
    for name, call in [("baseline", baseline_step), ("paired", paired_step)]:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - before
    record_property("incremental_fwb_peak_bytes", json.dumps(peaks))
