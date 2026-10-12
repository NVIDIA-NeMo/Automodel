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

"""Actual compiled grouped-Q RMSNorm layout, mathematics, and replay on CUDA."""

import json
from collections.abc import Callable

import pytest
import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.step5.norm import Step5RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compare tensors with the same semantic shape.

    Args:
        actual: Observed query [batch, sequence, heads, dim], QKV gradient
            [batch, sequence, groups, heads_per_group + 2, dim], or scale [dim].
        reference: Independent reference with the same shape and dtype.

    Returns:
        Relative L2, maximum error over reference RMS, and absolute error.
    """
    difference = actual.float() - reference.float()
    rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return {
        "relative_l2": (difference.square().mean().sqrt() / rms).item(),
        "max_over_rms": (difference.abs().max() / rms).item(),
        "max_abs": difference.abs().max().item(),
        "different_fraction": (actual != reference).float().mean().item(),
    }


@pytest.mark.parametrize("sequence", [4096, 8192])
def test_qnorm_grouped_cuda_math_replay_and_no_input_clone(
    sequence: int, record_property: Callable[[str, str], None]
) -> None:
    torch.manual_seed(1139)
    module = Step5RMSNorm(192, compile_norm=True).cuda()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192, device="cuda") + 0.00123)
    projection = torch.randn(1, sequence, 4, 18, 192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    query = projection[..., :16, :]
    assert not query.is_contiguous()
    normalized = module(query)
    assert normalized.is_contiguous()
    actual = normalized.flatten(2, 3)
    assert actual.untyped_storage().data_ptr() == normalized.untyped_storage().data_ptr()
    assert module.weight.dtype == torch.float32 and actual.dtype == torch.bfloat16
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (projection, module.weight), upstream)

    reference_projection = projection.detach().clone().requires_grad_()
    reference_weight = module.weight.detach().clone().requires_grad_()
    expected = (
        F.rms_norm(reference_projection[..., :16, :].float(), (192,), reference_weight + 1, module.variance_epsilon)
        .bfloat16()
        .flatten(2, 3)
    )
    reference_gradients = torch.autograd.grad(expected, (reference_projection, reference_weight), upstream)
    metrics = {}
    for name, observed, reference in zip(
        ("output", "qkv_gradient", "weight_gradient"), (actual, *actual_gradients), (expected, *reference_gradients)
    ):
        assert torch.isfinite(observed).all()
        metrics[name] = _metrics(observed, reference)
        # Existing Step5 norm-fusion gates; neither is widened for this layout.
        assert metrics[name]["relative_l2"] < 0.002, (name, metrics[name])
        assert metrics[name]["max_over_rms"] < 0.05, (name, metrics[name])
    assert torch.count_nonzero(actual_gradients[0][..., 16:, :]) == 0

    # The previous compiled two-dimensional path is a layout control, not the
    # mathematical oracle. It deliberately retains the pre-norm Q copy.
    legacy = module(query.reshape(1, sequence, 64, 192))
    legacy_gradients = torch.autograd.grad(legacy, (projection, module.weight), upstream)
    for name, observed, reference in zip(
        ("legacy_output", "legacy_qkv_gradient", "legacy_weight_gradient"),
        (actual, *actual_gradients),
        (legacy, *legacy_gradients),
    ):
        metrics[name] = _metrics(observed, reference)
        assert metrics[name]["relative_l2"] < 0.002, (name, metrics[name])
        assert metrics[name]["max_over_rms"] < 0.05, (name, metrics[name])

    replay = checkpoint(module, query, use_reentrant=False).flatten(2, 3)
    replay_gradients = torch.autograd.grad(replay, (projection, module.weight), upstream)
    repeat = module(query).flatten(2, 3)
    repeat_gradients = torch.autograd.grad(repeat, (projection, module.weight), upstream)
    with torch.no_grad():
        inference = module(query).flatten(2, 3)
    for observed, reference in zip(
        (replay, *replay_gradients, repeat, *repeat_gradients, inference),
        (actual, *actual_gradients, actual, *actual_gradients, actual),
    ):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    traces = {}
    peaks = {}
    for label in ("legacy", "grouped"):
        with torch.no_grad():
            # Compile no-grad specializations before observing actual execution.
            for _ in range(2):
                if label == "legacy":
                    result = module(query.reshape(1, sequence, 64, 192))
                else:
                    result = module(query).flatten(2, 3)
                del result
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
                if label == "legacy":
                    result = module(query.reshape(1, sequence, 64, 192))
                else:
                    result = module(query).flatten(2, 3)
                torch.cuda.synchronize()
            peaks[label] = torch.cuda.max_memory_allocated() - before
            del result
        events = trace.events()
        clones = [event.input_shapes for event in events if event.name == "aten::clone"]
        compiled = sorted({event.name for event in events if "Compiled" in event.name})
        kernels = sorted({event.name for event in events if event.device_type == torch.autograd.DeviceType.CUDA})
        traces[label] = {"clone_shapes": clones, "compiled_events": compiled, "cuda_kernels": kernels}
        assert compiled and kernels, traces[label]
    assert traces["legacy"]["clone_shapes"], traces
    assert not traces["grouped"]["clone_shapes"], traces
    # This proves materialization elimination in the executed compiled entry,
    # independently of the CPU/FakeTensor storage checks.
    expected_copy_bytes = query.numel() * query.element_size()
    assert peaks["legacy"] - peaks["grouped"] >= expected_copy_bytes, peaks
    record_property("numerical_metrics", json.dumps(metrics))
    record_property("compiled_execution", json.dumps(traces))
    record_property("incremental_forward_peak_bytes", json.dumps(peaks))
