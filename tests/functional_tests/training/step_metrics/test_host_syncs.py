# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Host synchronization budget for steady-state packed MoE training."""

import json
import math
from collections import Counter
from contextlib import nullcontext
from itertools import count
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

_PROFILED_STEPS = (1, 2)
_SYNC_APIS = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
}


def _host_syncs(trace_path: Path, ranges: tuple[str, ...]) -> dict[str, Counter[tuple[str, str]]]:
    events = json.loads(trace_path.read_text())["traceEvents"]
    # External ids link CUPTI calls to their issuing PyTorch operators, even
    # when backward runs on a worker thread. Explicit waits can have no operator.
    cpu_ops = {
        event["args"]["External id"]: event["name"]
        for event in events
        if event.get("cat") == "cpu_op" and "External id" in event.get("args", {})
    }
    syncs = []
    for event in events:
        if event.get("cat") not in ("cuda_runtime", "cuda_driver") or event.get("ph") != "X":
            continue
        api = event["name"].split("_", 1)[0]  # CUPTI may emit suffixes such as _v2 or _ptsz.
        blocking_copy = api.startswith(("cudaMemcpy", "cuMemcpy")) and "Async" not in api
        if api in _SYNC_APIS or blocking_copy:
            op = cpu_ops.get(event.get("args", {}).get("External id"), "<no ATen op>")
            syncs.append((event["ts"], api, op))

    counts = {}
    for name in ranges:
        # Ignore the corresponding GPU annotations and profiler setup/teardown.
        markers = [event for event in events if event.get("cat") == "user_annotation" and event["name"] == name]
        assert len(markers) == 1, f"Expected one CPU range named {name}, got {len(markers)} in {trace_path}"
        marker = markers[0]
        start, end = marker["ts"], marker["ts"] + marker["dur"]
        counts[name] = Counter((api, op) for ts, api, op in syncs if start <= ts < end)
    return counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA runtime profiling")
def test_packed_moe_training_step_host_sync_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile real accumulated optimizer steps through metric publication."""
    cfg = load_yaml_config(Path(__file__).with_name("tiny_qwen3_moe.yaml"))
    cfg.checkpoint.checkpoint_dir = str(tmp_path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    original_iter = StepScheduler.__iter__

    def record_steps(scheduler: StepScheduler):
        iterator = original_iter(scheduler)
        for step in count(scheduler.step):
            # Span batch loading through reporter polling. Step 0 warms up,
            # and the terminal step may wait for checkpoint logging.
            scope = record_function(f"train_step_{step}") if step in _PROFILED_STEPS else nullcontext()
            with scope:
                try:
                    batches = next(iterator)
                except StopIteration:
                    return
                yield batches

    try:
        recipe.setup()
        monkeypatch.setattr(StepScheduler, "__iter__", record_steps)
        device = recipe.dist_env.device
        scalar = torch.ones((), device=device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            with record_function("blocking_scalar_read_control"):
                scalar.item()
            recipe.run_train_validation_loop()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    trace_path = tmp_path / "training-trace.json"
    profiler.export_chrome_trace(str(trace_path))
    step_ranges = tuple(f"train_step_{step}" for step in _PROFILED_STEPS)
    counts = _host_syncs(trace_path, ("blocking_scalar_read_control", *step_ranges))
    assert counts["blocking_scalar_read_control"] == {("cudaStreamSynchronize", "aten::_local_scalar_dense"): 1}, (
        f"The profiler must see a known GPU scalar read before enforcing the budget. See {trace_path}"
    )

    # The Torch routing loop uses one dynamic nonzero per expert. Two layers,
    # four experts, and two microbatches, each with forward and recompute, give
    # 32 waits. Everything else in the step has a zero-wait budget.
    expected = {("cudaStreamSynchronize", "aten::nonzero"): 32}
    for name in step_ranges:
        assert counts[name] == expected, f"{name} host syncs {counts[name]}, expected {expected}. See {trace_path}"

    reports = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [report["step"] for report in reports] == list(range(cfg.step_scheduler.max_steps))
    for report in reports:
        for key in ("loss", "grad_norm", "step_time_seconds", "tps"):
            assert math.isfinite(report[key]) and report[key] > 0, report
        assert 0 < report["num_label_tokens"] <= report["num_tokens_per_step"]
        assert 0 < report["pack_fill"] <= 1
        assert report["tps"] == pytest.approx(report["num_tokens_per_step"] / report["step_time_seconds"])
