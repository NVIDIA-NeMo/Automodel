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

"""Tests for the Trackio logger: metric payloads, a real local run, and rank-0-only logging under torch.distributed."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.trackio_utils import TrackioLogger


class _RecordingRun:
    def __init__(self):
        self.calls = []
        self.finished = False

    def log(self, metrics, step=None):
        self.calls.append((metrics, step))

    def finish(self):
        self.finished = True


class TestTrackioLogger:
    def test_metrics_sample_drops_reserved_keys_and_keeps_epoch(self):
        """``MetricsSample.to_dict()`` carries ``step``/``timestamp``, which Trackio reserves; they must not be sent."""
        run = _RecordingRun()
        sample = MetricsSample(step=3, epoch=1, metrics={"loss": 0.5, "lr": 1e-4})

        TrackioLogger(run).log_metrics(sample.to_dict(), step=3)

        assert run.calls == [({"epoch": 1.0, "loss": 0.5, "lr": 1e-4}, 3)]

    def test_tensors_become_floats_and_non_numeric_values_are_skipped(self):
        run = _RecordingRun()

        TrackioLogger(run).log_metrics(
            {
                "scalar": torch.tensor(2.0),
                "vector": torch.tensor([1.0, 3.0]),
                "count": 7,
                "flag": True,
                "name": "default",
            },
            step=1,
        )

        assert run.calls == [({"scalar": 2.0, "vector": 2.0, "count": 7.0}, 1)]

    def test_nothing_is_logged_when_no_metric_survives(self):
        run = _RecordingRun()

        TrackioLogger(run).log_metrics({"step": 1, "timestamp": "2026-10-07T00:00:00Z"}, step=1)

        assert run.calls == []

    def test_finish_finishes_the_run(self):
        run = _RecordingRun()

        TrackioLogger(run).finish()

        assert run.finished


_ROUND_TRIP = textwrap.dedent(
    """
    import json, sys
    from nemo_automodel.components.loggers.loggers import TrackioConfig
    from trackio.sqlite_storage import SQLiteStorage

    logger = TrackioConfig(project="nemo-test", name="unit").build(run_config={"lr": 1e-3})
    logger.log_metrics({"step": 0, "timestamp": "t", "loss": 2.0, "lr": 1e-3}, step=0)
    logger.log_metrics({"loss": 1.0}, step=1)
    logger.finish()
    logs = SQLiteStorage.get_logs("nemo-test", "unit")
    print("RESULT " + json.dumps({"runs": SQLiteStorage.get_runs("nemo-test"), "loss": [log["loss"] for log in logs]}))
    """
)

_DISTRIBUTED = textwrap.dedent(
    """
    import json, sys
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from nemo_automodel.components.loggers.loggers import TrackioConfig


    def worker(rank, init_file):
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
        logger = TrackioConfig(project="nemo-dist").build(run_config={"world_size": 2}, model_name="org/model")
        if logger is not None:
            assert rank == 0
            for step in range(3):
                logger.log_metrics({"loss": float(step)}, step=step)
            logger.finish()
        dist.barrier()
        dist.destroy_process_group()


    if __name__ == "__main__":
        mp.spawn(worker, args=(sys.argv[1],), nprocs=2, join=True)
        from trackio.sqlite_storage import SQLiteStorage

        runs = SQLiteStorage.get_runs("nemo-dist")
        logs = SQLiteStorage.get_logs("nemo-dist", runs[0]) if runs else []
        print("RESULT " + json.dumps({"runs": runs, "steps": [log["step"] for log in logs]}))
    """
)


def _run_isolated(script: str, tmp_path, *args: str) -> dict:
    """Run ``script`` in a fresh interpreter whose Trackio store is ``tmp_path`` (Trackio reads it at import)."""
    path = tmp_path / "script.py"
    path.write_text(script)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACKIO_")}
    env["TRACKIO_DIR"] = str(tmp_path / "trackio")
    result = subprocess.run(
        [sys.executable, str(path), *args], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    # Trackio prints status lines of its own; the script tags its result line.
    (line,) = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    return json.loads(line.removeprefix("RESULT "))


@pytest.mark.timeout(150)
def test_real_trackio_round_trip(tmp_path):
    """A local Trackio run receives the metrics, without the reserved keys tripping a rename."""
    pytest.importorskip("trackio")

    result = _run_isolated(_ROUND_TRIP, tmp_path)

    assert result == {"runs": ["unit"], "loss": [2.0, 1.0]}


@pytest.mark.timeout(150)
def test_only_rank_zero_creates_a_run_under_torch_distributed(tmp_path):
    """With two gloo ranks both calling ``build``, exactly one run exists and holds every logged step."""
    pytest.importorskip("trackio")

    result = _run_isolated(_DISTRIBUTED, tmp_path, str(tmp_path / "init"))

    assert result == {"runs": ["org_model"], "steps": [0, 1, 2]}
