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

import types

import torch

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.trackio_utils import TrackioLogger


def _fake_run():
    """Minimal stand-in for ``trackio.Run`` capturing calls."""
    calls = {"log": [], "finish": 0}

    def _log(metrics, step=None):
        calls["log"].append((metrics, step))

    def _finish():
        calls["finish"] += 1

    return types.SimpleNamespace(log=_log, finish=_finish), calls


def test_log_metrics_converts_types_and_uses_step():
    run, calls = _fake_run()

    TrackioLogger(run).log_metrics(
        {
            "int_val": 3,
            "float_val": 2.5,
            "tensor_scalar": torch.tensor(4.0),
            "tensor_vec": torch.tensor([1.0, 3.0]),
            "flag": True,
            "name": "default",
        },
        step=5,
    )

    logged_metrics, step = calls["log"][-1]
    assert step == 5
    assert logged_metrics == {"int_val": 3.0, "float_val": 2.5, "tensor_scalar": 4.0, "tensor_vec": 2.0}
    assert all(isinstance(v, float) for v in logged_metrics.values())


def test_log_metrics_drops_trackio_reserved_keys():
    """``MetricsSample.to_dict()`` carries ``step``/``timestamp``, which Trackio reserves for its own fields."""
    run, calls = _fake_run()

    TrackioLogger(run).log_metrics(MetricsSample(step=3, epoch=1, metrics={"loss": 0.5}).to_dict(), step=3)

    assert calls["log"] == [({"epoch": 1.0, "loss": 0.5}, 3)]


def test_log_metrics_noop_when_no_metric_survives():
    run, calls = _fake_run()

    TrackioLogger(run).log_metrics({"step": 1, "timestamp": "2026-10-07T00:00:00Z"}, step=1)

    assert not calls["log"], "run.log should not be called with an empty payload"


def test_finish_calls_run_finish():
    run, calls = _fake_run()

    TrackioLogger(run).finish()

    assert calls["finish"] == 1
