# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import types
from dataclasses import dataclass

import pytest
import torch


def _install_fake_mlflow():
    """
    Install a minimal stub mlflow package into sys.modules capturing calls.
    """
    mlflow = types.ModuleType("mlflow")
    pytorch_mod = types.ModuleType("mlflow.pytorch")

    calls = {
        "set_tracking_uri": [],
        "get_experiment_by_name": [],
        "create_experiment": [],
        "start_run": [],
        "log_params": [],
        "log_metrics": [],
        "log_artifacts": [],
        "log_artifact": [],
        "log_model": [],
        "end_run": 0,
    }

    class _RunInfo:
        def __init__(self, run_id="run-123"):
            self.run_id = run_id

    class _Run:
        def __init__(self):
            self.info = _RunInfo()

    def set_tracking_uri(uri):
        calls["set_tracking_uri"].append(uri)

    def get_tracking_uri():
        return "file:///tmp/mlruns"

    @dataclass
    class _Experiment:
        experiment_id: str

    def get_experiment_by_name(name):
        calls["get_experiment_by_name"].append(name)
        # Return an existing experiment
        return _Experiment(experiment_id="exp-1")

    def create_experiment(name, artifact_location=None):
        calls["create_experiment"].append((name, artifact_location))
        return "exp-created"

    def start_run(**kwargs):
        calls["start_run"].append(kwargs)
        return _Run()

    def log_params(params):
        calls["log_params"].append(params)

    def log_metrics(metrics, step=None):
        calls["log_metrics"].append((metrics, step))

    def log_artifacts(local_dir, artifact_path=None):
        calls["log_artifacts"].append((local_dir, artifact_path))

    def log_artifact(local_path, artifact_path=None):
        calls["log_artifact"].append((local_path, artifact_path))

    def end_run():
        calls["end_run"] += 1

    def log_model(**kwargs):
        calls["log_model"].append(kwargs)

    # Bind functions
    mlflow.set_tracking_uri = set_tracking_uri
    mlflow.get_tracking_uri = get_tracking_uri
    mlflow.get_experiment_by_name = get_experiment_by_name
    mlflow.create_experiment = create_experiment
    mlflow.start_run = start_run
    mlflow.log_params = log_params
    mlflow.log_metrics = log_metrics
    mlflow.log_artifacts = log_artifacts
    mlflow.log_artifact = log_artifact
    mlflow.end_run = end_run
    pytorch_mod.log_model = log_model

    sys.modules.update(
        {
            "mlflow": mlflow,
            "mlflow.pytorch": pytorch_mod,
        }
    )
    return calls


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    original = set(sys.modules.keys())
    yield
    for name in list(sys.modules):
        if name.startswith("mlflow"):
            del sys.modules[name]
    for name in list(sys.modules):
        if name not in original and name.startswith("mlflow"):
            del sys.modules[name]


def test_build_mlflow_starts_run_and_sets_expected_tags(monkeypatch):
    calls = _install_fake_mlflow()

    # Pretend we're in rank 0 initialized process group
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    # Import after stubbing mlflow
    from nemo_automodel.components.loggers.mlflow_utils import build_mlflow

    # Minimal config stubs
    class Tags:
        def __init__(self, data):
            self._d = data

        def to_dict(self):
            return dict(self._d)

    class MlflowCfg:
        def __init__(self):
            self._data = {
                "experiment_name": "automodel-exp",
                "run_name": "run-A",
                "tracking_uri": "file:///tmp/mlruns",
                "artifact_location": "/tmp/mlruns",
                "tags": Tags({"task": "finetune"}),
            }

        def get(self, key, default=None):
            return self._data.get(key, default)

    class StepSchedulerCfg:
        def get(self, key, default=None):
            return {"global_batch_size": 64, "local_batch_size": 8}.get(key, default)

    class ModelCfg:
        pretrained_model_name_or_path = "dummy/model"

    class Cfg:
        def __init__(self):
            self.mlflow = MlflowCfg()
            self.model = ModelCfg()
            self.step_scheduler = StepSchedulerCfg()

        def get(self, key, default=None):
            return getattr(self, key, default)

        def to_dict(self):
            return {"foo": "bar"}

    cfg = Cfg()
    logger = build_mlflow(cfg)
    assert logger is not None
    # Ensure start_run was invoked with enriched tags
    assert calls["start_run"], "mlflow.start_run should have been called"
    start_kwargs = calls["start_run"][-1]
    tags = start_kwargs["tags"]
    assert tags["task"] == "finetune"
    assert tags["model"] == "dummy/model"
    assert tags["global_batch_size"] == "64"
    assert tags["local_batch_size"] == "8"


def test_log_params_flattens_and_stringifies(monkeypatch):
    calls = _install_fake_mlflow()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.mlflow_utils import MLflowLogger

    logger = MLflowLogger(experiment_name="exp", run_name="r1", tags={})
    logger.log_params({"a": 1, "b": {"x": 2, "y": True}, "c": [1, 2]})

    assert calls["log_params"], "mlflow.log_params not called"
    params = calls["log_params"][-1]
    assert params["a"] == "1"
    assert params["b.x"] == "2"
    assert params["b.y"] == "True"
    assert params["c"] == "[1, 2]"


def test_log_metrics_converts_types_and_uses_step(monkeypatch):
    calls = _install_fake_mlflow()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.mlflow_utils import MLflowLogger

    logger = MLflowLogger(experiment_name="exp", run_name="r1", tags={})
    metrics = {
        "int_val": 3,
        "float_val": 2.5,
        "tensor_scalar": torch.tensor(4.0),
        "tensor_vec": torch.tensor([1.0, 3.0]),
        "skip_obj": object(),
    }
    logger.log_metrics(metrics, step=5)

    assert calls["log_metrics"], "mlflow.log_metrics not called"
    logged_metrics, step = calls["log_metrics"][-1]
    assert step == 5
    assert isinstance(logged_metrics["int_val"], float) and logged_metrics["int_val"] == 3.0
    assert isinstance(logged_metrics["float_val"], float) and logged_metrics["float_val"] == 2.5
    assert isinstance(logged_metrics["tensor_scalar"], float) and logged_metrics["tensor_scalar"] == 4.0
    assert isinstance(logged_metrics["tensor_vec"], float), "tensor vectors should be averaged to float"
    assert "skip_obj" not in logged_metrics


def test_rank_guard_and_run_guard(monkeypatch):
    calls = _install_fake_mlflow()
    import torch.distributed as dist

    # Initialize as rank 0 to create the run
    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.mlflow_utils import MLflowLogger

    logger = MLflowLogger(experiment_name="exp", run_name="r1", tags={})
    # Switch to non-zero rank -> calls should NO-OP
    monkeypatch.setattr(dist, "get_rank", lambda: 1, raising=False)
    logger.log_metrics({"a": 1.0}, step=1)
    assert not calls["log_metrics"], "mlflow.log_metrics should not be called on non-main rank"

    # Reset to rank0 but clear the run -> NO-OP
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)
    logger.run = None
    logger.log_params({"x": 1})
    assert not calls["log_params"], "mlflow.log_params should not be called when run is None"


def test_context_manager_calls_end_run(monkeypatch):
    calls = _install_fake_mlflow()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.mlflow_utils import MLflowLogger

    with MLflowLogger(experiment_name="exp", run_name="rx", tags={}):
        pass
    assert calls["end_run"] == 1


