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

import pytest
import torch


def _install_fake_comet_ml():
    """
    Install a minimal stub comet_ml package into sys.modules capturing calls.
    """
    comet_ml = types.ModuleType("comet_ml")

    calls = {
        "experiment_init": [],
        "log_parameters": [],
        "log_metrics": [],
        "set_name": [],
        "add_tags": [],
        "end": 0,
    }

    class _FakeExperiment:
        url = "https://www.comet.com/test/fake-experiment"

        def __init__(self, **kwargs):
            calls["experiment_init"].append(kwargs)

        def log_parameters(self, params):
            calls["log_parameters"].append(params)

        def log_metrics(self, metrics, step=None):
            calls["log_metrics"].append((metrics, step))

        def set_name(self, name):
            calls["set_name"].append(name)

        def add_tags(self, tags):
            calls["add_tags"].append(tags)

        def end(self):
            calls["end"] += 1

    comet_ml.Experiment = _FakeExperiment
    sys.modules["comet_ml"] = comet_ml
    return calls


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    yield
    for name in list(sys.modules):
        if name.startswith("comet") or "comet_utils" in name:
            del sys.modules[name]


def test_build_comet_creates_experiment_with_config(monkeypatch):
    calls = _install_fake_comet_ml()

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import build_comet

    class CometCfg:
        def __init__(self):
            self._data = {
                "project_name": "test-project",
                "workspace": "test-workspace",
                "experiment_name": "test-run",
                "api_key": None,
                "tags": ["finetune", "llama"],
                "auto_metric_logging": False,
            }

        def get(self, key, default=None):
            return self._data.get(key, default)

    class ModelCfg:
        pretrained_model_name_or_path = "org/my-model"

    class Cfg:
        def __init__(self):
            self.comet = CometCfg()
            self.model = ModelCfg()

        def get(self, key, default=None):
            return getattr(self, key, default)

        def to_dict(self):
            return {"model": "org/my-model"}

    cfg = Cfg()
    logger = build_comet(cfg)
    assert logger is not None

    assert calls["experiment_init"], "comet_ml.Experiment should have been called"
    init_kwargs = calls["experiment_init"][-1]
    assert init_kwargs["project_name"] == "test-project"
    assert init_kwargs["workspace"] == "test-workspace"
    assert init_kwargs["auto_metric_logging"] is False

    assert calls["set_name"] == ["test-run"]
    assert calls["add_tags"], "add_tags should have been called"
    tags = calls["add_tags"][-1]
    assert "finetune" in tags
    assert "llama" in tags
    assert "model:org/my-model" in tags


def test_build_comet_auto_generates_experiment_name(monkeypatch):
    calls = _install_fake_comet_ml()

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import build_comet

    class CometCfg:
        def __init__(self):
            self._data = {"project_name": "test-project"}

        def get(self, key, default=None):
            return self._data.get(key, default)

    class ModelCfg:
        pretrained_model_name_or_path = "org/my-model"

    class Cfg:
        def __init__(self):
            self.comet = CometCfg()
            self.model = ModelCfg()

        def get(self, key, default=None):
            return getattr(self, key, default)

    build_comet(Cfg())
    assert calls["set_name"] == ["org_my-model"]


def test_build_comet_raises_without_config(monkeypatch):
    _install_fake_comet_ml()

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import build_comet

    class Cfg:
        def get(self, key, default=None):
            return default

    with pytest.raises(ValueError, match="Comet configuration not found"):
        build_comet(Cfg())


def test_log_params_delegates_to_experiment(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    logger.log_params({"lr": 0.001, "batch_size": 8})

    assert calls["log_parameters"], "experiment.log_parameters should have been called"
    params = calls["log_parameters"][-1]
    assert params["lr"] == 0.001
    assert params["batch_size"] == 8


def test_log_metrics_converts_types_and_uses_step(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    metrics = {
        "int_val": 3,
        "float_val": 2.5,
        "tensor_scalar": torch.tensor(4.0),
        "tensor_vec": torch.tensor([1.0, 3.0]),
        "skip_obj": object(),
    }
    logger.log_metrics(metrics, step=5)

    assert calls["log_metrics"], "experiment.log_metrics should have been called"
    logged_metrics, step = calls["log_metrics"][-1]
    assert step == 5
    assert isinstance(logged_metrics["int_val"], float) and logged_metrics["int_val"] == 3.0
    assert isinstance(logged_metrics["float_val"], float) and logged_metrics["float_val"] == 2.5
    assert isinstance(logged_metrics["tensor_scalar"], float) and logged_metrics["tensor_scalar"] == 4.0
    assert isinstance(logged_metrics["tensor_vec"], float), "tensor vectors should be averaged to float"
    assert "skip_obj" not in logged_metrics


def test_log_metrics_without_step(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    logger.log_metrics({"loss": 0.5})

    logged_metrics, step = calls["log_metrics"][-1]
    assert step is None
    assert logged_metrics["loss"] == 0.5


def test_rank_guard_blocks_non_rank_zero(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    # Switch to non-zero rank -> calls should NO-OP
    monkeypatch.setattr(dist, "get_rank", lambda: 1, raising=False)
    logger.log_metrics({"a": 1.0}, step=1)
    assert not calls["log_metrics"], "log_metrics should not be called on non-main rank"

    logger.log_params({"x": 1})
    assert not calls["log_parameters"], "log_parameters should not be called on non-main rank"


def test_experiment_none_guard(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    logger.experiment = None

    logger.log_metrics({"a": 1.0}, step=1)
    assert not calls["log_metrics"], "log_metrics should not be called when experiment is None"

    logger.log_params({"x": 1})
    assert not calls["log_parameters"], "log_params should not be called when experiment is None"


def test_end_calls_experiment_end(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    logger.end()
    assert calls["end"] == 1


def test_end_noop_when_no_experiment(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    logger.experiment = None
    logger.end()
    assert calls["end"] == 0


def test_context_manager_calls_end(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    with CometLogger(project_name="p"):
        pass
    assert calls["end"] == 1


def test_no_experiment_created_on_non_rank_zero(monkeypatch):
    calls = _install_fake_comet_ml()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: 1, raising=False)

    from nemo_automodel.components.loggers.comet_utils import CometLogger

    logger = CometLogger(project_name="p")
    assert logger.experiment is None
    assert not calls["experiment_init"], "Experiment should not be created on non-rank-0"
