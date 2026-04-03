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
from unittest import mock

import pytest

# Stub out nemo_run before importing utils so it is never required at
# module level.
_mock_run = mock.MagicMock()
_mock_exp = mock.MagicMock()
_mock_exp.__enter__ = mock.MagicMock(return_value=_mock_exp)
_mock_exp.__exit__ = mock.MagicMock(return_value=False)
_mock_run.Experiment.return_value = _mock_exp

sys.modules["nemo_run"] = _mock_run

from nemo_automodel.components.launcher.nemo_run.utils import (  # noqa: E402
    REMOTE_CONFIG_PATH,
    apply_overrides,
    load_executor_from_file,
    submit_nemo_run_job,
)


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset shared mocks before each test."""
    _mock_run.reset_mock()
    _mock_exp.reset_mock()
    _mock_exp.__enter__.reset_mock()
    _mock_exp.__exit__.reset_mock()
    _mock_run.Experiment.return_value = _mock_exp
    yield


# ---------------------------------------------------------------------------
# load_executor_from_file
# ---------------------------------------------------------------------------


class TestLoadExecutorFromFile:
    def test_loads_executor_from_map(self, tmp_path):
        executors_file = tmp_path / "executors.py"
        executors_file.write_text(
            "EXECUTOR_MAP = {\n"
            '    "my_cluster": type("Exec", (), {"launch": True})(),\n'
            "}\n"
        )
        executor = load_executor_from_file("my_cluster", str(executors_file))
        assert hasattr(executor, "launch")

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Executor definitions file not found"):
            load_executor_from_file("any", "/nonexistent/path/executors.py")

    def test_missing_key_raises_key_error(self, tmp_path):
        executors_file = tmp_path / "executors.py"
        executors_file.write_text(
            "EXECUTOR_MAP = {\n"
            '    "cluster_a": "exec_a",\n'
            '    "cluster_b": "exec_b",\n'
            "}\n"
        )
        with pytest.raises(KeyError, match="not found in EXECUTOR_MAP"):
            load_executor_from_file("nonexistent", str(executors_file))

    def test_missing_executor_map_raises_attribute_error(self, tmp_path):
        executors_file = tmp_path / "executors.py"
        executors_file.write_text("SOME_OTHER_VAR = 42\n")
        with pytest.raises(AttributeError, match="does not define an EXECUTOR_MAP"):
            load_executor_from_file("any", str(executors_file))

    def test_callable_executor_is_invoked(self, tmp_path):
        executors_file = tmp_path / "executors.py"
        # A callable that returns an object without a 'launch' attribute
        # should be called to produce the executor.
        executors_file.write_text(
            "def _make_exec():\n"
            '    return {"type": "slurm"}\n'
            "\n"
            'EXECUTOR_MAP = {"lazy": _make_exec}\n'
        )
        executor = load_executor_from_file("lazy", str(executors_file))
        assert executor == {"type": "slurm"}

    def test_executor_with_launch_attr_not_called(self, tmp_path):
        executors_file = tmp_path / "executors.py"
        # An object that has a 'launch' attribute should be returned as-is
        # even if it is callable.
        executors_file.write_text(
            "class MyExec:\n"
            "    launch = True\n"
            "    call_count = 0\n"
            "    def __call__(self):\n"
            "        self.call_count += 1\n"
            "        return self\n"
            "\n"
            'EXECUTOR_MAP = {"direct": MyExec()}\n'
        )
        executor = load_executor_from_file("direct", str(executors_file))
        assert executor.call_count == 0

    def test_key_error_shows_available_keys(self, tmp_path):
        executors_file = tmp_path / "executors.py"
        executors_file.write_text(
            'EXECUTOR_MAP = {"alpha": 1, "beta": 2, "gamma": 3}\n'
        )
        with pytest.raises(KeyError, match="alpha") as exc_info:
            load_executor_from_file("missing", str(executors_file))
        # All available keys should be listed
        msg = str(exc_info.value)
        assert "beta" in msg
        assert "gamma" in msg


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_nodes_override(self):
        executor = mock.MagicMock()
        executor.nodes = 1
        apply_overrides(executor, nodes=4, devices=None, container_image=None,
                        time=None, mounts=None, env_vars=None)
        assert executor.nodes == 4

    def test_devices_override_ntasks_per_node(self):
        executor = mock.MagicMock(spec=["ntasks_per_node"])
        executor.ntasks_per_node = 1
        apply_overrides(executor, nodes=None, devices=8, container_image=None,
                        time=None, mounts=None, env_vars=None)
        assert executor.ntasks_per_node == 8

    def test_devices_override_gpus_per_node(self):
        executor = mock.MagicMock(spec=["gpus_per_node"])
        executor.gpus_per_node = 1
        apply_overrides(executor, nodes=None, devices=4, container_image=None,
                        time=None, mounts=None, env_vars=None)
        assert executor.gpus_per_node == 4

    def test_devices_override_both_attrs(self):
        executor = mock.MagicMock(spec=["ntasks_per_node", "gpus_per_node"])
        executor.ntasks_per_node = 1
        executor.gpus_per_node = 1
        apply_overrides(executor, nodes=None, devices=8, container_image=None,
                        time=None, mounts=None, env_vars=None)
        assert executor.ntasks_per_node == 8
        assert executor.gpus_per_node == 8

    def test_container_image_override(self):
        executor = mock.MagicMock()
        apply_overrides(executor, nodes=None, devices=None,
                        container_image="nvcr.io/nvidia/nemo:24.05",
                        time=None, mounts=None, env_vars=None)
        assert executor.container_image == "nvcr.io/nvidia/nemo:24.05"

    def test_time_override(self):
        executor = mock.MagicMock()
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time="08:00:00", mounts=None, env_vars=None)
        assert executor.time == "08:00:00"

    def test_mounts_merge(self):
        executor = mock.MagicMock()
        executor.container_mounts = ["/data:/data"]
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=["/models:/models", "/scratch:/scratch"],
                        env_vars=None)
        assert executor.container_mounts == [
            "/data:/data",
            "/models:/models",
            "/scratch:/scratch",
        ]

    def test_mounts_merge_with_empty_existing(self):
        executor = mock.MagicMock()
        executor.container_mounts = []
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=["/new:/new"], env_vars=None)
        assert executor.container_mounts == ["/new:/new"]

    def test_mounts_merge_with_none_existing(self):
        executor = mock.MagicMock()
        executor.container_mounts = None
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=["/new:/new"], env_vars=None)
        assert executor.container_mounts == ["/new:/new"]

    def test_env_vars_merge(self):
        executor = mock.MagicMock()
        executor.env_vars = {"EXISTING": "val1"}
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=None, env_vars={"NEW": "val2"})
        assert executor.env_vars == {"EXISTING": "val1", "NEW": "val2"}

    def test_env_vars_merge_overwrites_existing(self):
        executor = mock.MagicMock()
        executor.env_vars = {"KEY": "old"}
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=None, env_vars={"KEY": "new"})
        assert executor.env_vars["KEY"] == "new"

    def test_env_vars_merge_with_none_existing(self):
        executor = mock.MagicMock()
        executor.env_vars = None
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=None, env_vars={"K": "V"})
        assert executor.env_vars == {"K": "V"}

    def test_none_values_not_applied(self):
        executor = mock.MagicMock()
        executor.nodes = 1
        executor.container_image = "original:v1"
        executor.time = "01:00:00"
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=None, env_vars=None)
        # None values should leave the executor unchanged
        assert executor.nodes == 1
        assert executor.container_image == "original:v1"
        assert executor.time == "01:00:00"

    def test_empty_mounts_not_applied(self):
        executor = mock.MagicMock()
        executor.container_mounts = ["/existing:/existing"]
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=[], env_vars=None)
        # Empty list is falsy, should not modify
        assert executor.container_mounts == ["/existing:/existing"]

    def test_empty_env_vars_not_applied(self):
        executor = mock.MagicMock()
        executor.env_vars = {"EXISTING": "val"}
        apply_overrides(executor, nodes=None, devices=None, container_image=None,
                        time=None, mounts=None, env_vars={})
        # Empty dict is falsy, should not modify
        assert executor.env_vars == {"EXISTING": "val"}

    def test_all_overrides_applied(self):
        executor = mock.MagicMock(spec=["nodes", "ntasks_per_node", "gpus_per_node",
                                        "container_image", "time",
                                        "container_mounts", "env_vars"])
        executor.nodes = 1
        executor.ntasks_per_node = 1
        executor.gpus_per_node = 1
        executor.container_image = "old:v1"
        executor.time = "01:00:00"
        executor.container_mounts = ["/old:/old"]
        executor.env_vars = {"OLD": "val"}

        apply_overrides(
            executor,
            nodes=4,
            devices=8,
            container_image="new:v2",
            time="08:00:00",
            mounts=["/new:/new"],
            env_vars={"NEW": "val2"},
        )

        assert executor.nodes == 4
        assert executor.ntasks_per_node == 8
        assert executor.gpus_per_node == 8
        assert executor.container_image == "new:v2"
        assert executor.time == "08:00:00"
        assert executor.container_mounts == ["/old:/old", "/new:/new"]
        assert executor.env_vars == {"OLD": "val", "NEW": "val2"}


# ---------------------------------------------------------------------------
# submit_nemo_run_job
# ---------------------------------------------------------------------------


class TestSubmitNemoRunJob:
    def test_returns_zero(self):
        script = mock.MagicMock()
        executor = mock.MagicMock()
        result = submit_nemo_run_job(script, executor, "my_job", detach=True,
                                     tail_logs=False)
        assert result == 0

    def test_experiment_created_with_job_name(self):
        script = mock.MagicMock()
        executor = mock.MagicMock()
        submit_nemo_run_job(script, executor, "test_exp", detach=True,
                            tail_logs=False)
        _mock_run.Experiment.assert_called_once_with("test_exp")

    def test_experiment_add_called(self):
        script = mock.MagicMock()
        executor = mock.MagicMock()
        submit_nemo_run_job(script, executor, "job1", detach=True,
                            tail_logs=False)
        _mock_exp.add.assert_called_once_with(
            script, executor=executor, name="job1",
        )

    def test_experiment_run_called_with_detach(self):
        script = mock.MagicMock()
        executor = mock.MagicMock()
        submit_nemo_run_job(script, executor, "job1", detach=True,
                            tail_logs=True)
        _mock_exp.run.assert_called_once_with(detach=True, tail_logs=True)

    def test_experiment_run_called_without_detach(self):
        script = mock.MagicMock()
        executor = mock.MagicMock()
        submit_nemo_run_job(script, executor, "job1", detach=False,
                            tail_logs=False)
        _mock_exp.run.assert_called_once_with(detach=False, tail_logs=False)

    def test_empty_job_name_uses_automodel(self):
        script = mock.MagicMock()
        executor = mock.MagicMock()
        submit_nemo_run_job(script, executor, "", detach=True,
                            tail_logs=False)
        _mock_run.Experiment.assert_called_once_with("automodel")
        _mock_exp.add.assert_called_once_with(
            script, executor=executor, name="automodel",
        )


# ---------------------------------------------------------------------------
# REMOTE_CONFIG_PATH constant
# ---------------------------------------------------------------------------


def test_remote_config_path_value():
    assert REMOTE_CONFIG_PATH == "/tmp/automodel_job_config.yaml"
