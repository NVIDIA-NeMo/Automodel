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

import mlflow
import pytest
import torch
import torch.distributed as dist

from nemo_automodel.components.loggers.mlflow_utils import (
    _install_mlflow_failure_hook,
    configure_mlflow,
    end_active_run_as_killed,
    flatten_params_for_mlflow,
    to_float_metrics,
)


class _DictWithToDict(dict):
    """Mimics ConfigNode's tags dict, which exposes a `.to_dict()` method."""

    def to_dict(self):
        return dict(self)


class _MlflowSubCfg:
    """Minimal stand-in for `cfg.mlflow` — supports `.get()` and truthiness."""

    def __init__(self, data: dict):
        # Wrap raw dict tags so they expose .to_dict() like ConfigNode does
        data = dict(data)
        if isinstance(data.get("tags"), dict) and not isinstance(data["tags"], _DictWithToDict):
            data["tags"] = _DictWithToDict(data["tags"])
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __bool__(self):
        return bool(self._data)


class _Cfg:
    """Minimal stand-in for the recipe's full Cfg — supports the surface
    `configure_mlflow` actually uses: `.get(...)` and `.to_yaml_dict(...)`.
    """

    def __init__(self, *, mlflow=None, checkpoint_dir=None, yaml_dict=None):
        self._mlflow = _MlflowSubCfg(mlflow) if mlflow else None
        self._checkpoint_dir = checkpoint_dir
        self._yaml_dict = yaml_dict if yaml_dict is not None else {"foo": "bar"}

    def get(self, key, default=None):
        if key == "mlflow":
            return self._mlflow if self._mlflow else default
        if key == "checkpoint.checkpoint_dir":
            return self._checkpoint_dir if self._checkpoint_dir is not None else default
        return default

    def to_yaml_dict(self, use_orig_values=True):
        return self._yaml_dict


class TestConfigureMlflow:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        # Default to rank-0 in an initialized process group
        monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
        monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)
        # Don't pick up MLFLOW_RUN_ID from the dev's environment
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        self.tmp_path = tmp_path
        self.tracking_uri = f"file://{tmp_path}/mlruns"
        mlflow.set_tracking_uri(self.tracking_uri)
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # configure_mlflow chains sys.excepthook; restore on teardown for isolation
        original_excepthook = sys.excepthook
        yield
        if mlflow.active_run() is not None:
            mlflow.end_run()
        sys.excepthook = original_excepthook

    def _make_cfg(self, *, ckpt_dir=None, mlflow_overrides=None, yaml_dict=None):
        ckpt_dir = ckpt_dir if ckpt_dir is not None else self.tmp_path / "ckpt"
        return _Cfg(
            mlflow={
                "experiment_name": "test-exp",
                "tracking_uri": self.tracking_uri,
                **(mlflow_overrides or {}),
            },
            checkpoint_dir=str(ckpt_dir),
            yaml_dict=yaml_dict,
        )

    def test_returns_none_when_not_rank_zero(self, monkeypatch):
        monkeypatch.setattr(dist, "get_rank", lambda: 1, raising=False)
        assert configure_mlflow(self._make_cfg()) is None
        assert mlflow.active_run() is None

    def test_returns_none_when_no_mlflow_config(self):
        cfg = _Cfg(mlflow=None, checkpoint_dir=str(self.tmp_path / "ckpt"))
        assert configure_mlflow(cfg) is None
        assert mlflow.active_run() is None

    def test_starts_run_with_config_tags(self):
        run = configure_mlflow(self._make_cfg(mlflow_overrides={"tags": {"task": "finetune"}}))
        assert run is not None
        stored = mlflow.get_run(run.info.run_id)
        assert stored.data.tags["task"] == "finetune"

    def test_description_maps_to_note_content_tag(self):
        run = configure_mlflow(self._make_cfg(mlflow_overrides={"description": "Quick smoke run"}))
        stored = mlflow.get_run(run.info.run_id)
        assert stored.data.tags["mlflow.note.content"] == "Quick smoke run"

    def test_logs_config_as_flattened_params(self):
        cfg = self._make_cfg(yaml_dict={"lr": 1e-3, "model": {"hidden": 64, "layers": 2}})
        run = configure_mlflow(cfg)
        params = mlflow.get_run(run.info.run_id).data.params
        assert params["lr"] == "0.001"
        assert params["model.hidden"] == "64"
        assert params["model.layers"] == "2"

    def test_logs_config_yaml_artifact(self):
        run = configure_mlflow(self._make_cfg())
        artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)]
        assert "config.yaml" in artifacts

    def test_writes_run_id_sidecar_for_resumption(self):
        ckpt_dir = self.tmp_path / "ckpt"
        run = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        assert (ckpt_dir / "mlflow_run_id").read_text() == run.info.run_id

    def test_resumes_from_sidecar_when_called_again(self):
        ckpt_dir = self.tmp_path / "ckpt"
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        mlflow.end_run()  # simulate process restart

        run2 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        assert run2.info.run_id == run1.info.run_id
        # And mlflow only sees one run, not two
        runs = mlflow.search_runs(experiment_names=["test-exp"])
        assert len(runs) == 1

    def test_resumes_from_env_var(self, monkeypatch):
        # First call creates a run we'll resume into
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-A"))
        mlflow.end_run()

        # Second call uses a *different* checkpoint dir (no sidecar match) but
        # MLFLOW_RUN_ID points at run1 — should resume run1
        monkeypatch.setenv("MLFLOW_RUN_ID", run1.info.run_id)
        run2 = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-B"))
        assert run2.info.run_id == run1.info.run_id

    def test_env_var_takes_precedence_over_sidecar(self, monkeypatch):
        # Two separate runs created via two separate ckpt_dirs
        run_a = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-A"))
        mlflow.end_run()
        run_b = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-B"))
        mlflow.end_run()
        assert run_a.info.run_id != run_b.info.run_id

        # ckpt-A's sidecar points at run_a; env var points at run_b → env wins
        monkeypatch.setenv("MLFLOW_RUN_ID", run_b.info.run_id)
        resumed = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-A"))
        assert resumed.info.run_id == run_b.info.run_id

    def test_resumed_run_skips_param_logging(self):
        ckpt_dir = self.tmp_path / "ckpt"
        # Initial launch logs params
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir, yaml_dict={"lr": 1e-3}))
        original_params = dict(mlflow.get_run(run1.info.run_id).data.params)
        mlflow.end_run()

        # Resume with different yaml — params shouldn't change (would raise if re-logged
        # with a different value; we don't even attempt the re-log on resume)
        configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir, yaml_dict={"lr": 5e-4}))
        resumed_params = dict(mlflow.get_run(run1.info.run_id).data.params)
        assert resumed_params == original_params

    def test_resumed_run_writes_timestamped_artifact(self):
        ckpt_dir = self.tmp_path / "ckpt"
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        mlflow.end_run()

        configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run1.info.run_id)}
        # Initial config.yaml plus a timestamped resume snapshot
        assert "config.yaml" in artifacts
        assert any(a.startswith("config.resumed-") and a.endswith(".yaml") for a in artifacts)

    def test_resume_false_ignores_existing_sidecar(self):
        ckpt_dir = self.tmp_path / "ckpt"
        # First run seeds the sidecar
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        mlflow.end_run()
        assert (ckpt_dir / "mlflow_run_id").read_text() == run1.info.run_id

        # Second launch with resume disabled — must start a new run
        run2 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir, mlflow_overrides={"resume": False}))
        assert run2.info.run_id != run1.info.run_id

    def test_resume_false_still_honors_env_var(self, monkeypatch):
        """`resume: false` gates the implicit sidecar lookup only — an
        explicit `MLFLOW_RUN_ID` env var is treated as a deliberate user
        override and still resumes the named run."""
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-A"))
        mlflow.end_run()

        monkeypatch.setenv("MLFLOW_RUN_ID", run1.info.run_id)
        run2 = configure_mlflow(self._make_cfg(ckpt_dir=self.tmp_path / "ckpt-B", mlflow_overrides={"resume": False}))
        assert run2.info.run_id == run1.info.run_id

    def test_resume_false_still_writes_sidecar(self):
        """A future `resume: true` launch (or recovery) must still be able to
        find the run, so the sidecar is always written even when resume is off."""
        ckpt_dir = self.tmp_path / "ckpt"
        run = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir, mlflow_overrides={"resume": False}))
        assert (ckpt_dir / "mlflow_run_id").read_text() == run.info.run_id

    def test_resume_false_overwrites_existing_sidecar(self):
        """Sidecar always points to the most recent run for this checkpoint_dir,
        so a `resume: false` launch overwrites any prior sidecar."""
        ckpt_dir = self.tmp_path / "ckpt"
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        mlflow.end_run()

        run2 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir, mlflow_overrides={"resume": False}))
        assert run2.info.run_id != run1.info.run_id
        assert (ckpt_dir / "mlflow_run_id").read_text() == run2.info.run_id

    def test_resume_default_is_true(self):
        """Omitting `resume` keeps the existing auto-resume behaviour."""
        ckpt_dir = self.tmp_path / "ckpt"
        run1 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        mlflow.end_run()

        # No `resume` key — should resume run1
        run2 = configure_mlflow(self._make_cfg(ckpt_dir=ckpt_dir))
        assert run2.info.run_id == run1.info.run_id


class TestEndActiveRunAsKilled:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        yield
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def test_marks_active_run_as_killed(self):
        run = mlflow.start_run()
        run_id = run.info.run_id

        end_active_run_as_killed()

        # mlflow's run should now be ended (no active run) and stored with KILLED status
        assert mlflow.active_run() is None
        assert mlflow.get_run(run_id).info.status == "KILLED"

    def test_no_op_when_no_active_run(self):
        assert mlflow.active_run() is None  # precondition
        end_active_run_as_killed()  # should not raise
        assert mlflow.active_run() is None

    def test_suppresses_errors_from_end_run(self, monkeypatch):
        # Signal-handler reentrancy with mlflow's logging can raise; the function
        # must swallow it so the SIGTERM path doesn't crash.
        mlflow.start_run()

        def _raises(*args, **kwargs):
            raise RuntimeError("simulated reentrancy failure")

        monkeypatch.setattr(mlflow, "end_run", _raises)
        end_active_run_as_killed()  # must not raise


class TestInstallMlflowFailureHook:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # The hook chains sys.excepthook globally; restore for isolation
        original_excepthook = sys.excepthook
        yield
        if mlflow.active_run() is not None:
            mlflow.end_run()
        sys.excepthook = original_excepthook

    def test_install_replaces_sys_excepthook_with_marked_wrapper(self):
        original = sys.excepthook
        _install_mlflow_failure_hook()
        assert sys.excepthook is not original
        assert getattr(sys.excepthook, "_mlflow_failure_hook", False) is True

    def test_hook_marks_run_failed_on_uncaught_exception(self):
        run = mlflow.start_run()
        run_id = run.info.run_id
        _install_mlflow_failure_hook()

        # Simulate an unhandled exception by invoking the excepthook directly
        try:
            raise RuntimeError("simulated training crash")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())

        assert mlflow.get_run(run_id).info.status == "FAILED"

    def test_hook_chains_to_previous_excepthook(self):
        called = []

        def previous(exc_type, exc_val, exc_tb):
            called.append((exc_type, str(exc_val)))

        sys.excepthook = previous
        _install_mlflow_failure_hook()

        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())

        # Previous excepthook ran (default traceback printing still happens)
        assert called == [(ValueError, "boom")]

    def test_install_is_idempotent(self):
        _install_mlflow_failure_hook()
        first_hook = sys.excepthook
        _install_mlflow_failure_hook()  # second call should be a no-op
        assert sys.excepthook is first_hook

    def test_hook_chains_without_calling_end_run_when_no_active_run(self, monkeypatch):
        assert mlflow.active_run() is None
        end_run_calls = []
        monkeypatch.setattr(mlflow, "end_run", lambda *a, **kw: end_run_calls.append((a, kw)))

        _install_mlflow_failure_hook()
        try:
            raise RuntimeError("no run active")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())

        assert end_run_calls == []  # end_run never called when no active run


class TestFlattenParamsForMlflow:
    def test_empty_dict_returns_empty(self):
        assert flatten_params_for_mlflow({}) == {}

    def test_already_flat_dict_stringifies_values(self):
        out = flatten_params_for_mlflow({"a": 1, "b": True, "c": "hi", "d": None})
        assert out == {"a": "1", "b": "True", "c": "hi", "d": "None"}

    def test_default_splits_one_level(self):
        out = flatten_params_for_mlflow({"model": {"hidden": 64, "layers": 2}, "lr": 1e-3})
        assert out == {"model.hidden": "64", "model.layers": "2", "lr": "0.001"}

    def test_default_stringifies_dicts_below_first_level(self):
        # Two levels deep — only the first gets split, second stays stringified
        out = flatten_params_for_mlflow({"model": {"text": {"output_hidden_states": True}}})
        assert out == {"model.text": "{'output_hidden_states': True}"}

    def test_max_depth_2_splits_two_levels(self):
        out = flatten_params_for_mlflow({"model": {"text": {"output_hidden_states": True}}}, max_depth=2)
        assert out == {"model.text.output_hidden_states": "True"}

    def test_max_depth_none_fully_recursive(self):
        out = flatten_params_for_mlflow({"a": {"b": {"c": {"d": 1}}}}, max_depth=None)
        assert out == {"a.b.c.d": "1"}

    def test_max_depth_zero_stringifies_top_level(self):
        out = flatten_params_for_mlflow({"model": {"hidden": 64}}, max_depth=0)
        assert out == {"model": "{'hidden': 64}"}

    def test_lists_and_tuples_stringified_not_split(self):
        out = flatten_params_for_mlflow({"betas": [0.9, 0.95], "shape": (3, 4)})
        assert out == {"betas": "[0.9, 0.95]", "shape": "(3, 4)"}

    def test_lists_inside_nested_dict_still_stringified(self):
        # Recursion shouldn't accidentally split list elements
        out = flatten_params_for_mlflow({"opt": {"betas": [0.9, 0.95]}})
        assert out == {"opt.betas": "[0.9, 0.95]"}


class TestToFloatMetrics:
    def test_empty_input_returns_empty(self):
        assert to_float_metrics({}) == {}

    def test_python_int_coerced_to_float(self):
        out = to_float_metrics({"step": 7})
        assert out == {"step": 7.0}
        assert isinstance(out["step"], float)

    def test_python_float_passes_through(self):
        out = to_float_metrics({"loss": 1.23})
        assert out == {"loss": 1.23}
        assert isinstance(out["loss"], float)

    def test_scalar_tensor_coerced_via_item(self):
        out = to_float_metrics({"loss": torch.tensor(2.5)})
        assert out == {"loss": 2.5}
        assert isinstance(out["loss"], float)

    def test_multi_element_tensor_reduced_via_mean(self):
        out = to_float_metrics({"grad_norm": torch.tensor([1.0, 3.0])})
        assert out == {"grad_norm": 2.0}
        assert isinstance(out["grad_norm"], float)

    def test_drops_string_value(self):
        # The `timestamp` field in MetricsSample.to_dict() is a string and
        # would crash mlflow.log_metrics if passed through.
        out = to_float_metrics({"loss": 1.0, "timestamp": "2026-05-06T12:00:00"})
        assert out == {"loss": 1.0}

    def test_drops_none_value(self):
        out = to_float_metrics({"loss": 1.0, "missing": None})
        assert out == {"loss": 1.0}

    def test_drops_arbitrary_object(self):
        out = to_float_metrics({"loss": 1.0, "weird": object()})
        assert out == {"loss": 1.0}
