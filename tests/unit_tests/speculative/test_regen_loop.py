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

"""Tests for the on-policy regeneration loop (config, runner cadence/consume, worker argv/main)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import types
from dataclasses import asdict

import pytest

from nemo_automodel.components.speculative.regen_loop import (
    RegenConfig,
    RegenRunner,
    _DONE_FILENAME,
    _SHARDS_DIRNAME,
    _regenerate_argv,
    _target_server_argv,
    main,
    resolve_regen_config,
)

_MODULE = "nemo_automodel.components.speculative.regen_loop"


def _config(tmp_path, **overrides):
    kwargs = dict(
        every_steps=10,
        cuda_visible_devices="7",
        target_model="/models/target",
        input_data="/data/train.jsonl",
        output_dir=str(tmp_path / "regen"),
    )
    kwargs.update(overrides)
    return RegenConfig(**kwargs)


class _FakeProc:
    def __init__(self, alive=True, pid=4242, return_code=0):
        self._alive = alive
        self.pid = pid
        self.returncode = None if alive else return_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):
        self._alive = False
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self._alive = False
        self.returncode = -9


def _ready_cycle(cfg, step):
    """Create a cycle dir with a shards subdir and the READY marker, as a finished worker would."""
    cycle_dir = os.path.join(cfg.output_dir, f"cycle_{step}")
    os.makedirs(os.path.join(cycle_dir, _SHARDS_DIRNAME), exist_ok=True)
    with open(os.path.join(cycle_dir, _DONE_FILENAME), "w") as f:
        f.write(str(step))
    return cycle_dir


# --------------------------------------------------------------------------- #
# resolve_regen_config
# --------------------------------------------------------------------------- #


def test_resolve_returns_none_when_absent_or_disabled(tmp_path):
    common = dict(default_target="/t", default_input_data="/d", output_dir=str(tmp_path))
    assert resolve_regen_config({}, **common) is None
    assert resolve_regen_config({"regen": None}, **common) is None
    assert resolve_regen_config({"regen": {"every_steps": 0}}, **common) is None


def test_resolve_requires_reserved_gpu(tmp_path):
    common = dict(default_target="/t", default_input_data="/d", output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="cuda_visible_devices is required"):
        resolve_regen_config({"regen": {"every_steps": 10}}, **common)
    with pytest.raises(ValueError, match="cuda_visible_devices is required"):
        resolve_regen_config({"regen": {"every_steps": 10, "cuda_visible_devices": ""}}, **common)


def test_resolve_requires_input_data_without_default(tmp_path):
    with pytest.raises(ValueError, match="input_data is required"):
        resolve_regen_config(
            {"regen": {"every_steps": 10, "cuda_visible_devices": "7"}},
            default_target="/t",
            default_input_data=None,
            output_dir=str(tmp_path),
        )


def test_resolve_rejects_unknown_option(tmp_path):
    with pytest.raises(ValueError, match="Unknown regen option"):
        resolve_regen_config(
            {"regen": {"every_steps": 10, "cuda_visible_devices": "7", "bogus": 1}},
            default_target="/t",
            default_input_data="/d",
            output_dir=str(tmp_path),
        )


def test_resolve_rejects_output_dir_override(tmp_path):
    # output_dir is a RegenConfig field (so it passes the unknown-option check) but
    # is always derived as <run>/regen; a user-set value must fail loudly, not be
    # silently discarded.
    with pytest.raises(ValueError, match="output_dir is not configurable"):
        resolve_regen_config(
            {"regen": {"every_steps": 10, "cuda_visible_devices": "7", "output_dir": "/somewhere/else"}},
            default_target="/t",
            default_input_data="/d",
            output_dir=str(tmp_path),
        )


def test_resolve_fills_defaults_and_applies_overrides(tmp_path):
    resolved = resolve_regen_config(
        {
            "regen": {
                "every_steps": 25,
                "cuda_visible_devices": "6,7",
                "temperature": 0.7,
                "max_new_tokens": 512,
            }
        },
        default_target="/models/target",
        default_input_data="/data/train.jsonl",
        output_dir=str(tmp_path / "run"),
    )
    assert resolved.every_steps == 25
    assert resolved.cuda_visible_devices == "6,7"
    # unset target/input fall back to the recipe-provided defaults
    assert resolved.target_model == "/models/target"
    assert resolved.input_data == "/data/train.jsonl"
    # overrides win, untouched fields keep dataclass defaults
    assert resolved.temperature == 0.7
    assert resolved.max_new_tokens == 512
    assert resolved.top_p == 1.0
    # output_dir is always nested under <run>/regen
    assert resolved.output_dir == os.path.join(str(tmp_path / "run"), "regen")


def test_resolve_accepts_block_with_to_dict(tmp_path):
    class _Block:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key, default)

        def to_dict(self):
            return dict(self._data)

    block = _Block({"every_steps": 10, "cuda_visible_devices": "7", "input_data": "/d"})
    resolved = resolve_regen_config(
        {"regen": block},
        default_target="/t",
        default_input_data=None,
        output_dir=str(tmp_path),
    )
    assert resolved is not None and resolved.input_data == "/d"


# --------------------------------------------------------------------------- #
# RegenRunner: cadence / launch
# --------------------------------------------------------------------------- #


def _runner_with_fake_popen(tmp_path, monkeypatch, launched):
    runner = RegenRunner(_config(tmp_path))
    monkeypatch.setattr(
        f"{_MODULE}.subprocess.Popen",
        lambda argv, **kw: launched.append(("popen", argv, kw)) or _FakeProc(),
    )
    return runner


def test_runner_launches_on_cadence_and_skips_while_running(tmp_path, monkeypatch):
    launched = []
    runner = _runner_with_fake_popen(tmp_path, monkeypatch, launched)

    assert not runner.maybe_launch(5)  # before the first cadence boundary
    assert runner.maybe_launch(10)
    assert not runner.maybe_launch(10)  # same bucket, no relaunch
    # A boundary while the previous cycle is still alive is skipped; once the
    # proc finishes, the next boundary launches again.
    assert not runner.maybe_launch(20)
    runner._proc._alive = False
    runner._proc.returncode = 0
    assert runner.maybe_launch(30)

    popen_calls = [entry for entry in launched if entry[0] == "popen"]
    assert len(popen_calls) == 2
    argv = popen_calls[0][1]
    assert argv[argv.index("--step") + 1] == "10"
    config_json = argv[argv.index("--config-json") + 1]
    with open(config_json) as f:
        sidecar = json.load(f)
    assert sidecar["cuda_visible_devices"] == "7"
    assert sidecar["target_model"] == "/models/target"
    assert popen_calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert popen_calls[0][2]["start_new_session"] is True


def test_launch_wipes_stale_cycle_dir(tmp_path, monkeypatch):
    launched = []
    runner = _runner_with_fake_popen(tmp_path, monkeypatch, launched)
    # A crashed prior worker left a half-written cycle: partial shards AND a stale
    # READY marker. Both must be wiped, else regenerate aborts on the non-empty
    # shard dir and the stale marker could make the cycle look ready.
    cycle_dir = _ready_cycle(runner.config, 10)
    stale_shard = os.path.join(cycle_dir, _SHARDS_DIRNAME, "shard-000000.parquet")
    with open(stale_shard, "w") as f:
        f.write("stale")

    assert runner.maybe_launch(10)
    assert not os.path.exists(os.path.join(cycle_dir, _DONE_FILENAME))
    assert not os.path.exists(stale_shard)  # leftover shards cleared for a clean regenerate


def test_reap_reports_failed_worker(tmp_path, caplog):
    runner = RegenRunner(_config(tmp_path))
    runner._proc = _FakeProc(alive=False, return_code=2)
    runner._launched_for_step = 10
    runner._worker_log_path = os.path.join(runner.config.output_dir, "worker.log")

    with caplog.at_level("ERROR"):
        runner._reap_finished_worker()

    assert "exited with code 2" in caplog.text
    assert runner._proc is None


def test_resume_from_step_suppresses_immediate_relaunch(tmp_path):
    runner = RegenRunner(_config(tmp_path))  # every_steps=10
    # A fresh runner (as rebuilt on resume) would fire at any step past the cadence.
    assert runner.due(1000)
    # Aligning to the restored step marks the current cadence region covered, so no
    # redundant relaunch at resume; the next cadence boundary still fires.
    runner.resume_from_step(1000)
    assert not runner.due(1000)
    assert not runner.due(1009)
    assert runner.due(1010)


def test_reap_is_noop_while_worker_runs(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    runner._proc = _FakeProc(alive=True)
    runner._reap_finished_worker()
    assert runner._proc is not None  # still alive → not released


# --------------------------------------------------------------------------- #
# RegenRunner: cycle discovery / consumption
# --------------------------------------------------------------------------- #


def test_cycle_dirs_orders_and_ignores_invalid(tmp_path, caplog):
    runner = RegenRunner(_config(tmp_path))
    for name in ("cycle_20", "cycle_5", "cycle_100", "cycle_final", "junk"):
        os.makedirs(os.path.join(runner.config.output_dir, name))

    with caplog.at_level("WARNING"):
        ordered = runner._cycle_dirs()

    assert [num for num, _ in ordered] == [5, 20, 100]
    assert "invalid cycle directory cycle_final" in caplog.text


def test_take_ready_returns_none_without_marker(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    os.makedirs(os.path.join(runner.config.output_dir, "cycle_10", _SHARDS_DIRNAME))  # no READY
    assert runner.take_ready_shards() is None


def test_cycle_dirs_empty_when_output_dir_missing(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    shutil.rmtree(runner.config.output_dir)  # e.g. cleaned between runs
    assert runner._cycle_dirs() == []
    assert runner.take_ready_shards() is None


def test_take_ready_returns_newest_and_consumes_once(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    _ready_cycle(runner.config, 10)
    cycle20 = _ready_cycle(runner.config, 20)

    shards = runner.take_ready_shards()
    assert shards == os.path.join(cycle20, _SHARDS_DIRNAME)
    # Both ready cycles are now consumed; a second call yields nothing.
    assert runner.take_ready_shards() is None


def test_take_ready_removes_superseded_active_cycle(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    cycle10 = _ready_cycle(runner.config, 10)
    assert runner.take_ready_shards() == os.path.join(cycle10, _SHARDS_DIRNAME)

    cycle20 = _ready_cycle(runner.config, 20)
    assert runner.take_ready_shards() == os.path.join(cycle20, _SHARDS_DIRNAME)
    # The previously active cycle's shards are freed; the new active one stays.
    assert not os.path.exists(cycle10)
    assert os.path.exists(cycle20)


def test_take_ready_frees_all_intermediate_cycles_in_one_call(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    # Several cycles complete within one epoch (cadence < epoch length); only the
    # freshest feeds training and the rest must not be stranded on disk.
    cycle10 = _ready_cycle(runner.config, 10)
    cycle20 = _ready_cycle(runner.config, 20)
    cycle30 = _ready_cycle(runner.config, 30)

    assert runner.take_ready_shards() == os.path.join(cycle30, _SHARDS_DIRNAME)
    assert not os.path.exists(cycle10)  # intermediate cycles freed, not just the active one
    assert not os.path.exists(cycle20)
    assert os.path.exists(cycle30)


def test_remove_cycle_swallows_missing_and_logs_oserror(tmp_path, monkeypatch, caplog):
    # Missing directory is a no-op.
    RegenRunner._remove_cycle(str(tmp_path / "does_not_exist"))

    existing = tmp_path / "cycle_10"
    existing.mkdir()
    monkeypatch.setattr(f"{_MODULE}.shutil.rmtree", lambda _p: (_ for _ in ()).throw(OSError("busy")))
    with caplog.at_level("ERROR"):
        RegenRunner._remove_cycle(str(existing))
    assert "failed to remove superseded cycle" in caplog.text


# --------------------------------------------------------------------------- #
# RegenRunner: shutdown
# --------------------------------------------------------------------------- #


def test_shutdown_escalates_to_process_group_kill(tmp_path, monkeypatch):
    runner = RegenRunner(_config(tmp_path))
    proc = _FakeProc()
    wait_calls = []

    def _wait(timeout=None):
        wait_calls.append(timeout)
        if len(wait_calls) == 1:
            raise subprocess.TimeoutExpired("worker", 30)
        return 0

    proc.wait = _wait
    runner._proc = proc
    signals = []
    monkeypatch.setattr(os, "getpgid", lambda _pid: 99)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))

    runner.shutdown()

    assert signals[0][0] == signals[1][0] == 99
    assert signals[0][1] != signals[1][1]  # SIGTERM then SIGKILL
    assert runner._proc is None


def test_shutdown_reaps_dead_proc_and_noop_when_none(tmp_path):
    runner = RegenRunner(_config(tmp_path))
    runner._proc = _FakeProc(alive=False, return_code=0)
    runner.shutdown()
    assert runner._proc is None
    # No proc at all: shutdown must be a safe no-op.
    runner.shutdown()
    assert runner._proc is None


def _timeout_then_ok_wait():
    """A wait() that raises TimeoutExpired on its first call, then returns 0."""
    calls = []

    def _wait(timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("worker", 30)
        return 0

    return _wait


def test_shutdown_falls_back_to_terminate_without_pgid(tmp_path, monkeypatch):
    runner = RegenRunner(_config(tmp_path))
    proc = _FakeProc()
    proc.wait = _timeout_then_ok_wait()
    runner._proc = proc
    # No process group available: SIGTERM/SIGKILL via killpg are unusable.
    monkeypatch.setattr(os, "getpgid", lambda _pid: (_ for _ in ()).throw(ProcessLookupError()))
    killpg_calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    runner.shutdown()

    assert killpg_calls == []  # never reached the group-signal path
    assert proc.terminated and proc.killed  # terminate(), then kill() after the wait timeout
    assert runner._proc is None


def test_shutdown_kills_directly_when_group_sigkill_fails(tmp_path, monkeypatch):
    runner = RegenRunner(_config(tmp_path))
    proc = _FakeProc()
    proc.wait = _timeout_then_ok_wait()
    runner._proc = proc

    def _killpg(pgid, sig):
        if sig == signal.SIGKILL:
            raise ProcessLookupError()  # group already gone by escalation time

    monkeypatch.setattr(os, "getpgid", lambda _pid: 99)
    monkeypatch.setattr(os, "killpg", _killpg)

    runner.shutdown()

    assert proc.killed  # SIGKILL on the group failed → direct proc.kill()
    assert runner._proc is None


# --------------------------------------------------------------------------- #
# Worker CLI: argv builders
# --------------------------------------------------------------------------- #


def test_target_server_argv_includes_optional_flags(tmp_path):
    base = _target_server_argv(_config(tmp_path), port=8123)
    assert base[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert base[base.index("--port") + 1] == "8123"
    assert "--max-model-len" not in base
    assert "--trust-remote-code" not in base

    full = _target_server_argv(
        _config(tmp_path, server_python="/venv/py", max_model_len=4096, trust_remote_code=True),
        port=8123,
    )
    assert full[0] == "/venv/py"
    assert full[full.index("--max-model-len") + 1] == "4096"
    assert "--trust-remote-code" in full


def test_regenerate_argv_includes_dataset_name_only_when_set(tmp_path):
    cfg = _config(tmp_path, temperature=0.5, shard_size=250)
    argv = _regenerate_argv(cfg, server="http://127.0.0.1:8000", shards_dir="/out/shards")
    assert argv[argv.index("--target-server") + 1] == "http://127.0.0.1:8000/v1"
    assert argv[argv.index("--output-dir") + 1] == "/out/shards"
    assert argv[argv.index("--temperature") + 1] == "0.5"
    assert argv[argv.index("--shard-size") + 1] == "250"
    assert "--dataset-name" not in argv

    argv_ds = _regenerate_argv(_config(tmp_path, dataset_name="hf/set"), server="http://s", shards_dir="/o")
    assert argv_ds[argv_ds.index("--dataset-name") + 1] == "hf/set"


# --------------------------------------------------------------------------- #
# Worker CLI: main
# --------------------------------------------------------------------------- #


def _run_main(tmp_path, monkeypatch, regenerate_rc):
    cfg = _config(tmp_path)
    cycle_dir = str(tmp_path / "cycle_10")
    os.makedirs(cycle_dir)
    config_json = str(tmp_path / "regen_config.json")
    with open(config_json, "w") as f:
        json.dump(asdict(cfg), f)

    monkeypatch.setattr(f"{_MODULE}._resolve_worker_port", lambda _p: 12345)
    monkeypatch.setattr(f"{_MODULE}._wait_for_server", lambda server, proc, timeout: None)
    monkeypatch.setattr(f"{_MODULE}.subprocess.Popen", lambda argv, **kw: _FakeProc(alive=False, return_code=0))

    calls = {}

    def _fake_regenerate_main(argv):
        calls["argv"] = argv
        return regenerate_rc

    fake_mod = types.ModuleType("nemo_automodel.components.speculative.regenerate")
    fake_mod.main = _fake_regenerate_main
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.speculative.regenerate", fake_mod)

    argv = ["--config-json", config_json, "--cycle-dir", cycle_dir, "--step", "10"]
    return cfg, cycle_dir, calls, main(argv)


def test_main_writes_ready_marker_on_success(tmp_path, monkeypatch):
    cfg, cycle_dir, calls, rc = _run_main(tmp_path, monkeypatch, regenerate_rc=0)
    assert rc == 0
    ready = os.path.join(cycle_dir, _DONE_FILENAME)
    assert os.path.exists(ready)
    with open(ready) as f:
        assert f.read() == "10"
    # regenerate ran against the shards subdir of this cycle
    assert calls["argv"][calls["argv"].index("--output-dir") + 1] == os.path.join(cycle_dir, _SHARDS_DIRNAME)


def test_main_raises_and_skips_marker_on_regenerate_failure(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="regenerate exited with code 3"):
        _run_main(tmp_path, monkeypatch, regenerate_rc=3)
    assert not os.path.exists(os.path.join(str(tmp_path / "cycle_10"), _DONE_FILENAME))


def test_main_terminates_live_target_server_on_exit(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cycle_dir = str(tmp_path / "cycle_10")
    os.makedirs(cycle_dir)
    config_json = str(tmp_path / "regen_config.json")
    with open(config_json, "w") as f:
        json.dump(asdict(cfg), f)

    # Server is still alive when main() returns, and ignores terminate() → must escalate to kill().
    server = _FakeProc(alive=True)
    server.wait = _timeout_then_ok_wait()
    monkeypatch.setattr(f"{_MODULE}._resolve_worker_port", lambda _p: 12345)
    monkeypatch.setattr(f"{_MODULE}._wait_for_server", lambda s, p, t: None)
    monkeypatch.setattr(f"{_MODULE}.subprocess.Popen", lambda argv, **kw: server)
    fake_mod = types.ModuleType("nemo_automodel.components.speculative.regenerate")
    fake_mod.main = lambda argv: 0
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.speculative.regenerate", fake_mod)

    rc = main(["--config-json", config_json, "--cycle-dir", cycle_dir, "--step", "10"])

    assert rc == 0
    assert server.terminated and server.killed
