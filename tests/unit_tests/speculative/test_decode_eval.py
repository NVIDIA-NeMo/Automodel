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

"""Tests for the periodic decode eval (config, snapshot export, runner, worker argv)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.decode_eval import (
    DecodeEvalConfig,
    DecodeEvalRunner,
    _bench_args,
    _resolve_worker_port,
    _serve_argv,
    _worker_env,
    export_draft_snapshot,
    resolve_decode_eval_config,
)
from nemo_automodel.components.speculative.serve_vllm import _find_draft_dir


def _config(tmp_path, **overrides):
    kwargs = dict(
        every_steps=10,
        cuda_visible_devices="7",
        target_model="/models/target",
        input_data="/data/train.jsonl",
        output_dir=str(tmp_path / "decode_eval"),
        num_speculative_tokens=7,
    )
    kwargs.update(overrides)
    return DecodeEvalConfig(**kwargs)


def test_resolve_returns_none_when_absent_or_disabled(tmp_path):
    assert (
        resolve_decode_eval_config(
            {},
            default_target="/t",
            default_input_data="/d",
            default_num_speculative_tokens=4,
            output_dir=str(tmp_path),
        )
        is None
    )
    cfg = {"decode_eval": {"every_steps": 0, "cuda_visible_devices": "7"}}
    assert (
        resolve_decode_eval_config(
            cfg,
            default_target="/t",
            default_input_data="/d",
            default_num_speculative_tokens=4,
            output_dir=str(tmp_path),
        )
        is None
    )


def test_resolve_requires_reserved_gpu(tmp_path):
    cfg = {"decode_eval": {"every_steps": 100}}
    with pytest.raises(ValueError, match="cuda_visible_devices"):
        resolve_decode_eval_config(
            cfg,
            default_target="/t",
            default_input_data="/d",
            default_num_speculative_tokens=4,
            output_dir=str(tmp_path),
        )


def test_resolve_fills_defaults_from_recipe(tmp_path):
    cfg = {"decode_eval": {"every_steps": 250, "cuda_visible_devices": 6}}
    resolved = resolve_decode_eval_config(
        cfg,
        default_target="/models/qwen",
        default_input_data="/data/messages",
        default_num_speculative_tokens=7,
        output_dir=str(tmp_path),
    )
    assert resolved.every_steps == 250
    assert resolved.cuda_visible_devices == "6"
    assert resolved.target_model == "/models/qwen"
    assert resolved.input_data == "/data/messages"
    assert resolved.output_dir == os.path.join(str(tmp_path), "decode_eval")
    assert resolved.num_speculative_tokens == 7
    assert resolved.port == 0


def test_resolve_validates_input_tokens_and_unknown_options(tmp_path):
    base = {"every_steps": 10, "cuda_visible_devices": "7"}
    with pytest.raises(ValueError, match="input_data"):
        resolve_decode_eval_config(
            {"decode_eval": base},
            default_target="/t",
            default_input_data=None,
            default_num_speculative_tokens=4,
            output_dir=str(tmp_path),
        )
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        resolve_decode_eval_config(
            {"decode_eval": {**base, "num_speculative_tokens": 0}},
            default_target="/t",
            default_input_data="/d",
            default_num_speculative_tokens=0,
            output_dir=str(tmp_path),
        )
    with pytest.raises(ValueError, match="num_prompt"):
        resolve_decode_eval_config(
            {"decode_eval": {**base, "num_prompt": 8}},
            default_target="/t",
            default_input_data="/d",
            default_num_speculative_tokens=4,
            output_dir=str(tmp_path),
        )


def test_export_draft_snapshot_is_serve_resolvable(tmp_path):
    """The snapshot must land in the model/consolidated layout serve_vllm probes."""

    class _TinyDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.register_buffer("d2t", torch.arange(4))
            self.config = LlamaConfig(hidden_size=4, num_attention_heads=1, num_hidden_layers=1)

    out_dir = str(tmp_path / "step_10")
    consolidated = export_draft_snapshot(_TinyDraft(), out_dir)
    assert os.path.exists(os.path.join(consolidated, "model.safetensors"))
    assert os.path.exists(os.path.join(consolidated, "config.json"))
    from safetensors import safe_open

    with safe_open(os.path.join(consolidated, "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    assert "d2t" in keys  # vocab-mapping buffers ride along
    assert _find_draft_dir(__import__("pathlib").Path(out_dir)) is not None


def test_worker_env_scrubs_torchrun_state(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc")
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    env = _worker_env("5")
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["HF_HOME"] == "/cache/hf"
    for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "TORCHELASTIC_RUN_ID"):
        assert key not in env


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


def _runner_with_fake_launch(tmp_path, monkeypatch, launched):
    runner = DecodeEvalRunner(_config(tmp_path))
    monkeypatch.setattr(
        "nemo_automodel.components.speculative.decode_eval.export_draft_snapshot",
        lambda model, out_dir: launched.append(("export", out_dir)) or out_dir,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.speculative.decode_eval.subprocess.Popen",
        lambda argv, **kw: launched.append(("popen", argv, kw)) or _FakeProc(),
    )
    return runner


def test_runner_launches_on_cadence_and_skips_while_running(tmp_path, monkeypatch):
    launched = []
    runner = _runner_with_fake_launch(tmp_path, monkeypatch, launched)
    model = object()

    assert not runner.maybe_launch(5, model)  # before the first boundary
    assert runner.maybe_launch(10, model)
    assert not runner.maybe_launch(10, model)  # same bucket, no relaunch
    # Next boundary while the previous eval is still alive: skipped, but the
    # bucket does not advance past it forever - once the proc finishes the next
    # boundary launches again.
    assert not runner.maybe_launch(20, model)
    runner._proc._alive = False
    runner._proc.returncode = 0
    assert runner.maybe_launch(30, model)

    popen_calls = [entry for entry in launched if entry[0] == "popen"]
    assert len(popen_calls) == 2
    argv = popen_calls[0][1]
    assert "--step" in argv and argv[argv.index("--step") + 1] == "10"
    config_json = argv[argv.index("--config-json") + 1]
    with open(config_json) as f:
        sidecar = json.load(f)
    assert sidecar["cuda_visible_devices"] == "7"
    assert sidecar["target_model"] == "/models/target"
    assert popen_calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert popen_calls[0][2]["start_new_session"] is True


def test_runner_collect_returns_each_result_once(tmp_path):
    runner = DecodeEvalRunner(_config(tmp_path))
    step_dir = os.path.join(runner.config.output_dir, "step_10")
    os.makedirs(step_dir)
    with open(os.path.join(step_dir, "result.json"), "w") as f:
        json.dump({"step": 10, "accept_length": 1.83}, f)

    first = runner.collect()
    assert len(first) == 1 and first[0]["accept_length"] == 1.83
    assert runner.collect() == []

    # A later result is picked up in step order.
    step_dir2 = os.path.join(runner.config.output_dir, "step_20")
    os.makedirs(step_dir2)
    with open(os.path.join(step_dir2, "result.json"), "w") as f:
        json.dump({"step": 20, "accept_length": 2.01}, f)
    second = runner.collect()
    assert [r["step"] for r in second] == [20]


def test_runner_resume_ignores_old_results_and_removes_snapshot(tmp_path):
    cfg = _config(tmp_path)
    step_dir = os.path.join(cfg.output_dir, "step_10")
    os.makedirs(os.path.join(step_dir, "model"))
    result_path = os.path.join(step_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump({"step": 10, "accept_length": 1.5}, f)

    runner = DecodeEvalRunner(cfg)

    assert runner.collect() == []
    assert not os.path.exists(os.path.join(step_dir, "model"))


def test_runner_ignores_invalid_step_directory(tmp_path):
    runner = DecodeEvalRunner(_config(tmp_path))
    os.makedirs(os.path.join(runner.config.output_dir, "step_final"))
    step_dir = os.path.join(runner.config.output_dir, "step_20")
    os.makedirs(step_dir)
    with open(os.path.join(step_dir, "result.json"), "w") as f:
        json.dump({"step": 20, "accept_length": 2.0}, f)

    assert [result["step"] for result in runner.collect()] == [20]


def test_failed_launch_retries_same_bucket_and_cleans_snapshot(tmp_path, monkeypatch):
    runner = DecodeEvalRunner(_config(tmp_path))
    step_dir = os.path.join(runner.config.output_dir, "step_10")

    def _export(_model, out_dir):
        os.makedirs(os.path.join(out_dir, "model"))
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr("nemo_automodel.components.speculative.decode_eval.export_draft_snapshot", _export)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        runner.maybe_launch(10, object())

    assert runner.due(10)
    assert not os.path.exists(os.path.join(step_dir, "model"))


def test_failed_worker_is_reported_and_snapshot_removed(tmp_path, caplog):
    runner = DecodeEvalRunner(_config(tmp_path))
    step_dir = os.path.join(runner.config.output_dir, "step_10")
    os.makedirs(os.path.join(step_dir, "model"))
    runner._proc = _FakeProc(alive=False, return_code=2)
    runner._launched_for_step = 10
    runner._worker_log_path = os.path.join(step_dir, "worker.log")

    runner.collect()

    assert "exited with code 2" in caplog.text
    assert runner._proc is None
    assert not os.path.exists(os.path.join(step_dir, "model"))


def test_shutdown_escalates_to_process_group_kill(tmp_path, monkeypatch):
    runner = DecodeEvalRunner(_config(tmp_path))
    step_dir = os.path.join(runner.config.output_dir, "step_10")
    os.makedirs(os.path.join(step_dir, "model"))
    runner._launched_for_step = 10
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
    assert signals[0][1] != signals[1][1]
    assert not os.path.exists(os.path.join(step_dir, "model"))


def test_serve_argv_builds_speculative_server_command(tmp_path, monkeypatch):
    """The worker must drive serve_vllm's library surface, not reimplement it."""
    seen = {}

    def _fake_build(serve_args):
        seen.update(vars(serve_args))
        return ["python", "-m", "vllm.entrypoints.openai.api_server"]

    monkeypatch.setattr("nemo_automodel.components.speculative.serve_vllm.build_vllm_argv", _fake_build)
    cfg = _config(tmp_path, port=8199)
    argv = _serve_argv(cfg, str(tmp_path / "step_10"))
    assert argv[0] == "python"
    assert seen["target"] == "/models/target"
    assert seen["draft"] == str(tmp_path / "step_10")
    assert seen["port"] == 8199
    assert seen["host"] == "127.0.0.1"
    assert seen["num_speculative_tokens"] == 7


def test_worker_port_auto_selects_and_rejects_collision():
    port = _resolve_worker_port(0)
    assert port > 0
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        occupied = sock.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            _resolve_worker_port(occupied)


def test_bench_args_carry_workload_settings(tmp_path):
    cfg = _config(tmp_path, num_prompts=16, concurrency=4, max_new_tokens=128, timeout_s=600.0)
    bench = _bench_args(cfg, "http://127.0.0.1:8199")
    assert bench.server == "http://127.0.0.1:8199"
    assert bench.model == "/models/target"
    assert bench.num_prompts == 16
    assert bench.baseline_server is None
    assert bench.prompt_context_column is None


def test_config_json_roundtrips_through_the_worker_boundary(tmp_path):
    """The sidecar the launcher writes must reconstruct the exact config."""
    import dataclasses

    cfg = _config(tmp_path, num_prompts=7, prompt_column="question")
    rebuilt = DecodeEvalConfig(**json.loads(json.dumps(dataclasses.asdict(cfg))))
    assert rebuilt == cfg


def test_recipe_hook_collects_logs_and_launches(monkeypatch):
    """The rank-0 log-point hook must log finished results to wandb (at the
    current step) and hand the launch decision to the runner."""
    from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.runtime = SimpleNamespace(global_step=42)
    recipe.draft_model = object()
    logged = []
    recipe._wandb_log = lambda data, step: logged.append((data, step))

    calls = SimpleNamespace(launch=[])

    class _StubRunner:
        def collect(self):
            return [{"step": 30, "accept_length": 1.9, "acceptance_rate": 0.45, "completed": 32, "failed": 0}]

        def maybe_launch(self, step, model):
            calls.launch.append((step, model))
            return True

    recipe.decode_eval_runner = _StubRunner()
    recipe._maybe_run_decode_eval()

    assert calls.launch == [(42, recipe.draft_model)]
    assert len(logged) == 1
    data, step = logged[0]
    assert step == 42  # wandb steps must not go backwards
    assert data["train/tau_real"] == 1.9
    assert data["train/tau_real_step"] == 30


def test_recipe_hook_noop_without_runner():
    from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe._maybe_run_decode_eval()  # must not raise (bare object, no setup)


def test_recipe_hook_can_collect_without_launching():
    from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.runtime = SimpleNamespace(global_step=42)
    recipe._wandb_log = lambda *_args, **_kwargs: None

    class _StubRunner:
        def collect(self):
            return []

        def maybe_launch(self, *_args):
            raise AssertionError("final collection must not launch a worker")

    recipe.decode_eval_runner = _StubRunner()
    recipe._maybe_run_decode_eval(launch=False)


def test_recipe_hook_contains_optional_eval_failures(caplog):
    from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.runtime = SimpleNamespace(global_step=42)
    recipe.draft_model = object()

    class _StubRunner:
        def collect(self):
            return []

        def maybe_launch(self, *_args):
            raise RuntimeError("snapshot failed")

    recipe.decode_eval_runner = _StubRunner()
    recipe._maybe_run_decode_eval()

    assert "training continues" in caplog.text
