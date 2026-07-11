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
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.decode_eval import (
    DecodeEvalConfig,
    DecodeEvalRunner,
    _bench_args,
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
    )
    kwargs.update(overrides)
    return DecodeEvalConfig(**kwargs)


def test_resolve_returns_none_when_absent_or_disabled(tmp_path):
    assert (
        resolve_decode_eval_config({}, default_target="/t", default_input_data="/d", output_dir=str(tmp_path)) is None
    )
    cfg = {"decode_eval": {"every_steps": 0, "cuda_visible_devices": "7"}}
    assert (
        resolve_decode_eval_config(cfg, default_target="/t", default_input_data="/d", output_dir=str(tmp_path)) is None
    )


def test_resolve_requires_reserved_gpu(tmp_path):
    cfg = {"decode_eval": {"every_steps": 100}}
    with pytest.raises(ValueError, match="cuda_visible_devices"):
        resolve_decode_eval_config(cfg, default_target="/t", default_input_data="/d", output_dir=str(tmp_path))


def test_resolve_fills_defaults_from_recipe(tmp_path):
    cfg = {"decode_eval": {"every_steps": 250, "cuda_visible_devices": 6}}
    resolved = resolve_decode_eval_config(
        cfg, default_target="/models/qwen", default_input_data="/data/messages", output_dir=str(tmp_path)
    )
    assert resolved.every_steps == 250
    assert resolved.cuda_visible_devices == "6"
    assert resolved.target_model == "/models/qwen"
    assert resolved.input_data == "/data/messages"
    assert resolved.output_dir == os.path.join(str(tmp_path), "decode_eval")


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
    def __init__(self, alive=True, pid=4242):
        self._alive = alive
        self.pid = pid

    def poll(self):
        return None if self._alive else 0


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
    assert runner.maybe_launch(30, model)

    popen_calls = [entry for entry in launched if entry[0] == "popen"]
    assert len(popen_calls) == 2
    argv = popen_calls[0][1]
    assert "--step" in argv and argv[argv.index("--step") + 1] == "10"
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


def test_serve_argv_builds_speculative_server_command(tmp_path, monkeypatch):
    """The worker must drive serve_vllm's library surface, not reimplement it."""
    seen = {}

    def _fake_build(serve_args):
        seen.update(vars(serve_args))
        return ["python", "-m", "vllm.entrypoints.openai.api_server"]

    monkeypatch.setattr("nemo_automodel.components.speculative.serve_vllm.build_vllm_argv", _fake_build)
    args = SimpleNamespace(
        target="/models/target",
        draft=str(tmp_path / "step_10"),
        num_speculative_tokens=None,
        port=8199,
        gpu_memory_utilization=0.8,
        max_model_len=None,
        trust_remote_code=False,
    )
    argv = _serve_argv(args)
    assert argv[0] == "python"
    assert seen["target"] == "/models/target"
    assert seen["draft"] == str(tmp_path / "step_10")
    assert seen["port"] == 8199
    assert seen["host"] == "127.0.0.1"


def test_bench_args_carry_workload_settings():
    args = SimpleNamespace(
        target="/models/target",
        input_data="/data/train.jsonl",
        num_prompts=16,
        concurrency=4,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        messages_column="messages",
        prompt_column=None,
        split="train",
        dataset_name=None,
        shuffle_seed=42,
        timeout_s=600.0,
    )
    bench = _bench_args(args, "http://127.0.0.1:8199")
    assert bench.server == "http://127.0.0.1:8199"
    assert bench.model == "/models/target"
    assert bench.num_prompts == 16
    assert bench.baseline_server is None
    assert bench.prompt_context_column is None


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
