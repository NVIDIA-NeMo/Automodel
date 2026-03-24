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
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

# Stub heavy optional deps before any module under test is imported.
sys.modules.setdefault("nemo_run", mock.MagicMock())
sys.modules.setdefault("torch.distributed.run", mock.MagicMock())

import nemo_automodel._cli.app as module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# _parse_gpus_per_node
# ---------------------------------------------------------------------------

def test_parse_gpus_per_node_standard():
    assert module._parse_gpus_per_node("T4:1") == 1
    assert module._parse_gpus_per_node("A100:8") == 8
    assert module._parse_gpus_per_node("V100:4") == 4


def test_parse_gpus_per_node_no_colon():
    # Falls back to 1 when format is unexpected.
    assert module._parse_gpus_per_node("T4") == 1


def test_parse_gpus_per_node_non_int():
    assert module._parse_gpus_per_node("T4:bad") == 1


# ---------------------------------------------------------------------------
# launch_with_skypilot – command construction
# ---------------------------------------------------------------------------

def _dummy_args(domain="llm", command="finetune"):
    return SimpleNamespace(domain=domain, command=command)


def test_launch_with_skypilot_single_node_command(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    job_conf_path = str(tmp_path / "job_config.yaml")
    _write_yaml(job_conf_path, {"model": "gpt2"})

    skypilot_cfg = {"cloud": "gcp", "accelerators": "T4:4"}
    module.launch_with_skypilot(
        _dummy_args(), job_conf_path, str(tmp_path), skypilot_cfg
    )

    cmd = captured["cfg"].command
    assert "torchrun" in cmd
    assert "--nproc_per_node=4" in cmd
    # No multi-node flags for single node
    assert "SKYPILOT_NUM_NODES" not in cmd
    assert "SKYPILOT_NODE_RANK" not in cmd


def test_launch_with_skypilot_multi_node_command(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    job_conf_path = str(tmp_path / "job_config.yaml")
    _write_yaml(job_conf_path, {"model": "llama"})

    skypilot_cfg = {"cloud": "aws", "accelerators": "A100:8", "num_nodes": 2}
    module.launch_with_skypilot(
        _dummy_args(), job_conf_path, str(tmp_path), skypilot_cfg
    )

    cmd = captured["cfg"].command
    assert "--nnodes=$SKYPILOT_NUM_NODES" in cmd
    assert "--node_rank=$SKYPILOT_NODE_RANK" in cmd
    assert "--master_addr=" in cmd
    assert "--nproc_per_node=8" in cmd


def test_launch_with_skypilot_explicit_gpus_per_node(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    job_conf_path = str(tmp_path / "job_config.yaml")
    _write_yaml(job_conf_path, {})

    # gpus_per_node overrides the value parsed from accelerators
    skypilot_cfg = {"cloud": "gcp", "accelerators": "T4:1", "gpus_per_node": 2}
    module.launch_with_skypilot(
        _dummy_args(), job_conf_path, str(tmp_path), skypilot_cfg
    )

    assert "--nproc_per_node=2" in captured["cfg"].command


def test_launch_with_skypilot_default_job_name(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    job_conf_path = str(tmp_path / "job_config.yaml")
    _write_yaml(job_conf_path, {})

    skypilot_cfg = {"cloud": "gcp"}
    module.launch_with_skypilot(
        _dummy_args(domain="llm", command="finetune"),
        job_conf_path,
        str(tmp_path),
        skypilot_cfg,
    )

    assert captured["cfg"].job_name == "llm_finetune"


def test_launch_with_skypilot_extra_args_appended(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    job_conf_path = str(tmp_path / "job_config.yaml")
    _write_yaml(job_conf_path, {})

    skypilot_cfg = {"cloud": "gcp"}
    module.launch_with_skypilot(
        _dummy_args(), job_conf_path, str(tmp_path), skypilot_cfg,
        extra_args=["--my-flag", "val"],
    )

    assert "--my-flag val" in captured["cfg"].command


# ---------------------------------------------------------------------------
# main() – skypilot branch
# ---------------------------------------------------------------------------

def test_main_skypilot_branch(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        f"""
skypilot:
  cloud: gcp
  accelerators: T4:1
  job_dir: {tmp_path / "skypilot_jobs"}
model:
  name: gpt2
"""
    )

    captured = {}

    def fake_launch(args, job_conf_path, job_dir, skypilot_config, extra_args=None):
        captured["skypilot_config"] = dict(skypilot_config)
        captured["job_dir"] = job_dir
        # Verify the stripped config (no skypilot key) was written
        with open(job_conf_path) as f:
            data = yaml.safe_load(f)
        assert "skypilot" not in data
        assert data.get("model", {}).get("name") == "gpt2"
        return 0

    monkeypatch.setattr(module, "launch_with_skypilot", fake_launch)
    monkeypatch.setattr(module.time, "time", lambda: 1234567890)
    monkeypatch.setattr("sys.argv", ["automodel", "finetune", "llm", "-c", str(cfg_file)])

    result = module.main()
    assert result == 0
    assert "cloud" in captured["skypilot_config"]
    assert captured["skypilot_config"]["cloud"] == "gcp"
    # job_dir should be popped before forwarding
    assert "job_dir" not in captured["skypilot_config"]
    assert captured["job_dir"].endswith("1234567890")


def test_main_skypilot_preserves_env_var_placeholders(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        f"""
skypilot:
  cloud: aws
  hf_token: ${{HF_TOKEN}}
model:
  name: llama
"""
    )

    captured = {}

    def fake_launch(args, job_conf_path, job_dir, skypilot_config, extra_args=None):
        captured["skypilot_config"] = dict(skypilot_config)
        return 0

    monkeypatch.setattr(module, "launch_with_skypilot", fake_launch)
    monkeypatch.setattr(module.time, "time", lambda: 9999999999)
    monkeypatch.setattr("sys.argv", ["automodel", "finetune", "llm", "-c", str(cfg_file)])

    module.main()
    # env-var placeholders must survive YAML round-trip unchanged
    assert captured["skypilot_config"]["hf_token"] == "${HF_TOKEN}"
