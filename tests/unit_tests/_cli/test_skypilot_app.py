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

import importlib
import shutil
import subprocess
import sys

import pytest
import yaml

from nemo_automodel.components.launcher.skypilot.launcher import (
    SkyPilotLauncher,
    _parse_gpus_per_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


RECIPE_TARGET = "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"


# ---------------------------------------------------------------------------
# _parse_gpus_per_node
# ---------------------------------------------------------------------------


def test_parse_gpus_per_node_standard():
    assert _parse_gpus_per_node("T4:1") == 1
    assert _parse_gpus_per_node("A100:8") == 8
    assert _parse_gpus_per_node("V100:4") == 4


def test_parse_gpus_per_node_no_colon():
    assert _parse_gpus_per_node("T4") == 1


def test_parse_gpus_per_node_non_int():
    assert _parse_gpus_per_node("T4:bad") == 1


# ---------------------------------------------------------------------------
# SkyPilotLauncher._build_command
# ---------------------------------------------------------------------------


def test_build_command_single_node():
    launcher = SkyPilotLauncher()
    cmd = launcher._build_command(RECIPE_TARGET, "/tmp/config.yaml", gpus_per_node=4, num_nodes=1)
    assert "PYTHONPATH=~/sky_workdir:$PYTHONPATH" in cmd
    assert "torchrun" in cmd
    assert "--nproc_per_node=4" in cmd
    assert "SKYPILOT_NUM_NODES" not in cmd
    assert "SKYPILOT_NODE_RANK" not in cmd


def test_build_command_multi_node():
    launcher = SkyPilotLauncher()
    cmd = launcher._build_command(RECIPE_TARGET, "/tmp/config.yaml", gpus_per_node=8, num_nodes=2)
    assert "--nnodes=$SKYPILOT_NUM_NODES" in cmd
    assert "--node_rank=$SKYPILOT_NODE_RANK" in cmd
    assert "--rdzv_backend=c10d" not in cmd
    assert "--master_addr=" in cmd
    assert "--nproc_per_node=8" in cmd


@pytest.fixture
def torch_run(monkeypatch):
    """Return the real ``torch.distributed.run`` module.

    ``test_app.py`` replaces it with a ``MagicMock`` in ``sys.modules`` at import time,
    so import the real module for the duration of the test.
    """
    import torch.distributed

    monkeypatch.delitem(sys.modules, "torch.distributed.run", raising=False)
    monkeypatch.delattr(torch.distributed, "run", raising=False)
    return importlib.import_module("torch.distributed.run")


def _torchrun_argv_under_bash(cmd, monkeypatch):
    """Run ``cmd`` the way SkyPilot does (via bash) and return the argv torchrun receives.

    ``torchrun`` is replaced by a shell function that prints one argument per line, and
    the SkyPilot node metadata is set as on a two-node cluster, where
    ``SKYPILOT_NODE_IPS`` holds one IP per line.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to expand the generated SkyPilot run command")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "1")
    script = 'torchrun() { printf "%s\\n" "$@"; }\n' + cmd
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_build_command_multi_node_master_addr_is_one_argument(monkeypatch, torch_run):
    launcher = SkyPilotLauncher()
    cmd = launcher._build_command(RECIPE_TARGET, "/tmp/config.yaml", gpus_per_node=8, num_nodes=2)
    argv = _torchrun_argv_under_bash(cmd, monkeypatch)

    args = torch_run.get_args_parser().parse_args(argv)
    assert args.master_addr == "10.0.0.1"
    assert args.training_script.endswith("train_ft.py")
    assert args.training_script_args == ["-c", "/tmp/config.yaml"]


def test_build_command_multi_node_rendezvous_uses_head_node(monkeypatch, torch_run):
    launcher = SkyPilotLauncher()
    cmd = launcher._build_command(RECIPE_TARGET, "/tmp/config.yaml", gpus_per_node=8, num_nodes=2)
    argv = _torchrun_argv_under_bash(cmd, monkeypatch)

    # config_from_args sets OMP_NUM_THREADS when it is unset; keep it scoped to this test.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    config, _, _ = torch_run.config_from_args(torch_run.get_args_parser().parse_args(argv))
    # Every node must rendezvous at the head node (first SkyPilot IP), not at its own localhost.
    assert config.rdzv_endpoint == "10.0.0.1:12375"
    assert config.min_nodes == config.max_nodes == 2


def test_build_command_extra_args():
    launcher = SkyPilotLauncher()
    cmd = launcher._build_command(
        RECIPE_TARGET,
        "/tmp/config.yaml",
        gpus_per_node=1,
        num_nodes=1,
        extra_args=["--my-flag", "val"],
    )
    assert "--my-flag" in cmd
    assert "val" in cmd


# ---------------------------------------------------------------------------
# SkyPilotLauncher.launch
# ---------------------------------------------------------------------------


def test_launch_single_node(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        captured["job_dir"] = job_dir
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    launcher = SkyPilotLauncher()
    config = {"model": {"name": "gpt2"}}
    skypilot_cfg = {
        "cloud": "gcp",
        "accelerators": "T4:4",
        "job_dir": str(tmp_path / "sky_jobs"),
    }

    result = launcher.launch(config, tmp_path / "cfg.yaml", RECIPE_TARGET, skypilot_cfg)
    assert result == 0
    assert "torchrun" in captured["cfg"].command
    assert "--nproc_per_node=4" in captured["cfg"].command
    assert "SKYPILOT_NUM_NODES" not in captured["cfg"].command


def test_launch_multi_node(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    launcher = SkyPilotLauncher()
    config = {"model": {"name": "llama"}}
    skypilot_cfg = {
        "cloud": "aws",
        "accelerators": "A100:8",
        "num_nodes": 2,
        "job_dir": str(tmp_path / "sky_jobs"),
    }

    launcher.launch(config, tmp_path / "cfg.yaml", RECIPE_TARGET, skypilot_cfg)
    assert "--nnodes=$SKYPILOT_NUM_NODES" in captured["cfg"].command
    assert "--node_rank=$SKYPILOT_NODE_RANK" in captured["cfg"].command
    assert "--nproc_per_node=8" in captured["cfg"].command


def test_launch_explicit_gpus_per_node(monkeypatch, tmp_path):
    captured = {}

    def fake_submit(cfg, job_dir):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    launcher = SkyPilotLauncher()
    skypilot_cfg = {
        "cloud": "gcp",
        "accelerators": "T4:1",
        "gpus_per_node": 2,
        "job_dir": str(tmp_path / "sky_jobs"),
    }

    launcher.launch({}, tmp_path / "cfg.yaml", RECIPE_TARGET, skypilot_cfg)
    assert "--nproc_per_node=2" in captured["cfg"].command


def test_launch_strips_skypilot_from_written_config(monkeypatch, tmp_path):
    written_configs = {}

    def fake_submit(cfg, job_dir):
        # Read back the job_config.yaml that was written
        import os

        conf_path = os.path.join(job_dir, "job_config.yaml")
        with open(conf_path) as f:
            written_configs["data"] = yaml.safe_load(f)
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.utils.submit_skypilot_job",
        fake_submit,
    )

    launcher = SkyPilotLauncher()
    config = {"model": {"name": "gpt2"}}
    skypilot_cfg = {
        "cloud": "gcp",
        "job_dir": str(tmp_path / "sky_jobs"),
    }

    launcher.launch(config, tmp_path / "cfg.yaml", RECIPE_TARGET, skypilot_cfg)
    # The written config should not contain the skypilot section
    assert "skypilot" not in written_configs["data"]
    assert written_configs["data"]["model"]["name"] == "gpt2"
