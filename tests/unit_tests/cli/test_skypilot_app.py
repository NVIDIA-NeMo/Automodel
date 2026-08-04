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
    assert "--rdzv_backend=c10d" in cmd
    assert "--master_addr=" in cmd
    assert "--nproc_per_node=8" in cmd


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
