# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
from nemo_automodel.components.launcher.slurm.template import render_script

# Base options shared across test cases
BASE_OPTS = {
    "account": "proj",
    "partition": "batch",
    "nodes": 2,
    "time": "01:00:00",
    "job_name": "test_job",
    "master_port": 12345,
    "wandb_key": "dummy",
    "hf_home": "/tmp/hf",
    "hf_token": "dummy_token",
    "chdir": "/workspace",
    "command": "echo hello",
    "container_image": "nvcr.io/containers/test:latest",
    "container_mounts": "/data:/data",
}


def _render(extra_opts=None):
    opts = BASE_OPTS.copy()
    if extra_opts:
        opts.update(extra_opts)
    return render_script(opts, job_dir="/tmp/jobdir")


def test_gpus_per_node_included():
    script = _render({"gpus_per_node": 4})
    # Directive should be present with correct value
    assert "#SBATCH --gpus-per-node=4" in script
    # NUM_GPUS exported should equal value
    assert "export NUM_GPUS=4" in script


def test_gpus_per_node_default():
    script = _render()
    # No explicit GPU directive
    assert "#SBATCH --gpus-per-node=" not in script
    # NUM_GPUS should fallback to Slurm env var default expression
    assert r"export NUM_GPUS=${SLURM_GPUS_PER_NODE:-8}" in script, script


def test_custom_env_vars():
    env_vars = {"FOO": "bar", "HELLO": "world"}
    script = _render({"env_vars": env_vars})
    # Each env var should be exported exactly once
    for key, value in env_vars.items():
        line = f"export {key}={value}"
        assert script.count(line) == 1
    # Custom env vars should appear after HF_TOKEN export for readability
    hf_index = script.index("export HF_TOKEN")
    custom_index = script.index("export FOO=bar")
    assert custom_index > hf_index


def test_mandatory_placeholders_filled():
    script = _render()
    assert r"export NUM_GPUS=${SLURM_GPUS_PER_NODE:-8}" in script, script
    script = script.replace(r"export NUM_GPUS=${SLURM_GPUS_PER_NODE:-8}", "export NUM_GPUS=8")
    # Ensure no unfilled curly braces remain
    assert "{" not in script and "}" not in script, script
    # Check that job directory and job name are substituted
    assert "/tmp/jobdir" in script
    assert "#SBATCH -J test_job" in script
