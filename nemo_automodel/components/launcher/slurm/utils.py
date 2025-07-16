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
import dataclasses
import logging
import os
import subprocess
from pathlib import Path
from typing import Union

from nemo_automodel.components.launcher.slurm.config import SlurmConfig, VolumeMapping
from nemo_automodel.components.launcher.slurm.template import render_script


def volume_map_to_str(val: Union[str, dict, VolumeMapping]) -> str:
    if isinstance(val, dict):
        assert "source" in val
        assert "dest" in val
        return f"{val['source']}:{val['dest']}"
    elif isinstance(val, VolumeMapping):
        return f"{val.source}:{val.dest}"
    elif isinstance(val, str):
        parts = val.split(":")
        if len(parts) == 1:
            # val = "/path"
            return f"{val}:{val}"
        elif len(parts) == 2:
            # val = "/path_a:/path_b"
            # fails on:
            #   val = ":/path_b"
            #   val = "/path_a:"
            #   val = ":"
            assert len(parts[0]) > 0 and len(parts[1]) > 0, parts
            return f"{parts[0]}:{parts[1]}"
        else:
            raise ValueError(val)
    else:
        raise ValueError(type(val))


def make_container_mounts(opts: dict) -> list:
    container_mounts = []
    if (hf_home := opts.get("hf_home", None)) and not hf_home.startswith("~/") and not hf_home.startswith("/home"):
        # HF_HOME may require both mount and env-var export.
        container_mounts.append(volume_map_to_str(hf_home))
    if val := opts.get("nemo_mount", None):
        container_mounts.append(volume_map_to_str(val))
    opts.pop("nemo_mount", None)
    for val in opts.get("extra_mounts", []):
        container_mounts.append(volume_map_to_str(val))
    opts.pop("extra_mounts", None)
    return container_mounts


def submit_slurm_job(config: SlurmConfig, job_dir) -> int:
    os.makedirs(job_dir, exist_ok=True)
    # Render the sbatch script
    opts = dataclasses.asdict(config)
    opts["container_mounts"] = ",".join(make_container_mounts(opts))
    sbatch_script = render_script(opts, job_dir)
    # write the sbatch script
    sbatch_script_path = os.path.join(job_dir, f"{config.job_name}.sbatch")
    with open(sbatch_script_path, "w") as fp:
        fp.write(sbatch_script)

    logging.info("Generated Slurm script ➜ {}".format(sbatch_script_path))

    proc = subprocess.Popen(["sbatch", sbatch_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = tuple(map(bytes.decode, proc.communicate()))
    logging.info(stdout)
    with open(Path(job_dir) / "subproc_sbatch.stdout", "w") as fp:
        fp.write(stdout)

    if proc.returncode != 0:
        logging.error(stderr)
    with open(Path(job_dir) / "subproc_sbatch.stderr", "w") as fp:
        fp.write(stderr)

    return proc.returncode
