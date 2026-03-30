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
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path


def submit_slurm_job(script_path: str, env_vars: dict[str, str], job_dir: str) -> int:
    """Submit an sbatch script with AUTOMODEL_* environment variables."""
    os.makedirs(job_dir, exist_ok=True)

    env = {**os.environ, **env_vars}
    logging.info("Submitting SLURM script: %s", script_path)
    for key, val in env_vars.items():
        logging.info("  %s=%s", key, val)

    proc = subprocess.Popen(
        ["sbatch", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    stdout, stderr = tuple(map(bytes.decode, proc.communicate()))
    logging.info(stdout)
    with open(Path(job_dir) / "subproc_sbatch.stdout", "w") as fp:
        fp.write(stdout)

    if proc.returncode != 0:
        logging.error(stderr)
    with open(Path(job_dir) / "subproc_sbatch.stderr", "w") as fp:
        fp.write(stderr)

    return proc.returncode
