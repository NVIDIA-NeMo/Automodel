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
from nemo_automodel.components.launcher.slurm.arg_parser import render_script
from pathlib import Path

import subprocess
import logging
import os


def submit_slurm_job(config, job_dir):
    os.makedirs(job_dir, exist_ok=True)
    sbatch_script = render_script(config, job_dir)
    sbatch_script_path = os.path.join(job_dir, f"{config.job_name}.sbatch")
    with open(sbatch_script_path, "w") as fp:
        fp.write(sbatch_script)

    logging.info("Generated Slurm script ➜ {}".format(sbatch_script_path))

    proc = subprocess.Popen(["sbatch", sbatch_script_path], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
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
