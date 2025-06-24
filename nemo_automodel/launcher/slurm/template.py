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

"""
make_and_submit.py  ·  Generate a Slurm batch script from CLI args (defined in a
dataclass), drop it to a temp file, and optionally submit it via sbatch.

Example
-------
python make_and_submit.py \
  --job-name llama3-test \
  --nodes 2 \
  --time 00:05:00 \
  --command "pip3 install torchdata; python3 /lustre/.../finetune.py --config cfg.yaml" \
  --dry-run          # inspect only
"""

import dataclasses, getpass, socket
from datetime import datetime
from nemo_automodel.launcher.slurm.config import SlurmConfig

HEADER = (
    "# -------------------------------------------------------------------\n"
    "# NeMo AutoModel sbatch script\n"
    "# User: {user}\n"
    "# Host: {host}\n"
    "# Date: {timestamp}\n"
    "# -------------------------------------------------------------------\n"
)

TEMPLATE = """#!/bin/bash
"""+ HEADER + """\
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node {ntasks}
#SBATCH --time {time}
#SBATCH --mail-type=FAIL
#SBATCH --exclusive
#SBATCH --output={log_dir}/slurm_%x_%j.out
#SBATCH -J {job_name}

# Container & mounts
CONTAINER_PATH="{container}"
NEMO_MOUNT="{nemo_mount}"
CACHE_MOUNT="{cache_mount}"
MOUNT="${{NEMO_MOUNT}}:{nemo_target},${{CACHE_MOUNT}}:{cache_target},{extra_mounts}"

# Multi-node env
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT={master_port}
export NUM_GPUS={gpus_per_node}
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))

# Experiment env
export WANDB_API_KEY={wandb_key}
export HF_HOME={hf_home}
export HF_TOKEN={hf_token}

# User command
read -r -d '' CMD <<'EOF'
{command}
EOF
echo "$CMD"

srun \\
    --mpi=pmix \\
    --container-entrypoint \\
    --no-container-mount-home \\
    --container-image=${{CONTAINER_PATH}} \\
    --container-mounts=${{MOUNT}} \\
    --export=ALL \\
    bash -c "$CMD"
"""

def render_script(opts: SlurmConfig) -> str:
    return TEMPLATE.format(
        user=getpass.getuser(),
        host=socket.gethostname(),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **dataclasses.asdict(opts),
    )
