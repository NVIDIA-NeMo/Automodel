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

from nemo_automodel.components.launcher.skypilot.config import SkyPilotConfig

# Fixed remote path where the job config YAML is uploaded.
REMOTE_CONFIG_PATH = "/tmp/automodel_job_config.yaml"

# Default setup command: install the package from the synced workdir.
_DEFAULT_SETUP = "cd ~/sky_workdir && pip install -e . --quiet"

_CLOUD_CLASSES = {
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "lambda": "Lambda",
    "kubernetes": "Kubernetes",
}


def _get_cloud(cloud_name: str):
    """Return a sky cloud object for the given cloud name string."""
    import sky

    cls_name = _CLOUD_CLASSES[cloud_name.lower()]
    return getattr(sky, cls_name)()


def submit_skypilot_job(config: SkyPilotConfig, job_dir: str) -> int:
    """
    Launch a training job on a cloud VM via SkyPilot.

    The local job config written to *job_dir*/job_config.yaml is uploaded to
    REMOTE_CONFIG_PATH on the remote VM.  The code in the current working
    directory is synced to ~/sky_workdir via SkyPilot's workdir mechanism.

    Args:
        config: Populated SkyPilotConfig (including the training command).
        job_dir: Local directory holding the job artifacts.

    Returns:
        0 on successful submission.
    """
    try:
        import sky
    except ImportError as exc:
        raise ImportError(
            "SkyPilot is not installed. "
            "Install it with: pip install skypilot[<cloud>]  "
            "(e.g. skypilot[gcp], skypilot[aws])"
        ) from exc

    local_config_path = os.path.join(job_dir, "job_config.yaml")

    # Build the environment variable dict for the remote task.
    envs: dict[str, str] = dict(config.env_vars)
    if config.hf_token:
        envs["HF_TOKEN"] = config.hf_token
    if config.wandb_key:
        envs["WANDB_API_KEY"] = config.wandb_key
    envs.setdefault("HF_HOME", config.hf_home)

    setup_cmd = config.setup if config.setup else _DEFAULT_SETUP

    task = sky.Task(
        name=config.job_name or "automodel_job",
        setup=setup_cmd,
        run=config.command,
        envs=envs,
        num_nodes=config.num_nodes,
    )
    task.workdir = "."

    task.set_file_mounts({REMOTE_CONFIG_PATH: local_config_path})

    task.set_resources(
        sky.Resources(
            cloud=_get_cloud(config.cloud),
            region=config.region,
            zone=config.zone,
            accelerators=config.accelerators,
            use_spot=config.use_spot,
            disk_size=config.disk_size,
            instance_type=config.instance_type,
        )
    )

    cluster_name = config.job_name or "automodel-cluster"
    logging.info(
        f"Submitting SkyPilot job '{cluster_name}' on {config.cloud} "
        f"({config.accelerators}, spot={config.use_spot}, nodes={config.num_nodes})"
    )

    sky.launch(
        task,
        cluster_name=cluster_name,
        detach_run=True,
        stream_logs=False,
    )
    return 0
