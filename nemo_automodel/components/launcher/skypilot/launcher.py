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

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nemo_automodel.components.launcher.base import Launcher
from nemo_automodel.components.launcher.skypilot.config import SkyPilotConfig
from nemo_automodel.components.launcher.skypilot.utils import REMOTE_CONFIG_PATH

logger = logging.getLogger(__name__)


def _parse_gpus_per_node(accelerators: str) -> int:
    """Extract GPU count from an accelerator string like ``'A100:8'``.

    Returns 1 when the string cannot be parsed.
    """
    parts = accelerators.split(":")
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 1


def _recipe_module_path(recipe_target: str, repo_root: str) -> str:
    module_path = recipe_target.rsplit(".", 1)[0]
    return os.path.join(repo_root, module_path.replace(".", "/") + ".py")


class SkyPilotLauncher(Launcher):
    """Launch a recipe job on a cloud VM via SkyPilot."""

    def _build_command(
        self,
        recipe_target: str,
        job_conf_path: str,
        gpus_per_node: int,
        num_nodes: int,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        repo_root = "~/sky_workdir"
        script_path = _recipe_module_path(recipe_target, repo_root)

        parts = [
            f"PYTHONPATH={repo_root}:$PYTHONPATH",
            "torchrun",
            f"--nproc_per_node={gpus_per_node}",
        ]

        if num_nodes > 1:
            parts += [
                "--nnodes=$SKYPILOT_NUM_NODES",
                "--node_rank=$SKYPILOT_NODE_RANK",
                "--rdzv_backend=c10d",
                "--master_addr=$(echo $SKYPILOT_NODE_IPS | head -n1)",
                "--master_port=12375",
            ]

        parts += [script_path, "-c", job_conf_path]

        if extra_args:
            parts.extend(extra_args)

        return " ".join(parts)

    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Dict[str, Any],
        extra_args: Optional[List[str]] = None,
    ) -> int:
        from nemo_automodel.components.launcher.skypilot.utils import submit_skypilot_job

        skypilot_cfg = dict(launcher_config)

        job_dir = os.path.join(
            skypilot_cfg.pop("job_dir", os.path.join(os.getcwd(), "skypilot_jobs")),
            str(int(time.time())),
        )
        os.makedirs(job_dir, exist_ok=True)

        # Write the training config (without skypilot section) for upload.
        job_conf_path = os.path.join(job_dir, "job_config.yaml")
        with open(job_conf_path, "w") as fp:
            yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
        logger.info("SkyPilot job artifacts in: %s", job_dir)

        accelerators = skypilot_cfg.get("accelerators", "T4:1")
        gpus_per_node = skypilot_cfg.pop("gpus_per_node", None) or _parse_gpus_per_node(accelerators)
        num_nodes = skypilot_cfg.get("num_nodes", 1)

        command = self._build_command(
            recipe_target,
            REMOTE_CONFIG_PATH,
            gpus_per_node,
            num_nodes,
            extra_args=extra_args,
        )

        job_name = skypilot_cfg.pop("job_name", "") or f"{recipe_target.rsplit('.', 1)[-1]}"

        sky_config = SkyPilotConfig(
            command=command,
            job_name=job_name,
            **{k: v for k, v in skypilot_cfg.items() if k in SkyPilotConfig.__dataclass_fields__},
        )

        return submit_skypilot_job(sky_config, job_dir)
