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
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nemo_automodel.components.launcher.base import Launcher
from nemo_automodel.components.launcher.nemo_run.config import NemoRunConfig
from nemo_automodel.components.launcher.nemo_run.utils import (
    REMOTE_CONFIG_PATH,
    apply_overrides,
    load_executor_from_file,
    submit_nemo_run_job,
)

logger = logging.getLogger(__name__)


def _recipe_module_path(recipe_target: str, repo_root: str = ".") -> str:
    """Convert a dotted recipe target to a filesystem path relative to *repo_root*."""
    module_path = recipe_target.rsplit(".", 1)[0]
    return os.path.join(repo_root, module_path.replace(".", "/") + ".py")


class NemoRunLauncher(Launcher):
    """Launch a recipe via NeMo-Run's executor API.

    Supports loading pre-configured executors from ``$NEMORUN_HOME/executors.py``
    (or a custom path) and submitting jobs as ``nemo_run.Script`` objects.
    Works with any NeMo-Run executor backend (Slurm, Kubernetes, Docker, local).
    """

    @staticmethod
    def _build_inline_script(
        config_yaml: str,
        recipe_target: str,
        devices: int,
        num_nodes: int,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        """Build an inline bash script that writes the config and runs torchrun.

        The config YAML is embedded as a heredoc so that no separate file
        transfer is needed -- the script is self-contained.
        """
        script_path = _recipe_module_path(recipe_target)

        parts = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "# Write training config (nemo_run section already stripped)",
            f"cat > {REMOTE_CONFIG_PATH} << 'AUTOMODEL_CONFIG_EOF'",
            config_yaml.rstrip(),
            "AUTOMODEL_CONFIG_EOF",
            "",
        ]

        # torchrun command
        torchrun_parts = ["torchrun", f"--nproc-per-node={devices}"]

        if num_nodes > 1:
            torchrun_parts += [
                f"--nnodes={num_nodes}",
                "--node-rank=$NODE_RANK",
                "--rdzv-backend=c10d",
                "--master-addr=$MASTER_ADDR",
                "--master-port=$MASTER_PORT",
            ]

        torchrun_parts += [script_path, f"-c {REMOTE_CONFIG_PATH}"]

        if extra_args:
            torchrun_parts.extend(extra_args)

        parts.append(" \\\n    ".join(torchrun_parts))
        return "\n".join(parts) + "\n"

    def _resolve_executor(self, nr_config: NemoRunConfig) -> Any:
        """Load a named executor or build a local one."""
        try:
            import nemo_run as run
        except ImportError:
            logger.error("nemo-run is not installed. Install with: pip install nemo-run")
            sys.exit(1)

        if nr_config.executor == "local":
            devices = nr_config.devices or 1
            executor = run.LocalExecutor(ntasks_per_node=devices)
            if nr_config.env_vars:
                executor.env_vars = dict(nr_config.env_vars)
            return executor

        # Named executor from executors file
        executor = load_executor_from_file(nr_config.executor, nr_config.executors_file)
        apply_overrides(
            executor,
            nodes=nr_config.nodes,
            devices=nr_config.devices,
            container_image=nr_config.container_image,
            time=nr_config.time,
            mounts=nr_config.mounts,
            env_vars=nr_config.env_vars,
        )
        return executor

    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Dict[str, Any],
        extra_args: Optional[List[str]] = None,
    ) -> int:
        try:
            import nemo_run as run
        except ImportError:
            logger.error("nemo-run is not installed. Install with: pip install nemo-run")
            sys.exit(1)

        nr_config = NemoRunConfig(**launcher_config)
        executor = self._resolve_executor(nr_config)

        # Determine devices (GPUs per node) for the torchrun command.
        devices = nr_config.devices
        if devices is None:
            val = getattr(executor, "ntasks_per_node", None)
            devices = int(val) if isinstance(val, (int, float)) and val else 1
        num_nodes = nr_config.nodes
        if num_nodes is None:
            val = getattr(executor, "nodes", None)
            num_nodes = int(val) if isinstance(val, (int, float)) and val else 1

        # Write the training config (without nemo_run section) for local record.
        job_dir = os.path.join(
            nr_config.job_dir or os.path.join(os.getcwd(), "nemo_run_jobs"),
            str(int(_time.time())),
        )
        os.makedirs(job_dir, exist_ok=True)
        job_conf_path = os.path.join(job_dir, "job_config.yaml")
        config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
        with open(job_conf_path, "w") as fp:
            fp.write(config_yaml)
        logger.info("NeMo-Run job artifacts in: %s", job_dir)

        # Build the inline script
        inline = self._build_inline_script(
            config_yaml=config_yaml,
            recipe_target=recipe_target,
            devices=devices,
            num_nodes=num_nodes,
            extra_args=extra_args,
        )

        script = run.Script(inline=inline)
        job_name = nr_config.job_name or f"{recipe_target.rsplit('.', 1)[-1]}"

        return submit_nemo_run_job(
            script=script,
            executor=executor,
            job_name=job_name,
            detach=nr_config.detach,
            tail_logs=nr_config.tail_logs,
        )
