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
    apply_overrides,
    load_executor_from_file,
    submit_nemo_run_job,
)

logger = logging.getLogger(__name__)

# Config filename and its path inside the container (/nemo_run/code/).
_CONFIG_FILENAME = "automodel_config.yaml"
_REMOTE_CONFIG_PATH = f"/nemo_run/code/{_CONFIG_FILENAME}"


class NemoRunLauncher(Launcher):
    """Launch a recipe via NeMo-Run's executor API.

    Supports loading pre-configured executors from ``$NEMORUN_HOME/executors.py``
    (or a custom path) and submitting jobs as ``nemo_run.Script`` objects.
    Works with any NeMo-Run executor backend (Slurm, Kubernetes, Docker, local).

    Uses NeMo-Run's native ``Torchrun`` launcher so that distributed training
    arguments (rendezvous, node rank, nproc-per-node) are managed automatically.
    The training config YAML is packaged via ``PatternPackager`` so it is
    available at ``/nemo_run/code/automodel_config.yaml`` inside the container.
    """

    def _resolve_executor(self, nr_config: NemoRunConfig) -> Any:
        """Load a named executor or build a local one."""
        try:
            import nemo_run as run
        except ImportError:
            logger.error("nemo-run is not installed. Install with: pip install nemo-run")
            sys.exit(1)

        if nr_config.executor == "local":
            executor = run.LocalExecutor()
            apply_overrides(executor, nr_config.overrides)
            return executor

        # Named executor from executors file
        executor = load_executor_from_file(nr_config.executor, nr_config.executors_file)
        apply_overrides(executor, nr_config.overrides)
        return executor

    @staticmethod
    def _configure_torchrun(executor: Any, devices: int) -> None:
        """Enable the native NeMo-Run Torchrun launcher on *executor*.

        Sets ``executor.launcher = "torchrun"`` and
        ``torchrun_nproc_per_node`` so NeMo-Run generates the correct
        ``torchrun --nproc-per-node=<N>`` invocation in the sbatch script.
        """
        executor.launcher = "torchrun"
        if hasattr(executor, "torchrun_nproc_per_node"):
            executor.torchrun_nproc_per_node = devices

    @staticmethod
    def _setup_packager(executor: Any, config_path: str) -> None:
        """Configure a ``PatternPackager`` that ships the config YAML.

        The packager tars the config file and NeMo-Run extracts it into
        ``{job_dir}/code/``, which is mounted at ``/nemo_run/code/`` inside
        the container.
        """
        try:
            import nemo_run as run
        except ImportError:
            return

        config_dir = os.path.dirname(config_path)
        executor.packager = run.PatternPackager(
            include_pattern=config_path,
            relative_path=config_dir,
        )

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

        nr_config = NemoRunConfig.from_dict(launcher_config)
        executor = self._resolve_executor(nr_config)

        # Determine devices (GPUs per node) via the executor's standard
        # nproc_per_node() method (defined on the base Executor class and
        # implemented by every backend).
        try:
            devices = executor.nproc_per_node()
        except NotImplementedError:
            devices = 1

        # Enable native Torchrun launcher (must be set *before* experiment.run
        # because NeMo-Run reads it during the packaging phase).
        self._configure_torchrun(executor, devices)

        # -- Write the training config for both local record and packaging. --
        job_dir = os.path.join(
            nr_config.job_dir or os.path.join(os.getcwd(), "nemo_run_jobs"),
            str(int(_time.time())),
        )
        os.makedirs(job_dir, exist_ok=True)
        config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)

        # Local record.
        local_config_path = os.path.join(job_dir, _CONFIG_FILENAME)
        with open(local_config_path, "w") as fp:
            fp.write(config_yaml)
        logger.info("NeMo-Run job artifacts in: %s", job_dir)

        # Set up PatternPackager so the config is shipped to the remote.
        self._setup_packager(executor, local_config_path)

        # Build the Script: use ``python -m <module>`` so the recipe is resolved
        # from the installed package, not a relative file path.
        module_path = recipe_target.rsplit(".", 1)[0]
        args = ["-c", _REMOTE_CONFIG_PATH]
        if extra_args:
            args.extend(extra_args)

        script = run.Script(
            path=module_path,
            m=True,
            entrypoint="python",
            args=args,
        )
        job_name = nr_config.job_name or f"{recipe_target.rsplit('.', 1)[-1]}"

        return submit_nemo_run_job(
            script=script,
            executor=executor,
            job_name=job_name,
            detach=nr_config.detach,
            tail_logs=nr_config.tail_logs,
        )
