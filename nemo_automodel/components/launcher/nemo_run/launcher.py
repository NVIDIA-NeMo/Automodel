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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo_automodel.components.launcher.base import Launcher
from nemo_automodel.components.launcher.interactive import resolve_recipe_cls
from nemo_automodel.components.launcher.nemo_run.config import NemoRunConfig

logger = logging.getLogger(__name__)


class NemoRunLauncher(Launcher):
    """Launch a recipe via NeMo-Run's executor API.

    Supports ``local``, ``slurm``, and ``k8s`` executor backends.
    Requires the optional ``nemo-run`` package to be installed.
    """

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
            logger.error(
                "nemo-run is not installed. Install with: pip install nemo-run"
            )
            sys.exit(1)

        nr_config = NemoRunConfig(**launcher_config)
        executor = self._build_executor(run, nr_config)

        recipe_cls = resolve_recipe_cls(recipe_target)

        from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

        cfg = parse_args_and_load_config(config_path)

        recipe = run.Partial(recipe_cls, cfg=cfg)

        with run.Experiment("automodel_job") as exp:
            exp.add(recipe, executor=executor, name="automodel")
            exp.run(sequential=True)

        return 0

    @staticmethod
    def _build_executor(run, nr_config: NemoRunConfig):
        if nr_config.executor == "local":
            return run.LocalExecutor(
                ntasks_per_node=nr_config.num_gpus_per_node,
                **nr_config.executor_kwargs,
            )
        elif nr_config.executor == "slurm":
            return run.SlurmExecutor(
                account=nr_config.account,
                partition=nr_config.partition,
                time=nr_config.time,
                nodes=nr_config.num_nodes,
                ntasks_per_node=nr_config.num_gpus_per_node,
                container_image=nr_config.container_image,
                **nr_config.executor_kwargs,
            )
        elif nr_config.executor == "k8s":
            return run.K8sExecutor(
                num_nodes=nr_config.num_nodes,
                num_gpus_per_node=nr_config.num_gpus_per_node,
                container_image=nr_config.container_image,
                **nr_config.executor_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown nemo_run executor: {nr_config.executor!r}. "
                f"Supported: local, slurm, k8s"
            )
