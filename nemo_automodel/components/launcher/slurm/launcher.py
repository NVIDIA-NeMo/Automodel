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
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nemo_automodel.components.launcher.base import Launcher

logger = logging.getLogger(__name__)


def _get_automodel_repo_root() -> Optional[Path]:
    cwd = Path.cwd()
    if (cwd / "nemo_automodel/components").exists() and (cwd / "examples/").exists():
        return cwd
    return None


def _recipe_module_path(recipe_target: str, repo_root: str) -> str:
    """Convert ``nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction``
    into ``<repo_root>/nemo_automodel/recipes/llm/train_ft.py``."""
    module_path = recipe_target.rsplit(".", 1)[0]
    return os.path.join(repo_root, module_path.replace(".", "/") + ".py")


class SlurmLauncher(Launcher):
    """Submit a recipe job to a SLURM cluster using a user-provided sbatch script.

    The ``slurm:`` YAML section requires a ``script`` field pointing to an
    sbatch script.  The CLI generates the torchrun command, writes the recipe
    config, and exports ``AUTOMODEL_*`` environment variables for the script
    to use.  See ``slurm.sub`` at the repo root for a reference template.
    """

    def _resolve_job_dir(self, slurm_config: Dict[str, Any]) -> str:
        job_dir = os.path.join(
            slurm_config.pop("job_dir", os.path.join(os.getcwd(), "slurm_jobs")),
            str(int(time.time())),
        )
        os.makedirs(job_dir, exist_ok=True)

        last_dir = Path(job_dir).parts[-1]
        assert len(last_dir) == 10 and last_dir.isdigit(), (
            "Expected last dir to be unix timestamp",
            job_dir,
        )
        return job_dir

    def _resolve_repo_root(self, slurm_config: Dict[str, Any]) -> str:
        if "repo_root" in slurm_config:
            repo_root = slurm_config.pop("repo_root")
            logger.info("Running job using source defined in yaml: %s", repo_root)
        else:
            detected = _get_automodel_repo_root()
            if detected:
                repo_root = str(detected)
                logger.info("Running job using source from: %s", repo_root)
            else:
                repo_root = "/opt/Automodel"
        logger.info("Using %s as code repo", repo_root)
        return repo_root

    def _build_command(
        self,
        recipe_target: str,
        repo_root: str,
        job_dir: str,
        job_conf_path: str,
        nsys_enabled: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        script_path = _recipe_module_path(recipe_target, repo_root)

        if nsys_enabled:
            profile_cmd = (
                f"nsys profile -s none "
                f"--trace=cuda,cudnn,nvtx "
                f"--cudabacktrace=all "
                f"--cuda-graph-trace=node "
                f"--python-backtrace=cuda "
                f"--wait all "
                f"-o {job_dir}/automodel_profile_%p.nsys-rep "
                f"--force-overwrite true "
                f"--capture-range=cudaProfilerApi "
                f"--capture-range-end=stop "
            )
        else:
            profile_cmd = ""

        command_parts = [
            f"PYTHONPATH={repo_root}:$PYTHONPATH",
            f"{profile_cmd}torchrun",
            "--nproc_per_node=${SLURM_GPUS_PER_NODE:-8}",
            "--nnodes=${SLURM_NNODES:-1}",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}",
                "See docs/launcher/slurm.md for examples."
            "-c",
            job_conf_path,
        ]
        if extra_args:
            command_parts.extend(extra_args)
        return " ".join(command_parts)

    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Dict[str, Any],
        extra_args: Optional[List[str]] = None,
    ) -> int:
        from nemo_automodel.components.launcher.slurm.utils import submit_slurm_job

        slurm_config = dict(launcher_config)

        # script is required (accept both "script" and legacy "custom_script")
        script = slurm_config.pop("script", None) or slurm_config.pop("custom_script", None)
        if script is None:
            raise ValueError(
                "slurm.script is required. Provide a path to your sbatch script.\n"
                "Copy the reference template to get started:\n"
                "  cp slurm.sub my_cluster.sub\n"
                "Then add to your YAML:\n"
                "  slurm:\n"
                "    script: my_cluster.sub\n"
                "See docs/launcher/cluster.md for examples."
            )

        script_path = Path(script).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"SLURM script not found: {script_path}")

        job_dir = self._resolve_job_dir(slurm_config)

        job_conf_path = os.path.join(job_dir, "job_config.yaml")
        with open(job_conf_path, "w") as fp:
            yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
        logger.info("Logging Slurm job in: %s", job_dir)

        repo_root = self._resolve_repo_root(slurm_config)

        command = self._build_command(
            recipe_target,
            repo_root,
            job_dir,
            job_conf_path,
            nsys_enabled=slurm_config.pop("nsys_enabled", False),
            extra_args=extra_args,
        )

        # Copy script to job dir for reproducibility
        copied_script = Path(job_dir) / script_path.name
        shutil.copy2(script_path, copied_script)

        env_vars = {
            "AUTOMODEL_COMMAND": command,
            "AUTOMODEL_CONFIG": job_conf_path,
            "AUTOMODEL_JOB_DIR": job_dir,
            "AUTOMODEL_REPO_ROOT": repo_root,
        }
        logger.info("Using SLURM script: %s", script_path)
        return submit_slurm_job(str(copied_script), env_vars, job_dir)
