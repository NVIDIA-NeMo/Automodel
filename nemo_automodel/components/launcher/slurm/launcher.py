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
    """Submit a recipe job to a SLURM cluster."""

    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Dict[str, Any],
        extra_args: Optional[List[str]] = None,
    ) -> int:
        from nemo_automodel.components.launcher.slurm.config import SlurmConfig, VolumeMapping
        from nemo_automodel.components.launcher.slurm.utils import submit_slurm_job

        slurm_config = dict(launcher_config)

        job_dir = os.path.join(
            slurm_config.pop("job_dir", os.path.join(os.getcwd(), "slurm_jobs")),
            str(int(time.time())),
        )
        os.makedirs(job_dir, exist_ok=True)

        job_conf_path = os.path.join(job_dir, "job_config.yaml")
        with open(job_conf_path, "w") as fp:
            yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
        logger.info("Logging Slurm job in: %s", job_dir)

        last_dir = Path(job_dir).parts[-1]
        assert len(last_dir) == 10 and last_dir.isdigit(), (
            "Expected last dir to be unix timestamp",
            job_dir,
        )

        if "hf_home" not in slurm_config:
            slurm_config["hf_home"] = str(Path(job_dir).parent / ".hf_home")
            os.makedirs(slurm_config["hf_home"], exist_ok=True)
        logger.info("Using HF_HOME= `%s`", slurm_config["hf_home"])

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

        if slurm_config.get("job_name", "") == "":
            slurm_config["job_name"] = "automodel_job"

        script_path = _recipe_module_path(recipe_target, repo_root)

        if slurm_config.get("nsys_enabled", False):
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
            f"uv sync --inexact --frozen $(cat /opt/uv_args.txt) && {profile_cmd}uv run --no-sync torchrun ",
            f"--nproc_per_node={slurm_config['ntasks_per_node']} ",
            f"--nnodes={slurm_config['nodes']} ",
            "--rdzv_backend=c10d ",
            f"--rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}}",
            script_path,
            "-c",
            f"{job_conf_path}",
        ]
        if extra_args:
            command_parts.extend(extra_args)
        command = " ".join(command_parts)

        if "extra_mounts" not in slurm_config:
            slurm_config["extra_mounts"] = []
        if Path(repo_root).exists():
            slurm_config["extra_mounts"].append(VolumeMapping(Path(repo_root), Path(repo_root)))

        return submit_slurm_job(
            SlurmConfig(**slurm_config, command=command, chdir=repo_root),
            job_dir,
        )
