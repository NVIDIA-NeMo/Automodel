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

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo_automodel.components.launcher.base import Launcher

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    """Return the repository root.  If CWD looks like an editable checkout,
    prepend it to ``PYTHONPATH`` so the local source takes precedence."""
    cwd = Path.cwd()
    if (cwd / "nemo_automodel/components").exists() and (cwd / "examples/").exists():
        new_pp = str(cwd)
        if "PYTHONPATH" in os.environ:
            new_pp += ":" + os.environ["PYTHONPATH"]
        os.environ["PYTHONPATH"] = new_pp
        logger.info("Running job using source from: %s", cwd)
        return cwd
    return Path(__file__).parents[3]


def resolve_recipe_cls(target_str: str):
    """Import and return the recipe class from a dotted path.

    Example *target_str*:
    ``nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction``
    """
    module_path, cls_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _recipe_module_path(recipe_target: str, repo_root: Path) -> Path:
    """Convert a dotted recipe target into an absolute filesystem path."""
    module_path = recipe_target.rsplit(".", 1)[0]
    relative = module_path.replace(".", "/") + ".py"
    return repo_root / relative


_INSTALL_MSG = (
    "Local/interactive execution requires PyTorch and the full nemo_automodel package.\n"
    "It looks like you have the lightweight CLI-only install (automodel[cli]).\n\n"
    "To run jobs locally, install the full package:\n"
    "  pip install nemo_automodel          # CPU/basic\n"
    "  pip install nemo_automodel[all]     # with CUDA & all extras\n\n"
    "For remote job submission from a login node (SLURM, k8s, NeMo-Run),\n"
    "add a slurm:, k8s:, or nemo_run: section to your YAML config instead.\n\n"
    "See: https://github.com/NVIDIA/NeMo-Automodel#readme"
)


class InteractiveLauncher(Launcher):
    """Launch a recipe locally on the current node using torchrun or in-process."""

    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Any = None,
        extra_args: Optional[List[str]] = None,
    ) -> int:
        try:
            from torch.distributed.run import determine_local_world_size, get_args_parser
            from torch.distributed.run import run as thrun
        except ImportError:
            logger.error(_INSTALL_MSG)
            return 1

        nproc_per_node: Optional[int] = launcher_config
        repo_root = _get_repo_root()
        script_path = _recipe_module_path(recipe_target, repo_root)

        num_devices = determine_local_world_size(nproc_per_node="gpu")
        assert num_devices > 0, "Expected num-devices to be > 0"

        if nproc_per_node == 1 or num_devices == 1:
            logger.info("Launching job locally on a single device")
            recipe_cls = resolve_recipe_cls(recipe_target)
            recipe = recipe_cls(config)
            recipe.setup()
            return recipe.run_train_validation_loop()
        else:
            effective_nproc = nproc_per_node if nproc_per_node is not None else num_devices
            logger.info("Launching job locally on %d devices", effective_nproc)

            torchrun_parser = get_args_parser()
            torchrun_args, _ = torchrun_parser.parse_known_args()
            torchrun_args.training_script = str(script_path)
            torchrun_args.training_script_args = ["-c", str(config_path)]
            if extra_args:
                torchrun_args.training_script_args.extend(extra_args)
            torchrun_args.nproc_per_node = effective_nproc
            return thrun(torchrun_args)
