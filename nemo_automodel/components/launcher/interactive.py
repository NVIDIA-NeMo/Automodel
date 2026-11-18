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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from nemo_automodel.components.launcher.base import Launcher

logger = logging.getLogger(__name__)


class _Recipe(Protocol):
    """What the interactive launcher needs from a resolved recipe.

    ``resolve_recipe_cls`` can return anything a dotted path points at,
    so the launcher only relies on this slice of the surface.
    """

    def setup(self) -> None: ...

    def run_train_validation_loop(self) -> int | None: ...


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


def resolve_recipe_cls(target_str: str) -> Callable[..., _Recipe]:
    """Import and return the recipe factory from a dotted path.

    Calling the returned object with the parsed config must produce an
    object satisfying :class:`_Recipe`; that contract lives with the
    launcher, not with the config string.
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
    "For SLURM clusters, use sbatch with the reference slurm.sub script.\n"
    "For SkyPilot or NeMo-Run, add a skypilot: or nemo_run: section to your YAML.\n\n"
    "See: https://github.com/NVIDIA-NeMo/Automodel#readme"
)


class InteractiveLauncher(Launcher):
    """Launch a recipe locally on the current node using torchrun or in-process."""

    @staticmethod
    def _is_torchrun_worker() -> bool:
        """Return True when this process was already spawned by torchrun.

        torchrun (``torch.distributed.run``) sets both ``LOCAL_RANK`` and
        ``TORCHELASTIC_RUN_ID`` in the environment of every worker it spawns.
        We check for both to avoid false positives from environments (e.g.
        SLURM) that may set ``LOCAL_RANK`` without an active torchrun session.

        When the user launches the CLI via
        ``torchrun --nproc-per-node N -m nemo_automodel.cli.app config.yaml``,
        each worker must run the recipe in-process instead of re-launching torchrun.
        """
        return "LOCAL_RANK" in os.environ and "TORCHELASTIC_RUN_ID" in os.environ

    def _run_recipe_in_process(self, recipe_target: str, config: dict[str, Any]) -> int:
        """Instantiate and run a recipe in the current process."""
        recipe = resolve_recipe_cls(recipe_target)(config)
        recipe.setup()
        rc = recipe.run_train_validation_loop()
        # Recipes currently fall off the end of the loop, so None counts as success.
        return 0 if rc is None else rc

    def launch(
        self,
        config: dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> int:
        """Run the recipe on this node, via torchrun or in-process.

        Args:
            config: Parsed YAML config dict (without the launcher section).
            config_path: Resolved path to the original YAML file.
            recipe_target: Dotted import path of the recipe class.
            launcher_config: Number of local processes, or None to auto-detect.
            extra_args: Additional CLI overrides forwarded to the recipe.

        Returns:
            Process exit code (0 = success).
        """
        try:
            from torch.distributed.run import determine_local_world_size, get_args_parser
            from torch.distributed.run import run as thrun
        except ImportError:
            logger.error(_INSTALL_MSG)
            return 1

        # Already inside a torchrun worker (e.g. user ran
        # ``torchrun --nproc-per-node N -m nemo_automodel.cli.app config.yaml``).
        # Run the recipe directly; do NOT re-launch torchrun.
        if self._is_torchrun_worker():
            logger.info(
                "Detected existing torchrun environment (LOCAL_RANK=%s); running recipe in-process.",
                os.environ["LOCAL_RANK"],
            )
            return self._run_recipe_in_process(recipe_target, config)

        nproc_per_node = launcher_config
        repo_root = _get_repo_root()
        script_path = _recipe_module_path(recipe_target, repo_root)

        num_devices = determine_local_world_size(nproc_per_node="gpu")
        assert num_devices > 0, "Expected num-devices to be > 0"

        if nproc_per_node == 1 or num_devices == 1:
            logger.info("Launching job locally on a single device")
            return self._run_recipe_in_process(recipe_target, config)
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
