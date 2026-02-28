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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class Launcher(ABC):
    """Base class for all job launchers (interactive, SLURM, k8s, nemo-run)."""

    @abstractmethod
    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Any,
        extra_args: Optional[List[str]] = None,
    ) -> int:
        """Launch a recipe job.

        Args:
            config: Parsed YAML config dict (without the launcher section).
            config_path: Resolved path to the original YAML file.
            recipe_target: Dotted import path of the recipe class.
            launcher_config: Launcher-specific configuration (dict, int, or None).
            extra_args: Additional CLI overrides forwarded to the recipe.

        Returns:
            Process exit code (0 = success).
        """
        ...
