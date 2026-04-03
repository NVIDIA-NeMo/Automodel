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

import os
from dataclasses import dataclass, field

# Default path to user-defined executor definitions.
# Respects the NEMORUN_HOME env var used by nemo-run itself (defaults to ~/.nemo_run).
_NEMORUN_HOME = os.environ.get("NEMORUN_HOME", os.path.join(os.path.expanduser("~"), ".nemo_run"))
DEFAULT_EXECUTORS_FILE = os.path.join(_NEMORUN_HOME, "executors.py")


@dataclass
class NemoRunConfig:
    """Configuration for the NeMo-Run launcher backend.

    The ``executor`` field is interpreted in two ways:

    1. **Named executor** -- if a name matches a key in the ``EXECUTOR_MAP``
       dictionary found in ``$NEMORUN_HOME/executors.py`` (or the file given by
       ``executors_file``), that pre-built executor object is loaded and any
       override fields (``nodes``, ``devices``, ``container_image``, ``time``,
       ``mounts``, ``env_vars``) are applied on top.

    2. **Inline executor type** -- the string ``"local"`` creates a
       ``run.LocalExecutor`` directly from the fields in this config.
    """

    # Executor selection: name from EXECUTOR_MAP or "local"
    executor: str = "local"

    # Compute resources (override values applied to named executors)
    nodes: int | None = None
    devices: int | None = None  # GPUs per node (maps to ntasks_per_node / gpus_per_node)

    # Container
    container_image: str | None = None

    # Time limit (e.g. Slurm wall time)
    time: str | None = None

    # Container mounts (appended to existing executor mounts)
    mounts: list[str] = field(default_factory=list)

    # Environment variables (merged into existing executor env_vars)
    env_vars: dict[str, str] = field(default_factory=dict)

    # Job metadata
    job_name: str = ""

    # Experiment behaviour
    detach: bool = True
    tail_logs: bool = False

    # Path to executor definitions file
    executors_file: str = field(default_factory=lambda: DEFAULT_EXECUTORS_FILE)

    # Local directory for job artifacts (config snapshot, logs)
    job_dir: str = ""

    def __post_init__(self) -> None:
        if self.nodes is not None and self.nodes < 1:
            raise ValueError(f"'nodes' must be >= 1, got: {self.nodes}")
        if self.devices is not None and self.devices < 1:
            raise ValueError(f"'devices' must be >= 1, got: {self.devices}")
