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

# Keys that belong to NemoRunConfig itself (not executor overrides).
_LAUNCHER_KEYS = frozenset(
    {
        "executor",
        "job_name",
        "detach",
        "tail_logs",
        "executors_file",
        "job_dir",
        "overrides",
    }
)


@dataclass
class NemoRunConfig:
    """Configuration for the NeMo-Run launcher backend.

    The ``executor`` field selects a named executor from
    ``$NEMORUN_HOME/executors.py``, or ``"local"`` for local execution.

    Any key not recognised as a launcher setting is collected into
    ``overrides`` and applied directly to the executor via ``setattr``.
    This means any executor attribute (``nodes``, ``partition``,
    ``container_image``, ``time``, ``env_vars``, etc.) can be overridden
    from YAML without changes to this config class.
    """

    # Executor selection: name from EXECUTOR_MAP or "local"
    executor: str = "local"

    # Job metadata
    job_name: str = ""

    # Experiment behaviour
    detach: bool = True
    tail_logs: bool = False

    # Path to executor definitions file
    executors_file: str = field(default_factory=lambda: DEFAULT_EXECUTORS_FILE)

    # Local directory for job artifacts (config snapshot, logs)
    job_dir: str = ""

    # Arbitrary executor attribute overrides (e.g. nodes, partition,
    # container_image, time, env_vars).  Populated automatically from
    # unrecognised YAML keys by ``from_dict``.
    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "NemoRunConfig":
        """Build from a raw YAML dict, splitting launcher keys from executor overrides."""
        launcher_kwargs = {}
        overrides = {}
        for k, v in d.items():
            if k in _LAUNCHER_KEYS:
                launcher_kwargs[k] = v
            else:
                overrides[k] = v
        launcher_kwargs.setdefault("overrides", {}).update(overrides)
        return cls(**launcher_kwargs)
