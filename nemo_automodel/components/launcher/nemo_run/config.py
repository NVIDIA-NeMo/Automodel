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
from collections.abc import Mapping
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
    overrides: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> NemoRunConfig:
        """Build from the raw ``nemo_run:`` YAML section, splitting launcher keys from executor overrides.

        Args:
            d: The parsed ``nemo_run:`` section. Launcher keys are validated here;
                every other key is an executor attribute override whose value is
                forwarded unchanged.

        Returns:
            The launcher configuration. A launcher key that is null in YAML takes its
            default, as if it were absent, except ``detach``: null keeps its existing
            meaning of ``False``.

        Raises:
            TypeError: If a non-null launcher key has the wrong type, or ``overrides``
                is not a mapping with string keys.
        """
        # A key that is absent or null in YAML (``tail_logs:``) means "use the default", except ``detach``.
        default = cls()
        executor = d.get("executor")
        executor = default.executor if executor is None else executor
        if not isinstance(executor, str):
            raise TypeError(f"nemo_run.executor must be a string, got {type(executor).__name__}")
        executors_file = d.get("executors_file")
        executors_file = default.executors_file if executors_file is None else executors_file
        if not isinstance(executors_file, str):
            raise TypeError(f"nemo_run.executors_file must be a string, got {type(executors_file).__name__}")
        job_name = d.get("job_name")
        job_name = default.job_name if job_name is None else job_name
        if not isinstance(job_name, str):
            raise TypeError(f"nemo_run.job_name must be a string, got {type(job_name).__name__}")
        job_dir = d.get("job_dir")
        job_dir = default.job_dir if job_dir is None else job_dir
        if not isinstance(job_dir, str):
            raise TypeError(f"nemo_run.job_dir must be a string, got {type(job_dir).__name__}")
        detach = d.get("detach", default.detach)
        # Unlike the other keys, a null ``detach`` has always meant "do not detach": it reached
        # ``Experiment.run`` as a falsy value. Keep that instead of switching to the default (True).
        detach = False if detach is None else detach
        if not isinstance(detach, bool):
            raise TypeError(f"nemo_run.detach must be true or false, got {type(detach).__name__}")
        tail_logs = d.get("tail_logs")
        tail_logs = default.tail_logs if tail_logs is None else tail_logs
        if not isinstance(tail_logs, bool):
            raise TypeError(f"nemo_run.tail_logs must be true or false, got {type(tail_logs).__name__}")

        explicit_overrides = d.get("overrides")
        explicit_overrides = {} if explicit_overrides is None else explicit_overrides
        if not isinstance(explicit_overrides, Mapping):
            raise TypeError(f"nemo_run.overrides must be a mapping, got {type(explicit_overrides).__name__}")
        # Explicit ``overrides`` first, then unrecognised top-level keys, which win on conflict.
        # Both become executor attribute names, so their keys must be strings.
        overrides: dict[str, object] = {}
        for source in (explicit_overrides, {k: v for k, v in d.items() if k not in _LAUNCHER_KEYS}):
            for k, v in source.items():
                if not isinstance(k, str):
                    raise TypeError(f"nemo_run executor override keys must be strings, got {type(k).__name__}")
                overrides[k] = v

        return cls(
            executor=executor,
            job_name=job_name,
            detach=detach,
            tail_logs=tail_logs,
            executors_file=executors_file,
            job_dir=job_dir,
            overrides=overrides,
        )
