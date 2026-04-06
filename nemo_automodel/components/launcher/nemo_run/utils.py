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

import importlib.util
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

def load_executor_from_file(name: str, executors_file: str) -> Any:
    """Load a named executor from a Python file containing an ``EXECUTOR_MAP``.

    The file (typically ``$NEMORUN_HOME/executors.py``) must define a module-level
    ``EXECUTOR_MAP`` dictionary whose keys are executor names and whose values
    are pre-built ``nemo_run`` executor instances (or zero-arg callables that
    return one).

    Args:
        name: Key to look up in ``EXECUTOR_MAP``.
        executors_file: Absolute path to the Python file.

    Returns:
        The executor object.

    Raises:
        FileNotFoundError: If *executors_file* does not exist.
        KeyError: If *name* is not found in the ``EXECUTOR_MAP``.
    """
    if not os.path.isfile(executors_file):
        raise FileNotFoundError(
            f"Executor definitions file not found: {executors_file}\n"
            f"Create it or set 'executors_file' in the nemo_run YAML section."
        )

    spec = importlib.util.spec_from_file_location("_nemo_run_executors", executors_file)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load executor definitions from {executors_file}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    executor_map = getattr(mod, "EXECUTOR_MAP", None)
    if executor_map is None:
        raise AttributeError(
            f"{executors_file} does not define an EXECUTOR_MAP dictionary."
        )

    if name not in executor_map:
        available = ", ".join(sorted(executor_map.keys()))
        raise KeyError(
            f"Executor '{name}' not found in EXECUTOR_MAP. Available: {available}"
        )

    executor = executor_map[name]
    # Support lazy callables: if the value is callable (but not an executor
    # instance), call it to get the executor.
    if callable(executor) and not hasattr(executor, "launch"):
        executor = executor()

    return executor


def apply_overrides(executor: Any, overrides: dict) -> None:
    """Apply arbitrary YAML overrides to an executor via ``setattr``.

    Dict and list values are *merged* with existing executor attributes
    (dicts are updated, lists are extended).  All other values are set
    directly.
    """
    for key, value in overrides.items():
        existing = getattr(executor, key, None)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged = dict(existing)
            merged.update(value)
            setattr(executor, key, merged)
        elif isinstance(value, list) and isinstance(existing, list):
            setattr(executor, key, list(existing) + list(value))
        else:
            setattr(executor, key, value)


def submit_nemo_run_job(script: Any, executor: Any, job_name: str,
                        detach: bool, tail_logs: bool) -> int:
    """Submit a job via NeMo-Run's Experiment API.

    Args:
        script: A ``nemo_run.Script`` object.
        executor: A NeMo-Run executor instance.
        job_name: Experiment and task name.
        detach: If True, return immediately after submission.
        tail_logs: If True, stream logs after submission.

    Returns:
        0 on successful submission.
    """
    try:
        import nemo_run as run
    except ImportError:
        raise ImportError(
            "nemo-run is not installed. Install with: pip install nemo-run"
        )

    exp_name = job_name or "automodel"
    task_name = job_name or "automodel"
    logger.info(
        "Submitting NeMo-Run experiment '%s' (executor=%s, detach=%s)",
        exp_name, type(executor).__name__, detach,
    )

    with run.Experiment(exp_name) as exp:
        exp.add(script, executor=executor, name=task_name)
        exp.run(detach=detach, tail_logs=tail_logs)

    return 0
