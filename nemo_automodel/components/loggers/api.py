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

"""Builder functions for remote loggers (WandB, MLflow, Comet).

Each builder accepts a typed config from ``config.py`` plus optional
runtime arguments, and returns the initialised logger / run object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from nemo_automodel.components.loggers.config import CometConfig, WandbConfig


def build_wandb(
    config: WandbConfig,
    run_config: Mapping[str, Any] | None = None,
    model_name: str | None = None,
) -> Any:
    """Initialise WandB and return the run.

    Args:
        config: WandB configuration.
        run_config: Full training config dict logged to the WandB run.
        model_name: Optional model name used to derive the run name
            when ``config.name`` is empty.

    Returns:
        Initialised ``wandb.Run``.
    """
    import wandb
    from wandb import Settings

    kwargs = asdict(config)
    # Remove None values so wandb.init() uses its own defaults.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if kwargs.get("name", "") == "" and model_name:
        kwargs["name"] = "_".join(model_name.split("/")[-2:])
    return wandb.init(
        **kwargs,
        config=dict(run_config) if run_config is not None else None,
        settings=Settings(silent=True),
    )


def build_comet(config: CometConfig) -> Any:
    """Initialise Comet ML and return the logger.

    Args:
        config: Comet configuration.

    Returns:
        ``CometLogger`` instance (only active on rank 0).
    """
    from nemo_automodel.components.loggers.comet_utils import CometLogger

    return CometLogger(
        project_name=config.project_name,
        workspace=config.workspace,
        api_key=config.api_key,
        experiment_name=config.experiment_name,
        tags=config.tags,
        auto_metric_logging=config.auto_metric_logging,
    )


__all__ = ["build_wandb", "build_comet"]
