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

"""Public config surface for the loggers component.

Look here for the typed parameters that drive WandB, MLflow, and Comet.
Look at ``api.py`` for the builder functions that consume these configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    """User-facing WandB configuration (maps to the YAML ``wandb:`` block).

    Fields are forwarded to ``wandb.init()``; any extra kwargs accepted by
    ``wandb.init`` can be added here as needed.

    Attributes:
        project: WandB project name.
        entity: WandB team / entity.  ``None`` uses the default from
            the wandb config or ``WANDB_ENTITY`` env var.
        name: Display name for the run.  When empty, the builder derives
            one from the model name.
        group: Group name for related runs.
        tags: List of string tags attached to the run.
        save_dir: Local directory for wandb files.
        notes: Free-text notes shown in the WandB UI.
    """

    project: str = "automodel"
    entity: str | None = None
    name: str = ""
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    save_dir: str | None = None
    notes: str | None = None


@dataclass
class MLflowConfig:
    """User-facing MLflow configuration (maps to the YAML ``mlflow:`` block).

    Attributes:
        experiment_name: MLflow experiment name.
        run_name: Display name for the run.
        tracking_uri: MLflow tracking server URI.  ``None`` uses the
            ``MLFLOW_TRACKING_URI`` env var or local ``./mlruns``.
        artifact_location: Root artifact store URI for the experiment.
        tags: Dictionary of string tags attached to the run.
        resume: When ``True`` (default), look for a ``mlflow_run_id``
            sidecar in the checkpoint dir and resume that run.
        description: Free-text description shown in the MLflow UI
            (sets the ``mlflow.note.content`` tag).
        flatten_depth: Nesting depth for ``mlflow.log_params``.
            ``1`` (default) splits one level; ``None`` is fully recursive.
    """

    experiment_name: str = "automodel-experiment"
    run_name: str = ""
    tracking_uri: str | None = None
    artifact_location: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    resume: bool = True
    description: str | None = None
    flatten_depth: int | None = 1


@dataclass
class CometConfig:
    """User-facing Comet ML configuration.

    Attributes:
        project_name: Comet project name.
        workspace: Comet workspace.  ``None`` uses the default.
        api_key: Comet API key.  ``None`` reads from ``COMET_API_KEY`` env var.
        experiment_name: Display name for this experiment run.
        tags: List of string tags attached to the experiment.
        auto_metric_logging: Enable Comet's automatic metric logging.
    """

    project_name: str = "automodel"
    workspace: str | None = None
    api_key: str | None = None
    experiment_name: str | None = None
    tags: list[str] = field(default_factory=list)
    auto_metric_logging: bool = False


__all__ = ["WandbConfig", "MLflowConfig", "CometConfig"]
