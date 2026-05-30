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

"""Typed configs for remote loggers (WandB, MLflow, Comet).

Each logger config is a plain dataclass exposing its YAML-configurable fields
plus a ``build(...)`` method that initialises and returns the logger / run
object.  Loggers are a closed, section-named set (no ``_target_`` dispatch), so
there is no free builder function — ``config.build(...)`` is the entry point.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WandbConfig:
    """User-facing WandB configuration (maps to the YAML ``wandb:`` block).

    Fields are forwarded to ``wandb.init()``; any extra kwargs accepted by
    ``wandb.init`` can be added here as needed.

    Attributes:
        project: WandB project name.
        entity: WandB team / entity.  ``None`` uses the default from
            the wandb config or ``WANDB_ENTITY`` env var.
        name: Display name for the run.  When empty, ``build`` derives
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

    def build(self, run_config: Mapping[str, Any] | None = None, model_name: str | None = None) -> Any:
        """Initialise WandB and return the run.

        Args:
            run_config: Full training config dict logged to the WandB run.
            model_name: Optional model name used to derive the run name
                when ``name`` is empty.

        Returns:
            Initialised ``wandb.Run``.
        """
        import wandb
        from wandb import Settings

        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        if kwargs.get("name", "") == "" and model_name:
            kwargs["name"] = "_".join(model_name.split("/")[-2:])
        return wandb.init(
            **kwargs,
            config=dict(run_config) if run_config is not None else None,
            settings=Settings(silent=True),
        )


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

    def build(self, checkpoint_dir: str | None = None, run_config: Mapping[str, Any] | None = None) -> Any:
        """Initialise MLflow on rank 0 and start (or resume) a run.

        Installs a ``sys.excepthook`` so crashed jobs report as FAILED rather
        than FINISHED.  On non-rank-0 processes returns ``None``.

        Args:
            checkpoint_dir: Checkpoint directory used to persist / read the
                ``mlflow_run_id`` sidecar for run resumption.
            run_config: Full training config dict logged as MLflow params
                and as a ``config.yaml`` artifact.

        Returns:
            Active ``mlflow.entities.Run`` on rank 0, or ``None``.
        """
        import logging as _logging
        import os
        from pathlib import Path

        import torch.distributed as dist

        _logger = _logging.getLogger(__name__)

        if not (dist.is_initialized() and dist.get_rank() == 0):
            return None

        try:
            import mlflow
        except ImportError as e:
            raise ImportError("MLflow is not installed. Please install it with: uv add mlflow") from e

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else mlflow.create_experiment(name=self.experiment_name, artifact_location=self.artifact_location)
            )
        except Exception as e:
            _logger.warning(f"Failed to create/get experiment: {e}")
            experiment_id = "0"

        tags = dict(self.tags)

        # Resume logic: env var always honoured; sidecar lookup gated by resume.
        sidecar = Path(checkpoint_dir) / "mlflow_run_id" if checkpoint_dir else None
        existing_run_id = os.environ.get("MLFLOW_RUN_ID") or (
            sidecar.read_text().strip() if self.resume and sidecar and sidecar.exists() else None
        )

        if self.description is not None:
            tags["mlflow.note.content"] = self.description

        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_id=existing_run_id,
            run_name=self.run_name,
            tags=tags,
        )

        # Persist run_id for future resume.
        if existing_run_id is None and sidecar is not None:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(run.info.run_id)

        # Install failure hook so crashed runs show as FAILED.
        from nemo_automodel.components.loggers.mlflow_utils import _install_mlflow_failure_hook

        _install_mlflow_failure_hook()

        # Log config as params + artifact.
        if run_config is not None:
            config_dict = dict(run_config)
            if existing_run_id is None:
                from nemo_automodel.components.loggers.mlflow_utils import flatten_params_for_mlflow

                mlflow.log_params(flatten_params_for_mlflow(config_dict, max_depth=self.flatten_depth))
                mlflow.log_dict(config_dict, "config.yaml")
            else:
                from datetime import datetime, timezone

                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                mlflow.log_dict(config_dict, f"config.resumed-{ts}.yaml")

        _logger.info(f"MLflow run started: {run.info.run_id}")
        _logger.info(f"View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")

        return run


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

    def build(self) -> Any:
        """Initialise Comet ML and return the logger (active on rank 0)."""
        from nemo_automodel.components.loggers.comet_utils import CometLogger

        return CometLogger(
            project_name=self.project_name,
            workspace=self.workspace,
            api_key=self.api_key,
            experiment_name=self.experiment_name,
            tags=self.tags,
            auto_metric_logging=self.auto_metric_logging,
        )


__all__ = ["CometConfig", "MLflowConfig", "WandbConfig"]
