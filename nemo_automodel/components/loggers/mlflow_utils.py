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

import logging
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MLflowLogger:
    """
    MLflow logger for experiment tracking and model management.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_location: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of the current run (optional)
            tracking_uri: MLflow tracking server URI (optional)
            tags: Dictionary of tags to add to the run
            artifact_location: Location to store artifacts (optional)
            **kwargs: Additional arguments passed to mlflow.start_run()
        """
        try:
            import mlflow
            import mlflow.pytorch
        except ImportError:
            raise ImportError("MLflow is not installed. Please install it with: uv add mlflow")

        self.mlflow = mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None

        if dist.is_initialized() and dist.get_rank() == 0:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
                else:
                    experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"Failed to create/get experiment: {e}")
                # fallback
                experiment_id = "0"

            self.run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=self.tags, **kwargs)

            logger.info(f"MLflow run started: {self.run.info.run_id}")
            logger.info(
                f"View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{self.run.info.run_id}"
            )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        if not dist.get_rank() == 0 or self.run is None:
            return

        str_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                str_params[key] = str(value)
            elif isinstance(value, (list, tuple)):
                str_params[key] = str(value)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    str_params[f"{key}.{nested_key}"] = str(nested_value)
            else:
                str_params[key] = str(value)

        self.mlflow.log_params(str_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics (optional)
        """
        if not dist.get_rank() == 0 or self.run is None:
            return

        try:
            float_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    float_metrics[key] = value.item() if value.numel() == 1 else float(value.mean().item())
                elif isinstance(value, (int, float)):
                    float_metrics[key] = float(value)
                else:
                    logger.warning(f"Skipping metric {key} with unsupported type: {type(value)}")

            # TODO: add system metrics to mlflow

            if step is not None:
                self.mlflow.log_metrics(float_metrics, step=step)
            else:
                self.mlflow.log_metrics(float_metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts to MLflow.

        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path within the run's artifact directory (optional)
        """
        if not dist.get_rank() == 0 or self.run is None:
            return

        try:
            self.mlflow.log_artifacts(local_dir, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a single artifact to MLflow.

        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the run's artifact directory (optional)
        """
        if not dist.get_rank() == 0 or self.run is None:
            return

        try:
            self.mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: Name for model registry (optional)
            **kwargs: Additional arguments for mlflow.pytorch.log_model()
        """
        if not dist.get_rank() == 0 or self.run is None:
            return

        self.mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path=artifact_path, registered_model_name=registered_model_name, **kwargs
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mlflow.end_run()
        logger.info("MLflow run ended successfully")


def build_mlflow(cfg) -> MLflowLogger:
    """Build MLflow logger from configuration.

    Args:
        cfg: Configuration object containing MLflow settings

    Returns:
        MLflowLogger instance
    """
    mlflow_config = cfg.get("mlflow", {})
    if not mlflow_config:
        raise ValueError("MLflow configuration not found in config")

    # Extract configuration parameters
    experiment_name = mlflow_config.get("experiment_name", "automodel-experiment")
    run_name = mlflow_config.get("run_name", "")
    tracking_uri = mlflow_config.get("tracking_uri", None)
    tags = mlflow_config.get("tags", {}).to_dict()
    artifact_location = mlflow_config.get("artifact_location", None)

    if hasattr(cfg, "model") and hasattr(cfg.model, "pretrained_model_name_or_path"):
        tags["model"] = cfg.model.pretrained_model_name_or_path

    if hasattr(cfg, "step_scheduler"):
        tags["global_batch_size"] = str(cfg.step_scheduler.get("global_batch_size", "unknown"))
        tags["local_batch_size"] = str(cfg.step_scheduler.get("local_batch_size", "unknown"))

    return MLflowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags,
        artifact_location=artifact_location,
    )
