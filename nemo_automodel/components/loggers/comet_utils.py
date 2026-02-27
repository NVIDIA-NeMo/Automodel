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


class CometLogger:
    """
    Comet ML logger for experiment tracking.
    """

    def __init__(
        self,
        project_name: str,
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[list] = None,
        auto_metric_logging: bool = False,
        **kwargs,
    ):
        """
        Initialize Comet ML logger.

        Args:
            project_name: Name of the Comet project
            workspace: Comet workspace (optional, uses default from config/env)
            api_key: Comet API key (optional, uses COMET_API_KEY env var)
            experiment_name: Name for this experiment run (optional)
            tags: List of tags to add to the experiment
            auto_metric_logging: Whether to enable Comet's auto metric logging
            **kwargs: Additional arguments passed to comet_ml.Experiment()
        """
        try:
            import comet_ml
        except ImportError:
            raise ImportError("comet_ml is not installed. Please install it with: uv add comet_ml")

        self.comet_ml = comet_ml
        self.experiment = None

        if dist.is_initialized() and dist.get_rank() == 0:
            init_kwargs = {"project_name": project_name, "auto_metric_logging": auto_metric_logging, **kwargs}
            if api_key:
                init_kwargs["api_key"] = api_key
            if workspace:
                init_kwargs["workspace"] = workspace

            self.experiment = comet_ml.Experiment(**init_kwargs)

            if experiment_name:
                self.experiment.set_name(experiment_name)
            if tags:
                self.experiment.add_tags(tags)

            logger.info(f"Comet experiment: {self.experiment.url}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to Comet.

        Args:
            params: Dictionary of parameters to log
        """
        if not dist.get_rank() == 0 or self.experiment is None:
            return

        self.experiment.log_parameters(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to Comet.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics (optional)
        """
        if not dist.get_rank() == 0 or self.experiment is None:
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

            self.experiment.log_metrics(float_metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def end(self) -> None:
        """End the Comet experiment."""
        if self.experiment is not None:
            self.experiment.end()
            logger.info("Comet experiment ended successfully")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


def build_comet(cfg) -> CometLogger:
    """Build Comet logger from configuration.

    Args:
        cfg: Configuration object containing Comet settings

    Returns:
        CometLogger instance
    """
    comet_config = cfg.get("comet", {})
    if not comet_config:
        raise ValueError("Comet configuration not found in config")

    project_name = comet_config.get("project_name", "automodel-experiment")
    workspace = comet_config.get("workspace", None)
    api_key = comet_config.get("api_key", None)
    experiment_name = comet_config.get("experiment_name", "")
    tags = list(comet_config.get("tags", []))
    auto_metric_logging = comet_config.get("auto_metric_logging", False)

    if hasattr(cfg, "model") and hasattr(cfg.model, "pretrained_model_name_or_path"):
        tags.append(f"model:{cfg.model.pretrained_model_name_or_path}")

    if not experiment_name and hasattr(cfg, "model") and hasattr(cfg.model, "pretrained_model_name_or_path"):
        experiment_name = "_".join(cfg.model.pretrained_model_name_or_path.split("/")[-2:])

    return CometLogger(
        project_name=project_name,
        workspace=workspace,
        api_key=api_key,
        experiment_name=experiment_name,
        tags=tags,
        auto_metric_logging=auto_metric_logging,
    )
