# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Callable, Mapping
from typing import Any

from nemo_automodel.components.checkpoint import build_checkpoint_config as _build_checkpoint_config
from nemo_automodel.components.loggers.mlflow_utils import build_mlflow as _build_mlflow
from nemo_automodel.components.loggers.wandb_utils import build_wandb as _build_wandb
from nemo_automodel.components.loss import build_loss_fn as _build_loss_fn
from nemo_automodel.components.optim import build_lr_scheduler as _build_lr_scheduler
from nemo_automodel.components.optim import build_optimizer as _build_optimizer
from nemo_automodel.components.training import build_step_scheduler as _build_step_scheduler


def _as_dict(cfg: Any | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Expected a mapping-like config, got {type(cfg).__name__}")


def _callable_and_kwargs(cfg: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    if hasattr(cfg, "to_dict") or isinstance(cfg, Mapping):
        cfg_dict = _as_dict(cfg)
        target = cfg_dict.pop("_target_", None)
        if target is not None:
            return target, cfg_dict
    target = getattr(cfg, "_target_", None)
    if target is not None:
        return target, {}
    if callable(cfg):
        return cfg, {}
    if hasattr(cfg, "instantiate"):
        return cfg.instantiate, {}
    raise AttributeError("Config must provide _target_, be callable, or provide instantiate()")


def build_checkpoint_config(
    cfg_ckpt: Any,
    cache_dir: str | None,
    model_repo_id: str | None,
    is_peft: bool,
):
    return _build_checkpoint_config(
        checkpoint_kwargs=_as_dict(cfg_ckpt) if cfg_ckpt is not None else None,
        cache_dir=cache_dir,
        model_repo_id=model_repo_id,
        is_peft=is_peft,
    )


def build_loss_fn(cfg_loss: Any) -> Any:
    loss_factory, loss_kwargs = _callable_and_kwargs(cfg_loss)
    return _build_loss_fn(loss_factory=loss_factory, loss_kwargs=loss_kwargs)


def build_optimizer(model: Any, cfg_opt: Any, distributed_config: Any, device_mesh: Any):
    optimizer_factory, optimizer_kwargs = _callable_and_kwargs(cfg_opt)
    return _build_optimizer(
        model=model,
        optimizer_factory=optimizer_factory,
        optimizer_kwargs=optimizer_kwargs,
        distributed_config=distributed_config,
        device_mesh=device_mesh,
    )


def build_lr_scheduler(cfg: Any, optimizer: Any, step_scheduler: Any):
    return _build_lr_scheduler(
        scheduler_kwargs=_as_dict(cfg) if cfg is not None else None,
        optimizer=optimizer,
        step_scheduler=step_scheduler,
    )


def build_step_scheduler(cfg: Any, dataloader: Any, dp_group_size: int, local_batch_size: int):
    scheduler_kwargs = _as_dict(cfg) if cfg is not None else None
    assert scheduler_kwargs is None or "_target_" not in scheduler_kwargs, "_target_ not permitted in step scheduler"
    return _build_step_scheduler(
        scheduler_kwargs=scheduler_kwargs,
        dataloader=dataloader,
        dp_group_size=dp_group_size,
        local_batch_size=local_batch_size,
    )


def _model_name_from_cfg(cfg_model: Any) -> str | None:
    pretrained = cfg_model.get("pretrained_model_name_or_path", None)
    if pretrained is not None:
        return pretrained
    model_config = cfg_model.get("config", None)
    if model_config is not None:
        if isinstance(model_config, str):
            return model_config
        return model_config.get("pretrained_model_name_or_path", None)
    return None


def build_wandb(cfg: Any):
    model_name = _model_name_from_cfg(cfg.model) if hasattr(cfg, "model") else None
    return _build_wandb(
        wandb_kwargs=_as_dict(cfg.wandb),
        run_config=_as_dict(cfg),
        model_name=model_name,
    )


def build_mlflow(cfg: Any):
    model_name = _model_name_from_cfg(cfg.model) if hasattr(cfg, "model") else None
    return _build_mlflow(
        mlflow_kwargs=_as_dict(cfg.mlflow),
        model_name=model_name,
        step_scheduler_kwargs=_as_dict(cfg.step_scheduler) if hasattr(cfg, "step_scheduler") else None,
    )
