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

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.distributed.config import DistributedConfig
    from nemo_automodel.components.training.step_scheduler import StepScheduler

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.optim.config import LRSchedulerConfig, OptimizerConfig, _resolve_optimizer
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.optim.utils import build_dion_optimizer, is_dion_optimizer
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)


def _fully_shard_megatron_optimizer(model_part: torch.nn.Module, optimizer: torch.optim.Optimizer):
    from nemo_automodel.components.distributed import megatron_fsdp

    if not megatron_fsdp.HAS_MEGATRON_FSDP:
        return optimizer
    return megatron_fsdp.fully_shard_optimizer(model_part, optimizer)


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig | None = None,
    *,
    optimizer_factory: Callable[..., torch.optim.Optimizer] | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    distributed_config: DistributedConfig | None = None,
    device_mesh: DeviceMesh | None = None,
):
    """Build optimizers for a model or model parts.

    Accepts either an ``OptimizerConfig`` (preferred for external integrations
    like veRL) or an explicit ``(optimizer_factory, optimizer_kwargs)`` pair
    (used by ``_component_builders`` when resolving from YAML).

    Args:
        model: The model to build optimizers for.
        config: Typed optimizer config.  When provided, ``optimizer_factory``
            and ``optimizer_kwargs`` are derived from it.
        optimizer_factory: Callable or class that creates an optimizer.
            Ignored when ``config`` is provided.
        optimizer_kwargs: Keyword arguments forwarded to the optimizer factory.
            Ignored when ``config`` is provided.
        distributed_config: Distributed strategy configuration.
        device_mesh: Device mesh used for tensor/data parallelism.

    Returns:
        List of optimizers, one per model part.
    """
    if config is not None:
        optimizer_factory = _resolve_optimizer(config.name)
        optimizer_kwargs = {"lr": config.lr, "weight_decay": config.weight_decay, **config.extra_kwargs}
    elif optimizer_factory is None:
        raise ValueError("Either config or optimizer_factory must be provided")

    optimizer_kwargs = dict(optimizer_kwargs or {})

    # Resolve dtype strings (e.g. "torch.bfloat16") to torch.dtype objects for
    # optimizers like TE FusedAdam that accept dtype kwargs.
    for attr in ("master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        val = optimizer_kwargs.get(attr, None)
        if isinstance(val, str):
            optimizer_kwargs[attr] = dtype_from_str(val)

    if device_mesh is not None and "tp" in device_mesh.mesh_dim_names and device_mesh["tp"].size() > 1:
        # TP does not support foreach
        optimizer_kwargs["foreach"] = False

    optimizer = []
    has_dion_optimizer = is_dion_optimizer(optimizer_factory)
    for part in getattr(model, "parts", [model]):
        trainable_params = list(filter(lambda x: x.requires_grad, part.parameters()))
        assert len(trainable_params) > 0, "trainable_params cannot be empty"
        # TODO(@akoumparouli): no branching for building the optimizer, refactor.
        if has_dion_optimizer:
            tmp_optimizer = build_dion_optimizer(
                optimizer_factory=optimizer_factory,
                optimizer_kwargs=optimizer_kwargs,
                model=part,
                distributed_mesh=device_mesh,
            )
        else:
            tmp_optimizer = optimizer_factory(params=trainable_params, **optimizer_kwargs)
        if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
            assert not has_dion_optimizer, "Dion optimizer does not support fully_shard_optimizer"
            tmp_optimizer = _fully_shard_megatron_optimizer(part, tmp_optimizer)
        optimizer.append(tmp_optimizer)

    return optimizer


def build_lr_scheduler(
    config: LRSchedulerConfig | None,
    optimizer: list[torch.optim.Optimizer] | torch.optim.Optimizer,
    step_scheduler: StepScheduler,
) -> list[OptimizerParamScheduler] | None:
    """Build the learning rate scheduler.

    Args:
        config: LR scheduler configuration.  ``None`` disables scheduling.
        optimizer: The optimizer(s) to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        Configured optimizer parameter schedulers, or None if not configured.
    """
    if config is None:
        return None

    # Calculate total steps for the training run
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

    # Total optimizer steps (accounting for gradient accumulation)
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    optimizer_param_schedulers = []
    for opt in optimizer:
        base_lr = opt.param_groups[0]["lr"]
        base_wd = opt.param_groups[0].get("weight_decay", 0.0)

        scheduler = OptimizerParamScheduler(
            optimizer=opt,
            init_lr=config.init_lr if config.init_lr is not None else base_lr * 0.1,
            max_lr=config.max_lr if config.max_lr is not None else base_lr,
            min_lr=config.min_lr if config.min_lr is not None else base_lr * 0.01,
            lr_warmup_steps=config.lr_warmup_steps
            if config.lr_warmup_steps is not None
            else min(1000, total_steps // 10),
            lr_decay_steps=config.lr_decay_steps if config.lr_decay_steps is not None else total_steps,
            lr_decay_style=config.lr_decay_style,
            start_wd=config.start_wd if config.start_wd is not None else base_wd,
            end_wd=config.end_wd if config.end_wd is not None else base_wd,
            wd_incr_steps=config.wd_incr_steps if config.wd_incr_steps is not None else total_steps,
            wd_incr_style=config.wd_incr_style,
            use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=config.override_opt_param_scheduler,
            wsd_decay_steps=config.wsd_decay_steps,
            lr_wsd_decay_style=config.lr_wsd_decay_style,
        )
        optimizer_param_schedulers.append(scheduler)

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={optimizer_param_schedulers[0].lr_warmup_steps}, "
        f"decay_style={config.lr_decay_style}"
    )

    return optimizer_param_schedulers
