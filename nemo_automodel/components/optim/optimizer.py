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

"""Typed optimizer + LR scheduler configs (TorchTitan-style).

Each optimizer config is a plain dataclass exposing the full parameter surface
as named fields (no opaque ``**kwargs``) and a ``build(params)`` method that
constructs a single optimizer directly.  Reading the dataclass tells you exactly
what you can configure.

:func:`build_optimizer` is the single orchestration entry point.  It loops over
``model.parts`` and applies the per-part concerns (TP ``foreach``, Dion param
grouping, Megatron-FSDP sharding), dispatching on its second argument:

- a typed :class:`OptimizerConfig` instance — the Automodel-native path; per-part
  construction delegates to ``config.build(...)``.
- an optimizer *dotted path* or *class* plus arbitrary ``**optimizer_kwargs`` —
  the escape hatch for external integrations (e.g. veRL).  Adding a new typed
  config never requires the integration to change: it can keep passing a name
  and kwargs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.optim.dion import build_dion_optimizer, is_dion_optimizer
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.distributed.config import DistributedConfig
    from nemo_automodel.components.training.step_scheduler import StepScheduler

logger = logging.getLogger(__name__)

_DTYPE_FIELDS = ("master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")


# ---------------------------------------------------------------------------
# Typed optimizer configs
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Base optimizer config.  Subclasses expose their full field surface and
    implement :meth:`build`."""

    lr: float = 1e-4
    weight_decay: float = 0.01

    def build(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        """Construct the optimizer for ``params``."""
        raise NotImplementedError(f"{type(self).__name__} must implement build()")


@dataclass
class AdamConfig(OptimizerConfig):
    """``torch.optim.Adam``."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    def build(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            foreach=foreach,
        )


@dataclass
class AdamWConfig(OptimizerConfig):
    """``torch.optim.AdamW``."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    fused: bool = False

    def build(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            fused=self.fused,
            # foreach and fused are mutually exclusive; only pass foreach when not fused.
            **({} if self.fused else {"foreach": foreach}),
        )


@dataclass
class FusedAdamConfig(OptimizerConfig):
    """``transformer_engine.pytorch.optimizers.FusedAdam``."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    adam_w_mode: bool = True
    bias_correction: bool = True
    master_weights: bool = True
    master_weight_dtype: str | None = None

    def build(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        from transformer_engine.pytorch.optimizers import FusedAdam

        kwargs: dict[str, Any] = dict(
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            adam_w_mode=self.adam_w_mode,
            bias_correction=self.bias_correction,
            master_weights=self.master_weights,
        )
        if self.master_weight_dtype is not None:
            kwargs["master_weight_dtype"] = dtype_from_str(self.master_weight_dtype)
        return FusedAdam(params, **kwargs)


@dataclass
class FlashAdamWConfig(OptimizerConfig):
    """``flashoptim.FlashAdamW``."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    master_weight_bits: int = 24

    def build(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        from flashoptim import FlashAdamW

        return FlashAdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            master_weight_bits=self.master_weight_bits,
        )


@dataclass
class MuonConfig(OptimizerConfig):
    """``dion.Muon`` — matrix-aware update for 2D+ params, scalar fallback for 1D.

    Dion needs the model (for parameter grouping) and the device mesh, so it is
    built via :meth:`build_dion` rather than the plain ``build(params)`` path.
    """

    lr: float = 5e-4
    weight_decay: float = 0.0
    mu: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"
    scalar_opt: str = "adamw"
    scalar_betas: tuple[float, float] = (0.9, 0.999)
    scalar_eps: float = 1e-8

    def build_dion(self, model: torch.nn.Module, device_mesh: DeviceMesh | None) -> torch.optim.Optimizer:
        from dion import Muon

        return build_dion_optimizer(
            optimizer_factory=Muon,
            optimizer_kwargs=dict(
                lr=self.lr,
                weight_decay=self.weight_decay,
                mu=self.mu,
                betas=self.betas,
                epsilon=self.epsilon,
                adjust_lr=self.adjust_lr,
                scalar_opt=self.scalar_opt,
                scalar_betas=self.scalar_betas,
                scalar_eps=self.scalar_eps,
            ),
            model=model,
            distributed_mesh=device_mesh,
        )


# ---------------------------------------------------------------------------
# LR scheduler config
# ---------------------------------------------------------------------------


@dataclass
class LRSchedulerConfig:
    """LR scheduler configuration.  ``None`` fields are computed from the
    training schedule (total steps, optimizer base LR/WD) by
    :func:`build_lr_scheduler`."""

    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    lr_decay_style: str = "cosine"
    init_lr: float | None = None
    max_lr: float | None = None
    min_lr: float | None = None
    start_wd: float | None = None
    end_wd: float | None = None
    wd_incr_steps: int | None = None
    wd_incr_style: str = "constant"
    use_checkpoint_opt_param_scheduler: bool = True
    override_opt_param_scheduler: bool = False
    wsd_decay_steps: int | None = None
    lr_wsd_decay_style: str | None = None


# ---------------------------------------------------------------------------
# Shared per-part construction concerns
# ---------------------------------------------------------------------------


def _foreach_for_mesh(device_mesh: DeviceMesh | None) -> bool | None:
    """Return ``False`` when TP > 1 (foreach is unsupported), else ``None``."""
    if device_mesh is not None and "tp" in device_mesh.mesh_dim_names and device_mesh["tp"].size() > 1:
        return False
    return None


def _fully_shard_megatron_optimizer(model_part: torch.nn.Module, optimizer: torch.optim.Optimizer):
    from nemo_automodel.components.distributed import megatron_fsdp

    if not megatron_fsdp.HAS_MEGATRON_FSDP:
        return optimizer
    return megatron_fsdp.fully_shard_optimizer(model_part, optimizer)


def _maybe_shard_megatron(
    model_part: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    distributed_config: DistributedConfig | None,
    *,
    allow: bool = True,
) -> torch.optim.Optimizer:
    if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
        assert allow, "Dion optimizer does not support fully_shard_optimizer"
        return _fully_shard_megatron_optimizer(model_part, optimizer)
    return optimizer


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_optimizer(
    model: torch.nn.Module,
    optimizer: OptimizerConfig | Callable[..., torch.optim.Optimizer],
    *,
    distributed_config: DistributedConfig | None = None,
    device_mesh: DeviceMesh | None = None,
    **optimizer_kwargs: Any,
) -> list[torch.optim.Optimizer]:
    """Build one optimizer per ``model.parts`` (or ``[model]``).

    Single orchestration entry point.  It applies the per-part concerns
    (TP ``foreach``, Dion param grouping, Megatron-FSDP sharding) and dispatches
    on ``optimizer``:

    - **Typed config** (:class:`OptimizerConfig` instance) — the Automodel-native
      path.  Hyperparameters come from the config; ``**optimizer_kwargs`` must be
      empty.  Per-part construction delegates to ``config.build(...)``.
    - **Optimizer class / callable** (e.g. ``torch.optim.AdamW``) plus
      ``**optimizer_kwargs`` — the integration escape hatch (e.g. veRL).  The
      caller resolves any dotted path to a callable; the component never does
      string resolution.

    Args:
        model: Model (or model with ``.parts``) to optimize.
        optimizer: Typed :class:`OptimizerConfig` instance, or an optimizer
            class/callable to construct with ``**optimizer_kwargs``.
        distributed_config: Distributed strategy config; triggers Megatron-FSDP
            optimizer sharding when it is a :class:`MegatronFSDPConfig`.
        device_mesh: Device mesh used for tensor/data parallelism.
        **optimizer_kwargs: Constructor kwargs for the class/callable form
            (dtype strings such as ``"torch.bfloat16"`` are resolved).  Must be
            empty when ``optimizer`` is a typed config.

    Returns:
        One optimizer per model part.
    """
    foreach = _foreach_for_mesh(device_mesh)
    is_config = isinstance(optimizer, OptimizerConfig)

    if is_config:
        if optimizer_kwargs:
            raise ValueError(
                "Optimizer hyperparameters must be set on the config, not passed as keyword "
                f"arguments to build_optimizer (got {sorted(optimizer_kwargs)})."
            )
        is_dion = isinstance(optimizer, MuonConfig)
    else:
        if isinstance(optimizer, type) and issubclass(optimizer, OptimizerConfig):
            raise TypeError(
                f"Pass an OptimizerConfig instance, not the class {optimizer.__name__} "
                f"(e.g. {optimizer.__name__}(lr=1e-4))."
            )
        if not callable(optimizer):
            raise TypeError(
                "build_optimizer expects an OptimizerConfig or an optimizer class/callable, "
                f"got {type(optimizer).__name__}.  Resolve dotted paths in the caller."
            )
        factory = optimizer
        kwargs = dict(optimizer_kwargs)
        for attr in _DTYPE_FIELDS:
            val = kwargs.get(attr, None)
            if isinstance(val, str):
                kwargs[attr] = dtype_from_str(val)
        if foreach is not None:
            kwargs.setdefault("foreach", foreach)
        is_dion = is_dion_optimizer(factory)

    optimizers: list[torch.optim.Optimizer] = []
    for part in getattr(model, "parts", [model]):
        trainable_params = [p for p in part.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "trainable_params cannot be empty"
        if is_dion and is_config:
            opt = optimizer.build_dion(part, device_mesh)
        elif is_dion:
            opt = build_dion_optimizer(
                optimizer_factory=factory,
                optimizer_kwargs=kwargs,
                model=part,
                distributed_mesh=device_mesh,
            )
        elif is_config:
            opt = optimizer.build(trainable_params, foreach=foreach)
        else:
            opt = factory(params=trainable_params, **kwargs)
        opt = _maybe_shard_megatron(part, opt, distributed_config, allow=not is_dion)
        optimizers.append(opt)
    return optimizers


def build_lr_scheduler(
    config: LRSchedulerConfig | None,
    optimizer: list[torch.optim.Optimizer] | torch.optim.Optimizer,
    step_scheduler: StepScheduler,
) -> list[OptimizerParamScheduler] | None:
    """Build the learning rate scheduler(s).

    Args:
        config: LR scheduler configuration.  ``None`` disables scheduling.
        optimizer: The optimizer(s) to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        Configured optimizer parameter schedulers, or ``None`` if not configured.
    """
    if config is None:
        return None

    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

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


__all__ = [
    "AdamConfig",
    "AdamWConfig",
    "FlashAdamWConfig",
    "FusedAdamConfig",
    "LRSchedulerConfig",
    "MuonConfig",
    "OptimizerConfig",
    "build_lr_scheduler",
    "build_optimizer",
]
