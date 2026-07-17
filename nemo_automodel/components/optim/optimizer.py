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
as named fields (no opaque ``**kwargs``).  Reading the dataclass tells you
exactly what you can configure.

Every config owns its own construction via ``config.build(model, ...)``, which
loops over ``model.parts`` and applies the per-part concerns (TP ``foreach``,
Megatron-FSDP sharding).  Subclasses only implement the small
``_build_optimizer(params)`` hook; configs with bespoke construction needs
(e.g. :class:`MuonConfig`'s Dion parameter grouping) override ``build`` directly.

:func:`build_optimizer` is a thin dispatcher: it normalizes its
``optimizer_config`` argument to an :class:`OptimizerConfig` and returns
``config.build(model, ...)``.  The argument is either:

- a typed :class:`OptimizerConfig` instance — the Automodel-native path; or
- a ``(name_or_path, kwargs)`` tuple, where ``name_or_path`` is a short registry
  name (``"adam"``, ``"adamw"``, ``"muon"``, ...) or a dotted import path
  (``"torch.optim.AdamW"``).  It is resolved and constructed with ``kwargs``: a
  typed config from its fields, or — for any other callable — the escape hatch
  for external integrations (e.g. veRL) via :class:`OptimizerFromFactoryConfig`.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.optim.dion import build_dion_optimizer, is_dion_optimizer
from nemo_automodel.components.optim.precision_warnings import warn_if_torch_adam_with_bf16_params
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.training.step_scheduler import StepScheduler

logger = logging.getLogger(__name__)

_DTYPE_FIELDS = ("master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")


# ---------------------------------------------------------------------------
# Typed optimizer configs
# ---------------------------------------------------------------------------


# Config fields that are not optimizer-constructor kwargs and must be stripped
# from ``asdict(self)`` before splatting into the underlying optimizer.
_NON_CONSTRUCTOR_FIELDS = frozenset({"param_group_overrides"})


@dataclass
class ParamGroupOverride:
    """Per-parameter-group learning-rate / weight-decay override.

    Parameters whose (module-qualified) name matches :attr:`pattern` are placed
    in their own optimizer parameter group carrying :attr:`lr_mult` /
    :attr:`wd_mult`, which the LR scheduler multiplies into the group's learning
    rate and weight decay every step (see
    :meth:`OptimizerParamScheduler.step`). This mirrors Megatron-LM's per-group
    ``lr_mult`` scaling of ``max_lr`` / ``min_lr``.

    Attributes:
        pattern: Python regular expression matched against each parameter name
            with :func:`re.search` (so a plain substring like ``"router"`` also
            works).
        lr_mult: Multiplier applied to this group's learning rate.
        wd_mult: Multiplier applied to this group's weight decay.
    """

    pattern: str
    lr_mult: float = 1.0
    wd_mult: float = 1.0


def _coerce_param_group_overrides(overrides: list[Any]) -> list[ParamGroupOverride]:
    """Coerce a list of dicts (as delivered by YAML) or objects into ``ParamGroupOverride``."""
    coerced: list[ParamGroupOverride] = []
    for override in overrides:
        if isinstance(override, ParamGroupOverride):
            coerced.append(override)
        elif isinstance(override, dict):
            coerced.append(ParamGroupOverride(**override))
        else:
            raise TypeError(f"param_group_overrides entries must be dict or ParamGroupOverride, got {override!r}")
    return coerced


def _build_param_groups(
    named_params: list[tuple[str, torch.nn.Parameter]],
    overrides: list[ParamGroupOverride],
) -> list[dict[str, Any]]:
    """Partition ``named_params`` into optimizer groups by name-pattern ``overrides``.

    Each parameter joins the group of the first override whose ``pattern`` matches
    its name; unmatched parameters form the default group. Override groups carry
    only ``lr_mult`` / ``wd_mult`` (read by :meth:`OptimizerParamScheduler.step`),
    not a pre-scaled ``lr`` / ``weight_decay``: every group therefore inherits the
    optimizer's base ``lr`` / ``weight_decay`` at construction, and the scheduler
    (which calls ``step(0)`` on init) applies the multipliers. Keeping the stored
    ``lr`` unscaled is what lets ``LRSchedulerConfig.build`` read an accurate base
    LR from ``param_groups[0]`` even when the default group is empty. Empty groups
    (a pattern matching nothing) are dropped with a warning.

    Args:
        named_params: ``(name, parameter)`` pairs for the trainable params of one
            model part.
        overrides: The per-group overrides to apply, in priority order.

    Returns:
        A list of parameter-group dicts suitable for a torch optimizer, default
        (unmatched) group first.
    """
    compiled = [re.compile(override.pattern) for override in overrides]
    default_params: list[torch.nn.Parameter] = []
    matched_params: list[list[torch.nn.Parameter]] = [[] for _ in overrides]
    for name, param in named_params:
        for idx, regex in enumerate(compiled):
            if regex.search(name):
                matched_params[idx].append(param)
                break
        else:
            default_params.append(param)

    groups: list[dict[str, Any]] = []
    if default_params:
        groups.append({"params": default_params})
    for idx, override in enumerate(overrides):
        if not matched_params[idx]:
            logger.warning("param_group_overrides pattern %r matched no parameters; skipping", override.pattern)
            continue
        groups.append({"params": matched_params[idx], "lr_mult": override.lr_mult, "wd_mult": override.wd_mult})
    return groups


def _trainable_params_or_groups(part: torch.nn.Module, overrides: list[ParamGroupOverride]) -> list:
    """Return one model part's trainable params, grouped by ``overrides`` when set.

    Without overrides this is the flat trainable-parameter list (unchanged
    behavior); with overrides it is the list of parameter-group dicts from
    :func:`_build_param_groups`.
    """
    named_params = [(name, p) for name, p in part.named_parameters() if p.requires_grad]
    if not named_params:
        raise ValueError("optimizer received no trainable parameters")
    if overrides:
        return _build_param_groups(named_params, overrides)
    return [p for _, p in named_params]


@dataclass
class OptimizerConfig:
    """Base optimizer config.

    Subclasses expose their full field surface and implement
    :meth:`_build_optimizer`, the per-part hook that constructs a single
    optimizer from a list of parameters.  :meth:`build` owns the shared
    orchestration (per-part loop, TP ``foreach``, per-group LR overrides) and is
    rarely overridden — only by configs whose construction does not fit the
    ``parameters -> optimizer`` shape (e.g. :class:`MuonConfig`).  Megatron-FSDP
    optimizer sharding is no longer applied here; the recipe layer re-applies it
    via ``shard_optimizers_for_megatron_fsdp(...)``.
    """

    # Whether this optimizer can be sharded for Megatron-FSDP. The recipe layer
    # reads this to decide whether to shard the built optimizers.
    supports_megatron_fsdp_sharding: ClassVar[bool] = True

    # Per-group LR/WD overrides matched by parameter name. Empty = single group
    # (unchanged behavior). Honored by the standard torch optimizers (typed configs
    # and the factory path). Dion-family configs do their own grouping and warn if
    # this is set.
    param_group_overrides: list[ParamGroupOverride] = field(default_factory=list)

    def __post_init__(self) -> None:
        # YAML delivers overrides as plain dicts; coerce them to the typed form.
        self.param_group_overrides = _coerce_param_group_overrides(self.param_group_overrides)

    def build(
        self,
        model: torch.nn.Module,
        *,
        device_mesh: DeviceMesh | None = None,
        is_peft: bool = False,
    ) -> list[torch.optim.Optimizer]:
        """Build one optimizer per ``model.parts`` (or ``[model]``).

        Applies the shared per-part concerns (TP ``foreach`` disabling, per-group
        LR/WD overrides) and delegates the actual optimizer instantiation to
        :meth:`_build_optimizer`. Megatron-FSDP optimizer sharding is applied by
        the recipe layer, not here.

        Args:
            model: Model (or model with ``.parts``) to optimize.
            device_mesh: Device mesh used for tensor/data parallelism.
            is_peft: Whether the model is being trained with PEFT (suppresses the
                bf16 torch-Adam precision warning).

        Returns:
            One optimizer per model part.
        """
        foreach = _foreach_for_mesh(device_mesh)
        optimizers: list[torch.optim.Optimizer] = []
        for part in getattr(model, "parts", [model]):
            params = _trainable_params_or_groups(part, self.param_group_overrides)
            optimizers.append(self._build_optimizer(params, foreach=foreach))
        warn_if_torch_adam_with_bf16_params(optimizer=optimizers, is_peft=is_peft, context="optim", logger=logger)
        return optimizers

    def _build_optimizer(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        """Construct a single optimizer for ``params`` (one model part)."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build_optimizer()")

    def _constructor_kwargs(self) -> dict[str, Any]:
        """``asdict(self)`` with non-constructor (grouping) fields removed."""
        return {k: v for k, v in asdict(self).items() if k not in _NON_CONSTRUCTOR_FIELDS}

    def build_from_param_groups(
        self,
        param_groups: list[dict[str, Any]],
        *,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.optim.Optimizer:
        """Build one optimizer from caller-defined parameter groups."""
        foreach = _foreach_for_mesh(device_mesh)
        return self._build_optimizer(param_groups, foreach=foreach)


@dataclass
class AdamConfig(OptimizerConfig):
    """``torch.optim.Adam``."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    def _build_optimizer(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            **self._constructor_kwargs(),
            foreach=foreach,
        )


@dataclass
class AdamWConfig(OptimizerConfig):
    """``torch.optim.AdamW``."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    fused: bool = False

    def _build_optimizer(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            **self._constructor_kwargs(),
            # foreach and fused are mutually exclusive; only pass foreach when not fused.
            foreach=foreach and not self.fused,
        )


@dataclass
class FusedAdamConfig(OptimizerConfig):
    """``transformer_engine.pytorch.optimizers.FusedAdam``."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    adam_w_mode: bool = True
    bias_correction: bool = True
    master_weights: bool = True
    master_weight_dtype: str | None = None

    def _build_optimizer(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        from transformer_engine.pytorch.optimizers import FusedAdam

        kwargs = self._constructor_kwargs()
        master_weight_dtype = kwargs.pop("master_weight_dtype", None)
        if master_weight_dtype is not None:
            master_weight_dtype = dtype_from_str(master_weight_dtype)
        return FusedAdam(_drop_empty_local_shards(params), **kwargs, master_weight_dtype=master_weight_dtype)


@dataclass
class FlashAdamWConfig(OptimizerConfig):
    """``flashoptim.FlashAdamW``."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    master_weight_bits: int = 24

    def _build_optimizer(self, params, *, foreach: bool | None = None) -> torch.optim.Optimizer:
        from flashoptim import FlashAdamW

        return FlashAdamW(params, **self._constructor_kwargs())


# Fields consumed by build_dion_optimizer for parameter grouping; these are NOT dion
# constructor kwargs, so they are stripped from asdict(self) before instantiating the optimizer.
_DION_GROUPING_FIELDS = frozenset({"scalar_opt", "scalar_betas", "scalar_eps", "scalar_lr", "embed_lr", "lm_head_lr"})


@dataclass
class _DionConfigBase(OptimizerConfig):
    """Shared base for the dion-family typed configs (Muon / NorMuon / Dion2 / Dion).

    Dion optimizers need Dion's parameter grouping (built from the model) and the
    device mesh rather than a flat parameter list, so :meth:`build` runs grouping
    per model part.  The grouping-only fields below (``scalar_*`` / ``*_lr``) are
    consumed by :func:`build_dion_optimizer` and stripped from the constructor
    kwargs.  Dion is incompatible with Megatron-FSDP optimizer sharding; this is
    enforced at the recipe layer (``supports_megatron_fsdp_sharding = False``
    drives an ``allow=False`` sharding call that asserts rather than silently
    returning an unsharded optimizer).
    """

    # Dion optimizers are incompatible with Megatron-FSDP optimizer sharding.
    supports_megatron_fsdp_sharding: ClassVar[bool] = False

    lr: float = 5e-4
    weight_decay: float = 0.0
    scalar_opt: str = "adamw"
    scalar_betas: tuple[float, float] = (0.9, 0.999)
    scalar_eps: float = 1e-8
    scalar_lr: float | None = None
    embed_lr: float | None = None
    lm_head_lr: float | None = None

    # Name of the dion constructor argument that receives the resolved device mesh.
    _mesh_kwarg: ClassVar[str] = "distributed_mesh"

    def _make_optimizer(self, param_groups: Any, ctor_kwargs: dict[str, Any]) -> torch.optim.Optimizer:
        """Instantiate the concrete dion optimizer from grouped params + filtered kwargs."""
        raise NotImplementedError

    def build(
        self,
        model: torch.nn.Module,
        *,
        device_mesh: DeviceMesh | None = None,
        is_peft: bool = False,
    ) -> list[torch.optim.Optimizer]:
        if self.param_group_overrides:
            logger.warning("param_group_overrides is ignored by Dion-family optimizers, which do their own grouping")
        optimizers: list[torch.optim.Optimizer] = []
        for part in getattr(model, "parts", [model]):
            param_groups, mesh_kwargs = build_dion_optimizer(
                self, part, device_mesh=device_mesh, mesh_kwarg=self._mesh_kwarg
            )
            ctor_kwargs = {
                k: v
                for k, v in asdict(self).items()
                if k not in _DION_GROUPING_FIELDS and k not in _NON_CONSTRUCTOR_FIELDS
            }
            opt = self._make_optimizer(param_groups, {**ctor_kwargs, **mesh_kwargs})
            optimizers.append(opt)
        return optimizers


@dataclass
class MuonConfig(_DionConfigBase):
    """``dion.Muon`` — matrix-aware update for 2D+ params, scalar fallback for 1D."""

    mu: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"
    nesterov: bool = False
    flatten: bool = False
    use_triton: bool = False

    def _make_optimizer(self, param_groups, ctor_kwargs):
        from dion import Muon

        return Muon(param_groups, **ctor_kwargs)


@dataclass
class NorMuonConfig(_DionConfigBase):
    """``dion.NorMuon`` — Muon variant with neuron-wise normalization."""

    mu: float = 0.95
    muon_beta2: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"

    def _make_optimizer(self, param_groups, ctor_kwargs):
        from dion import NorMuon

        return NorMuon(param_groups, **ctor_kwargs)


@dataclass
class Dion2Config(_DionConfigBase):
    """``dion.Dion2`` — recommended successor to the legacy Dion optimizer."""

    fraction: float = 0.25
    ef_decay: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"

    def _make_optimizer(self, param_groups, ctor_kwargs):
        from dion import Dion2

        return Dion2(param_groups, **ctor_kwargs)


@dataclass
class DionConfig(_DionConfigBase):
    """``dion.Dion`` — legacy low-rank optimizer (prefer :class:`Dion2Config`).

    Legacy Dion takes separate replicate/outer/inner shard meshes; for FSDP2 the
    resolved 1-D shard submesh maps to ``outer_shard_mesh``.
    """

    mu: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    rank_fraction: float = 1.0
    rank_multiple_of: int = 1
    power_iters: int = 1
    qr_method: str = "rcqr"

    _mesh_kwarg: ClassVar[str] = "outer_shard_mesh"

    def _make_optimizer(self, param_groups, ctor_kwargs):
        from dion import Dion

        return Dion(param_groups, **ctor_kwargs)


# ---------------------------------------------------------------------------
# Escape hatch: build from an arbitrary factory callable
# ---------------------------------------------------------------------------


@dataclass
class OptimizerFromFactoryConfig(OptimizerConfig):
    """Build an optimizer from an arbitrary factory callable plus kwargs.

    The integration escape hatch (e.g. veRL): rather than exposing typed fields,
    it wraps an optimizer class/callable and the ``**kwargs`` to construct it.
    This keeps the factory path on the same ``config.build(model, ...)`` contract
    as the typed configs, so :func:`build_optimizer` never has to special-case it.

    Hyperparameters live in :attr:`kwargs`; the inherited ``lr``/``weight_decay``
    fields are unused.  The factory is called as ``factory(params=..., **kwargs)``;
    Dion-family optimizers (which need parameter grouping) should use the typed
    :class:`MuonConfig` instead.  A ``param_group_overrides`` entry in
    :attr:`kwargs` is consumed here (not forwarded to the factory) to drive
    per-group LR/WD, matching the typed-config behavior.
    """

    factory: Callable[..., torch.optim.Optimizer] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(
        self,
        model: torch.nn.Module,
        *,
        device_mesh: DeviceMesh | None = None,
        is_peft: bool = False,
    ) -> list[torch.optim.Optimizer]:
        assert callable(self.factory), "OptimizerFromFactoryConfig.factory must be a callable"
        foreach = _foreach_for_mesh(device_mesh)

        kwargs = dict(self.kwargs)
        # For the factory path, per-group overrides normally arrive inside ``kwargs``
        # (like every other hyperparameter, since the typed ``lr``/``weight_decay``
        # fields are unused here); pop them so they drive grouping rather than being
        # forwarded to the optimizer constructor. Fall back to the inherited field
        # so an override set either way is honored.
        kwargs_overrides = kwargs.pop("param_group_overrides", [])
        overrides = self.param_group_overrides or _coerce_param_group_overrides(kwargs_overrides)
        for attr in _DTYPE_FIELDS:
            val = kwargs.get(attr, None)
            if isinstance(val, str):
                kwargs[attr] = dtype_from_str(val)
        # Only inject ``foreach`` for factories that actually accept it. The TP>1 path sets
        # ``foreach=False`` via ``_foreach_for_mesh``; passing it to a factory that does not take
        # ``foreach`` (e.g. TE ``FusedAdam``) would raise.  Honour an explicit user-provided value.
        if foreach is not None and "foreach" not in kwargs and _factory_accepts_foreach(self.factory):
            kwargs["foreach"] = foreach

        optimizers: list[torch.optim.Optimizer] = []
        for part in getattr(model, "parts", [model]):
            # ``_trainable_params_or_groups`` already raises when a part has no
            # trainable parameters, and returns either a flat param list or the
            # per-group dicts.
            params = _trainable_params_or_groups(part, overrides)
            # TE FusedAdam's multi_tensor_apply faults on zero-numel local shards; see
            # _drop_empty_local_shards.  Same guard as FusedAdamConfig, for the YAML
            # ``_target_: transformer_engine...FusedAdam`` escape hatch.
            if _is_te_fused_adam(self.factory):
                params = _drop_empty_local_shards(params)
            optimizers.append(self.factory(params=params, **kwargs))
        warn_if_torch_adam_with_bf16_params(optimizer=optimizers, is_peft=is_peft, context="optim", logger=logger)
        return optimizers

    def build_from_param_groups(
        self,
        param_groups: list[dict[str, Any]],
        *,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.optim.Optimizer:
        assert callable(self.factory), "OptimizerFromFactoryConfig.factory must be a callable"
        foreach = _foreach_for_mesh(device_mesh)

        kwargs = dict(self.kwargs)
        for attr in _DTYPE_FIELDS:
            val = kwargs.get(attr, None)
            if isinstance(val, str):
                kwargs[attr] = dtype_from_str(val)
        if foreach is not None and "foreach" not in kwargs and _factory_accepts_foreach(self.factory):
            kwargs["foreach"] = foreach

        if _is_te_fused_adam(self.factory):
            param_groups = _drop_empty_local_shards(param_groups)
        return self.factory(params=param_groups, **kwargs)


# ---------------------------------------------------------------------------
# LR scheduler config
# ---------------------------------------------------------------------------


@dataclass
class LRSchedulerConfig:
    """LR scheduler configuration.  ``None`` fields are computed by
    :meth:`build` from the training schedule (total steps, optimizer base LR/WD)."""

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

    def build(
        self,
        optimizer: list[torch.optim.Optimizer] | torch.optim.Optimizer,
        step_scheduler: StepScheduler,
    ) -> list[OptimizerParamScheduler]:
        """Build one LR scheduler per optimizer.

        ``None`` fields are filled from the training schedule and each
        optimizer's base LR/WD.

        Args:
            optimizer: The optimizer(s) to schedule.
            step_scheduler: The step scheduler, used to derive total steps.

        Returns:
            One :class:`OptimizerParamScheduler` per optimizer.
        """
        # ``epoch_len`` is already expressed in optimizer steps (StepScheduler computes it as
        # ``ceil(len(dataloader) / grad_acc_steps)``) and is ``None`` for iterable/streaming
        # dataloaders, where ``len()`` is undefined.  Never call ``len(dataloader)`` here.
        if step_scheduler.epoch_len is not None:
            total_steps = step_scheduler.num_epochs * step_scheduler.epoch_len
            if step_scheduler.max_steps is not None:
                total_steps = min(total_steps, step_scheduler.max_steps)
        elif step_scheduler.max_steps is not None:
            total_steps = step_scheduler.max_steps
        else:
            raise ValueError(
                "Cannot infer total steps for an iterable/streaming dataset; set step_scheduler.max_steps."
            )
        lr_decay_steps = self.lr_decay_steps if self.lr_decay_steps is not None else total_steps
        wd_incr_steps = self.wd_incr_steps if self.wd_incr_steps is not None else total_steps
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizers = [optimizer]
        else:
            optimizers = list(optimizer)
        schedulers = []
        for opt in optimizers:
            base_lr = opt.param_groups[0]["lr"]
            base_wd = opt.param_groups[0].get("weight_decay", 0.0)
            schedulers.append(
                OptimizerParamScheduler(
                    optimizer=opt,
                    init_lr=self.init_lr if self.init_lr is not None else base_lr * 0.1,
                    max_lr=self.max_lr if self.max_lr is not None else base_lr,
                    min_lr=self.min_lr if self.min_lr is not None else base_lr * 0.01,
                    lr_warmup_steps=self.lr_warmup_steps
                    if self.lr_warmup_steps is not None
                    else min(1000, total_steps // 10),
                    lr_decay_steps=lr_decay_steps,
                    lr_decay_style=self.lr_decay_style,
                    start_wd=self.start_wd if self.start_wd is not None else base_wd,
                    end_wd=self.end_wd if self.end_wd is not None else base_wd,
                    wd_incr_steps=wd_incr_steps,
                    wd_incr_style=self.wd_incr_style,
                    use_checkpoint_opt_param_scheduler=self.use_checkpoint_opt_param_scheduler,
                    override_opt_param_scheduler=self.override_opt_param_scheduler,
                    wsd_decay_steps=self.wsd_decay_steps,
                    lr_wsd_decay_style=self.lr_wsd_decay_style,
                )
            )

        logger.info(
            f"Building LR scheduler with total_steps={total_steps}, "
            f"warmup_steps={schedulers[0].lr_warmup_steps}, "
            f"decay_style={self.lr_decay_style}"
        )
        return schedulers


# ---------------------------------------------------------------------------
# Shared per-part construction concerns
# ---------------------------------------------------------------------------


def _foreach_for_mesh(device_mesh: DeviceMesh | None) -> bool | None:
    """Return ``False`` when TP > 1 (foreach is unsupported), else ``None``."""
    if (
        device_mesh is not None
        and device_mesh.mesh_dim_names is not None
        and "tp" in device_mesh.mesh_dim_names
        and device_mesh["tp"].size() > 1
    ):
        return False
    return None


def _local_numel(param: torch.Tensor) -> int:
    """Number of elements of ``param`` owned by this rank.

    Args:
        param: Parameter tensor of arbitrary shape. For a ``DTensor`` (e.g. an
            FSDP2 parameter with a dim-0 ``Shard`` placement), the global shape
            may be non-empty while this rank's local shard holds zero elements;
            the local (``to_local()``) element count is returned. For a plain
            tensor, local and global element counts coincide (``numel()``).

    Returns:
        Element count of the rank-local shard; 0 when this rank owns no slice.
    """
    if isinstance(param, DTensor):
        return param.to_local().numel()
    return param.numel()


def _drop_empty_local_shards(params: list[Any]) -> list[Any]:
    """Drop parameters whose rank-local shard holds zero elements.

    FSDP2 shards every parameter along dim-0 across the shard group, so any
    parameter with dim-0 smaller than the group — e.g. the biases, norm
    weights, or class/position embeddings of a small dense vision tower
    sharded over a wide mesh — leaves zero-numel local shards on the tail
    ranks.  TransformerEngine FusedAdam's ``multi_tensor_apply`` kernel has no
    empty-tensor guard and faults (CUDA misaligned address / illegal memory
    access) at the first optimizer step.  Dropping locally-empty shards is
    exact, not an approximation: every element of those parameters lives on
    other ranks, whose optimizers update them; this rank has nothing to do.

    Args:
        params: Flat list of parameters, or list of param-group dicts with a
            ``"params"`` list.  Parameters are plain tensors or ``DTensor``s;
            a ``DTensor`` parameter carries a non-empty global shape whose
            dim-0-sharded local shard may be empty on this rank.

    Returns:
        ``params`` with zero-numel local shards removed.  Param groups keep
        their other options unchanged.

    Raises:
        ValueError: If every parameter of the flat list — or of any single
            param group — is locally empty on this rank.  Neither outcome has
            a safe representation: dropping a whole group makes
            ``optimizer.param_groups`` rank-asymmetric (LR/WD schedulers
            address groups positionally, e.g. ``param_groups[0]``, so ranks
            would silently schedule different lr/wd for shards other ranks
            own), while keeping an empty group breaks torch DCP's flattened
            optimizer-state load, which indexes the first param of each group.
    """
    remedy = (
        "This happens when FSDP2 shards parameters whose dim-0 is smaller than the shard group "
        "(e.g. tiny biases/norm weights of a small module on a wide mesh). Merge such parameters "
        "into a group that keeps at least one non-empty local shard on every rank, or shrink the "
        "sharding mesh so every rank owns at least one element."
    )
    params = list(params)
    dropped = 0
    if params and isinstance(params[0], dict):
        filtered: list[Any] = []
        for idx, group in enumerate(params):
            kept = [p for p in group["params"] if _local_numel(p) > 0]
            dropped += len(group["params"]) - len(kept)
            if not kept:
                raise ValueError(
                    f"Every parameter in optimizer param group {idx} has a zero-numel local shard on "
                    f"this rank; TE FusedAdam cannot hold them (multi_tensor_apply faults on empty "
                    f"tensors) and the group can be neither dropped nor kept empty. {remedy}"
                )
            filtered.append({**group, "params": kept})
    else:
        filtered = [p for p in params if _local_numel(p) > 0]
        dropped = len(params) - len(filtered)
        if params and not filtered:
            raise ValueError(
                f"Every trainable parameter of this model part has a zero-numel local shard on this "
                f"rank, leaving TE FusedAdam with an empty parameter list. {remedy}"
            )
    if dropped:
        logger.warning(
            "Dropped %d parameter(s) with zero-numel local shards from TE FusedAdam on this rank: "
            "TransformerEngine's multi_tensor_apply faults on empty tensors, and every element of "
            "the dropped parameters is owned (and updated) by other ranks.",
            dropped,
        )
    return filtered


def _is_te_fused_adam(factory: Callable[..., Any]) -> bool:
    """Return ``True`` if ``factory`` is TransformerEngine's ``FusedAdam`` (or a subclass).

    ``functools.partial`` wrappers are unwrapped (iteratively, for nested
    partials) before the identity check, so ``partial(FusedAdam, ...)``
    factories are recognized.  Other wrapper callables (closures, custom
    factory functions) are opaque and are NOT recognized; such factories must
    guard against zero-numel local shards themselves.

    Identity-based and import-free: TE is an optional dependency, so this never
    imports it.  If TE has not been imported yet, ``factory`` cannot be TE's
    ``FusedAdam`` class and the check is trivially ``False``.
    """
    while isinstance(factory, functools.partial):
        factory = factory.func
    te_optimizers = sys.modules.get("transformer_engine.pytorch.optimizers")
    if te_optimizers is None:
        return False
    te_fused_adam = getattr(te_optimizers, "FusedAdam", None)
    return isinstance(te_fused_adam, type) and isinstance(factory, type) and issubclass(factory, te_fused_adam)


def _factory_accepts_foreach(factory: Callable[..., Any]) -> bool:
    """Return ``True`` if ``factory`` accepts a ``foreach`` kwarg.

    ``torch.optim`` optimizers take ``foreach``; external factories such as TE
    ``FusedAdam`` do not, so passing it would raise ``TypeError``.
    """
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "foreach" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

# Short names accepted by :func:`build_optimizer`, mapped to their typed configs.
OPTIMIZER_CONFIG_REGISTRY: dict[str, type[OptimizerConfig]] = {
    "adam": AdamConfig,
    "adamw": AdamWConfig,
    "fused_adam": FusedAdamConfig,
    "flash_adamw": FlashAdamWConfig,
    "muon": MuonConfig,
    "normuon": NorMuonConfig,
    "dion": DionConfig,
    "dion2": Dion2Config,
}

# Maps a resolved dion optimizer class name (e.g. from YAML ``_target_: dion.Muon``) to the
# typed config that performs Dion's parameter grouping.  Keyed by ``cls.__name__``.
_DION_CONFIG_FOR: dict[str, type[OptimizerConfig]] = {
    "Muon": MuonConfig,
    "NorMuon": NorMuonConfig,
    "Dion2": Dion2Config,
    "Dion": DionConfig,
}


def _import_from_path(path: str) -> Any:
    """Import an object from a dotted path, e.g. ``"torch.optim.AdamW"``."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a dotted import path like 'torch.optim.AdamW', got {path!r}.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module {module_name!r} from optimizer path {path!r}.") from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Module {module_name!r} has no attribute {attr!r} (from optimizer path {path!r}).") from e


def build_optimizer_config(
    target: OptimizerConfig | str | type[OptimizerConfig] | Callable[..., torch.optim.Optimizer],
    kwargs: dict[str, Any] | None = None,
) -> OptimizerConfig:
    """Normalize an optimizer ``target`` plus ``kwargs`` into an :class:`OptimizerConfig`.

    This is the single normalization entry point shared by the recipe layer
    (which resolves a YAML ``_target_`` to a Python object) and
    :func:`build_optimizer` (which accepts ``(name_or_path, kwargs)`` tuples).

    ``target`` is one of:

    - an :class:`OptimizerConfig` instance — returned as-is (``kwargs`` ignored,
      since the instance already carries its typed fields).
    - an :class:`OptimizerConfig` subclass — instantiated from its typed fields
      with ``**kwargs``.
    - a string — a registry short name (see :data:`OPTIMIZER_CONFIG_REGISTRY`,
      e.g. ``"adamw"``) or a dotted import path (e.g. ``"torch.optim.AdamW"``);
      it is resolved and then handled as a subclass or callable.
    - any other optimizer callable/class — wrapped in an
      :class:`OptimizerFromFactoryConfig` (the escape hatch for external
      integrations, e.g. veRL).

    Args:
        target: The optimizer config instance/subclass, registry name or import
            path, or optimizer callable to normalize.
        kwargs: Constructor arguments for the resolved config/callable.

    Returns:
        An :class:`OptimizerConfig` ready to ``build(...)``.
    """
    if isinstance(target, OptimizerConfig):
        return target

    if isinstance(target, str):
        resolved = OPTIMIZER_CONFIG_REGISTRY.get(target.lower())
        if resolved is None:
            resolved = _import_from_path(target)
        target = resolved

    kwargs = dict(kwargs or {})
    if isinstance(target, type) and issubclass(target, OptimizerConfig):
        return target(**kwargs)
    # Dion-family optimizers need parameter grouping; route a resolved dion class
    # (e.g. YAML ``_target_: dion.Muon``) to its typed config rather than the flat-params
    # factory escape hatch, which would lose grouping and leak grouping-only kwargs.
    if is_dion_optimizer(target):
        dion_config = _DION_CONFIG_FOR.get(getattr(target, "__name__", ""))
        if dion_config is None:
            raise TypeError(
                f"Unsupported dion optimizer {getattr(target, '__name__', target)!r}; "
                f"supported: {sorted(_DION_CONFIG_FOR)}."
            )
        return dion_config(**kwargs)
    if callable(target):
        return OptimizerFromFactoryConfig(factory=target, kwargs=kwargs)
    raise TypeError(
        f"Optimizer target resolved to {target!r}, which is not an OptimizerConfig instance/subclass or a callable."
    )


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig | tuple[str, dict[str, Any]],
    *,
    device_mesh: DeviceMesh | None = None,
) -> list[torch.optim.Optimizer]:
    """Build one optimizer per ``model.parts`` (or ``[model]``).

    Thin dispatcher: it normalizes ``config`` to an :class:`OptimizerConfig` and
    returns ``config.build(model, ...)``.  Per-part concerns (TP ``foreach``,
    Dion param grouping) live on the config.  Megatron-FSDP optimizer sharding is
    re-applied separately by the recipe layer.

    ``config`` is one of:

    - a typed :class:`OptimizerConfig` instance — the Automodel-native path.
    - a ``(name_or_path, kwargs)`` tuple, where ``name_or_path`` is a short
      registry name (see :data:`OPTIMIZER_CONFIG_REGISTRY`, e.g. ``"adamw"``) or a
      dotted import path (e.g. ``"torch.optim.AdamW"``), and ``kwargs`` are the
      constructor arguments.  A registry/import-path that resolves to an
      :class:`OptimizerConfig` subclass is built from its typed fields; any other
      callable is wrapped in an :class:`OptimizerFromFactoryConfig` (the escape
      hatch for external integrations, e.g. veRL).

    Args:
        model: Model (or model with ``.parts``) to optimize.
        config: An :class:`OptimizerConfig` instance or a ``(name_or_path,
            kwargs)`` tuple.
        device_mesh: Device mesh used for tensor/data parallelism.

    Returns:
        One optimizer per model part.
    """
    if isinstance(config, OptimizerConfig):
        optimizer_config = config
    elif isinstance(config, tuple):
        if len(config) != 2:
            raise TypeError(f"Expected a (name_or_path, kwargs) tuple of length 2, got length {len(config)}.")
        name, kwargs = config
        if not isinstance(name, str):
            raise TypeError(
                f"The first tuple element must be a registry name or import-path string, got {type(name).__name__}."
            )
        if kwargs is not None and not isinstance(kwargs, dict):
            raise TypeError(f"The second tuple element must be a dict of kwargs, got {type(kwargs).__name__}.")
        optimizer_config = build_optimizer_config(name, kwargs)
    else:
        raise TypeError(
            "build_optimizer expects an OptimizerConfig instance or a (name_or_path, kwargs) tuple, "
            f"got {type(config).__name__}."
        )
    return optimizer_config.build(model, device_mesh=device_mesh)


__all__ = [
    "OPTIMIZER_CONFIG_REGISTRY",
    "AdamConfig",
    "AdamWConfig",
    "Dion2Config",
    "DionConfig",
    "FlashAdamWConfig",
    "FusedAdamConfig",
    "LRSchedulerConfig",
    "MuonConfig",
    "NorMuonConfig",
    "OptimizerConfig",
    "OptimizerFromFactoryConfig",
    "build_optimizer",
    "build_optimizer_config",
]
