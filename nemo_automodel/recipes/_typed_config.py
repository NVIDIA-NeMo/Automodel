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

"""Typed view over the recipe's YAML ``ConfigNode``.

The YAML→typed coercion happens **here**, at the recipe input boundary
(``recipe.__init__`` wraps the raw ``ConfigNode`` in ``RecipeConfig``), so the
recipe body only ever sees typed component configs and calls
``self.cfg.<section>.build(...)`` directly.

Known sections are exposed as cached, typed attributes that own a ``build()``:
``wandb``/``mlflow``/``step_scheduler``/``lr_scheduler`` map to component config
dataclasses, while the ``_target_``-based ``optimizer``/``loss_fn`` and the
``checkpoint`` block map to small recipe-layer "spec" wrappers (``OptimizerSpec``,
``LossSpec``, ``CheckpointSpec``) that resolve the YAML and delegate to the pure
component builders.  Sections with no typed view (e.g. ``model``, ``comet``) fall
through to the raw ``ConfigNode`` via ``__getattr__``.

This is the recipe layer, so it is allowed to know the YAML schema (which keys
are runtime args, ``_target_`` resolution, etc.); the components themselves stay
YAML-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from nemo_automodel.components.loggers.loggers import MLflowConfig, WandbConfig
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig

if TYPE_CHECKING:
    from nemo_automodel.components.config.loader import ConfigNode

# Keys present in the YAML ``step_scheduler:`` block that are runtime args passed
# to ``StepSchedulerConfig.build(...)`` separately (not config fields).
_STEP_SCHEDULER_RUNTIME_KEYS = ("local_batch_size", "dp_size", "dataloader")


def _section_kwargs(node: Any) -> dict[str, Any]:
    """``**kwargs`` for a typed config from a ConfigNode section (drops ``_target_``)."""
    d = node.to_dict() if hasattr(node, "to_dict") else dict(node)
    d.pop("_target_", None)
    return d


def _as_dict(cfg: Any | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Expected a mapping-like config, got {type(cfg).__name__}")


def _callable_and_kwargs(cfg: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve a ``_target_``-style section to its factory callable plus kwargs."""
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


@dataclass(frozen=True)
class OptimizerSpec:
    """Resolved ``optimizer:`` section: a factory callable plus its kwargs.

    ``build`` delegates to the pure component ``build_optimizer`` (which accepts a
    callable or an ``OptimizerConfig``); the YAML/``_target_`` knowledge stays here
    in the recipe layer.
    """

    factory: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, model: Any, *, distributed_config: Any = None, device_mesh: Any = None) -> Any:
        from nemo_automodel.components.optim import build_optimizer

        return build_optimizer(
            model,
            self.factory,
            distributed_config=distributed_config,
            device_mesh=device_mesh,
            **self.kwargs,
        )


@dataclass(frozen=True)
class LossSpec:
    """Resolved ``loss_fn:`` section: a factory callable plus its kwargs."""

    factory: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        from nemo_automodel.components.loss import build_loss_fn

        return build_loss_fn(self.factory, **self.kwargs)


@dataclass(frozen=True)
class CheckpointSpec:
    """The ``checkpoint:`` YAML block; ``build`` binds runtime args and delegates
    to the component ``build_checkpoint_config`` (which owns the merge/PEFT logic).
    """

    checkpoint_kwargs: dict[str, Any] | None

    def build(self, *, cache_dir: str | None, model_repo_id: str | None, is_peft: bool) -> Any:
        from nemo_automodel.components.checkpoint import build_checkpoint_config

        return build_checkpoint_config(
            checkpoint_kwargs=self.checkpoint_kwargs,
            cache_dir=cache_dir,
            model_repo_id=model_repo_id,
            is_peft=is_peft,
        )


class RecipeConfig:
    """Typed view over the YAML config consumed by recipes.

    ``wandb``, ``mlflow``, ``step_scheduler``, ``lr_scheduler``, ``optimizer``,
    ``loss_fn`` and ``checkpoint`` are exposed as typed objects that own a
    ``.build(...)``; all other attributes delegate to the underlying
    ``ConfigNode``.
    """

    def __init__(self, raw: ConfigNode):
        self._raw = raw

    @cached_property
    def wandb(self) -> WandbConfig | None:
        node = self._raw.get("wandb", None)
        return WandbConfig(**_section_kwargs(node)) if node is not None else None

    @cached_property
    def mlflow(self) -> MLflowConfig | None:
        node = self._raw.get("mlflow", None)
        return MLflowConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def step_scheduler(self) -> StepSchedulerConfig:
        node = self._raw.get("step_scheduler", None)
        if node is None:
            return StepSchedulerConfig()
        kwargs = {k: v for k, v in _section_kwargs(node).items() if k not in _STEP_SCHEDULER_RUNTIME_KEYS}
        return StepSchedulerConfig(**kwargs)

    @cached_property
    def lr_scheduler(self) -> LRSchedulerConfig | None:
        node = self._raw.get("lr_scheduler", None)
        return LRSchedulerConfig(**_section_kwargs(node)) if node is not None else None

    @cached_property
    def optimizer(self) -> OptimizerSpec | None:
        node = self._raw.get("optimizer", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return OptimizerSpec(factory, kwargs)

    @cached_property
    def loss_fn(self) -> LossSpec | None:
        node = self._raw.get("loss_fn", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return LossSpec(factory, kwargs)

    @cached_property
    def checkpoint(self) -> CheckpointSpec:
        node = self._raw.get("checkpoint", None)
        return CheckpointSpec(_as_dict(node) if node is not None else None)

    # --- everything else delegates to the raw ConfigNode -------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._raw, name)

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._raw.to_dict()

    def to_yaml_dict(self, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self._raw, "to_yaml_dict"):
            return self._raw.to_yaml_dict(**kwargs)
        return self.to_dict()
