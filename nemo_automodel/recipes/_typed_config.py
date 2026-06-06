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
dataclasses; the ``optimizer`` and ``loss_fn`` blocks resolve to a component
:class:`~nemo_automodel.components.optim.optimizer.OptimizerConfig` /
:class:`~nemo_automodel.components.loss.loss.LossConfig` via
``build_optimizer_config`` / ``build_loss_config`` (which own a ``build()``),
while the ``checkpoint`` block is coerced directly into a component
:class:`~nemo_automodel.components.checkpoint.config.CheckpointingConfig` (the
model-derived ``model_repo_id`` / ``model_cache_dir`` / ``is_peft`` are filled
in here from the surrounding YAML).  Sections with no typed view (e.g.
``model``, ``comet``) fall through to the raw ``ConfigNode`` via
``__getattr__``.

This is the recipe layer, so it is allowed to know the YAML schema (which keys
are runtime args, ``_target_`` resolution, etc.); the components themselves stay
YAML-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any

from nemo_automodel.components.loggers.loggers import CometConfig, MLflowConfig, WandbConfig
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig

if TYPE_CHECKING:
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.datasets.loader import DataloaderConfig
    from nemo_automodel.components.loss.loss import LossConfig
    from nemo_automodel.components.loss.mtp import MTPLossConfig
    from nemo_automodel.components.optim.optimizer import OptimizerConfig

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


def _target_kwargs(node: Any) -> tuple[Any, dict[str, Any]]:
    """``(target, kwargs)`` for a loader ``make_*`` resolver from a config node."""
    if node is None:
        return None, {}
    if isinstance(node, str):
        return node, {}
    return _callable_and_kwargs(node)


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


class RecipeConfig:
    """Typed view over the YAML config consumed by recipes.

    ``wandb``, ``mlflow``, ``step_scheduler``, ``lr_scheduler``, ``optimizer``,
    ``loss_fn`` and ``checkpoint`` are exposed as typed objects that own a
    ``.build(...)`` (``optimizer`` is an
    :class:`~nemo_automodel.components.optim.optimizer.OptimizerConfig`,
    ``checkpoint`` a
    :class:`~nemo_automodel.components.checkpoint.config.CheckpointingConfig`);
    all other attributes delegate to the underlying ``ConfigNode``.
    """

    def __init__(self, raw: "ConfigNode"):
        self._raw = raw

    @cached_property
    def wandb(self) -> WandbConfig | None:
        node = self._raw.get("wandb", None)
        return WandbConfig.from_kwargs(**_section_kwargs(node)) if node else None

    @cached_property
    def mlflow(self) -> MLflowConfig | None:
        node = self._raw.get("mlflow", None)
        return MLflowConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def comet(self) -> CometConfig | None:
        node = self._raw.get("comet", None)
        return CometConfig(**_section_kwargs(node)) if node else None

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
        return LRSchedulerConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def optimizer(self) -> "OptimizerConfig" | None:
        from nemo_automodel.components.optim.optimizer import build_optimizer_config

        node = self._raw.get("optimizer", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return build_optimizer_config(factory, kwargs)

    @cached_property
    def dataloader(self) -> "DataloaderConfig" | None:
        from nemo_automodel.components.datasets.loader import (
            build_dataloader_config,
            make_collate_fn,
            make_dataset_config,
            make_packing_config,
        )

        node = self._raw.get("dataset", None)
        if node is None:
            return None
        target, ds_kwargs = _callable_and_kwargs(node)
        ds_kwargs.pop("tokenizer", None)
        dataset_config = make_dataset_config(target, ds_kwargs)
        dl = _as_dict(self._raw.get("dataloader", None))
        collate = make_collate_fn(*_target_kwargs(dl.pop("collate_fn", None)))
        ps = _as_dict(self._raw.get("packed_sequence", None))
        packing = None
        if ps.get("packed_sequence_size", 0) > 0:
            packing = make_packing_config(ps.pop("packing_strategy", "thd"), ps)
        return build_dataloader_config(
            dataset_config,
            dataloader=dl,
            packing=packing,
            collate_fn=collate,
            seed=self._raw.get("seed", 42),
            local_batch_size=self._raw.get("step_scheduler.local_batch_size", 1),
        )

    @cached_property
    def validation_dataloaders(self) -> dict[str, "DataloaderConfig"]:
        """One :class:`DataloaderConfig` per ``validation_dataset*`` block."""
        from nemo_automodel.components.datasets.loader import (
            build_dataloader_config,
            make_collate_fn,
            make_dataset_config,
            make_packing_config,
        )

        def _name(key: str) -> str:
            key = key.replace("validation_dataset", "")
            if len(key) > 1 and key[0] in ("_", "-", "."):
                key = key[1:]
            return key or "default"

        val_dl_node = self._raw.get("validation_dataloader", None)
        ps = _as_dict(self._raw.get("packed_sequence", None))
        packing = None
        if ps.get("packed_sequence_size", 0) > 0:
            packing = make_packing_config(ps.pop("packing_strategy", "thd"), ps)
        out: dict[str, DataloaderConfig] = {}
        for key in filter(lambda k: k.startswith("validation_dataset"), self._raw.to_dict().keys()):
            node = self._raw.get(key, None)
            if node is None:
                continue
            target, ds_kwargs = _callable_and_kwargs(node)
            ds_kwargs.pop("tokenizer", None)
            dataset_config = make_dataset_config(target, ds_kwargs)
            dl = _as_dict(val_dl_node)
            collate = make_collate_fn(*_target_kwargs(dl.pop("collate_fn", None)))
            out[_name(key)] = build_dataloader_config(
                dataset_config,
                dataloader=dl,
                packing=packing,
                collate_fn=collate,
                seed=self._raw.get("seed", 42),
                local_batch_size=self._raw.get("step_scheduler.local_batch_size", 1),
            )
        return out

    @cached_property
    def loss_fn(self) -> "LossConfig" | None:
        from nemo_automodel.components.loss import build_loss_config

        node = self._raw.get("loss_fn", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return build_loss_config(factory, **kwargs)

    @cached_property
    def mtp(self) -> "MTPLossConfig":
        # MTP loss params are model-driven (scaling_factor comes from the model
        # output / get_mtp_loss_scaling_factor; ignore_index is fixed) and are not
        # exposed via YAML.  This typed accessor just lets recipes build MTP through
        # the typed-config boundary like the other sections.
        from nemo_automodel.components.loss.mtp import MTPLossConfig

        return MTPLossConfig()

    @cached_property
    def checkpoint(self) -> "CheckpointingConfig":
        from nemo_automodel.components.checkpoint.config import CheckpointingConfig

        node = self._raw.get("checkpoint", None)
        kwargs = _as_dict(node) if node is not None else {}
        kwargs.pop("restore_from", None)  # consumed separately at load time, not a config field
        model = self._raw.get("model", None)
        # Model-derived values; YAML overrides win if explicitly set.
        kwargs |= {
            "model_repo_id": _model_name_from_cfg(model) if model is not None else None,
            "model_cache_dir": self._raw.get("model.cache_dir", None),
            "is_peft": bool(self._raw.get("peft", None)),
        }
        return CheckpointingConfig(**kwargs)

    # everything else delegates to the raw ConfigNode
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
