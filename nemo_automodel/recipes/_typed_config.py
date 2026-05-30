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

Known sections that map to a typed component config with a ``build()`` are
exposed as cached, typed attributes.  Sections that are ``_target_``-based
(``model``, ``optimizer``, ``loss``) or have no typed config fall through to the
raw ``ConfigNode`` via ``__getattr__`` — those are still built by the recipe via
their own builders.

This is the recipe layer, so it is allowed to know the YAML schema (which keys
are runtime args, etc.); the components themselves stay YAML-free.
"""

from __future__ import annotations

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


class RecipeConfig:
    """Typed view over the YAML config consumed by recipes.

    ``wandb``, ``mlflow``, ``step_scheduler`` and ``lr_scheduler`` are exposed as
    typed dataclass instances (with ``.build(...)``); all other attributes
    delegate to the underlying ``ConfigNode``.
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
