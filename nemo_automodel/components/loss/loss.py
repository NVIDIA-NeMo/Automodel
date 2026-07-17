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

"""Typed loss configs + builder (TorchTitan-style).

Each loss config is a plain dataclass exposing its full parameter surface as
named fields (no opaque ``**kwargs``) and a ``build()`` method that constructs
the loss module directly — lazy imports keep optional kernel deps out of module
load.  Reading the dataclass tells you exactly what you can configure.

:func:`build_loss_config` normalizes any of these into a :class:`LossConfig`,
and :func:`build_loss_module` is the single build entry point — a thin wrapper that
returns ``build_loss_config(...).build()``.  Both dispatch on the argument:

- a typed :class:`LossConfig` instance — the Automodel-native path; per-loss
  construction delegates to ``config.build()``.
- a registry name or a known loss *class* (see :data:`LOSS_CONFIG_REGISTRY`,
  e.g. ``MaskedCrossEntropy`` → :class:`MaskedCrossEntropyConfig`) — upgraded to
  its typed config so the YAML recipe path gets the same field validation.
- any other loss *class* / *callable* plus arbitrary ``**loss_kwargs`` — the
  escape hatch (wrapped in :class:`LossFromFactoryConfig`) for external
  integrations (e.g. veRL).  Adding a new typed config never forces the caller
  to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Typed loss configs
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Base loss config.  Subclasses expose their full field surface and
    implement :meth:`build`."""

    def build(self) -> nn.Module:
        """Construct the loss module."""
        raise NotImplementedError(f"{type(self).__name__} must implement build()")


@dataclass
class MaskedCrossEntropyConfig(LossConfig):
    """``MaskedCrossEntropy``.

    Attributes:
        fp32_upcast: Cast logits to float32 before computing CE.
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode — ``"sum"``, ``"mean"``, or ``"none"``.
    """

    fp32_upcast: bool = True
    ignore_index: int = -100
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        return MaskedCrossEntropy(
            fp32_upcast=self.fp32_upcast,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


@dataclass
class QuackCrossEntropyConfig(LossConfig):
    """``QuackCrossEntropy`` fused CUDA loss.

    Attributes:
        fp32_upcast: Cast logits to float32 before computing CE.
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode — ``"sum"``, ``"mean"``, or ``"none"``.
    """

    fp32_upcast: bool = False
    ignore_index: int = -100
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.quack_ce import QuackCrossEntropy

        return QuackCrossEntropy(
            fp32_upcast=self.fp32_upcast,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


@dataclass
class FusedLinearCEConfig(LossConfig):
    """``FusedLinearCrossEntropy``.

    Attributes:
        ignore_index: Label value marking padding tokens.
        logit_softcapping: Softcap logits before CE (0 = disabled).
        reduction: Reduction mode.
    """

    ignore_index: int = -100
    logit_softcapping: float = 0.0
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        return FusedLinearCrossEntropy(
            ignore_index=self.ignore_index,
            logit_softcapping=self.logit_softcapping,
            reduction=self.reduction,
        )


@dataclass
class TEParallelCEConfig(LossConfig):
    """``TEParallelCrossEntropy``.

    Attributes:
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode.
        tp_group: Tensor-parallel process group the loss reduces over (runtime
            arg; usually left as ``None`` in YAML and supplied programmatically).
    """

    ignore_index: int = -100
    reduction: str = "sum"
    tp_group: Any = None

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.te_parallel_ce import TEParallelCrossEntropy

        return TEParallelCrossEntropy(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            tp_group=self.tp_group,
        )


@dataclass
class KDLossConfig(LossConfig):
    """``KDLoss`` (knowledge distillation).

    Attributes:
        ignore_index: Label value marking padding tokens.
        temperature: Softmax temperature T.  Loss is scaled by T².
        fp32_upcast: Cast logits to float32 for numerical stability.
        tp_group: Tensor-parallel process group the loss reduces over (runtime
            arg; usually left as ``None`` in YAML and supplied programmatically).
        chunk_size: Vocab chunk size for the KD loss (0 = no chunking).
    """

    ignore_index: int = -100
    temperature: float = 1.0
    fp32_upcast: bool = True
    tp_group: Any = None
    chunk_size: int = 0

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.kd_loss import KDLoss

        return KDLoss(
            ignore_index=self.ignore_index,
            temperature=self.temperature,
            fp32_upcast=self.fp32_upcast,
            tp_group=self.tp_group,
            chunk_size=self.chunk_size,
        )


@dataclass
class LossFromFactoryConfig(LossConfig):
    """Escape hatch for external integrations (e.g. veRL) and the YAML recipe path.

    Rather than exposing typed fields, it wraps a loss class/callable and the
    ``**kwargs`` to construct it, keeping the factory path on the same
    ``config.build()`` contract as the typed configs so callers never have to
    special-case it.  The factory is called as ``factory(**kwargs)``.
    """

    factory: Callable[..., nn.Module] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> nn.Module:
        assert callable(self.factory), "LossFromFactoryConfig.factory must be a callable"
        return self.factory(**self.kwargs)


# ---------------------------------------------------------------------------
# Registry + builder
# ---------------------------------------------------------------------------

# Loss classes (by ``__name__``) that have a typed config.  Used by
# :func:`build_loss_config` to resolve a short string name and to upgrade a bare
# loss class (e.g. a YAML ``_target_`` resolved to ``MaskedCrossEntropy``) to its
# typed config, so the YAML path benefits from the same field validation as the
# Automodel-native path.
LOSS_CONFIG_REGISTRY: dict[str, type[LossConfig]] = {
    "MaskedCrossEntropy": MaskedCrossEntropyConfig,
    "QuackCrossEntropy": QuackCrossEntropyConfig,
    "FusedLinearCrossEntropy": FusedLinearCEConfig,
    "TEParallelCrossEntropy": TEParallelCEConfig,
    "KDLoss": KDLossConfig,
}


def build_loss_config(loss: LossConfig | str | Callable[..., nn.Module], **loss_kwargs: Any) -> LossConfig:
    """Normalize a loss ``loss`` target plus ``kwargs`` into a :class:`LossConfig`.

    The single normalization entry point shared by the recipe layer (which
    resolves a YAML ``_target_`` to a Python object) and :func:`build_loss_module`.
    It dispatches on ``loss``:

    - a typed :class:`LossConfig` instance — returned as-is; ``**loss_kwargs``
      must be empty (hyperparameters live on the config).
    - a :class:`LossConfig` subclass (the class) — rejected; pass an instance.
    - a registry name (see :data:`LOSS_CONFIG_REGISTRY`, e.g.
      ``"MaskedCrossEntropy"``) — built from the matching typed config.
    - a loss class / callable plus ``**loss_kwargs`` — if it is a registered loss
      class and the kwargs fit the config's fields, it is upgraded to that typed
      config; otherwise it is wrapped in a :class:`LossFromFactoryConfig`.  The
      caller resolves any dotted path to a callable; the component never does
      dotted-path resolution.

    Returns:
        A :class:`LossConfig` ready to ``build()``.
    """
    if isinstance(loss, LossConfig):
        if loss_kwargs:
            raise ValueError(
                "Loss hyperparameters must be set on the config, not passed as keyword "
                f"arguments (got {sorted(loss_kwargs)})."
            )
        return loss
    if isinstance(loss, type) and issubclass(loss, LossConfig):
        raise TypeError(f"Pass a LossConfig instance, not the class {loss.__name__} (e.g. {loss.__name__}()).")
    if isinstance(loss, str):
        config_cls = LOSS_CONFIG_REGISTRY.get(loss)
        if config_cls is None:
            raise TypeError(
                f"Unknown loss name {loss!r}: expected a registry name "
                f"({sorted(LOSS_CONFIG_REGISTRY)}), a LossConfig, or a loss class/callable.  "
                "Resolve dotted paths in the caller."
            )
        return config_cls(**loss_kwargs)
    if not callable(loss):
        raise TypeError(
            f"build_loss_config expects a LossConfig, a registry name, or a loss class/callable, "
            f"got {type(loss).__name__}.  Resolve dotted paths in the caller."
        )
    config_cls = LOSS_CONFIG_REGISTRY.get(getattr(loss, "__name__", ""))
    if config_cls is not None and set(loss_kwargs).issubset({f.name for f in fields(config_cls)}):
        return config_cls(**loss_kwargs)
    return LossFromFactoryConfig(factory=loss, kwargs=dict(loss_kwargs))


def build_loss_module(loss: LossConfig | Callable[..., nn.Module], **loss_kwargs: Any) -> nn.Module:
    """Build a loss function.

    Thin dispatcher: it normalizes ``loss`` to a :class:`LossConfig` via
    :func:`build_loss_config` and returns ``config.build()``.  Dispatches on
    ``loss``:

    - **Typed config** (:class:`LossConfig` instance) — the Automodel-native
      path.  Hyperparameters come from the config; ``**loss_kwargs`` must be
      empty.
    - **Loss class / callable** (e.g. ``MaskedCrossEntropy``) plus
      ``**loss_kwargs`` — the integration / YAML escape hatch.  The caller
      resolves any dotted path to a callable; the component never does string
      resolution.

    Args:
        loss: Typed :class:`LossConfig` instance, or a loss class/callable to
            construct with ``**loss_kwargs``.
        **loss_kwargs: Constructor kwargs for the class/callable form.  Must be
            empty when ``loss`` is a typed config.

    Returns:
        Instantiated loss function.
    """
    return build_loss_config(loss, **loss_kwargs).build()


__all__ = [
    "LOSS_CONFIG_REGISTRY",
    "FusedLinearCEConfig",
    "KDLossConfig",
    "LossConfig",
    "LossFromFactoryConfig",
    "MaskedCrossEntropyConfig",
    "QuackCrossEntropyConfig",
    "TEParallelCEConfig",
    "build_loss_config",
    "build_loss_module",
]
