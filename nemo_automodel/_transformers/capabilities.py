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

"""Model capabilities introspection and input validation.

Provides :class:`ModelSupports` (a read-only descriptor of what a model can
do) and :func:`attach_capabilities_and_validate` which attaches
``model.supports``, ``model.supports_*``, and ``model.validate_for_mesh``
to any ``nn.Module``.

Capabilities are derived from code introspection -- class attributes, mixin
inheritance, forward-signature inspection -- so they stay in sync as models
evolve without manual feature tables.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from nemo_automodel.components.distributed.mesh import MeshContext

logger = logging.getLogger(__name__)


def _has_optimized_tp_plan(model_cls: type) -> bool:
    """Check if *model_cls* has an entry in ``PARALLELIZE_FUNCTIONS``."""
    from nemo_automodel.components.distributed.optimized_tp_plans import (
        PARALLELIZE_FUNCTIONS,
        _get_class_qualname,
    )

    return _get_class_qualname(model_cls) in PARALLELIZE_FUNCTIONS


def _is_moe(model_cls: type) -> bool:
    from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin

    return issubclass(model_cls, MoEFSDPSyncMixin)


def _supports_seq_lens(model: "nn.Module") -> bool:
    """True when ``model.forward()`` accepts a ``seq_lens`` kwarg."""
    # @akoumparouli: this is a bit of a hack, but it's the best we can do for now
    # TODO: improve this
    fwd = getattr(model, "forward", None)
    if not callable(fwd):
        return False
    try:
        params = inspect.signature(fwd).parameters
        if "seq_lens" in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        return False


def _has_backend(model: "nn.Module") -> bool:
    """True for custom models that carry a ``BackendConfig``."""
    backend = getattr(model, "backend", None)
    return backend is not None and hasattr(backend, "attn")


def _uses_te_attention(model: "nn.Module") -> bool:
    """True when the model was constructed with the TE attention backend."""
    backend = getattr(model, "backend", None)
    return getattr(backend, "attn", None) == "te"


def _is_hybrid(model: "nn.Module") -> bool:
    """True when the model mixes attention with non-attention layers (e.g. Mamba/SSM).

    Detected via config attributes used by NemotronH (``layers_block_type``)
    and HF hybrid models (``hybrid_override_pattern``, ``is_hybrid_model``).
    """
    config = getattr(model, "config", None)
    if config is None:
        return False
    for attr in ("layers_block_type", "hybrid_override_pattern"):
        pattern = getattr(config, attr, None)
        if pattern and any(str(c).upper() == "M" for c in pattern):
            return True
    return getattr(config, "is_hybrid_model", False) is True


class ModelSupports:
    """Queryable feature-support descriptor attached to a model instance.

    Every property is derived from introspection of the live model so it
    reflects the actual class hierarchy and forward signature, not a
    hand-maintained table.

    Usage::

        model = NeMoAutoModelForCausalLM.from_pretrained(...)
        model.supports.tp   # True / False
        model.supports.pp   # ...
    """

    __slots__ = ("_model", "_model_cls", "_mesh")

    def __init__(self, model: "nn.Module", mesh: "MeshContext | None" = None) -> None:
        self._model = model
        self._model_cls = type(model)
        self._mesh = mesh

    def __repr__(self) -> str:
        names = (
            "tp",
            "pp",
            "cp",
            "ep",
            "sequence_packing",
            "gradient_checkpointing",
            "generate",
        )
        flags = ", ".join("{}={}".format(name, getattr(self, "supports_" + name)) for name in names)
        flags += ", is_custom_model={}".format(self.is_custom_model)
        return "ModelSupports({})".format(flags)

    # model kind

    @property
    def is_custom_model(self) -> bool:
        """True when the model class has a custom (non-HF) implementation in the registry."""
        from nemo_automodel._transformers.registry import ModelRegistry

        return ModelRegistry.has_custom_model(self._model_cls.__name__)

    # parallelism

    @property
    def supports_tp(self) -> bool:
        """Model has an optimized or HF-native tensor-parallel plan."""
        return _has_optimized_tp_plan(self._model_cls) or getattr(self._model, "_tp_plan", None) is not None

    @property
    def supports_pp(self) -> bool:
        """Model supports pipeline parallelism.

        True when the model either declares a ``_pp_plan`` or inherits from
        ``MoEFSDPSyncMixin`` (MoE models handle PP via
        ``patched_backward_maybe_with_nosync``).
        """
        return getattr(self._model, "_pp_plan", None) is not None or _is_moe(self._model_cls)

    # alias

    @property
    def supports_tp_plan(self) -> bool:
        return self.supports_tp

    @property
    def supports_pp_plan(self) -> bool:
        return self.supports_pp

    @property
    def supports_cp(self) -> bool:
        """Model supports context parallelism.

        +------------------+----------------+---------+
        | Model kind       | Attention      | CP?     |
        +------------------+----------------+---------+
        | Custom           | TE             | Yes     |
        | Custom           | FlexAttention  | No      |
        | HF (pure attn)   | SDPA           | Yes     |
        | HF (pure attn)   | no SDPA        | No      |
        | HF hybrid (Mamba)| any            | No      |
        +------------------+----------------+---------+
        """
        if _has_backend(self._model):
            return _uses_te_attention(self._model)
        if _is_hybrid(self._model):
            return False
        return getattr(self._model, "_supports_sdpa", False) is True

    @property
    def supports_ep(self) -> bool:
        """Model is a Mixture-of-Experts that supports expert parallelism."""
        return _is_moe(self._model_cls)

    # misc

    @property
    def supports_sequence_packing(self) -> bool:
        """``forward()`` accepts ``seq_lens`` for packed-sequence training."""
        sp_attn_backend = getattr(self._model, "_supports_sdpa", False) is True or _uses_te_attention(self._model)
        return _supports_seq_lens(self._model) and sp_attn_backend

    @property
    def supports_generate(self) -> bool:
        """Model has a ``generate()`` method for autoregressive inference."""
        return callable(getattr(self._model, "generate", None))

    @property
    def supports_gradient_checkpointing(self) -> bool:
        """Gradient checkpointing is supported."""
        if self.supports_ep:
            return False
        for cls in type(self._model).__mro__:
            if "supports_gradient_checkpointing" in cls.__dict__:
                val = cls.__dict__["supports_gradient_checkpointing"]
                if isinstance(val, (property, classmethod, staticmethod)):
                    continue
                return val is True
        return False

    # mesh-aware helpers

    @property
    def cp_size(self) -> int:
        return getattr(self._mesh, "cp_size", 1)

    @property
    def tp_size(self) -> int:
        return getattr(self._mesh, "tp_size", 1)

    @property
    def pp_size(self) -> int:
        return getattr(self._mesh, "pp_size", 1)

    @property
    def ep_size(self) -> int:
        return getattr(self._mesh, "ep_size", 1)

    @property
    def supports_cp_with_sequence_packing(self) -> bool:
        """CP + packed sequences requires TE attention backend."""
        if self.cp_size <= 1:
            return self.supports_sequence_packing
        return self.supports_sequence_packing and _uses_te_attention(self._model)


def validate_for_mesh(model: "nn.Module", mesh: "MeshContext") -> None:
    """Validate *mesh* parallelism sizes against this model's capabilities.

    Works both as a bound method (``model.validate_for_mesh()``) and as a
    standalone call (``validate_for_mesh(model)``).

    Raises :class:`ValueError` with one bullet per violation.
    """
    if mesh is None:
        return

    # If capabilities haven't been attached yet, use a temporary ModelSupports.
    if not hasattr(model, "supports"):
        supports = ModelSupports(model, mesh)
    else:
        supports = model.supports

    tp_size = getattr(mesh, "tp_size", 1)
    pp_size = getattr(mesh, "pp_size", 1)
    ep_size = getattr(mesh, "ep_size", 1)
    cp_size = getattr(mesh, "cp_size", 1)

    arch = type(model).__name__
    errors: list[str] = []

    if tp_size > 1 and not supports.supports_tp:
        errors.append(
            f"Tensor parallelism (tp_size={tp_size}) requested but {arch} "
            f"has no TP plan (not in PARALLELIZE_FUNCTIONS and no `_tp_plan` attribute).\n"
            f"Please re-run with --distributed.tp_size=1 or\n"
            f"modify distributed YAML config section:\n"
            f"distributed:\n"
            f"  tp_size: 1"
        )

    if pp_size > 1 and not supports.supports_pp:
        errors.append(
            f"Pipeline parallelism (pp_size={pp_size}) requires a _pp_plan "
            f"attribute on {arch}, but none was found.\n"
            f"Please re-run with --distributed.pp_size=1 or\n"
            f"modify distributed YAML config section:\n"
            f"distributed:\n"
            f"  pp_size: 1"
        )

    if cp_size > 1 and not supports.supports_cp:
        if _is_hybrid(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) is not supported for "
                f"hybrid model {arch} (contains Mamba/SSM layers).\n"
                f"Please re-run with --distributed.cp_size=1 or\n"
                f"modify distributed YAML config section:\n"
                f"distributed:\n"
                f"  cp_size: 1"
            )
        elif _has_backend(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) for {arch} requires "
                f"the TE attention backend (backend.attn='te').\n"
                f"Please re-run with --distributed.cp_size=1 or switch to TE attention:\n"
                f"model:\n"
                f"  backend:\n"
                f"    attn: te"
            )
        else:
            errors.append(
                f"Context parallelism (cp_size={cp_size}) not supported with {arch} "
                f"(model does not declare _supports_sdpa).\n"
                f"Please re-run with --distributed.cp_size=1 or\n"
                f"modify distributed YAML config section:\n"
                f"distributed:\n"
                f"  cp_size: 1"
            )

    if ep_size > 1 and not supports.supports_ep:
        errors.append(
            f"Expert parallelism (ep_size={ep_size}) requires a MoE model, "
            f"but {arch} does not inherit from MoEFSDPSyncMixin.\n"
            f"Please re-run with --distributed.ep_size=1 or\n"
            f"modify distributed YAML config section:\n"
            f"distributed:\n"
            f"  ep_size: 1"
        )

    if errors:
        raise ValueError(f"Unsupported configuration for {arch}:\n" + "\n".join(f"  - {e}" for e in errors))


def _supports_forwarding_property(name: str) -> property:
    """Property that forwards ``model.<name>`` to ``model.supports.<name>``."""

    def fget(self: "nn.Module") -> bool:
        return getattr(self.supports, name)

    fget.__name__ = name
    return property(fget)


def _lazy_supports_property(self: "nn.Module") -> ModelSupports:
    try:
        return self._supports  # type: ignore[attr-defined]
    except AttributeError:
        self._supports = ModelSupports(self, getattr(self, "_mesh", None))  # type: ignore[attr-defined]
        return self._supports  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=1)
def _build_class_dict() -> dict[str, property | type]:
    cls_dict: dict[str, property | type] = {
        "supports": property(_lazy_supports_property),
    }
    for attr in dir(ModelSupports):
        if attr.startswith("supports_") or attr == "is_custom_model":
            cls_dict[attr] = _supports_forwarding_property(attr)
    return cls_dict


def attach_capabilities_and_validate(model: "nn.Module", mesh: "MeshContext") -> "nn.Module":
    """Attach ``model.supports`` and ``model.supports_*`` and call validate_for_mesh.

    Injects a thin dynamic subclass so that property descriptors (supports_*)
        resolve via ``__getattribute__`` with no ``__getattr__`` overhead,
        which avoids triggering ModelCapabilitiesMixin.__getattr__ for models
        that lack the attribute.
    Safe to call more than once -- subsequent calls are no-ops.
    """
    if "supports" not in type(model).__dict__:
        orig_cls = model.__class__
        new_cls = type(
            orig_cls.__name__,
            (orig_cls,),
            _build_class_dict(),
        )
        new_cls.__module__ = orig_cls.__module__
        new_cls.__qualname__ = orig_cls.__qualname__
        model.__class__ = new_cls
    validate_for_mesh(model, mesh)
    return model
