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
do) and :class:`ModelCapabilitiesMixin` which attaches ``model.supports`` and
a ``validate_for_mesh`` helper to any ``nn.Module``.

Capabilities are derived from code introspection -- class attributes, mixin
inheritance, forward-signature inspection -- so they stay in sync as models
evolve without manual feature tables.
"""

from __future__ import annotations

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
    )

    return model_cls in PARALLELIZE_FUNCTIONS or model_cls.__name__ in {k.__name__ for k in PARALLELIZE_FUNCTIONS}


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


def _uses_te_attention(model: "nn.Module") -> bool:
    """True when the model was constructed with the TE attention backend."""
    backend = getattr(model, "backend", None)
    return getattr(backend, "attn", None) == "te"


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
        )
        flags = ", ".join("{}={}".format(name, getattr(self, "supports_" + name)) for name in names)
        return "ModelSupports({})".format(flags)

    # parallelism

    @property
    def supports_tp(self) -> bool:
        """Model has an optimized or HF-native tensor-parallel plan."""
        return _has_optimized_tp_plan(self._model_cls) or getattr(self._model, "_tp_plan", None) is not None

    @property
    def supports_pp(self) -> bool:
        """Model declares a ``_pp_plan`` for pipeline parallelism."""
        return getattr(self._model, "_pp_plan", None) is not None

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

        * Non-MoE models: requires ``_supports_sdpa``.
        * MoE models: requires TE attention backend (runtime check).
        """
        if self.supports_ep:
            return _uses_te_attention(self._model)
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
    def supports_gradient_checkpointing(self) -> bool:
        """Gradient checkpointing is supported."""
        if self.supports_ep:
            return False
        # Walk MRO directly to avoid triggering ModelCapabilitiesMixin.__getattr__,
        # which would recurse back here for models that lack the attribute.
        for cls in type(self._model).__mro__:
            if "supports_gradient_checkpointing" in cls.__dict__:
                return cls.__dict__["supports_gradient_checkpointing"] is True
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

    # @akoumparouli: quantisation support + peft
    # @property
    # def supports_fp8(self) -> bool:
    #     """FP8 training is available (torchao works with any model)."""
    #     return True

    # @property
    # def supports_nvfp4(self) -> bool:
    #     """NVFP4 quantisation path (not yet implemented)."""
    #     return False

    # @property
    # def supports_qlora(self) -> bool:
    #     """Quantised LoRA via BitsAndBytes."""
    #     return True

    # @property
    # def supports_peft(self) -> bool:
    #     """Parameter-efficient fine-tuning (LoRA) support."""
    #     return True


class ModelCapabilitiesMixin:
    """Mixin injected at model-creation time to expose capabilities.

    Provides:
    * ``model.supports`` -- a :class:`ModelSupports` descriptor.
    * ``model.validate_for_mesh(mesh)`` -- raises ``ValueError`` with
      actionable messages when the mesh configuration is incompatible.
    """

    @property
    def supports(self) -> ModelSupports:
        try:
            return self._supports
        except AttributeError:
            self._supports = ModelSupports(self, getattr(self, "_mesh", None))
            return self._supports

    def __getattr__(self, name: str):
        if name.startswith("supports_"):
            return getattr(self.supports, name)
        raise AttributeError(name)

    def validate_for_mesh(self) -> None:
        """Validate *mesh* parallelism sizes against this model's capabilities.

        Raises :class:`ValueError` with one bullet per violation.
        """
        mesh = getattr(self, "_mesh", None)
        tp_size = getattr(mesh, "tp_size", 1)
        pp_size = getattr(mesh, "pp_size", 1)
        ep_size = getattr(mesh, "ep_size", 1)
        cp_size = getattr(mesh, "cp_size", 1)

        arch = type(self).__name__
        sup = self.supports
        errors: list[str] = []

        if tp_size > 1 and not sup.supports_tp:
            errors.append(
                f"Tensor parallelism (tp_size={tp_size}) requested but {arch} "
                f"has no TP plan (not in PARALLELIZE_FUNCTIONS and no `_tp_plan` attribute).\n"
                f"Please re-run with --distributed.tp_size=1 or\n"
                f"modify distributed YAML config section:\n"
                f"distributed:\n"
                f"  tp_size: 1"
            )

        if pp_size > 1 and not sup.supports_pp:
            errors.append(
                f"Pipeline parallelism (pp_size={pp_size}) requires a _pp_plan "
                f"attribute on {arch}, but none was found.\n"
                f"Please re-run with --distributed.pp_size=1 or\n"
                f"modify distributed YAML config section:\n"
                f"distributed:\n"
                f"  pp_size: 1"
            )

        if cp_size > 1 and not sup.supports_cp:
            if _is_moe(type(self)):
                errors.append(
                    f"Context parallelism (cp_size={cp_size}) for {arch} requires "
                    f"the TE attention backend (backend.attn='te').\n"
                    f"Please re-run with --distributed.cp_size=1 or switch to TE attention."
                )
            else:
                errors.append(
                    f"Context parallelism (cp_size={cp_size}) not supported with {arch} model.\n"
                    f"Please re-run with --distributed.cp_size=1 or\n"
                    f"modify distributed YAML config section:\n"
                    f"distributed:\n"
                    f"  cp_size: 1"
                )
        if cp_size > 1 and not sup.supports_cp_with_sequence_packing:
            logger.warning(
                f"Context parallelism (cp_size={cp_size}) + sequence packing is not supported with {arch} model."
            )

        if ep_size > 1 and not sup.supports_ep:
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


def attach_capabilities_and_validate(model: "nn.Module", mesh: "MeshContext") -> "nn.Module":
    """Inject :class:`ModelCapabilitiesMixin` into *model* (in-place).

    After this call ``model.supports`` and ``model.validate_for_mesh`` are
    available.  Safe to call more than once -- subsequent calls are no-ops.
    """
    if isinstance(model, ModelCapabilitiesMixin):
        return model
    model.__class__ = type(
        model.__class__.__name__,
        (ModelCapabilitiesMixin, model.__class__),
        {},
    )
    model._mesh = mesh  # type: ignore[attr-defined]
    model.validate_for_mesh()
    return model
