# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Precision of the Transformer Engine GEMMs of Nemotron 3 models under ``te_nvfp4``.

The map knows the Mamba, attention and MoE blocks of the hybrid pattern and the MTP head. A dense MLP block
(block type ``mlp`` of Nemotron-H) has no module class here: its projections are rejected like every other
GEMM module the map does not know. The output projection (``lm_head``) is an ``LMHead``, a ``torch.nn.Linear``
that Transformer Engine does not quantize: it runs in the model dtype under every map and has no class.
"""

from __future__ import annotations

import logging
import re
import types
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
from torch import nn

from nemo_automodel.components.models.common import utils as common_utils
from nemo_automodel.components.models.common.te_precision import (
    TE_PRECISIONS,
    TEModulePrecision,
    TEPrecision,
    TEPrecisionMap,
    TEPrecisionPlan,
)
from nemo_automodel.components.models.common.utils import _EXCLUDED_FORWARD_ATTR, _forward_outside_fp8

logger = logging.getLogger(__name__)

NemotronV3ModuleClass: TypeAlias = Literal[
    "mamba_in_proj",
    "mamba_out_proj",
    "attention",
    "routed_experts",
    "shared_experts",
    "latent_proj",
    "mtp",
]

# Activation checkpointing replaces a module by a wrapper that holds it under this attribute: the block
# (``activation_checkpointing: true``) or the MoE mixer of the block (``activation_checkpointing: moe``).
_WRAPPER = r"(?:_checkpoint_wrapped_module\.)?"
# "<prefix>model.layers.<index>": a decoder layer; "<prefix>mtp.layers.<index>": a block of the MTP head.
_DECODER_BLOCK = re.compile(rf"^(?P<block>(?:.+\.)?model\.layers\.(?P<layer>\d+)\.{_WRAPPER})(?P<module>.+)$")
_MTP_BLOCK = re.compile(rf"^(?P<block>(?:.+\.)?mtp\.layers\.\d+\.{_WRAPPER})(?P<module>.+)$")
# "mixer.<module>" or, for a mixer inside its own checkpoint wrapper, "mixer._checkpoint_wrapped_module.<module>".
_MIXER_MODULE = re.compile(rf"^(?P<mixer>mixer\.{_WRAPPER})(?P<module>.+)$")

_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_ROUTED_EXPERT_PROJECTIONS = ("experts.gate_up_linear", "experts.down_linear")
_SHARED_EXPERT_PROJECTIONS = ("shared_experts.up_proj", "shared_experts.gate_proj", "shared_experts.down_proj")
_LATENT_PROJECTIONS = ("fc1_latent_proj", "fc2_latent_proj")


def _classify_block_module(block: str, module: str) -> tuple[NemotronV3ModuleClass, str] | None:
    """Module class and scope of a module that runs GEMMs inside a decoder or MTP block.

    Args:
        block: Qualified name of the block with a trailing dot, ``"<prefix>model.layers.3."``, including the
            checkpoint wrapper of the block if it has one.
        module: Name of the module relative to the block, e.g. ``"mixer.experts.down_linear"`` or, with the
            mixer inside a checkpoint wrapper, ``"mixer._checkpoint_wrapped_module.experts.down_linear"``.

    Returns:
        The module class and the qualified name of the module whose forward selects the precision, or ``None``
        for a module without GEMMs of its own. The attention mixer is the scope of its projections, so the
        attention kernel between them follows them. The expert containers are the scope of their GEMMs: the
        routed experts read the active recipe to pad their token groups. A scope is always the module a
        checkpoint recompute calls again, never a checkpoint wrapper around it.
    """
    match = _MIXER_MODULE.match(module)
    if match is None:
        return None
    mixer = f"{block}{match['mixer']}".removesuffix(".")
    projection = match["module"]
    if projection == "in_proj":
        return "mamba_in_proj", f"{block}{module}"
    if projection == "out_proj":
        return "mamba_out_proj", f"{block}{module}"
    if projection in _ATTENTION_PROJECTIONS:
        return "attention", mixer
    if projection in _ROUTED_EXPERT_PROJECTIONS:
        return "routed_experts", f"{mixer}.experts"
    if projection in _SHARED_EXPERT_PROJECTIONS:
        return "shared_experts", f"{mixer}.shared_experts"
    if projection in _LATENT_PROJECTIONS:
        return "latent_proj", f"{block}{module}"
    return None


def _is_expert_container(module: str) -> bool:
    """Whether ``module``, a name relative to its block, is the routed-expert container of an MoE mixer."""
    match = _MIXER_MODULE.match(module)
    return match is not None and match["module"] == "experts"


_MXFP8_FORWARD_ATTR = "_te_mxfp8_forward"
"""Instance attribute holding the forward a module had before a precision plan routed it to MXFP8."""


def _forward_in_mxfp8(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Run ``module``'s forward under Transformer Engine's MXFP8 recipe where it would run a quantized recipe.

    Bound per instance by :func:`apply_precision_plan` for the scopes a plan gives MXFP8. The forward the
    module had before is looked up on the instance at call time, so the binding stays valid after
    ``copy.deepcopy``. MXFP8 replaces the recipe of an enclosing Transformer Engine autocast.
    Outside one (a forward that the caller runs in high precision, such as generation) and inside a disabled
    one the module runs in the model dtype like every other module.

    Args:
        module: The module whose forward runs. Any module can be routed, so the arguments and the result are
            those of its forward and carry no type of their own here.
        *args: Positional arguments of that forward, passed on unchanged.
        **kwargs: Keyword arguments of that forward, passed on unchanged.

    Returns:
        What that forward returns.
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.quantization import autocast as te_autocast

    installed = vars(module).get(_MXFP8_FORWARD_ATTR)
    if not FP8GlobalStateManager.is_fp8_enabled():
        if installed is not None:
            return installed(*args, **kwargs)
        return type(module).forward(module, *args, **kwargs)
    with te_autocast(enabled=True, recipe=MXFP8BlockScaling()):
        if installed is not None:
            return installed(*args, **kwargs)
        return type(module).forward(module, *args, **kwargs)


def _bind_forward(module: nn.Module, wrapper: types.FunctionType, saved_attr: str) -> None:
    """Make ``wrapper`` the instance forward of ``module`` and keep the instance forward it replaces.

    Args:
        module: Module whose calls go through ``wrapper`` afterwards.
        wrapper: Function called as ``wrapper(module, *args, **kwargs)``.
        saved_attr: Instance attribute that receives the instance forward ``module`` had, if it had one.
    """
    installed = vars(module).get("forward")
    if isinstance(installed, types.MethodType) and installed.__func__ is wrapper:
        return
    if installed is not None:
        setattr(module, saved_attr, installed)
    module.forward = types.MethodType(wrapper, module)


def te_gemm_modules(model: nn.Module) -> list[tuple[str, str]]:
    """List the Transformer Engine modules of ``model`` that run GEMMs and can therefore run quantized.

    Args:
        model: Model (or pipeline stage) to inspect.

    Returns:
        ``(qualified name, Transformer Engine class name)`` of every ``Linear``, ``GroupedLinear``,
        ``LayerNormLinear`` and ``LayerNormMLP``, in ``named_modules`` order; empty without Transformer Engine.
    """
    if not common_utils.HAVE_TE:
        return []
    from transformer_engine.pytorch import GroupedLinear, LayerNormLinear, LayerNormMLP, Linear

    # Transformer Engine has no registry of its GEMM modules; these are the classes that quantize operands.
    gemm_classes = (GroupedLinear, LayerNormLinear, LayerNormMLP, Linear)
    found: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        gemm_class = next((cls for cls in gemm_classes if isinstance(module, cls)), None)
        if gemm_class is not None:
            found.append((name, gemm_class.__name__))
    return found


def apply_precision_plan(model: nn.Module, plan: TEPrecisionPlan) -> None:
    """Give every scope of ``plan`` the forward of its precision and log the plan on rank 0; idempotent.

    A scope that does not run NVFP4 gets an instance ``forward``: outside the Transformer Engine autocast for
    high precision, under the MXFP8 recipe for MXFP8 (:func:`_forward_in_mxfp8`).

    Args:
        model: Model (or pipeline stage) that holds the modules of the plan.
        plan: Precision of its Transformer Engine GEMM modules.
    """
    modules = dict(model.named_modules())
    for scope, precision in plan.scopes().items():
        if precision == "high_precision":
            _bind_forward(modules[scope], _forward_outside_fp8, _EXCLUDED_FORWARD_ATTR)
        elif precision == "mxfp8":
            _bind_forward(modules[scope], _forward_in_mxfp8, _MXFP8_FORWARD_ATTR)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(
            "te_nvfp4: Transformer Engine GEMM modules by precision (high_precision = outside the "
            "quantized autocast, in the model dtype)\n%s",
            plan.summary(),
        )


@dataclass(frozen=True, kw_only=True)
class NemotronV3PrecisionMap(TEPrecisionMap):
    """Precision of every Transformer Engine GEMM of a Nemotron 3 language model, by module class.

    Set as ``te_nvfp4.precision_map``. It covers a language model built from Mamba, attention and MoE blocks
    (``NemotronHForCausalLM`` with that pattern) on its own and inside a vision-language model, where the
    names carry a ``language_model.`` prefix, with or without activation checkpointing of the blocks or of the
    MoE mixers. A model with dense MLP blocks is rejected. The router, the norms, the embedding, the Mamba
    convolution and scan and the output projection have no Transformer Engine GEMM and are not affected.

    Every decoder layer is one block of the hybrid pattern (Mamba, attention or MoE), so
    ``high_precision_tail_layers=14`` on an 88-layer model keeps layers 74-87 in high precision whatever
    their class says.
    """

    high_precision_tail_layers: int
    """Number of decoder layers at the end of the stack whose GEMMs all run in high precision."""
    mamba_in_proj: TEPrecision
    """Input projection of the Mamba mixers."""
    mamba_out_proj: TEPrecision
    """Output projection of the Mamba mixers."""
    attention: TEPrecision
    """Query, key, value and output projections of the attention mixers."""
    routed_experts: TEPrecision
    """Grouped up and down projections of the routed experts."""
    shared_experts: TEPrecision
    """Up and down projections of the shared expert."""
    latent_proj: TEPrecision
    """Projections into and out of the latent space of the routed experts."""
    mtp: TEPrecision
    """Every GEMM of the multi-token-prediction head, whatever its kind."""

    def __post_init__(self) -> None:
        """Reject a negative tail and precisions Transformer Engine does not have."""
        if isinstance(self.high_precision_tail_layers, bool) or not isinstance(self.high_precision_tail_layers, int):
            raise ValueError(f"high_precision_tail_layers must be an integer, got {self.high_precision_tail_layers!r}")
        if self.high_precision_tail_layers < 0:
            raise ValueError(f"high_precision_tail_layers must be at least 0, got {self.high_precision_tail_layers}")
        for module_class, precision in self.precision_by_class().items():
            if precision not in TE_PRECISIONS:
                raise ValueError(f"{module_class} must be one of {', '.join(TE_PRECISIONS)}, got {precision!r}")

    def precision_by_class(self) -> dict[NemotronV3ModuleClass, TEPrecision]:
        """The configured precision of every module class, before the high-precision tail is applied."""
        return {
            "mamba_in_proj": self.mamba_in_proj,
            "mamba_out_proj": self.mamba_out_proj,
            "attention": self.attention,
            "routed_experts": self.routed_experts,
            "shared_experts": self.shared_experts,
            "latent_proj": self.latent_proj,
            "mtp": self.mtp,
        }

    def resolve(self, module_names: Sequence[str], gemm_module_names: Collection[str]) -> TEPrecisionPlan:
        """Assign a precision to every Transformer Engine GEMM module of a Nemotron 3 model.

        Args:
            module_names: Qualified names of all modules of the model, in ``named_modules`` order.
            gemm_module_names: The names among them that are Transformer Engine GEMM modules.

        Returns:
            Plan with one entry per Transformer Engine GEMM module, in ``module_names`` order.

        Raises:
            ValueError: If the names hold no decoder layer; if a Transformer Engine GEMM module is not one the
                map has a class for (the projections of a dense MLP block among them); if a module that the
                map quantizes is not a Transformer Engine module (a
                Mamba projection built with ``backend.mamba_linear="torch"``, another projection built with
                ``backend.linear="torch"``, experts without Transformer Engine grouped GEMMs), which would
                silently leave it in high precision; if the decoder layers are not
                numbered 0..N-1 (a pipeline stage holds only some of them, so the tail cannot be located); or
                if the tail is longer than the stack.
        """
        precision_by_class = self.precision_by_class()
        gemm_modules = set(gemm_module_names)
        # (name, module class, decoder layer, scope) of every module that runs GEMMs in this model family.
        sites: list[tuple[str, NemotronV3ModuleClass, int | None, str]] = []
        layers: set[int] = set()
        for name in module_names:
            decoder = _DECODER_BLOCK.match(name)
            mtp = _MTP_BLOCK.match(name)
            if decoder is not None:
                layers.add(int(decoder["layer"]))
                kind = _classify_block_module(decoder["block"], decoder["module"])
                if kind is not None:
                    sites.append((name, kind[0], int(decoder["layer"]), kind[1]))
                elif _is_expert_container(decoder["module"]) and f"{name}.gate_up_linear" not in gemm_modules:
                    # Experts of another backend hold their weights themselves; they are one site.
                    sites.append((name, "routed_experts", int(decoder["layer"]), name))
            elif mtp is not None:
                # Inside the head only the scope depends on the kind of module.
                kind = _classify_block_module(mtp["block"], mtp["module"])
                if kind is not None:
                    sites.append((name, "mtp", None, kind[1]))
                elif mtp["module"] == "eh_proj":
                    sites.append((name, "mtp", None, name))
                elif _is_expert_container(mtp["module"]) and f"{name}.gate_up_linear" not in gemm_modules:
                    sites.append((name, "mtp", None, name))
        if not layers:
            raise ValueError(
                "NemotronV3PrecisionMap found no module named 'model.layers.<index>.<module>': the model is not "
                "a Nemotron 3 model, so nothing would be quantized as configured."
            )
        unknown = sorted(gemm_modules - {name for name, _, _, _ in sites})
        if unknown:
            raise ValueError(
                f"NemotronV3PrecisionMap does not know {len(unknown)} Transformer Engine GEMM module(s) and "
                f"will not guess their precision: {', '.join(unknown)}. It has classes for the projections of "
                "Mamba, attention and MoE blocks and for the MTP head; a dense MLP block (block type 'mlp') "
                "and modules outside the decoder layers and the MTP head have none."
            )
        num_layers = len(layers)
        if sorted(layers) != list(range(num_layers)):
            raise ValueError(
                f"NemotronV3PrecisionMap needs the whole decoder stack to locate its last layers, but the model "
                f"holds layers {sorted(layers)}; pipeline stages are not supported."
            )
        if self.high_precision_tail_layers > num_layers:
            raise ValueError(
                f"high_precision_tail_layers={self.high_precision_tail_layers} exceeds the {num_layers} decoder layers"
            )
        first_tail_layer = num_layers - self.high_precision_tail_layers
        modules: list[TEModulePrecision] = []
        not_quantizable: list[str] = []
        for name, module_class, layer, scope in sites:
            in_tail = layer is not None and layer >= first_tail_layer
            precision: TEPrecision = "high_precision" if in_tail else precision_by_class[module_class]
            if name in gemm_modules:
                modules.append(
                    TEModulePrecision(
                        name=name, module_class=module_class, layer=layer, precision=precision, scope=scope
                    )
                )
            elif precision != "high_precision":
                not_quantizable.append(f"{name} ({module_class}: {precision})")
        if not_quantizable:
            raise ValueError(
                f"NemotronV3PrecisionMap quantizes {len(not_quantizable)} module(s) that are not Transformer Engine "
                "GEMM modules and would silently stay in high precision; build them with "
                "backend.mamba_linear='te' (Mamba projections), backend.linear='te' (other projections) / "
                "backend.experts='te' (with an expert-parallel dispatcher on more than one rank) or set their "
                f"class to high_precision: {', '.join(not_quantizable[:8])}"
                + (f" and {len(not_quantizable) - 8} more" if len(not_quantizable) > 8 else "")
            )
        return TEPrecisionPlan(
            modules=tuple(modules), num_layers=num_layers, high_precision_tail_layers=self.high_precision_tail_layers
        )

    def prepare(self, model: nn.Module) -> None:
        """Give every Transformer Engine GEMM module of ``model`` the forward of its precision; idempotent.

        ``TENvfp4Config.exclude_modules`` calls this for ``te_nvfp4.precision_map``.

        Args:
            model: Model (or pipeline stage) whose modules are inspected by qualified name.

        Raises:
            ValueError: If the model has no Transformer Engine GEMM module, or :meth:`resolve` rejects it.
        """
        if not common_utils.HAVE_TE:
            return
        gemm_modules = te_gemm_modules(model)
        if not gemm_modules:
            raise ValueError(
                "te_nvfp4.precision_map is set but the model has no Transformer Engine GEMM module "
                "(backend.linear='te' or backend.experts='te' with an expert-parallel dispatcher): "
                "nothing would be quantized"
            )
        plan = self.resolve([name for name, _ in model.named_modules()], [name for name, _ in gemm_modules])
        apply_precision_plan(model, plan)
