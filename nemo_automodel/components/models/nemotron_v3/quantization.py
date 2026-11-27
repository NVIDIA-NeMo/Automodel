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

"""NVFP4 training of Nemotron 3 Super models with the precision layout published for them.

:class:`NemotronSuperNvfp4Config` is the ``model.backend.te_nvfp4`` config of these models. Which module runs
NVFP4, MXFP8 or high precision is fixed here and has no setting. It is the layout of the Nemotron 3 Super
technical report, arXiv 2604.12374, section 2.2 "Precision by Layer Type".

The layout knows the Mamba, attention and MoE blocks of the hybrid pattern and the MTP head. A dense MLP block
(block type ``mlp`` of Nemotron-H) has no module class here: its projections are rejected like every other
GEMM module the layout does not know. The output projection (``lm_head``) is an ``LMHead``, a
``torch.nn.Linear`` that Transformer Engine does not quantize: it runs in the model dtype and has no class.
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
from nemo_automodel.components.models.common.utils import _EXCLUDED_FORWARD_ATTR, TENvfp4Config, _forward_outside_fp8

logger = logging.getLogger(__name__)

TEPrecision: TypeAlias = Literal["nvfp4", "mxfp8", "high_precision"]
"""How the GEMMs of a module run inside the NVFP4 autocast: under the NVFP4 recipe, under Transformer Engine's
MXFP8 recipe, or outside the autocast in the model dtype."""

TE_PRECISIONS: tuple[TEPrecision, ...] = ("nvfp4", "mxfp8", "high_precision")

NemotronV3ModuleClass: TypeAlias = Literal[
    "mamba_in_proj",
    "mamba_out_proj",
    "attention",
    "routed_experts",
    "shared_experts",
    "latent_proj",
    "mtp",
]

HIGH_PRECISION_TAIL_LAYERS = 14
"""Number of decoder layers at the end of the stack whose GEMMs all run in high precision, whatever their
class: layers 74-87 of the 88."""

PRECISION_BY_CLASS: dict[NemotronV3ModuleClass, TEPrecision] = {
    "mamba_in_proj": "nvfp4",
    "mamba_out_proj": "mxfp8",
    "attention": "high_precision",
    "routed_experts": "nvfp4",
    "shared_experts": "nvfp4",
    "latent_proj": "high_precision",
    "mtp": "high_precision",
}
"""Precision of every module class in the layers before the high-precision tail; ``mtp`` is every GEMM of the
multi-token-prediction head, whatever its kind."""


@dataclass(frozen=True)
class TEModulePrecision:
    """Precision of one Transformer Engine GEMM module (``Linear`` or ``GroupedLinear``)."""

    name: str
    """Qualified name of the GEMM module in ``model.named_modules()``."""
    module_class: str
    """Label of the kind of module, as the model family names it, e.g. ``"mamba_in_proj"``."""
    layer: int | None
    """Index of the decoder layer that holds the module; ``None`` outside the decoder stack."""
    precision: TEPrecision
    """How the GEMMs of the module run."""
    scope: str
    """Qualified name of the module whose forward selects the precision: ``name`` or one of its ancestors.
    An ancestor is used when it reads the active recipe itself, as grouped experts do to pad their inputs."""

    def __post_init__(self) -> None:
        """Reject a scope that does not contain the module."""
        if self.scope != self.name and not self.name.startswith(f"{self.scope}."):
            raise ValueError(f"Scope {self.scope!r} is not {self.name!r} or one of its ancestors")


@dataclass(frozen=True)
class TEPrecisionPlan:
    """Precision of every Transformer Engine GEMM module of one model."""

    modules: tuple[TEModulePrecision, ...]
    """One entry per GEMM module."""
    num_layers: int
    """Number of decoder layers; 0 when the plan has no notion of layers."""
    high_precision_tail_layers: int
    """Number of decoder layers at the end of the stack whose GEMMs all run in high precision."""

    def __post_init__(self) -> None:
        """Reject scopes with two precisions and scopes inside other scopes."""
        scopes = self.scopes()
        for module in self.modules:
            if scopes[module.scope] != module.precision:
                raise ValueError(
                    f"Scope {module.scope!r} holds modules with different precisions "
                    f"({scopes[module.scope]} and {module.precision})"
                )
        for scope in scopes:
            parent = next((other for other in scopes if scope.startswith(f"{other}.")), None)
            if parent is not None:
                raise ValueError(f"Scope {scope!r} lies inside scope {parent!r}")

    def scopes(self) -> dict[str, TEPrecision]:
        """Precision by scope name, in the order of the first module of each scope."""
        scopes: dict[str, TEPrecision] = {}
        for module in self.modules:
            scopes.setdefault(module.scope, module.precision)
        return scopes

    def counts(self) -> dict[str, dict[TEPrecision, int]]:
        """Number of GEMM modules by module class and precision, classes in order of first appearance."""
        counts: dict[str, dict[TEPrecision, int]] = {}
        for module in self.modules:
            by_precision = counts.setdefault(module.module_class, dict.fromkeys(TE_PRECISIONS, 0))
            by_precision[module.precision] += 1
        return counts

    def summary(self) -> str:
        """Table of :meth:`counts` with a line for the layer range, for the startup log."""
        lines = []
        if self.num_layers > 0:
            first_tail_layer = self.num_layers - self.high_precision_tail_layers
            layer_range = f"decoder layers: {self.num_layers}"
            if first_tail_layer > 0:
                layer_range += f"; layers 0-{first_tail_layer - 1} follow the precision of their module class"
            if self.high_precision_tail_layers > 0:
                layer_range += f"; layers {first_tail_layer}-{self.num_layers - 1} run in high precision"
            lines.append(layer_range)
        counts = self.counts()
        width = max([len("module class"), *(len(module_class) for module_class in counts)])
        header = f"{'module class':<{width}}" + "".join(f"  {precision:>14}" for precision in TE_PRECISIONS)
        lines.append(header)
        totals = dict.fromkeys(TE_PRECISIONS, 0)
        for module_class, by_precision in counts.items():
            lines.append(
                f"{module_class:<{width}}" + "".join(f"  {by_precision[precision]:>14}" for precision in TE_PRECISIONS)
            )
            for precision in TE_PRECISIONS:
                totals[precision] += by_precision[precision]
        lines.append(f"{'total':<{width}}" + "".join(f"  {totals[precision]:>14}" for precision in TE_PRECISIONS))
        return "\n".join(lines)


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


def resolve_precision_plan(module_names: Sequence[str], gemm_module_names: Collection[str]) -> TEPrecisionPlan:
    """Assign the precision of the published layout to every Transformer Engine GEMM module of a model.

    Args:
        module_names: Qualified names of all modules of the model, in ``named_modules`` order.
        gemm_module_names: The names among them that are Transformer Engine GEMM modules.

    Returns:
        Plan with one entry per Transformer Engine GEMM module, in ``module_names`` order.

    Raises:
        ValueError: If the names hold no decoder layer; if a Transformer Engine GEMM module is not one the
            layout has a class for (the projections of a dense MLP block among them); if a module that the
            layout quantizes is not a Transformer Engine module (a Mamba projection built with
            ``backend.mamba_linear="torch"``, another projection built with ``backend.linear="torch"``,
            experts without Transformer Engine grouped GEMMs), which would silently leave it in high
            precision; if the decoder layers are not numbered 0..N-1 (a pipeline stage holds only some of
            them, so the tail cannot be located); or if the model has fewer decoder layers than the tail.
    """
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
            "NemotronSuperNvfp4Config found no module named 'model.layers.<index>.<module>': the model is not "
            "a Nemotron 3 model, so nothing would be quantized as published."
        )
    unknown = sorted(gemm_modules - {name for name, _, _, _ in sites})
    if unknown:
        raise ValueError(
            f"NemotronSuperNvfp4Config does not know {len(unknown)} Transformer Engine GEMM module(s) and "
            f"will not guess their precision: {', '.join(unknown)}. It has classes for the projections of "
            "Mamba, attention and MoE blocks and for the MTP head; a dense MLP block (block type 'mlp') "
            "and modules outside the decoder layers and the MTP head have none."
        )
    num_layers = len(layers)
    if sorted(layers) != list(range(num_layers)):
        raise ValueError(
            f"NemotronSuperNvfp4Config needs the whole decoder stack to locate its last layers, but the model "
            f"holds layers {sorted(layers)}; pipeline stages are not supported."
        )
    if HIGH_PRECISION_TAIL_LAYERS > num_layers:
        raise ValueError(
            f"NemotronSuperNvfp4Config keeps the last {HIGH_PRECISION_TAIL_LAYERS} decoder layers in high "
            f"precision, which exceeds the {num_layers} decoder layers of the model"
        )
    first_tail_layer = num_layers - HIGH_PRECISION_TAIL_LAYERS
    modules: list[TEModulePrecision] = []
    not_quantizable: list[str] = []
    for name, module_class, layer, scope in sites:
        in_tail = layer is not None and layer >= first_tail_layer
        precision: TEPrecision = "high_precision" if in_tail else PRECISION_BY_CLASS[module_class]
        if name in gemm_modules:
            modules.append(
                TEModulePrecision(name=name, module_class=module_class, layer=layer, precision=precision, scope=scope)
            )
        elif precision != "high_precision":
            not_quantizable.append(f"{name} ({module_class}: {precision})")
    if not_quantizable:
        raise ValueError(
            f"NemotronSuperNvfp4Config quantizes {len(not_quantizable)} module(s) that are not Transformer Engine "
            "GEMM modules and would silently stay in high precision; build them with "
            "backend.mamba_linear='te' (Mamba projections), backend.linear='te' (other projections) / "
            "backend.experts='te' (with an expert-parallel dispatcher on more than one rank): "
            f"{', '.join(not_quantizable[:8])}"
            + (f" and {len(not_quantizable) - 8} more" if len(not_quantizable) > 8 else "")
        )
    return TEPrecisionPlan(
        modules=tuple(modules), num_layers=num_layers, high_precision_tail_layers=HIGH_PRECISION_TAIL_LAYERS
    )


@dataclass(kw_only=True)
class NemotronSuperNvfp4Config(TENvfp4Config):
    """``te_nvfp4`` with the precision layout published for Nemotron 3 Super models.

    Set as ``model.backend.te_nvfp4`` through its ``_target_``. The layout (:data:`PRECISION_BY_CLASS`,
    :data:`HIGH_PRECISION_TAIL_LAYERS`) is fixed and has no setting; the options of the NVFP4 recipe are those
    of ``TENvfp4Config``. It covers a language model built from Mamba, attention and MoE blocks
    (``NemotronHForCausalLM`` with that pattern) on its own and inside a vision-language model, where the
    names carry a ``language_model.`` prefix, with or without activation checkpointing of the blocks or of the
    MoE mixers. A model with dense MLP blocks is rejected. The router, the norms, the embedding, the Mamba
    convolution and scan and the output projection have no Transformer Engine GEMM and are not affected.

    Every decoder layer is one block of the hybrid pattern (Mamba, attention or MoE), so the 88-layer model
    keeps layers 74-87 in high precision whatever their class says.
    """

    exclude_fqns: tuple[str, ...] = ()
    """Stays empty: the layout keeps modules in high precision itself."""

    def __post_init__(self) -> None:
        """Reject excluded module names next to the fixed layout."""
        super().__post_init__()
        if self.exclude_fqns:
            raise ValueError(
                "NemotronSuperNvfp4Config has a fixed precision layout and takes no te_nvfp4.exclude_fqns, got "
                f"{list(self.exclude_fqns)!r}"
            )

    def exclude_modules(self, model: nn.Module) -> list[str]:
        """Give every Transformer Engine GEMM module of ``model`` the forward of its precision; idempotent.

        Args:
            model: Model (or pipeline stage) whose modules are inspected by qualified name.

        Returns:
            Qualified names of the modules now kept out of the autocast, in the order of their first GEMM
            module; empty without Transformer Engine.

        Raises:
            ValueError: If the model has no Transformer Engine GEMM module, or :func:`resolve_precision_plan`
                rejects it.
        """
        if not common_utils.HAVE_TE:
            return []
        gemm_modules = te_gemm_modules(model)
        if not gemm_modules:
            raise ValueError(
                "NemotronSuperNvfp4Config is set but the model has no Transformer Engine GEMM module "
                "(backend.linear='te' or backend.experts='te' with an expert-parallel dispatcher): "
                "nothing would be quantized"
            )
        plan = resolve_precision_plan([name for name, _ in model.named_modules()], [name for name, _ in gemm_modules])
        apply_precision_plan(model, plan)
        return [scope for scope, precision in plan.scopes().items() if precision == "high_precision"]
