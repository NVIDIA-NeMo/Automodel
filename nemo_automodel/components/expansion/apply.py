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

"""Applying model expansion to a causal-LM and freezing everything else."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import torch.nn as nn

from nemo_automodel.components.expansion.dual_stream import (
    patch_layer_for_expansion,
    patch_model_for_pipeline,
    patch_norm_for_merge,
)
from nemo_automodel.components.expansion.expanded_linear import (
    ExpandedLinear,
    patch_linear_for_expansion,
)

logger = logging.getLogger(__name__)

__all__ = ["ExpansionConfig", "apply_expansion", "freeze_non_expansion_parameters", "initialize_expansion"]

#: Projections that write into the residual stream. Starting these at zero is what makes
#: the expanded model reproduce its parent exactly before the first optimizer step.
DEFAULT_ZERO_INIT_MODULES = ("o_proj", "down_proj")

#: Projections expanded by default; the leaf names shared by the llama-style families.
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class ExpansionConfig:
    """Configuration for dual-stream model expansion.

    Attributes:
        enabled: Whether to expand at all.
        layers: Zero-based decoder-layer indices to expand. ``None`` expands every layer.
            Layers left out still carry both streams, in ``skip`` mode, which is exact and
            costs no extra compute.
        target_modules: Leaf module names to expand within an expanded layer.
        zero_init_modules: Subset of ``target_modules`` whose expansion weight starts at
            zero rather than as a copy of the pretrained weight.
        merge_weight: ``lambda`` in ``h = h_a + lambda * (h_b - h_a)``, applied just before
            the decoder stack's final norm.
    """

    enabled: bool = False
    layers: list[int] | None = None
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    zero_init_modules: list[str] = field(default_factory=lambda: list(DEFAULT_ZERO_INIT_MODULES))
    merge_weight: float = 0.5

    def __post_init__(self) -> None:
        """Validate the field combination.

        Raises:
            ValueError: If ``merge_weight`` is outside ``[0, 1]`` or ``zero_init_modules``
                names something absent from ``target_modules``.
        """
        if not 0.0 <= self.merge_weight <= 1.0:
            raise ValueError(f"merge_weight must be in [0, 1], got {self.merge_weight}")
        unknown = set(self.zero_init_modules) - set(self.target_modules)
        if unknown:
            raise ValueError(
                f"zero_init_modules contains {sorted(unknown)}, which are not in "
                f"target_modules {sorted(self.target_modules)}"
            )


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder-layer list of a causal-LM.

    Args:
        model: A causal-LM whose decoder exposes ``.layers``, the convention both stock
            Hugging Face models and this repository's own implementations follow.

    Returns:
        The ``nn.ModuleList`` of decoder layers.

    Raises:
        AttributeError: If the model does not follow that convention, which means it needs
            explicit support rather than a silently wrong expansion.
    """
    for attr in ("model", "language_model", "transformer"):
        inner = getattr(model, attr, None)
        if inner is not None and getattr(inner, "layers", None) is not None:
            return inner.layers
    raise AttributeError(
        f"{type(model).__name__} does not expose a decoder layer list at "
        "`.model.layers` / `.language_model.layers` / `.transformer.layers`; model "
        "expansion needs explicit support for this architecture."
    )


def _hidden_size(model: nn.Module) -> int:
    """Read the width of one hidden-state stream from the model config.

    The two streams travel between decoder layers concatenated on the hidden axis, and a
    layer tells that carrier from the embedding output by its width, so the number has to
    be known at patch time.

    Args:
        model: The causal-LM being expanded.

    Returns:
        The model's hidden size.

    Raises:
        AttributeError: If the config does not report one, which means this architecture
            needs explicit support rather than a guess.
    """
    config = getattr(model, "config", None)
    for holder in (config, getattr(config, "text_config", None)):
        hidden_size = getattr(holder, "hidden_size", None)
        if isinstance(hidden_size, int):
            return hidden_size
    # Models built without a Hugging Face config still have a final norm, and a decoder
    # stack's final norm normalizes over exactly one stream's width.
    weight = getattr(_final_norm(model), "weight", None)
    if weight is not None and weight.dim() >= 1:
        return weight.shape[-1]
    raise AttributeError(
        f"{type(model).__name__} reports no `config.hidden_size` and its final norm has no "
        "weight to read a width from; model expansion needs the hidden size to recognize "
        "the two-stream carrier passed between decoder layers."
    )


def _final_norm(model: nn.Module) -> nn.Module:
    """Locate the norm applied after the decoder stack.

    Args:
        model: A causal-LM.

    Returns:
        The final norm module.

    Raises:
        AttributeError: If it cannot be found, since the stream pair has to be merged
            somewhere before the LM head.
    """
    for attr in ("model", "language_model", "transformer"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        for norm_attr in ("norm", "final_layernorm", "ln_f"):
            norm = getattr(inner, norm_attr, None)
            if norm is not None:
                return norm
    raise AttributeError(
        f"{type(model).__name__} does not expose a final norm; model expansion needs one "
        "to merge the two hidden-state streams before the LM head."
    )


def _unreachable_grouped_weights(layer: nn.Module) -> list[str]:
    """Find weights in a decoder layer that expansion cannot reach.

    Expansion patches ``nn.Linear`` modules. A mixture-of-experts block in this repository
    does not use them: every expert's projection lives in one stacked parameter of shape
    ``[experts, in, out]``, so :func:`apply_expansion` walks straight past it and expands
    only the attention projections beside it. The resulting model is half-expanded and
    nothing about it says so.

    Args:
        layer: A decoder layer being expanded.

    Returns:
        Names of stacked (3-D or higher) floating-point parameters, which is what a grouped
        expert stack looks like. Empty for a dense layer.
    """
    return [
        name
        for name, param in layer.named_parameters()
        if param.dim() >= 3 and param.is_floating_point() and not is_expansion_parameter(name)
    ]


def apply_expansion(model: nn.Module, config: ExpansionConfig, initialize: bool = True) -> nn.Module:
    """Expand a causal-LM in place.

    Every decoder layer is patched to carry two hidden-state streams; the layers named by
    ``config.layers`` additionally get expansion weights on their target projections. The
    final norm is patched to merge the streams before it normalizes.

    Every module is patched in place rather than wrapped, so module paths and state-dict
    keys are unchanged. Tensor-parallel plans match on paths and ``to_hf`` / ``from_hf``
    match on keys; both would break if a wrapper inserted a level. Because the expansion
    weights are ordinary parameters on ordinarily-named modules, checkpointing needs no
    special handling: they are saved and restored with everything else.

    Allocating an expansion weight has to happen **before the model is parallelized**,
    because the tensor-parallel plan and ``fully_shard`` are what distribute it. Giving it
    a value has to happen **after the pretrained weights are loaded**, because it copies
    them. On a single process both moments are the same one and the default
    ``initialize=True`` covers it; a parallel run materializes its weights only after
    sharding, so it passes ``initialize=False`` here and calls
    :func:`initialize_expansion` once the checkpoint is in.

    A checkpoint written before expansion has no expansion keys, so loading one into an
    already-expanded model needs ``Checkpointer.load_model(...,
    allow_checkpoint_key_subset=True)``, which keeps the values set here.

    Under pipeline parallelism ``config.layers`` has to name at least one layer in every
    stage's range. A stage owns a contiguous slice of the decoder stack, and one holding no
    expanded layer has nothing to train, which leaves its optimizer with no parameters.

    Args:
        model: The pretrained causal-LM to expand. Modified in place.
        config: Which layers and projections to expand, and how to merge.
        initialize: Also give the expansion weights their values. See above.

    Returns:
        The same model, expanded.

    Raises:
        ValueError: If ``config.layers`` names an index outside the decoder stack.
        RuntimeError: If ``initialize`` is set and the weights are not ready for it. See
            :meth:`ExpandedLinear.initialize_expansion`.
    """
    if not config.enabled:
        return model

    layers = _decoder_layers(model)
    selected = set(range(len(layers)) if config.layers is None else config.layers)
    invalid = sorted(i for i in selected if not 0 <= i < len(layers))
    if invalid:
        raise ValueError(f"expansion layers {invalid} are outside the decoder stack of {len(layers)} layers")

    hidden_size = _hidden_size(model)
    targets = set(config.target_modules)
    zero_init = set(config.zero_init_modules)
    n_expanded = 0
    for index, layer in enumerate(layers):
        if index in selected:
            for parent in layer.modules():
                for child_name, child in parent.named_children():
                    if child_name in targets and isinstance(child, nn.Linear):
                        patch_linear_for_expansion(child, zero_init=child_name in zero_init, initialize=initialize)
                        n_expanded += 1
        patch_layer_for_expansion(layer, "expand" if index in selected else "skip", hidden_size)
    patch_norm_for_merge(_final_norm(model), config.merge_weight, hidden_size)
    # Pipeline parallelism precomputes inter-stage shapes from the config rather than
    # inferring them, so the doubled carrier width has to be declared.
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(vocab_size, int):
        patch_model_for_pipeline(model, hidden_size, vocab_size)

    unreachable = {index: _unreachable_grouped_weights(layers[index]) for index in sorted(selected)}
    unreachable = {index: names for index, names in unreachable.items() if names}
    if unreachable:
        index, names = next(iter(unreachable.items()))
        raise NotImplementedError(
            f"Layer {index} holds stacked expert weights that expansion cannot reach "
            f"(for example {names[0]!r}, shape {dict(layers[index].named_parameters())[names[0]].shape}). "
            "Expansion patches nn.Linear modules, and a mixture-of-experts block keeps every "
            "expert's projection in one stacked parameter instead. Expanding this layer would "
            "silently grow its attention and leave its experts untouched. Layers affected: "
            f"{sorted(unreachable)}."
        )

    logger.info(
        "Model expansion: %d/%d layers expanded, %d expansion weights added, merge_weight=%s",
        len(selected),
        len(layers),
        n_expanded,
        config.merge_weight,
    )
    return model


def initialize_expansion(model: nn.Module) -> int:
    """Give every expansion weight in an expanded model its value.

    The counterpart to ``apply_expansion(..., initialize=False)``, for the case where the
    pretrained weights arrive after the model has been built and sharded. Idempotent: it
    recomputes each weight from the pretrained one, so calling it twice is harmless, and
    calling it after training has started would discard what was learned.

    Args:
        model: A model expanded with ``initialize=False``. Modified in place.

    Returns:
        The number of expansion weights initialized.

    Raises:
        ValueError: If the model has no expansion weights, which means ``apply_expansion``
            was never called and the initialization would silently do nothing.
        RuntimeError: If the weights are not ready. See
            :meth:`ExpandedLinear.initialize_expansion`.
    """
    linears = [module for _, module in expanded_linears(model)]
    if not linears:
        raise ValueError("initialize_expansion found no expanded linears; was apply_expansion called first?")
    for linear in linears:
        linear.initialize_expansion()
    logger.info("Model expansion: initialized %d expansion weights", len(linears))
    return len(linears)


def is_expansion_parameter(name: str) -> bool:
    """Whether a parameter name belongs to an expansion weight.

    Args:
        name: A dotted parameter name as produced by ``named_parameters()``.

    Returns:
        True for expansion weights, False for pretrained ones.
    """
    return ".expansion." in name


def freeze_non_expansion_parameters(model: nn.Module) -> tuple[int, int]:
    """Freeze every parameter except the expansion weights.

    Freezing the rest is what keeps stream A out of the autograd graph entirely, so no
    activations are stored for it and backward only runs over the expanded layers.

    Args:
        model: The expanded model. Modified in place.

    Returns:
        ``(trainable, frozen)`` parameter counts, in elements.
    """
    trainable = frozen = 0
    for name, param in model.named_parameters():
        param.requires_grad = is_expansion_parameter(name)
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    if not trainable:
        raise ValueError(
            "freeze_non_expansion_parameters left no trainable parameters; was apply_expansion called first?"
        )
    return trainable, frozen


def expansion_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Iterate the expansion weights of an expanded model.

    Args:
        model: An expanded model.

    Yields:
        ``(name, parameter)`` for each expansion weight.
    """
    for name, param in model.named_parameters():
        if is_expansion_parameter(name):
            yield name, param


def expanded_linears(model: nn.Module) -> Iterable[tuple[str, ExpandedLinear]]:
    """Iterate the expanded linear layers of an expanded model.

    Args:
        model: An expanded model.

    Yields:
        ``(name, module)`` for each :class:`ExpandedLinear`.
    """
    for name, module in model.named_modules():
        if isinstance(module, ExpandedLinear):
            yield name, module
