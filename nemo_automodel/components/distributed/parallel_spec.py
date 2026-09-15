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

"""Model-owned parallelization contract consumed by ``components.distributed``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy


@dataclass(frozen=True)
class ParallelSpec:
    """Everything ``fsdp2_strategy_parallelize`` needs to know about one architecture.

    A model class declares its contract as a ``parallel_spec`` class attribute, which
    every dynamically created wrapper subclass (``HFCheckpointingMixin``, capability
    injection) inherits. For architectures Automodel does not own -- stock ``transformers``
    classes, ``trust_remote_code`` checkpoints, ``diffusers`` transformers -- the
    ``_transformers`` / ``_diffusers`` bridge sets the attribute on the wrapper class it
    creates. Every field defaults to the generic llama-style behaviour, so a dense causal
    LM with ``model.layers`` needs no spec.

    Attributes:
        tp_plan: ``(model, sequence_parallel) -> {module FQN pattern: ParallelStyle}``.
            ``None`` falls back to the model's HuggingFace ``_tp_plan``, then the base plan.
        adjust_tp_plan: ``(model, plan) -> plan`` applied to the final plan whatever its source
            (explicit, optimized or fallback), for architectures whose forward constrains it.
        layer_groups: Transformer-block containers per role (``"language"``, ``"vision"``),
            each as candidate FQNs; the first that resolves wins, so one spec covers several
            ``transformers`` module-tree layouts. ``None`` uses ``model.model.layers`` or
            the largest ``ModuleList`` in the model.
        text_config_path: Dotted attribute path from the model to the config holding
            ``num_attention_heads`` / ``num_key_value_heads`` for TP validation, e.g.
            ``"config.text_config"``. ``None`` reads ``model.config``.
        hf_tp_plan_prefix: Candidate FQNs of the submodule whose ``_tp_plan`` keys are
            relative to it; the first that resolves is used and prefixes those keys.
        validate_tp: ``(model, tp_size) -> None`` replacing the generic head-divisibility
            check.
        strategy: Whole-flow ``ParallelizationStrategy`` override; ``None`` uses the default.
    """

    tp_plan: Callable[[nn.Module, bool], dict[str, ParallelStyle]] | None = None
    adjust_tp_plan: Callable[[nn.Module, dict[str, ParallelStyle]], dict[str, ParallelStyle]] | None = None
    layer_groups: dict[str, tuple[str, ...]] | None = None
    text_config_path: str | None = None
    hf_tp_plan_prefix: tuple[str, ...] = ("model",)
    validate_tp: Callable[[nn.Module, int], None] | None = None
    strategy: ParallelizationStrategy | None = None
