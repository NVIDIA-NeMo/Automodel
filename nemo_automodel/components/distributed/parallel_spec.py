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

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.distributed.tensor.parallel import ParallelStyle

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy


@dataclass(frozen=True)
class ParallelSpec:
    """Everything ``fsdp2_strategy_parallelize`` needs to know about one architecture, as data.

    A model class declares its contract as a ``parallel_spec`` class attribute, which every
    dynamically created wrapper subclass (``HFCheckpointingMixin``, capability injection)
    inherits. Architectures the repository does not re-implement declare it in
    ``components/models/<family>/parallelization.py`` and the loaders bind it onto the wrapper
    they create. Every field defaults to the generic llama-style behaviour, so a dense causal LM
    with ``model.layers`` needs no spec.

    Every field is declarative: plans are dictionaries, constraints are module names. Behaviour
    that genuinely depends on the model instance belongs in a ``ParallelizationStrategy``
    (``strategy``), never in the spec.

    Attributes:
        tp_plan: ``{module FQN pattern: ParallelStyle}`` applied under tensor parallelism.
            ``None`` falls back to the model's HuggingFace ``_tp_plan``, then the base plan.
        sequence_parallel_plan: Entries overlaid on ``tp_plan`` when sequence parallelism is
            requested (sequence-sharded norms, reduce-scattering projections). ``None`` means the
            architecture has no sequence-parallel variant; the request is then ignored with a
            warning.
        layer_groups: Transformer-block containers per role (``"language"``, ``"vision"``),
            each as candidate FQNs; the first that resolves wins, so one spec covers several
            ``transformers`` module-tree layouts. ``None`` uses ``model.model.layers`` or
            the largest ``ModuleList`` in the model.
        text_config_path: Dotted attribute path from the model to the config holding
            ``num_attention_heads`` / ``num_key_value_heads`` for TP validation, e.g.
            ``"config.text_config"``. ``None`` reads ``model.config``.
        hf_tp_plan_prefix: Candidate FQNs of the submodule whose ``_tp_plan`` keys are
            relative to it; the first that resolves is used and prefixes those keys.
        sharded_output_only: Module FQNs whose plan entry is dropped unless it produces a sharded
            output, whatever the plan's source (declared, HuggingFace or user-supplied). A head
            whose forward combines its output with a sharded weight (e.g. weight-normalized
            logits) cannot take a replicated output, so it is left un-parallelized instead.
        strategy: Whole-flow ``ParallelizationStrategy`` override; ``None`` uses the default.
    """

    tp_plan: dict[str, ParallelStyle] | None = None
    sequence_parallel_plan: dict[str, ParallelStyle] | None = None
    layer_groups: dict[str, tuple[str, ...]] | None = None
    text_config_path: str | None = None
    hf_tp_plan_prefix: tuple[str, ...] = ("model",)
    sharded_output_only: tuple[str, ...] = ()
    strategy: ParallelizationStrategy | None = None

    def resolved_tp_plan(self, sequence_parallel: bool = False) -> dict[str, ParallelStyle] | None:
        """The declared plan for one run: ``tp_plan`` with ``sequence_parallel_plan`` overlaid when requested.

        Styles are shallow-copied so the shared declaration is never mutated -- LoRA translation
        rewrites a style's class in place -- and ``None`` is returned when no plan is declared.
        """
        if self.tp_plan is None:
            return None
        plan = dict(self.tp_plan)
        if sequence_parallel and self.sequence_parallel_plan:
            plan.update(self.sequence_parallel_plan)
        return {fqn: copy.copy(style) for fqn, style in plan.items()}
