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

"""Dispatch registry mapping target architecture -> DFlash draft model.

Mirrors the EAGLE registry (``components/speculative/eagle/registry.py``). The
DFlash draft is a non-causal Qwen3-style stack and is config-driven, so adding a
Qwen3-shaped architecture is a one-line append.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import Qwen3DFlash2DraftModel


@dataclass(frozen=True)
class DFlashDraftSpec:
    """How to build a DFlash draft model for a particular target architecture.

    Attributes:
        draft_cls: DFlash draft class (also the base for the Domino / JetSpec
            recipes, which only change the head or the objective).
        draft2_cls: DFlash 2 draft class -- the same backbone plus the in-block
            convolutions and the candidate selector.
    """

    draft_cls: type[PreTrainedModel]
    draft2_cls: type[PreTrainedModel]


# Qwen3-shaped dense / MoE targets. The DFlash draft only consumes post-block
# hidden states captured via forward hooks, so an MoE target (e.g.
# ``Qwen3MoeForCausalLM``) is handled identically to a dense one.
_QWEN3_ARCHITECTURES: tuple[str, ...] = (
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    # Qwen3.5-family targets, which is what ``Qwen/Qwen3.8-27B`` ships as
    # (``model_type: qwen3_5``). The ``*ForConditionalGeneration`` variants keep
    # their decoder hyper-parameters on a nested ``text_config``; the recipe and
    # the target wrapper unwrap it via ``resolve_text_config``.
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
)


DFLASH_DRAFT_REGISTRY: dict[str, DFlashDraftSpec] = {
    arch: DFlashDraftSpec(draft_cls=Qwen3DFlashDraftModel, draft2_cls=Qwen3DFlash2DraftModel)
    for arch in _QWEN3_ARCHITECTURES
}


def resolve_dflash_draft_spec(architectures: list[str]) -> DFlashDraftSpec:
    """Return the first registered DFlash draft spec matching any architecture in the list."""
    for arch in architectures:
        spec = DFLASH_DRAFT_REGISTRY.get(arch)
        if spec is not None:
            return spec
    raise ValueError(
        f"TrainDFlashRecipe: no DFlash draft spec registered for any of {architectures}. "
        f"Supported architectures: {sorted(DFLASH_DRAFT_REGISTRY)}."
    )
