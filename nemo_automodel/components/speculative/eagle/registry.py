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

"""Dispatch registry mapping target architecture -> EAGLE draft model.

Mirrors SpecForge's ``modeling/auto.py`` pattern but keyed by HF
``architectures`` string instead of by config class. The string key avoids a
hard import dependency on every individual HF ``*Config`` class (which can
differ between transformers versions) and matches NeMo AutoModel's existing
``_transformers/registry.MODEL_ARCH_MAPPING`` style.

Today the dense draft (``LlamaEagle3DraftModel`` / ``LlamaEagleDraftModel``)
covers all registered architectures: the implementation is config-driven and
reads ``attention_bias``, ``mlp_bias``, ``head_dim``, ``rope_theta`` /
``rope_scaling``, and ``rms_norm_eps`` directly from the target config.
Adding an architecture that fits this shape is a one-line registry
append. Architectures that need a different draft (e.g. gpt-oss with its
alternating sliding-window attention and non-standard RoPE) get a new
``draft_cls`` entry pointing at a dedicated draft module.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel


@dataclass(frozen=True)
class DraftSpec:
    """How to build an EAGLE draft model for a particular target architecture."""

    draft_cls: type[PreTrainedModel]


# Llama-style dense LLMs. The dense draft works for any architecture in this
# tuple as long as the target supports HuggingFace's
# ``output_hidden_states=True`` mechanism (all standard causal-LM HF models do,
# including MoE backbones -- the draft only consumes the post-block hidden
# states, not the per-expert routing internals).
_DENSE_ARCHITECTURES: tuple[str, ...] = (
    "LlamaForCausalLM",
    "Phi3ForCausalLM",
)


EAGLE3_DRAFT_REGISTRY: dict[str, DraftSpec] = {
    arch: DraftSpec(draft_cls=LlamaEagle3DraftModel) for arch in _DENSE_ARCHITECTURES
}


EAGLE1_DRAFT_REGISTRY: dict[str, DraftSpec] = {
    arch: DraftSpec(draft_cls=LlamaEagleDraftModel) for arch in _DENSE_ARCHITECTURES
}


def _resolve(architectures: list[str], registry: dict[str, DraftSpec], recipe_name: str) -> DraftSpec:
    """Return the first registered draft spec matching any architecture in the list."""
    for arch in architectures:
        spec = registry.get(arch)
        if spec is not None:
            return spec
    raise ValueError(
        f"{recipe_name}: no EAGLE draft spec registered for any of {architectures}. "
        f"Supported architectures: {sorted(registry)}."
    )


def resolve_eagle3_draft_spec(architectures: list[str]) -> DraftSpec:
    """Resolve the EAGLE-3 draft spec for a target's ``config.architectures`` field."""
    return _resolve(architectures, EAGLE3_DRAFT_REGISTRY, "TrainEagle3Recipe")


def resolve_eagle1_draft_spec(architectures: list[str]) -> DraftSpec:
    """Resolve the EAGLE-1 / EAGLE-2 draft spec for a target's ``config.architectures`` field."""
    return _resolve(architectures, EAGLE1_DRAFT_REGISTRY, "TrainEagle1Recipe")
