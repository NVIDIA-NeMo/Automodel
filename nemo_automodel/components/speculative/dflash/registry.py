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
Qwen3 DFlash draft is a non-causal Qwen3-style stack and is config-driven, so
adding a Qwen3-shaped architecture is a one-line append; a target whose backbone
differs (Kimi K3's MLA) registers its own draft class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch import nn
from transformers import PretrainedConfig

from nemo_automodel.components.speculative.dflash.draft_kimi_k3 import (
    KimiK3DFlashDraftModel,
    build_kimi_k3_dflash_draft_config,
    build_kimi_k3_dflash_target_kwargs,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    build_qwen3_dflash_draft_config,
)


def _no_target_kwargs(recipe_cfg) -> dict:
    """Default: a target needs no architecture-specific ``from_pretrained`` kwargs."""
    del recipe_cfg
    return {}


@dataclass(frozen=True)
class DFlashDraftSpec:
    """How to build a DFlash draft model for a particular target architecture.

    Attributes:
        draft_cls: The draft model class.
        build_draft_config: Builds the draft config from the target's text
            config; called with the keyword arguments ``num_draft_layers``,
            ``num_target_layers``, ``block_size``, ``dflash_config``, and
            ``attention_backend``.
        build_target_kwargs: Extra ``from_pretrained`` keyword arguments for the
            frozen target, derived from the recipe's ``recipe_args``.
        attention_backends: The attention backends this draft can run. The
            trainer's mask format is driven by the same knob and must agree (a
            flex ``BlockMask`` only works with the flex attention function, a
            dense additive mask only with sdpa / eager).
        supports_context_parallel: Whether the frozen target can be sharded by
            the DFlash context-parallel path, which installs a key/value-gather
            hook on the target's SDPA call. False for a target that shards the
            sequence itself (it declares ``_owns_cp_attention`` and never routes
            through that hook). This is declared here rather than read off the
            loaded target because the recipe's other CP gates -- which force the
            target onto HuggingFace SDPA -- run before the target exists, and
            would otherwise report a misleading error first.
    """

    draft_cls: type[nn.Module]
    build_draft_config: Callable[..., PretrainedConfig]
    build_target_kwargs: Callable[[Any], dict] = _no_target_kwargs
    attention_backends: tuple[str, ...] = ("flex_attention", "sdpa", "eager")
    supports_context_parallel: bool = True


# Qwen3-shaped dense / MoE targets. The DFlash draft only consumes post-block
# hidden states captured via forward hooks, so an MoE target (e.g.
# ``Qwen3MoeForCausalLM``) is handled identically to a dense one.
_QWEN3_ARCHITECTURES: tuple[str, ...] = (
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
)
# Kimi K3. Both the text-only causal LM and the multimodal wrapper map to the same
# dense MLA draft: DFlash captures hidden states from the text backbone only, so the
# vision tower is irrelevant to the draft. That draft attends over the dense additive
# mask and has no FlexAttention path, and the target owns context parallelism itself.
_KIMI_K3_ARCHITECTURES: tuple[str, ...] = (
    "KimiK3ForCausalLM",
    "KimiK3ForConditionalGeneration",
)


DFLASH_DRAFT_REGISTRY: dict[str, DFlashDraftSpec] = {
    **{
        arch: DFlashDraftSpec(
            draft_cls=Qwen3DFlashDraftModel,
            build_draft_config=build_qwen3_dflash_draft_config,
        )
        for arch in _QWEN3_ARCHITECTURES
    },
    **{
        arch: DFlashDraftSpec(
            draft_cls=KimiK3DFlashDraftModel,
            build_draft_config=build_kimi_k3_dflash_draft_config,
            build_target_kwargs=build_kimi_k3_dflash_target_kwargs,
            attention_backends=("sdpa",),
            supports_context_parallel=False,
        )
        for arch in _KIMI_K3_ARCHITECTURES
    },
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
