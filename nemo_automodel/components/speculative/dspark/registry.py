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

"""Dispatch registry mapping a target architecture string to a DSpark draft model.

Mirrors the EAGLE / DFlash registries: keyed by the target's HF ``architectures``
string so adding a target family is a one-line append, with no recipe change.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel

from nemo_automodel.components.speculative.dspark.draft_deepseek_v4 import DeepseekV4DSparkModel
from nemo_automodel.components.speculative.dspark.draft_glm_5_2 import Glm5_2DSparkModel
from nemo_automodel.components.speculative.dspark.draft_qwen3 import Qwen3DSparkModel


@dataclass(frozen=True)
class DraftSpec:
    """How to build a DSpark draft model for a particular target architecture."""

    draft_cls: type[PreTrainedModel]


# Qwen3-style dense (and MoE) targets: the draft only consumes the target's
# post-block hidden states, so an MoE backbone is handled like a dense one.
_DENSE_ARCHITECTURES: tuple[str, ...] = (
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
)

DSPARK_DRAFT_REGISTRY: dict[str, DraftSpec] = {
    arch: DraftSpec(draft_cls=Qwen3DSparkModel) for arch in _DENSE_ARCHITECTURES
}
# DeepSeek V4 target: a V4-attention draft (Q-LoRA, single shared K=V latent,
# grouped O-LoRA, interleaved partial RoPE), registered separately from the
# Qwen3-style drafts because its backbone differs.
DSPARK_DRAFT_REGISTRY["DeepseekV4ForCausalLM"] = DraftSpec(draft_cls=DeepseekV4DSparkModel)
# GLM-5.2 target (HF arch GlmMoeDsaForCausalLM): a dense GLM MLA draft (DeepSeek-V3-style
# Q-LoRA + compressed KV latent + interleaved complex RoPE), with the DSA indexer and MoE
# dropped. Registered separately because its MLA backbone differs from V4's.
DSPARK_DRAFT_REGISTRY["GlmMoeDsaForCausalLM"] = DraftSpec(draft_cls=Glm5_2DSparkModel)


def resolve_dspark_draft_spec(architectures: list[str]) -> DraftSpec:
    """Return the first registered DSpark draft spec matching ``architectures``."""
    for arch in architectures:
        spec = DSPARK_DRAFT_REGISTRY.get(arch)
        if spec is not None:
            return spec
    raise ValueError(
        f"TrainDSparkRecipe: no DSpark draft spec registered for any of {architectures}. "
        f"Supported architectures: {sorted(DSPARK_DRAFT_REGISTRY)}."
    )


def build_target_layer_ids(num_target_layers: int, num_feature_layers: int) -> list[int]:
    """Evenly spread ``num_feature_layers`` feature layers across the target depth.

    Used as the default when ``target_layer_ids`` is not given. Returns strictly
    increasing ids in ``[1, num_target_layers - 1]`` (the embedding output, id
    ``-1``/``0``, is excluded by default to match the paper's choice of mid/late
    feature layers).
    """
    num_target_layers = int(num_target_layers)
    num_feature_layers = int(num_feature_layers)
    if num_target_layers < 2:
        raise ValueError(f"num_target_layers must be >= 2, got {num_target_layers}")
    num_feature_layers = max(1, min(num_feature_layers, num_target_layers - 1))
    last = num_target_layers - 1
    if num_feature_layers == 1:
        return [last]
    step = (last - 1) / (num_feature_layers - 1)
    ids = sorted({int(round(1 + i * step)) for i in range(num_feature_layers)})
    return ids


__all__ = ["DraftSpec", "DSPARK_DRAFT_REGISTRY", "resolve_dspark_draft_spec", "build_target_layer_ids"]
