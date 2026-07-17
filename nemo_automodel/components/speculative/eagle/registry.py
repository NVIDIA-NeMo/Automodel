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

The dense draft (``LlamaEagle3DraftModel`` / ``LlamaEagleDraftModel``) covers
most registered architectures: the implementation is config-driven and reads
``attention_bias``, ``mlp_bias``, ``head_dim``, ``rope_theta`` /
``rope_scaling``, and ``rms_norm_eps`` directly from the target config.
Adding an architecture that fits this shape is a one-line registry
append. Architectures that need a different draft get a new ``draft_cls``
entry pointing at a dedicated draft module -- e.g. gpt-oss
(``GptOssForCausalLM``), whose YaRN RoPE (``rope_type="yarn"``) is not
implemented by ``LlamaRotaryEmbedding``, uses :class:`GptOssEagle3DraftModel`
(a thin subclass that swaps in gpt-oss's YaRN rotary so the draft stays
positionally consistent with the target). See ``draft_gpt_oss.py`` for the
rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel

from nemo_automodel.components.speculative.eagle.draft_deepseek import DeepseekV3Eagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_gemma import Gemma4Eagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_gpt_oss import GptOssEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel


@dataclass(frozen=True)
class DraftSpec:
    """How to build an EAGLE draft model for a particular target architecture."""

    draft_cls: type[PreTrainedModel]


# Llama-style dense LLMs. The dense draft works for any architecture in this
# tuple as long as the target supports HuggingFace's
# ``output_hidden_states=True`` mechanism. This includes MoE backbones
# (e.g. ``Qwen3MoeForCausalLM``): the draft only consumes the post-block
# hidden states emitted by ``register_forward_hook`` on each decoder layer,
# not the per-expert routing internals -- so an MoE target is treated
# identically to a dense target end-to-end.
_DENSE_ARCHITECTURES: tuple[str, ...] = (
    "LlamaForCausalLM",
    "Phi3ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
)


EAGLE3_DRAFT_REGISTRY: dict[str, DraftSpec] = {
    arch: DraftSpec(draft_cls=LlamaEagle3DraftModel) for arch in _DENSE_ARCHITECTURES
}

# gpt-oss MoE target. The draft is still Llama-style dense (it only consumes
# post-block hidden states, like any MoE target), but it needs a dedicated
# draft class to reproduce gpt-oss's YaRN RoPE -- see ``draft_gpt_oss.py``.
# Only EAGLE-3 is wired up; EAGLE-1/2 for gpt-oss is not validated yet.
EAGLE3_DRAFT_REGISTRY["GptOssForCausalLM"] = DraftSpec(draft_cls=GptOssEagle3DraftModel)

# DeepSeek-V3 MLA target. Multi-head Latent Attention (low-rank q/kv projections,
# interleaved RoPE on the rope slice, and a value head dim that differs from the
# q/k head dim) cannot be represented by the Llama-style dense draft, so it gets a
# dedicated MLA draft -- see ``draft_deepseek.py``. Only EAGLE-3 is wired up.
EAGLE3_DRAFT_REGISTRY["DeepseekV3ForCausalLM"] = DraftSpec(draft_cls=DeepseekV3Eagle3DraftModel)

# Gemma4 multimodal target (``Gemma4ForConditionalGeneration``). The draft is
# Llama-style dense (it consumes only post-block hidden states from the frozen
# target's text backbone), but the Gemma4 *text* config needs two quirks
# reconciled before the shared draft can build -- the ``hidden_activation`` key
# and the nested per-attention-type ``rope_parameters`` -- so it gets a thin
# dedicated draft class. See ``draft_gemma.py`` for the rationale. The target's
# decoder config is nested under ``config.text_config`` (the draft is built from
# it via ``config.get_text_config()`` in the recipe) and its decoder layers live
# under ``model.language_model.layers`` (handled by the target wrapper's
# ``_get_transformer_layers``). Only EAGLE-3 is wired up.
EAGLE3_DRAFT_REGISTRY["Gemma4ForConditionalGeneration"] = DraftSpec(draft_cls=Gemma4Eagle3DraftModel)


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
