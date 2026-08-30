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

"""EAGLE-3 draft model for Gemma4 targets (``Gemma4ForConditionalGeneration``).

Gemma4 is a multimodal decoder whose *text* backbone differs from a Llama-style
dense LLM in a handful of ways: a scaled word embedding (``* sqrt(hidden)``),
zero-centred ``(1 + w)`` RMSNorm, per-head Q/K norm, GeGLU (``gelu_pytorch_tanh``)
MLPs, alternating sliding-window / full attention with two separate RoPE
schedules, and (on the E2B/E4B checkpoints) Gemma-3n-style per-layer inputs and
AltUp. Almost none of that reaches the EAGLE-3 *draft*: the draft is a single
from-scratch decoder layer that consumes only the post-block auxiliary hidden
states emitted by the frozen target (via ``register_forward_hook``) and
re-projects its own Q/K/V, so it never sees the target's sliding mask, per-layer
inputs, AltUp, or experts -- structurally it is the same Llama-style dense draft
used for every other registry entry. The draft's own RMSNorms are trained from
scratch, so the target's zero-centred norm form does not need to be reproduced,
and the constant embedding scale is normalised away by the draft's
``input_layernorm`` before the fused ``[embed, hidden]`` attention input.

Three config quirks *do* have to be reconciled before the shared Llama draft can
build, and that is all this module does (see :func:`_normalize_gemma4_draft_config`):

1. **Activation key.** Gemma text configs name the MLP activation
   ``hidden_activation`` (``gelu_pytorch_tanh``); the shared MLP reads
   ``config.hidden_act``. The GeGLU structure is identical to the draft's SwiGLU
   wiring once the activation is ``gelu_pytorch_tanh``, so we copy the value
   across.
2. **RoPE.** Gemma4 stores a nested, per-attention-type ``rope_parameters``
   (``full_attention``: ``rope_theta=1e6``, ``rope_type="proportional"``,
   ``partial_rotary_factor=0.25``; ``sliding_attention``: ``rope_theta=1e4``,
   ``rope_type="default"``) that the shared ``LlamaRotaryEmbedding`` cannot read
   -- its ``_get_rope_config`` expects a flat ``rope_theta`` and would silently
   fall back to base ``10000``. The single draft layer runs *full* causal
   attention over the whole sequence, so it mirrors the target's **global**
   (full-attention) schedule. We flatten that schedule to a standard
   full-rotary Llama RoPE (``rope_theta`` = the global theta, ``head_dim`` from
   the config, no partial rotary). Training and inference are then
   self-consistent: the saved checkpoint keeps the canonical
   ``architectures: ["LlamaEagle3DraftModel"]`` string and a flat ``rope_theta``,
   so SGLang / vLLM reproduce the exact same rotary with their existing EAGLE-3
   Llama head -- no Gemma-specific inference support is required. (Reproducing
   the target's ``proportional`` + partial-rotary global schedule instead would
   be more faithful to the target's long-context frequencies but no inference
   engine can currently serve it, so it is intentionally not done here.)
3. **MoE FFN width.** On a MoE Gemma4 (``enable_moe_block``) the config
   ``intermediate_size`` is the *per-expert* FFN width (e.g. 2112 on 26B-A4B,
   *below* ``hidden_size`` 2816). The dense draft would otherwise build a
   contracting MLP and be starved of capacity, so the draft MLP is sized to the
   target's *active* FFN width (``top_k_experts * intermediate_size``). Dense
   targets are untouched.

Everything else (GQA, the EAGLE-3 TTT cache attention, the ``fc`` projection,
the draft ``lm_head`` and vocab mapping) is inherited unchanged from
:class:`LlamaEagle3DraftModel`, and the on-disk state-dict layout is identical.
"""

from __future__ import annotations

import logging

from transformers import PretrainedConfig

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel

logger = logging.getLogger(__name__)

# Fallback global RoPE base if a Gemma4 config omits the nested ``rope_parameters``
# (all shipped Gemma4 checkpoints set it; this only guards a hand-written config).
_DEFAULT_GLOBAL_ROPE_THETA = 1_000_000.0


def _extract_global_rope_theta(config: PretrainedConfig) -> float:
    """Return the full-attention (global) RoPE base for the Gemma4 text config.

    Gemma4 nests RoPE parameters per attention type under
    ``config.rope_parameters`` -- ``{"full_attention": {"rope_theta": ...},
    "sliding_attention": {"rope_theta": ...}}``. The draft runs full causal
    attention, so it uses the ``full_attention`` base. Falls back to a flat
    ``rope_theta`` attribute and then to :data:`_DEFAULT_GLOBAL_ROPE_THETA`.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        full = rope_params.get("full_attention")
        if isinstance(full, dict) and full.get("rope_theta") is not None:
            return float(full["rope_theta"])
    flat_theta = getattr(config, "rope_theta", None)
    if flat_theta is not None:
        return float(flat_theta)
    return _DEFAULT_GLOBAL_ROPE_THETA


def _normalize_gemma4_draft_config(config: PretrainedConfig) -> None:
    """Reconcile Gemma4 text-config quirks in place so the Llama draft can build.

    Sets ``hidden_act`` from Gemma's ``hidden_activation`` and flattens the
    nested ``rope_parameters`` to a standard full-rotary Llama RoPE keyed on the
    global (full-attention) ``rope_theta`` (see the module docstring). Mutates
    ``config`` so the draft's serialized ``config.json`` carries the flattened,
    inference-reproducible RoPE.
    """
    # 1. Activation: gemma names it ``hidden_activation``; the shared MLP reads
    #    ``hidden_act``. Only set when absent so an explicit override is honored.
    if getattr(config, "hidden_act", None) is None:
        hidden_activation = getattr(config, "hidden_activation", None)
        if hidden_activation is None:
            raise ValueError(
                "Gemma4Eagle3DraftModel: config exposes neither 'hidden_act' nor 'hidden_activation'; "
                "cannot build the draft MLP activation."
            )
        config.hidden_act = hidden_activation

    # 2. RoPE: flatten the nested per-attention-type schedule to a standard
    #    full-rotary Llama RoPE on the global base. Clearing ``rope_parameters``
    #    routes ``_get_rope_config`` down the flat ``rope_theta`` path.
    config.rope_theta = _extract_global_rope_theta(config)
    config.rope_parameters = None
    config.rope_scaling = None
    # The draft uses the config ``head_dim`` (256 on all Gemma4 sizes) with full
    # rotary; the target's global-attention ``partial_rotary_factor`` (0.25) and
    # larger ``global_head_dim`` (512) apply only inside the frozen target.
    config.partial_rotary_factor = 1.0

    # 3. MoE targets: the draft is a dense model, but on a MoE Gemma4
    #    (``enable_moe_block``) the config ``intermediate_size`` is the *per-expert*
    #    FFN width -- e.g. 2112 on the 26B-A4B target, which is smaller than
    #    ``hidden_size`` (2816). Left as-is the dense draft would build a
    #    *contracting* MLP (down-projecting below the residual width), starving it
    #    of capacity relative to every dense sibling (whose ``intermediate_size``
    #    is the full ~4x FFN). Size the draft MLP to the target's *active* FFN
    #    width instead -- ``top_k_experts`` experts run per token -- so it matches
    #    the per-token feed-forward compute the target actually applies. Dense
    #    Gemma4 targets have ``enable_moe_block`` False and are left untouched.
    if getattr(config, "enable_moe_block", False):
        top_k = int(getattr(config, "top_k_experts", None) or 1)
        config.intermediate_size = int(config.intermediate_size) * top_k


class Gemma4Eagle3DraftModel(LlamaEagle3DraftModel):
    """EAGLE-3 draft model for Gemma4 targets.

    Identical to :class:`LlamaEagle3DraftModel` except that the Gemma4 text
    config is normalized (activation key + flattened global RoPE) before the
    shared draft is constructed. See the module docstring for the rationale;
    the on-disk checkpoint is byte-for-byte a standard ``LlamaEagle3DraftModel``
    export so SGLang / vLLM load it with their existing EAGLE-3 Llama head.
    """

    def __init__(self, config: PretrainedConfig):
        _normalize_gemma4_draft_config(config)
        super().__init__(config)
