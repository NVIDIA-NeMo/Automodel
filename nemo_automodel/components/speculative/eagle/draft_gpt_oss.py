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

"""EAGLE-3 draft model for gpt-oss targets (``GptOssForCausalLM``).

gpt-oss is a Mixture-of-Experts decoder whose target backbone differs from a
Llama-style dense LLM in three ways: alternating sliding-window / full
attention layers, learnable attention sinks, and YaRN-scaled RoPE. The first
two never reach the EAGLE-3 *draft*: the draft is a single from-scratch decoder
layer that consumes only the post-block auxiliary hidden states emitted by the
frozen target (via ``register_forward_hook``) and re-projects its own Q/K, so
it never sees the target's experts, sinks, or sliding mask -- structurally it
is the same Llama-style dense draft used for every other registry entry.

RoPE is the exception, and it must match the target. During speculative
decoding the draft runs at the *same token positions* as the target, and gpt-oss
is a long-context model whose rotary frequencies are reshaped by YaRN
(NTK-by-parts with a concentration scale, base 150000, ``factor=32`` to extend
4096 -> 131072). A draft trained with a different rotary schedule is
positionally inconsistent with the target: it may converge at the short context
used during training, but its acceptance rate collapses at the long positions
gpt-oss is built for, because its notion of "position p" diverges from the
target's. (SpecForge trains the gpt-oss draft from a plain ``model_type: llama``
config with standard RoPE; that is a latent bug masked by short-context
training, not a recipe to copy.)

The shared ``LlamaRotaryEmbedding`` cannot represent YaRN -- it implements only
``{"default", "llama3"}`` and silently falls back to the llama3 NTK schedule for
``rope_type="yarn"``. So this draft swaps in :class:`GptOssDraftRotaryEmbedding`,
which reproduces gpt-oss's exact YaRN ``inv_freq`` and concentration (reusing the
target's own ``components/models/gpt_oss/rope_utils.RotaryEmbedding``) but returns
``(cos, sin)`` in the duplicated ``[..., head_dim]`` layout. gpt-oss's interleaved
``apply_rotary_emb`` and the draft's ``rotate_half``-based ``apply_rotary_pos_emb``
are algebraically identical under that layout, so the draft's rotation is
bit-faithful to the target's.

Everything else (GQA, attention/MLP bias, RMSNorm, the EAGLE-3 TTT cache
attention, the ``fc`` projection, the draft ``lm_head`` and vocab mapping) is
inherited unchanged from ``LlamaEagle3DraftModel``. The on-disk state-dict layout
and the saved ``architectures: ["LlamaEagle3DraftModel"]`` string are unchanged,
so checkpoints trained here load into SGLang exactly like the Llama draft.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from nemo_automodel.components.models.common.utils import get_rope_config
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding as GptOssRotaryEmbedding
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel


class GptOssDraftRotaryEmbedding(nn.Module):
    """gpt-oss YaRN RoPE exposed through the ``LlamaRotaryEmbedding`` ``(cos, sin)`` API.

    Produces the same per-position rotary frequencies as the gpt-oss target
    (YaRN NTK-by-parts with concentration), reusing the target's own
    ``RotaryEmbedding`` so the YaRN math lives in one place. Unlike that class --
    which builds ``cos``/``sin`` of size ``rotary_dim // 2`` and applies them with
    an interleaved split -- this returns them duplicated to size ``head_dim`` so
    the draft attention's ``rotate_half``-based ``apply_rotary_pos_emb`` performs
    the identical rotation. The values in ``position_ids`` are honored (EAGLE-3
    TTT passes ``arange(seq_len) + step_idx``).
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        base, rope_parameters, partial_rotary_factor = get_rope_config(config)
        # The draft reuses the Llama attention's full-head_dim apply_rotary_pos_emb,
        # which has no partial-rotary pass-through. gpt-oss uses full rotary; guard
        # in case a variant does not so the failure is explicit, not a silent
        # shape mismatch.
        if partial_rotary_factor != 1.0:
            raise ValueError(
                f"GptOssDraftRotaryEmbedding only supports full rotary "
                f"(partial_rotary_factor=1.0), got {partial_rotary_factor}."
            )
        # Same parameter mapping the gpt-oss target uses to build its rotary
        # (see components/models/gpt_oss/model.py): factor -> scaling_factor,
        # beta_slow -> ntk_alpha, beta_fast -> ntk_beta.
        self._rope = GptOssRotaryEmbedding(
            head_dim=head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_parameters.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_parameters.get("factor", 1.0),
            ntk_alpha=rope_parameters.get("beta_slow", 1.0),
            ntk_beta=rope_parameters.get("beta_fast", 32.0),
            device=None,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concentration, inv_freq = self._rope._compute_concentration_and_inv_freq()
        inv_freq = inv_freq.to(device=x.device, dtype=torch.float32)
        # angles: [batch, seq_len, rotary_dim // 2]
        angles = position_ids[..., None].to(torch.float32) * inv_freq
        # Duplicate to head_dim so rotate_half consumes it like LlamaRotaryEmbedding.
        emb = torch.cat((angles, angles), dim=-1)
        cos = (emb.cos() * concentration).to(x.dtype)
        sin = (emb.sin() * concentration).to(x.dtype)
        return cos, sin


class GptOssEagle3DraftModel(LlamaEagle3DraftModel):
    """EAGLE-3 draft model for gpt-oss targets.

    Identical to :class:`LlamaEagle3DraftModel` except that the single draft
    layer's rotary embedding is replaced with :class:`GptOssDraftRotaryEmbedding`
    so the draft reproduces gpt-oss's YaRN RoPE instead of the YaRN-incapable
    ``LlamaRotaryEmbedding``.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        # Swap each draft layer's rotary embedding for the gpt-oss YaRN one (the
        # draft is single-layer today, but iterate so this does not silently miss
        # layers if that changes). The LlamaRotaryEmbedding built by the parent is
        # discarded here -- it has no trained parameters.
        for layer in self.model.layers:
            layer.self_attn.rotary_emb = GptOssDraftRotaryEmbedding(config)
