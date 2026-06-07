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

"""P-EAGLE (parallel-drafting) draft-model behavior, split out of ``draft_llama.py``.

The P-EAGLE forward path is provided as mixins so the EAGLE-3 draft classes in
``draft_llama.py`` opt into it by inheritance without interleaving
``parallel_drafting`` branches through the shared EAGLE-3 code:

* :class:`_PeagleAttentionMixin` -> ``Eagle3LlamaAttention``
* :class:`_PeagleDecoderLayerMixin` -> ``Eagle3LlamaDecoderLayer``
* :class:`_PeagleVanillaLayerMixin` -> ``Eagle3LlamaPeagleLayer``
* :class:`_PeagleDraftMixin` -> ``LlamaEagle3DraftModel``

The mixins reference only attributes the host classes already define (``self``)
plus the helpers in this module, so the dependency stays one-way
(``draft_llama`` -> ``peagle_draft``) with no circular import.
"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo_automodel.components.models.llama.rope_utils import apply_rotary_pos_emb
from nemo_automodel.components.speculative.eagle.peagle_attention import create_peagle_mask_mod

# flex_attention compiled for the CUDA training path. Inductor's flex backend is
# not available on CPU, and it currently requires query/key/value head dimensions
# of at least 16. ``_peagle_flex_attention`` dispatches to eager
# ``flex_attention`` for unsupported cases -- correct, just slower -- which keeps
# P-EAGLE unit tests and small CPU/GPU smoke checks runnable. The compiled
# callable is lazy, so importing this module on CPU costs nothing.
_peagle_flex_attention_compiled = torch.compile(
    flex_attention,
    mode="max-autotune-no-cudagraphs",
)


def _peagle_compile_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return whether Inductor's flex-attention lowering supports these tensors."""
    return q.is_cuda and q.shape[-1] >= 16 and k.shape[-1] >= 16 and v.shape[-1] >= 16


def _peagle_flex_attention(q, k, v, *, block_mask, scale):
    """Run the P-EAGLE flex attention, compiling only when Inductor supports it."""
    flex = _peagle_flex_attention_compiled if _peagle_compile_supported(q, k, v) else flex_attention
    return flex(q, k, v, block_mask=block_mask, scale=scale)


class _PeagleAttentionMixin:
    """P-EAGLE single parallel-group attention for ``Eagle3LlamaAttention``."""

    def forward_peagle(
        self,
        combined_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """P-EAGLE single parallel-group attention.

        Unlike the EAGLE-3 ``cache_hidden`` recurrence, P-EAGLE flattens all COD
        depths into one sequence and attends in a single pass: there is no
        per-step rotary phase offset (the depth is baked into ``position_ids =
        anchor_pos + depth``) and no diagonal-extension cache. Cross-depth
        visibility is enforced entirely by ``block_mask`` (see
        :func:`create_peagle_mask_mod`), so this is plain scaled-dot-product
        attention through ``flex_attention``.
        """
        batch_size, seq_len, _ = combined_states.shape
        q, k, v = self._project_qkv(combined_states)
        cos, sin = self.rotary_emb(combined_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = self._repeat_kv(k, v)
        attn_output = _peagle_flex_attention(q, k, v, block_mask=block_mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class _PeagleDecoderLayerMixin:
    """P-EAGLE forward for the fused first layer ``Eagle3LlamaDecoderLayer``."""

    def forward_peagle(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Decoder-layer variant for the P-EAGLE single parallel forward.

        Mirrors :meth:`forward` (same norms, residuals, MLP and ``[embeds,
        hidden]`` concatenation) but routes attention through
        ``self_attn.forward_peagle`` with a COD ``block_mask`` instead of the
        ``cache_hidden`` recurrence.
        """
        residual = hidden_states
        norm_input_embeds = self.input_layernorm(input_embeds)
        norm_hidden_states = self.hidden_norm(hidden_states)
        combined_states = torch.cat((norm_input_embeds, norm_hidden_states), dim=-1)
        hidden_states = residual + self.self_attn.forward_peagle(combined_states, position_ids, block_mask)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class _PeagleVanillaLayerMixin:
    """P-EAGLE forward for the vanilla deep layer ``Eagle3LlamaPeagleLayer``."""

    def forward_peagle(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Standard pre-norm Llama block over ``H`` hidden states with the COD mask."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn.forward_peagle(hidden_states, position_ids, block_mask)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class _PeagleDraftMixin:
    """P-EAGLE draft-model methods for ``LlamaEagle3DraftModel``."""

    def _init_peagle_parameters(self, config) -> None:
        """Register the learnable P-EAGLE ``mask_hidden`` placeholder.

        A single learnable placeholder that substitutes for the target auxiliary
        hidden states at every masked multi-token-prediction position (COD depths
        >= 1). It lives at the *pre-``fc``* concatenated-aux dimension
        (``num_aux_hidden_states * target_hidden_size`` == ``model.fc.in_features``)
        so it flows through ``project_hidden_states`` -- and ``fc_norm`` when set --
        exactly like a real aux-hidden vector. Shape ``[1, 1, 3 * H]`` and the
        on-disk key ``mask_hidden`` mirror speculators
        (https://github.com/vllm-project/speculators/pull/480) so the checkpoint
        loads into vLLM's parallel-drafting runtime unchanged. Called only when
        ``parallel_drafting`` is set so EAGLE-3 / EAGLE-3.1 checkpoints round-trip
        with no extra keys.
        """
        # Initialized with unit-variance noise to match speculators'
        # ``torch.randn(1, 1, 3 * hidden_size)`` exactly (NOT the 0.02
        # ``initializer_range`` used for ordinary weights).
        self.mask_hidden = nn.Parameter(torch.empty(1, 1, self.model.fc.in_features))
        nn.init.normal_(self.mask_hidden, mean=0.0, std=1.0)

    def masked_projected_hidden(self) -> torch.Tensor:
        """Project the learnable P-EAGLE ``mask_hidden`` placeholder to draft hidden size.

        Returns a ``[1, hidden_size]`` tensor obtained by running the
        ``[1, 1, num_aux_hidden_states * target_hidden_size]`` placeholder through
        the same ``project_hidden_states`` path (``fc`` plus optional
        ``fc_norm``) used for real auxiliary hidden states. The P-EAGLE trainer
        scatters the result into every masked COD depth. Only valid when the
        draft was built with ``config.parallel_drafting=True``.
        """
        return self.project_hidden_states(self.mask_hidden.view(1, -1))

    def forward_peagle(
        self,
        sampled_input_ids: torch.Tensor,
        sampled_projected_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Run the P-EAGLE single parallel-group forward.

        All COD depths are already flattened into one ``[1, total_sampled]``
        sequence by the caller:

        * ``sampled_input_ids`` -- real token ids at depth-0 slots, the masked
          ``mask_token_id`` at depth >= 1 slots;
        * ``sampled_projected_hidden`` -- ``fc``-projected target aux states at
          depth-0 slots, the projected ``mask_hidden`` placeholder elsewhere;
        * ``position_ids`` -- ``anchor_pos + depth`` (the reference position);
        * ``block_mask`` -- the COD cross-depth visibility mask.

        Returns the pre-logits hidden states (post-``norm`` when
        ``config.norm_output`` is set), one row per sampled element.
        """
        draft_input_embeds = self.embed_input_ids(sampled_input_ids)
        # Layer 0 fuses ``[embed, hidden]`` (2H); deeper layers refine plain H.
        hidden_states = self.model.layers[0].forward_peagle(
            input_embeds=draft_input_embeds,
            hidden_states=sampled_projected_hidden,
            position_ids=position_ids,
            block_mask=block_mask,
        )
        for layer in self.model.layers[1:]:
            hidden_states = layer.forward_peagle(hidden_states, position_ids, block_mask)
        if getattr(self.config, "norm_output", False):
            hidden_states = self.model.norm(hidden_states)
        return hidden_states

    def build_peagle_block_mask(self, anchor_pos, depth, lengths, total_seq_len):
        """Construct the COD ``flex_attention`` block mask for one sequence."""
        mask_mod = create_peagle_mask_mod(
            anchor_pos=anchor_pos, depth=depth, lengths=lengths, total_seq_len=total_seq_len
        )
        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=anchor_pos.shape[0],
            KV_LEN=anchor_pos.shape[0],
            device=anchor_pos.device,
        )
