# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Llama-style dense LLM draft model for EAGLE-3 / EAGLE-3.1 training.

The implementation is config-driven and supports any HuggingFace dense
decoder-only architecture whose layout matches Llama: GQA attention with
optional Q/K/V/O bias (`config.attention_bias`), SwiGLU MLP with optional
bias (`config.mlp_bias`), RMSNorm, and rotary position embeddings parameterized
by `config.rope_theta` / `config.rope_scaling`. This currently covers Llama,
Phi-3, and Qwen3 dense (Phi-3 omits `attention_bias` / `mlp_bias`, which
the attention and MLP layers already read via
`getattr(config, "<field>", False)`; Qwen3 decouples `head_dim` from
`hidden_size / num_attention_heads`, which the attention layer reads via
`getattr(config, "head_dim", ...)`).

Class names and the public `architectures` string remain ``LlamaEagle3*`` for
backward compatibility with already-trained checkpoints and with SGLang's
``LlamaForCausalLMEagle3.load_weights`` (the saved state dict layout is
unchanged):

    model.embed_tokens.weight
    model.fc.weight
    model.layers.0.input_layernorm.weight
    model.layers.0.hidden_norm.weight
    model.layers.0.post_attention_layernorm.weight
    model.layers.0.self_attn.{q,k,v,o}_proj.weight
    model.layers.0.mlp.{gate,up,down}_proj.weight
    model.norm.weight
    lm_head.weight

SGLang merges ``q_proj/k_proj/v_proj`` into a single ``qkv_proj`` and
``gate_proj/up_proj`` into ``gate_up_proj`` via its ``stacked_params_mapping``
at load time, so the un-fused storage above is the canonical on-disk format.

EAGLE-3.1 introduces two optional drafter-side toggles that together address
the "attention drift" failure mode observed when speculation depth grows:

* ``config.fc_norm`` (bool, default False) -- when True, an
  ``nn.ModuleList`` of ``num_aux_hidden_states`` independent RMSNorms (each
  of size ``target_hidden_size``) is applied per chunk before the
  concatenated auxiliary hidden states enter ``model.fc``. The on-disk keys
  are ``model.fc_norm.0.weight``, ``model.fc_norm.1.weight``, ...; the
  module layout matches vLLM's EAGLE-3.1 integration in PR
  https://github.com/vllm-project/vllm/pull/42764 so checkpoints trained
  here load directly into vLLM / SGLang.
* ``config.norm_output`` (bool, default False) -- when True, the existing
  final RMSNorm (``model.norm``) is applied to the per-step hidden state
  returned by ``forward`` so that the next TTT step (and the lm_head)
  consume the post-norm state instead of the raw decoder output. Adds no
  new parameters.

Both flags default to False so EAGLE-3 checkpoints continue to load and
behave identically. Enabling them applies the EAGLE-3.1 drafter toggles to
the Llama-style draft used here; the MLA-backbone Kimi K2.6 draft
(``Eagle3DeepseekV2ForCausalLM`` in ``lightseekorg/kimi-k2.6-eagle3.1-mla``)
is a separate architecture and is not covered by this module.

P-EAGLE (parallel-drafting EAGLE-3) adds one further optional toggle:

* ``config.parallel_drafting`` (bool, default False) -- when True, the draft
  registers a single learnable ``mask_hidden`` placeholder of shape
  ``[1, 1, num_aux_hidden_states * target_hidden_size]`` (the pre-``fc``
  concatenated-aux dimension) and exposes :meth:`LlamaEagle3DraftModel.forward_peagle`,
  a single parallel forward over a flat, COD-subsampled sequence with a
  ``flex_attention`` cross-depth mask (see ``peagle_attention.py`` /
  ``peagle_data.py``). The trainer feeds the ``mask_hidden`` placeholder --
  projected through the same ``project_hidden_states`` path as real aux states --
  at every masked depth (``>= 1``), together with the masked token
  ``config.mask_token_id``, so the draft predicts all ``config.num_depths`` tokens
  in one forward instead of autoregressively. The shape, the on-disk key
  ``mask_hidden``, and the COD config (``num_depths`` / ``down_sample_ratio`` /
  ``mask_token_id``) mirror speculators
  (https://github.com/vllm-project/speculators/pull/480) so the checkpoint loads
  into vLLM's parallel-drafting runtime unchanged. The masked token slot reuses
  ``embed_tokens[config.mask_token_id]``. SGLang does not serve a P-EAGLE head
  today (https://github.com/sgl-project/sglang/issues/23171). The flag only ever
  adds the ``mask_hidden`` key, so EAGLE-3 / EAGLE-3.1 checkpoints round-trip
  unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from nemo_automodel.components.datasets.llm.packed_sequence import build_block_causal_additive_mask
from nemo_automodel.components.models.common import initialize_rms_norm_module
from nemo_automodel.components.models.llama.rope_utils import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from nemo_automodel.components.speculative.eagle.peagle_draft import (
    _PeagleAttentionMixin,
    _PeagleDecoderLayerMixin,
    _PeagleDraftMixin,
    _PeagleVanillaLayerMixin,
)
from nemo_automodel.shared.import_utils import safe_import_from

logger = logging.getLogger(__name__)


def _load_flash_attn_func() -> tuple[bool, object | None, object | None]:
    """Best-effort load of flash-attn without breaking eager-only users.

    ``safe_import_from`` already handles missing modules and missing symbols, but
    some broken ``flash-attn`` installs fail with lower-level loader errors
    (e.g. ABI / shared-library issues) that should not prevent importing this
    module for the eager path. Returns the dense ``flash_attn_func`` and the
    ``flash_attn_varlen_func`` (used by the packed block-causal path).
    """
    try:
        has_fa, flash_attn_func = safe_import_from("flash_attn", "flash_attn_func")
        _, flash_attn_varlen_func = safe_import_from("flash_attn", "flash_attn_varlen_func")
    except Exception as exc:  # pragma: no cover - depends on local flash-attn loader failures.
        logger.warning("Failed to import flash_attn.flash_attn_func; FlashAttention-2 path will be disabled: %s", exc)
        return False, None, None
    if not has_fa:
        return False, None, None
    return True, flash_attn_func, flash_attn_varlen_func


_HAS_FA, _flash_attn_func, _flash_attn_varlen_func = _load_flash_attn_func()

_SUPPORTED_ATTN_IMPLEMENTATIONS = ("eager", "flash_attention_2")


def _build_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a standard causal + padding mask for SDPA/eager attention."""
    batch_size, seq_len = attention_mask.shape
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=attention_mask.device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    expanded = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return causal + expanded


def _is_right_padded_attention_mask(attention_mask: torch.Tensor) -> bool:
    """Return True when each row is a contiguous valid-prefix followed by padding."""
    mask_bool = attention_mask.to(dtype=torch.bool)
    return not bool((mask_bool[:, 1:] & ~mask_bool[:, :-1]).any())


def _seq_lens_to_cu_seqlens(seq_lens: torch.Tensor, seq_length: int) -> tuple[torch.Tensor, int]:
    """Build FlashAttention varlen ``cu_seqlens`` (int32) from packed ``seq_lens``.

    Documents are flattened row-major to match the varlen attention's
    ``reshape(B*T, ...)`` token order. Returns ``(cu_seqlens, max_seqlen)``.
    """
    doc_lens = seq_lens[seq_lens > 0]
    cu_seqlens = torch.zeros(doc_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens[1:] = torch.cumsum(doc_lens, dim=0).to(torch.int32)
    expected_total = seq_lens.shape[0] * seq_length
    if int(cu_seqlens[-1].item()) != expected_total:
        raise ValueError(
            f"Packed seq_lens sum to {int(cu_seqlens[-1].item())} but expected B*T={expected_total}; "
            "each row's document lengths (with trailing padding folded into the last document) "
            "must sum to the packed sequence length."
        )
    # ``doc_lens`` is non-empty here: an all-zero ``seq_lens`` would sum to 0,
    # which the ``expected_total`` check above already rejects.
    max_seqlen = int(doc_lens.max().item())
    return cu_seqlens, max_seqlen


class Eagle3LlamaAttention(_PeagleAttentionMixin, nn.Module):
    """EAGLE-3 draft attention over ``[input_emb, hidden]`` 2H features.

    Driven through a shared ``cache_hidden = [K_list, V_list]`` pair. At
    step ``k`` (0-indexed), with ``K_list`` and ``V_list`` already holding
    entries from steps ``0..k-1``:

    1. ``step_idx = len(K_list)`` (equal to ``k``) gives the rotary phase
       shift, so the draft's ``K_k`` encodes "this is ``k`` tokens into
       the future". The shifted ``cos`` / ``sin`` are computed from
       ``position_ids + step_idx``.
    2. The freshly projected K, V (after GQA expansion) are appended to
       the cache lists in place.
    3. The attention output is the EAGLE-3 mixed pattern:

       ``attn_weights = [ Q @ K_0^T / sqrt(d) + mask ]  ||  diag_1  ||  ...  ||  diag_k``

       where ``diag_i[t] = (Q_t * K_i_t).sum(-1) / sqrt(d)``. The softmax
       is taken over the full extended column axis of length ``T + k``.
       Output is

       ``out = attn_probs[..., :T] @ V_0  +  sum_{i=1..k} attn_probs[..., T+i-1, None] * V_i``.

       In English: Q at position ``t`` attends to all K_0 positions (the
       regular ``T x T`` causal block), and additionally to the *same*
       position ``t`` in each previous draft step ``i >= 1``.
       Implementation-wise we replace SpecForge ``llama3_eagle.py``'s
       two ``O(k^2)`` ``cat`` / ``add`` Python loops with single
       vectorized ``einsum`` calls.

    ``cache_hidden`` is mutated in place; callers are responsible for
    re-initializing it to ``[[], []]`` at the start of each training
    batch.
    """

    def __init__(self, config: PretrainedConfig, fuse_input: bool = True):
        super().__init__()
        self.config = config
        # ``fuse_input`` toggles the q/k/v input width. The EAGLE-3 first layer
        # consumes the concatenated ``[embed, hidden]`` (2H); P-EAGLE's deeper
        # layers (layer_id >= 1) are vanilla Llama layers on plain hidden (H).
        self.fuse_input = fuse_input
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        # Read only the explicit ``attn_implementation`` field; HF's private
        # ``_attn_implementation`` is owned by ``PreTrainedModel`` and may be
        # auto-set to "sdpa"/"flash_attention_2" by HF before this module
        # supports those backends.
        attn_impl = getattr(config, "attn_implementation", None) or "eager"
        if attn_impl not in _SUPPORTED_ATTN_IMPLEMENTATIONS:
            raise ValueError(
                f"Eagle3LlamaAttention: unsupported attn_implementation={attn_impl!r}; "
                f"expected one of {_SUPPORTED_ATTN_IMPLEMENTATIONS}"
            )
        if attn_impl == "flash_attention_2" and not _HAS_FA:
            raise ImportError(
                "Eagle3LlamaAttention: attn_implementation='flash_attention_2' requires the "
                "'flash-attn' package to be installed."
            )
        self.attn_implementation = attn_impl

        in_features = config.hidden_size * 2 if fuse_input else config.hidden_size
        self.q_proj = nn.Linear(
            in_features, self.num_heads * self.head_dim, bias=getattr(config, "attention_bias", False)
        )
        self.k_proj = nn.Linear(
            in_features, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False)
        )
        self.v_proj = nn.Linear(
            in_features, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False)
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=getattr(config, "attention_bias", False)
        )
        self.rotary_emb = LlamaRotaryEmbedding(config)
        # Set by attach_eagle3_cp_attention when the draft runs under context
        # parallelism: the sequence is sharded across this cp group and the mixed
        # causal/TTT-diagonal attention runs as a differentiable ring. None otherwise.
        self._cp_group = None
        # When True the cp ring uses the load-balanced zig-zag layout (inputs must
        # be sharded in zig-zag order); False uses the contiguous ring.
        self._cp_zigzag = False

    def _project_qkv(self, combined_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = combined_states.shape
        q = self.q_proj(combined_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(combined_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(combined_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v

    def _repeat_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        return k, v

    def forward(
        self,
        combined_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = combined_states.shape
        q, k, v = self._project_qkv(combined_states)

        # ``step_idx`` is the cache length BEFORE this step's append; it
        # equals the 0-indexed TTT step number and doubles as the rotary
        # phase shift. After the append below the cache holds
        # ``step_idx + 1`` entries (indices ``0..step_idx``). On the first
        # call ``cache_hidden = [[], []]`` so ``step_idx = 0`` and the
        # diagonal-extension blocks below collapse to a plain causal
        # attention, equivalent to the non-cached path.
        cache_k, cache_v = cache_hidden
        step_idx = len(cache_k)

        cos, sin = self.rotary_emb(combined_states, position_ids + step_idx)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = self._repeat_kv(k, v)
        cache_k.append(k)
        cache_v.append(v)

        if self._cp_group is not None:
            attn_output = self._cp_ring_attention_forward(q, cache_k, cache_v, batch_size, seq_len)
        elif self.attn_implementation == "flash_attention_2":
            attn_output = self._flash_attention_forward(
                q, cache_k, cache_v, step_idx, batch_size, seq_len, cu_seqlens, max_seqlen
            )
        else:
            attn_output = self._eager_attention_forward(
                q, cache_k, cache_v, attention_mask, step_idx, batch_size, seq_len
            )
        return self.o_proj(attn_output)

    def _cp_ring_attention_forward(
        self,
        q: torch.Tensor,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Context-parallel counterpart of ``_eager_attention_forward``.

        The sequence is sharded across ``self._cp_group``; block-0 (``Q @ K_0^T``,
        causal over the full sequence) runs as a ring while the per-position TTT
        diagonals stay local. Inputs are ``[B, H, T_local, D]``; the ring works in
        FlashAttention ``[B, T_local, H, D]`` layout.
        """
        from nemo_automodel.components.distributed.ring_attention import (
            cached_ring_attention,
            cached_zigzag_ring_attention,
        )

        # RoPE upcasts q/k to fp32; the ring uses FlashAttention (fp16/bf16 only), so
        # cast to the module's compute dtype (matches the plain FA2 path).
        dt = self.o_proj.weight.dtype
        qf = q.transpose(1, 2).contiguous().to(dt)
        ckf = [t.transpose(1, 2).contiguous().to(dt) for t in cache_k]
        cvf = [t.transpose(1, 2).contiguous().to(dt) for t in cache_v]
        ring = cached_zigzag_ring_attention if self._cp_zigzag else cached_ring_attention
        out = ring(qf, ckf, cvf, self._cp_group, self.scaling)  # [B, T, H, D]
        return out.reshape(batch_size, seq_len, -1)

    def _eager_attention_forward(
        self,
        q: torch.Tensor,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        attention_mask: torch.Tensor,
        step_idx: int,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        # Block 1: full T x T causal attention against the step-0 keys.
        k0, v0 = cache_k[0], cache_v[0]
        attn_weights = torch.matmul(q, k0.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask  # [B, 1, T, T] additive mask

        # Block 2: diagonal extensions for cached later steps ``i = 1..step_idx``.
        # Each contributes one column ``(Q_t * K_i_t).sum(-1) / sqrt(d)``,
        # i.e. Q at position ``t`` attends only to position ``t`` of
        # ``K_i``. Replaces SpecForge's ``O(k^2)`` cat-in-loop with a
        # single ``einsum`` + single ``cat``.
        if step_idx >= 1:
            later_k = torch.stack(cache_k[1:], dim=0)  # [step_idx, B, H, T, D]
            diag = torch.einsum("bhtd,sbhtd->bhts", q, later_k) * self.scaling
            attn_weights = torch.cat((attn_weights, diag), dim=-1)

        # Block 3: softmax over the extended ``T + step_idx`` key axis.
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)

        # Block 4: output =
        #   ``attn_probs[..., :T] @ V_0``  (regular T x T block)
        # + ``sum_{i=1..step_idx} attn_probs[..., T+i-1, None] * V_i``
        # Same fusion as Block 2 -- one ``einsum`` instead of an O(k^2)
        # accumulator loop.
        attn_output = torch.matmul(attn_probs[..., :seq_len], v0)
        if step_idx >= 1:
            later_v = torch.stack(cache_v[1:], dim=0)  # [step_idx, B, H, T, D]
            diag_probs = attn_probs[..., seq_len:]  # [B, H, T, step_idx]
            attn_output = attn_output + torch.einsum("bhts,sbhtd->bhtd", diag_probs, later_v)

        return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        step_idx: int,
        batch_size: int,
        seq_len: int,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """EAGLE-3 attention via FlashAttention-2 for the T x T causal block.

        FA2 covers Block 1 (causal attention against ``K_0``) and returns its
        log-sum-exp. The diagonal Block 2 (cached steps ``i >= 1``) is computed
        eagerly and merged via the log-space identity
        ``lse_full = logaddexp(lse_fa, logsumexp(diag))``: the FA output is scaled
        by ``exp(lse_fa - lse_full)`` and each diagonal by ``exp(diag - lse_full)``.

        With ``cu_seqlens`` (packing), Block 1 uses ``flash_attn_varlen_func`` for
        document-level causal attention; the position-wise Block 2 is unchanged.
        """
        # FA2 expects (B, T, H, D); eager cache is (B, H, T, D).
        k0, v0 = cache_k[0], cache_v[0]
        q_fa = q.transpose(1, 2).contiguous()
        k0_fa = k0.transpose(1, 2).contiguous()
        v0_fa = v0.transpose(1, 2).contiguous()
        if cu_seqlens is not None:
            attn_output_bhtd, lse_fa = self._flash_block1_varlen(
                q_fa, k0_fa, v0_fa, cu_seqlens, max_seqlen, batch_size, seq_len
            )
        else:
            # ``softmax_lse`` is fp32 with shape (B, H, T): the log-sum-exp of the
            # SCALED Block-1 logits (the FA kernel folds in ``softmax_scale``).
            out_fa, lse_fa, _ = _flash_attn_func(
                q_fa,
                k0_fa,
                v0_fa,
                softmax_scale=self.scaling,
                causal=True,
                return_attn_probs=True,
            )
            # FA output is (B, T, H, D); bring back to (B, H, T, D) for the merge.
            attn_output_bhtd = out_fa.transpose(1, 2)

        if step_idx >= 1:
            # Diagonal logits use the same scaling as FA, so the LSEs are commensurate.
            later_k = torch.stack(cache_k[1:], dim=0)  # [step_idx, B, H, T, D]
            diag_logits = torch.einsum("bhtd,sbhtd->bhts", q, later_k) * self.scaling

            lse_fa_f32 = lse_fa.float()  # [B, H, T]
            diag_f32 = diag_logits.float()  # [B, H, T, step_idx]
            diag_lse = torch.logsumexp(diag_f32, dim=-1)  # [B, H, T]
            lse_full = torch.logaddexp(lse_fa_f32, diag_lse)  # [B, H, T]

            w1 = torch.exp(lse_fa_f32 - lse_full).to(q.dtype)  # [B, H, T]
            w2 = torch.exp(diag_f32 - lse_full.unsqueeze(-1)).to(q.dtype)  # [B, H, T, step_idx]

            attn_output_bhtd = attn_output_bhtd * w1.unsqueeze(-1)
            later_v = torch.stack(cache_v[1:], dim=0)  # [step_idx, B, H, T, D]
            attn_output_bhtd = attn_output_bhtd + torch.einsum("bhts,sbhtd->bhtd", w2, later_v)

        return attn_output_bhtd.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def _flash_block1_varlen(
        self,
        q_fa: torch.Tensor,
        k0_fa: torch.Tensor,
        v0_fa: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Document-level causal Block 1 via ``flash_attn_varlen_func``.

        Flattens ``(B, T, H, D)`` to varlen ``(total_tokens, H, D)`` and reshapes
        outputs back to ``[B, H, T, D]`` / ``[B, H, T]`` for the dense-path merge.
        Note varlen ``softmax_lse`` is ``[H, total_tokens]`` (head-major), unlike
        the dense ``[B, H, T]`` -- hence the explicit reshape + shape check.
        """
        if _flash_attn_varlen_func is None:
            raise ImportError(
                "Eagle3LlamaAttention: packed FlashAttention-2 requires flash_attn.flash_attn_varlen_func."
            )
        num_heads, head_dim = q_fa.shape[2], q_fa.shape[3]
        total_tokens = batch_size * seq_len
        q_flat = q_fa.reshape(total_tokens, num_heads, head_dim)
        k0_flat = k0_fa.reshape(total_tokens, num_heads, head_dim)
        v0_flat = v0_fa.reshape(total_tokens, num_heads, head_dim)
        out_flat, lse_flat, _ = _flash_attn_varlen_func(
            q_flat,
            k0_flat,
            v0_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=True,
            return_attn_probs=True,
        )
        # out_flat: [total_tokens, H, D] -> [B, H, T, D]
        attn_output_bhtd = out_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        # lse_flat: [H, total_tokens] -> [B, H, T]
        if lse_flat.shape != (num_heads, total_tokens):
            raise RuntimeError(
                f"Unexpected varlen softmax_lse shape {tuple(lse_flat.shape)}; "
                f"expected {(num_heads, total_tokens)}. Verify the installed flash-attn version."
            )
        lse_fa = lse_flat.transpose(0, 1).reshape(batch_size, seq_len, num_heads).permute(0, 2, 1)
        return attn_output_bhtd, lse_fa


def attach_eagle3_cp_attention(model: nn.Module, cp_group, zigzag: bool = False) -> None:
    """Route every EAGLE-3 draft attention through the context-parallel ring path.

    Sets ``_cp_group`` on each :class:`Eagle3LlamaAttention` so its forward runs the
    differentiable causal-ring + TTT-diagonal attention over the cp-sharded sequence.
    A no-op ``cp_group`` of size 1 leaves the plain per-rank path in place. With
    ``zigzag=True`` the load-balanced zig-zag ring is used (the inputs must then be
    sharded in zig-zag order); otherwise the contiguous ring.

    Raises if the draft has no :class:`Eagle3LlamaAttention` to route (e.g. the
    DeepSeek MLA draft): the trainer would still shard/shift the sequence and
    renormalize the loss while each rank's attention silently saw only its own
    shard, so context parallelism must be refused rather than run wrong.
    """
    matched = 0
    for module in model.modules():
        if isinstance(module, Eagle3LlamaAttention):
            module._cp_group = cp_group
            module._cp_zigzag = zigzag
            matched += 1
    if matched == 0:
        raise NotImplementedError(
            "Context parallelism (cp_size>1) for the EAGLE-3 draft is only implemented for the "
            f"Eagle3LlamaAttention ring path; {type(model).__name__} exposes no such attention "
            "module (e.g. the DeepSeek MLA draft). Set cp_size=1 for this draft architecture."
        )


class Eagle3LlamaMLP(nn.Module):
    """Standard Llama-style SwiGLU MLP on hidden-size activations."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        from transformers.activations import ACT2FN

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=getattr(config, "mlp_bias", False)
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=getattr(config, "mlp_bias", False)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Eagle3LlamaDecoderLayer(_PeagleDecoderLayerMixin, nn.Module):
    """Single decoder layer used by the minimal EAGLE-3 draft model.

    Attribute names mirror SGLang's ``LlamaDecoderLayer`` in
    ``sglang/srt/models/llama_eagle3.py``: ``input_layernorm`` is applied
    to the per-step token embeddings (``embeds`` in SGLang),
    ``hidden_norm`` is applied to the carried hidden state.
    ``is_input_layer`` is the layer-0 flag that gates the ``[embeds,
    hidden]`` concatenation (always true for our single-layer draft).
    """

    def __init__(self, config: PretrainedConfig, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.is_input_layer = layer_id == 0
        self.input_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.hidden_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)
        self.post_attention_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.self_attn = Eagle3LlamaAttention(config)
        self.mlp = Eagle3LlamaMLP(config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_input_embeds = self.input_layernorm(input_embeds)
        norm_hidden_states = self.hidden_norm(hidden_states)
        combined_states = torch.cat((norm_input_embeds, norm_hidden_states), dim=-1)
        hidden_states = residual + self.self_attn(
            combined_states,
            attention_mask,
            position_ids,
            cache_hidden=cache_hidden,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Eagle3LlamaPeagleLayer(_PeagleVanillaLayerMixin, nn.Module):
    """Vanilla Llama decoder layer for P-EAGLE depths ``>= 1``.

    The EAGLE-3 first layer (:class:`Eagle3LlamaDecoderLayer`) fuses the token
    embedding and the projected target hidden state (``2H`` attention input).
    P-EAGLE stacks ``num_hidden_layers`` layers; every layer after the first is
    a standard Llama block operating on plain hidden states (``H``), matching
    speculators' ``decoder_layer_class`` (a vanilla ``LlamaDecoderLayer``). Only
    the P-EAGLE flex-attention path is implemented (these deeper layers do not
    participate in the EAGLE-3 ``cache_hidden`` TTT recurrence).
    """

    def __init__(self, config: PretrainedConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.self_attn = Eagle3LlamaAttention(config, fuse_input=False)
        self.mlp = Eagle3LlamaMLP(config)


class Eagle3LlamaModel(nn.Module):
    """Inner backbone matching SGLang's ``LlamaModel`` in ``llama_eagle3.py``.

    Owns ``embed_tokens``, the ``fc`` projection from concatenated target
    aux hidden states to draft hidden size, the (single-element) draft
    ``layers`` ModuleList, and the final ``norm``. The ``LlamaEagle3DraftModel``
    wrapper around this module adds the top-level ``lm_head`` and the
    training-facing public API.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        # SGLang uses ``num_aux_hidden_states`` (default 3) to size ``fc``'s
        # input dim. We mirror that convention so the weight shape is
        # identical and the key ``model.fc.weight`` round-trips cleanly.
        num_aux_hidden_states = getattr(config, "num_aux_hidden_states", 3)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.fc = nn.Linear(target_hidden_size * num_aux_hidden_states, config.hidden_size, bias=False)
        # EAGLE-3.1 ``fc_norm``: one RMSNorm of size ``target_hidden_size`` per
        # auxiliary hidden-state chunk, applied *before* the chunks are
        # concatenated and fed into ``fc``. The norms are independent
        # parameters (``nn.ModuleList``, NOT a single shared scale), which
        # matches the upstream vLLM EAGLE-3.1 implementation
        # (https://github.com/vllm-project/vllm/pull/42764) and the community
        # checkpoint at ``lightseekorg/kimi-k2.6-eagle3.1-mla``. The on-disk
        # keys are therefore ``model.fc_norm.0.weight``,
        # ``model.fc_norm.1.weight``, ... (one per chunk). The module is only
        # registered when ``fc_norm`` is set so EAGLE-3 checkpoints continue
        # to round-trip with no extra keys.
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    initialize_rms_norm_module("torch", target_hidden_size, eps=config.rms_norm_eps, device=None)
                    for _ in range(num_aux_hidden_states)
                ]
            )
        # EAGLE-3 / EAGLE-3.1 TTT uses a single fused first layer. P-EAGLE stacks
        # ``num_hidden_layers`` layers: the fused first layer (2H) plus
        # ``num_hidden_layers - 1`` vanilla Llama layers (H), matching speculators'
        # ``[first_layer] + [decoder_layer for i in range(1, num_layers)]``. The
        # single-layer construction is preserved for the non-parallel path so the
        # merged EAGLE checkpoints round-trip unchanged.
        layers: list[nn.Module] = [Eagle3LlamaDecoderLayer(config, layer_id=0)]
        if getattr(config, "parallel_drafting", False):
            num_layers = max(1, int(getattr(config, "num_hidden_layers", 1)))
            layers.extend(Eagle3LlamaPeagleLayer(config, layer_id=i) for i in range(1, num_layers))
        self.layers = nn.ModuleList(layers)
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)


class LlamaEagle3DraftModel(_PeagleDraftMixin, PreTrainedModel):
    """Llama-style dense EAGLE-3 draft model (Llama, Phi-3, Qwen3).

    State dict keys match SGLang's ``LlamaForCausalLMEagle3`` so the saved
    checkpoint can be loaded by SGLang's inference engine without any
    remapping (SGLang's ``load_weights`` fuses ``q/k/v_proj`` into
    ``qkv_proj`` and ``gate/up_proj`` into ``gate_up_proj`` via its
    standard ``stacked_params_mapping``).

    The class name is retained for checkpoint-architectures compatibility; the
    implementation is config-driven and works for any HF dense decoder-only
    config that exposes ``hidden_size``, ``num_attention_heads``,
    ``num_key_value_heads``, ``attention_bias``, ``mlp_bias``, ``rope_theta``,
    and ``rms_norm_eps``. A decoupled ``head_dim`` is read via
    ``getattr(config, "head_dim", ...)`` in the attention layer.

    Scope:
    - single draft decoder layer
    - no KV-cache optimization
    - no speculative runtime integration
    """

    config_class = PretrainedConfig
    base_model_prefix = "model"
    # Declare the attention backends this draft actually implements so
    # ``PreTrainedModel.__init__`` allows them. ``Eagle3LlamaAttention`` supports
    # ``eager`` and ``flash_attention_2`` (see ``_SUPPORTED_ATTN_IMPLEMENTATIONS``)
    # but NOT SDPA; without this flag transformers defaults ``_supports_flash_attn``
    # to ``False`` and rejects ``attn_implementation="flash_attention_2"``.
    _supports_flash_attn = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        # Activation checkpointing for the P-EAGLE draft layers (training-only
        # memory knob; toggled via ``gradient_checkpointing_enable`` and read by
        # ``_PeagleDraftMixin.forward_peagle``). Off by default; never serialized.
        self.gradient_checkpointing = False

        self.model = Eagle3LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        # Persistent d2t/t2d vocab-remap buffers, serialized into
        # ``model.safetensors`` for vLLM/SGLang to consume at inference (see
        # :meth:`set_vocab_mapping` for the mapping and why it matters). Created
        # only when the draft vocab is compressed (``draft_vocab_size <
        # vocab_size``); otherwise the draft logits are already in target space
        # and inference engines expect no mapping. Populated by
        # :meth:`set_vocab_mapping` during training, or restored from a
        # checkpoint on resume / inference.
        self.has_vocab_compression = self.draft_vocab_size < config.vocab_size
        if self.has_vocab_compression:
            self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long), persistent=True)
            self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool), persistent=True)

        # P-EAGLE (parallel drafting): register the learnable ``mask_hidden``
        # placeholder only when enabled so EAGLE-3 / EAGLE-3.1 checkpoints
        # round-trip with no extra keys (see ``_PeagleDraftMixin``).
        if getattr(config, "parallel_drafting", False):
            self._init_peagle_parameters(config)

        self.post_init()

    def copy_embeddings_from_target(self, target_embedding: nn.Embedding) -> None:
        """Initialize draft embeddings from the target model embeddings.

        When the target model is wrapped with FSDP2, ``target_embedding.weight``
        is a ``DTensor`` sharded across ranks.  The draft embedding is a plain
        ``nn.Parameter`` (the draft is not FSDP-wrapped), so a direct
        ``copy_`` of a DTensor into a regular tensor raises a mixed-type
        distributed-operator error.  Gather to a full local tensor first.
        """
        target_weight = target_embedding.weight
        if hasattr(target_weight, "full_tensor"):
            target_weight = target_weight.full_tensor()
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(target_weight)

    def set_vocab_mapping(self, selected_token_ids: torch.Tensor) -> None:
        """Populate the ``d2t`` / ``t2d`` vocab-remap buffers from the draft->target id map.

        ``selected_token_ids`` has shape ``[draft_vocab_size]``; entry ``i`` is
        the *target* vocab id of draft id ``i`` (the frequency-pruned mapping
        built by ``build_eagle3_token_mapping``). This writes the two buffers
        inference engines consume:

        - ``d2t[i] = selected_token_ids[i] - i`` -- the offset form vLLM expects
          (``target_id = draft_id + d2t[draft_id]``);
        - ``t2d[target_id] = True`` for every selected target id -- the boolean
          presence mask SGLang consumes.

        These must be in the saved checkpoint: without them vLLM/SGLang find no
        mapping, silently align draft ids to the first ``draft_vocab_size``
        target ids, and acceptance rate collapses.

        No-op when the draft vocab is not compressed (the buffers do not exist
        and the draft logits are already in target space).
        """
        if not self.has_vocab_compression:
            return
        selected = selected_token_ids.reshape(-1).to(dtype=torch.long, device=self.d2t.device)
        if selected.numel() != self.draft_vocab_size:
            raise ValueError(
                "set_vocab_mapping expected selected_token_ids of length "
                f"draft_vocab_size={self.draft_vocab_size}, got {selected.numel()}."
            )
        base = torch.arange(self.draft_vocab_size, dtype=torch.long, device=selected.device)
        self.d2t.copy_(selected - base)
        self.t2d.zero_()
        self.t2d[selected] = True

    def freeze_embeddings(self) -> None:
        """Freeze draft input embeddings."""
        self.model.embed_tokens.weight.requires_grad_(False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable activation checkpointing for the P-EAGLE draft layers.

        Training-only memory knob: recomputes each ``forward_peagle`` layer in the
        backward instead of storing its activations (the EAGLE-3 TTT ``forward``
        path is unaffected). ``gradient_checkpointing_kwargs`` is accepted for
        HF-API parity but ignored -- recompute is always non-reentrant, the only
        mode compatible with the non-tensor ``block_mask``.
        """
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing for the P-EAGLE draft layers."""
        self.gradient_checkpointing = False

    def project_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target aux states from ``num_aux * H_target`` to draft hidden size.

        When ``config.fc_norm`` is set (EAGLE-3.1), the input is split into
        ``num_aux_hidden_states`` equal chunks along the last dim and each
        chunk is passed through its own RMSNorm in ``model.fc_norm`` (the
        modules are independent, matching vLLM's upstream implementation).
        The normalized chunks are then re-concatenated and fed to ``fc``,
        stabilising the per-aux-state scale before the projection mixes them
        and removing the speculation-depth drift observed with raw inputs.
        """
        if getattr(self.config, "fc_norm", False):
            num_aux = len(self.model.fc_norm)
            chunks = aux_hidden_states.chunk(num_aux, dim=-1)
            aux_hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)],
                dim=-1,
            )
        return self.model.fc(aux_hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids with the draft embedding table."""
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute draft logits on the configured draft vocabulary.

        With ``config.norm_output`` unset (EAGLE-3 default) the input is the
        raw decoder-layer output and the final ``model.norm`` is applied
        here. With ``config.norm_output`` set (EAGLE-3.1) ``forward`` has
        already returned the post-norm state, so ``lm_head`` is applied
        directly to avoid a double normalisation.
        """
        if getattr(self.config, "norm_output", False):
            return self.lm_head(hidden_states)
        return self.lm_head(self.model.norm(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        projected_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache_hidden: Optional[list[list[torch.Tensor]]] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one full-sequence draft update step.

        ``cache_hidden`` is the EAGLE-3 TTT cache. Pass ``[[], []]`` on
        the first step of a TTT unroll and the same list object on each
        subsequent step; the attention layer appends the per-step K and V
        to it. If ``None`` is passed (e.g. from a one-shot evaluation
        call) a fresh ``[[], []]`` is allocated locally -- step 0 of TTT
        is mathematically equivalent to a plain causal forward.

        ``seq_lens`` (packing) makes Block-1 attention document-level block-causal
        (eager mask / FA2 varlen); callers must pass per-document ``position_ids``.
        """
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(input_ids.shape[0], -1)
        if cache_hidden is None:
            cache_hidden = [[], []]
        is_fa2 = self.model.layers[0].self_attn.attn_implementation == "flash_attention_2"
        cu_seqlens: torch.Tensor | None = None
        max_seqlen: int | None = None
        if seq_lens is not None:
            # Packed: structure comes from seq_lens (cu_seqlens for FA2 / block-causal
            # mask for eager), so the right-padding check below does not apply.
            seq_length = input_ids.shape[1]
            if is_fa2:
                # FA2 attends document-wise through cu_seqlens and never reads the 4D
                # additive mask, so skip materializing the [B, 1, T, T] block-causal mask.
                causal_mask = None
                cu_seqlens, max_seqlen = _seq_lens_to_cu_seqlens(seq_lens, seq_length)
            else:
                causal_mask = build_block_causal_additive_mask(
                    seq_lens, seq_length=seq_length, dtype=projected_hidden_states.dtype, device=input_ids.device
                )
        else:
            if is_fa2 and not _is_right_padded_attention_mask(attention_mask):
                raise ValueError(
                    "LlamaEagle3DraftModel: attn_implementation='flash_attention_2' requires a right-padded "
                    "attention_mask (each row must be contiguous 1s followed by 0s)."
                )
            causal_mask = _build_causal_mask(attention_mask=attention_mask, dtype=projected_hidden_states.dtype)

        draft_input_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.model.layers[0](
            input_embeds=draft_input_embeds,
            hidden_states=projected_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_hidden=cache_hidden,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # EAGLE-3.1 ``norm_output``: route the post-norm hidden state to both
        # the next TTT step (fed back via ``cur_hidden_states`` in the trainer
        # loop) and to ``compute_logits``. ``compute_logits`` detects the
        # flag and skips re-norming. With the flag unset this branch is a
        # no-op and EAGLE-3 behavior is preserved.
        if getattr(self.config, "norm_output", False):
            hidden_states = self.model.norm(hidden_states)
        return hidden_states
