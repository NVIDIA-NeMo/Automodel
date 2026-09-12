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

"""DFlash 2 draft model (Qwen3-style).

DFlash 2 (https://inco.ai/blog/dflash2/) keeps DFlash's one-pass parallel block
draft -- the block is still predicted in a single non-causal forward -- and adds
two cheap modules on top of it:

* A **two-tap dynamic depthwise convolution** before and after each attention and
  MLP sublayer of every draft layer::

      Conv(x)_t = k_{t,0} * x_t + k_{t,1} * x_{t-1}

  Each coefficient is a learned base kernel plus a small correction predicted
  from the current hidden state and shared across ``conv_group_size`` channels.
  The taps never cross a draft-block boundary, so block position 1 reads block
  position 0 -- the last verified (anchor) token -- and block position 0 reads
  zero padding. This moves the short-range within-block work off attention (which
  goes back to reading the target context) and removes most of DFlash's *suffix
  decay*: prediction quality at the end of the block, for ~3% extra parameters.

* A **pairwise path selector**. DFlash picks every position's top-1 candidate
  independently, so neighbours can disagree (a repeated word, a broken phrase)
  and the block is cut short at verification -- even though the right token is
  usually already in the position's candidate list. The selector keeps each
  position's top ``selector_top_k`` candidates and scores every adjacent pair in
  one shot::

      S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>

  ``U_t(b)`` is DFlash's own logit for candidate ``b``; ``A`` and ``B`` are
  rank-``selector_rank`` token codebooks matched under a context gate ``H(h_t)``
  -- a low-rank bilinear score over adjacent candidates. Scoring is fully
  parallel; only the final walk over the precomputed scores is sequential, and it
  touches no backbone or LM head.

Both modules are initialised to the identity (zero conv correction, zero
successor codebook), so a freshly constructed DFlash 2 draft starts out
numerically equal to plain DFlash and training moves it away from there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    GREEDY_TEMPERATURE_EPS,
    Qwen3DFlashDecoderLayer,
    Qwen3DFlashDraftModel,
    assert_target_supports_rollback,
    extract_context_feature,
    resolve_output_head,
    sample,
    sampling_probs,
    validate_sampling,
)


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    group_size: int,
    block_size: int,
) -> torch.Tensor:
    """Depthwise convolution over draft-block positions with content-adaptive taps.

    Tap ``offset`` reads position ``t - offset`` of the *same* draft block; the
    leading ``offset`` positions of every block read zero padding instead of the
    previous block's tail, which is what keeps the packed ``blocks * block_size``
    training layout equivalent to drafting one block at a time.

    Args:
        hidden: Tensor of shape [batch, sequence, hidden]; ``sequence`` is a whole
            number of ``block_size``-long draft blocks laid end to end.
        dynamic: Tensor of shape [batch, sequence, kernel, groups]; the per-position
            correction added to ``base_kernel``, shared by the ``group_size``
            channels of each group.
        base_kernel: Tensor of shape [kernel, hidden]; the learned base taps.
        group_size: Number of channels sharing one dynamic correction.
        block_size: Draft-block length; taps never cross a block boundary.

    Returns:
        Tensor of shape [batch, sequence, hidden]; a fresh tensor that neither
        aliases nor mutates ``hidden``.
    """
    batch, seq_len, hidden_size = hidden.shape
    groups = hidden_size // group_size
    kernel_size = base_kernel.shape[0]
    n_blocks = seq_len // block_size
    blocks = hidden.view(batch, n_blocks, block_size, groups, group_size)
    dynamic = dynamic.view(batch, n_blocks, block_size, kernel_size, groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(kernel_size):
        # Shift along the in-block position axis (dim 2); block position < offset
        # reads zeros rather than the previous block's tail.
        values = blocks if offset == 0 else F.pad(blocks[:, :, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base_kernel[offset].view(1, 1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, :, offset], values)
    return output.view(batch, seq_len, hidden_size)


class GroupedDynamicCausalConv(nn.Module):
    """Two-tap dynamic depthwise convolution wrapped around one draft sublayer.

    One instance covers the convolution *before* a sublayer (:meth:`prepare`) and
    the one *after* it (:meth:`finish`). Both sets of dynamic coefficients are
    predicted from the sublayer's input, so :meth:`prepare` returns the
    coefficients :meth:`finish` needs and the projection runs once per sublayer.

    Args:
        hidden_size: Channel count of the draft hidden states.
        kernel_size: Number of taps; DFlash 2 uses 2 (self and predecessor).
        group_size: Channels sharing one dynamic correction (16 in DFlash 2).
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int) -> None:
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}.")
        if kernel_size < 1:
            raise ValueError(f"conv_kernel_size must be >= 1, got {kernel_size}.")
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        # Index 0 is the pre-sublayer convolution, index 1 the post-sublayer one.
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the convolution to the identity: unit self-tap, no correction."""
        nn.init.zeros_(self.kernel_projection.weight)
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0, :] = 1.0

    def prepare(self, hidden: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the pre-sublayer convolution and emit the post-sublayer coefficients.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden]; the sublayer input.
            block_size: Draft-block length; taps never cross a block boundary.

        Returns:
            Tuple of ``(convolved, dynamic)``: ``convolved`` is a Tensor of shape
            [batch, sequence, hidden] to feed the sublayer, and ``dynamic`` a
            Tensor of shape [batch, sequence, kernel, groups] to pass back to
            :meth:`finish`. Neither aliases ``hidden``.
        """
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, groups)
        convolved = _grouped_dynamic_convolve(
            hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size, block_size
        )
        return convolved, dynamic[..., 1, :, :]

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor, block_size: int) -> torch.Tensor:
        """Apply the post-sublayer convolution using the coefficients from :meth:`prepare`.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden]; the sublayer output.
            dynamic: Tensor of shape [batch, sequence, kernel, groups]; the second
                half of :meth:`prepare`'s coefficients.
            block_size: Draft-block length; taps never cross a block boundary.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size, block_size)


class CandidateSelector(nn.Module):
    """Pairwise path selector over each draft position's top-k candidates.

    Scores ``S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>`` for predecessor token
    ``a`` and candidate ``b``: DFlash's own logit plus a low-rank bilinear match
    between the two tokens' codebook embeddings, gated by the draft hidden state.

    The codebooks are plain ``[vocab, rank]`` parameters rather than
    ``nn.Embedding`` modules so the saved keys are ``candidate_selector.*_codebook``
    -- the names the published DFlash 2 drafters and the serving runtimes use --
    instead of ``candidate_selector.*_codebook.weight``.

    Args:
        vocab_size: Size of the (shared with the target) token vocabulary.
        hidden_size: Channel count of the draft hidden states.
        rank: Codebook / gate width (``selector_rank``; 256 in DFlash 2).
        top_k: Candidates kept per position (``selector_top_k``; 16 in DFlash 2).
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int, top_k: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"selector_rank must be >= 1, got {rank}.")
        if not 1 <= top_k <= vocab_size:
            raise ValueError(f"selector_top_k must be in [1, vocab_size={vocab_size}], got {top_k}.")
        self.rank = rank
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the selector to a no-op: every score collapses to the draft logit.

        ``S_t`` is bilinear in the two codebooks, so collapsing it needs one factor
        at zero. Zeroing the successor codebook does that; it still receives
        gradient immediately, while the predecessor codebook and the context gate
        multiply it and therefore only start training on the second step.
        """
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=0.02)
            self.successor_codebook.zero_()

    def pair_scores(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score every (predecessor, candidate) pair for a batch of draft positions.

        Fully parallel: no position depends on another's score. Training calls this
        once with the ground-truth predecessors; the decode-time walk in
        :meth:`walk` calls it one position at a time with the token it just picked.

        Args:
            hidden: Tensor of shape [..., hidden]; the draft hidden state at each
                scored position, with arbitrary leading dimensions.
            unary: Tensor of shape [..., candidates]; ``U_t(b)``, the draft logit of
                each candidate, with the same leading dimensions as ``hidden``.
            candidate_ids: Long tensor of shape [..., candidates]; the candidate
                token ids, with the same leading dimensions as ``hidden``.
            predecessor_ids: Long tensor of shape [...]; the token preceding each
                scored position, with the same leading dimensions as ``hidden``.

        Returns:
            Tensor of shape [..., candidates] holding ``S_t(a, b)``.
        """
        gate = F.embedding(predecessor_ids, self.predecessor_codebook) * self.hidden_projection(hidden)
        pairwise = torch.einsum("...r,...kr->...k", gate, F.embedding(candidate_ids, self.successor_codebook))
        return unary + pairwise

    def walk(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Trace one coherent path through the per-position candidate lists.

        Below ``GREEDY_TEMPERATURE_EPS`` (matching ``sample``'s own greedy
        threshold -- dividing by a positive-but-tiny temperature blows the scores
        up enough that the softmax can turn to NaN) this follows the best
        successor at each step; otherwise the step is sampled from the softmax
        over the candidate scores, and the returned per-step distribution is the
        draft proposal ``q`` that :func:`dflash2_rejection_sample` needs to stay
        lossless.

        Args:
            hidden: Tensor of shape [batch, draft, hidden]; the draft hidden states
                of the block's predicted positions.
            logits: Tensor of shape [batch, draft, vocab]; the draft logits at those
                positions.
            anchor_ids: Long tensor of shape [batch]; the last verified token, i.e.
                the predecessor of draft position 0.
            temperature: Sampling temperature; below ``GREEDY_TEMPERATURE_EPS``
                selects greedily.

        Returns:
            Tuple ``(path, candidate_ids, draft_probs)``: ``path`` is a Long tensor
            of shape [batch, draft] with the selected token per position;
            ``candidate_ids`` a Long tensor of shape [batch, draft, candidates] with
            the scored candidates; ``draft_probs`` a Tensor of shape
            [batch, draft, candidates] with the per-position proposal distribution
            over those candidates, or ``None`` when decoding greedily.
        """
        unary, candidate_ids = torch.topk(logits, self.top_k, dim=-1)
        gate_hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path: list[torch.Tensor] = []
        prob_rows: list[torch.Tensor] = []
        for position in range(hidden.shape[1]):
            gate = F.embedding(predecessor, self.predecessor_codebook) * gate_hidden[:, position]
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk", gate, F.embedding(candidate_ids[:, position], self.successor_codebook)
            )
            if temperature >= GREEDY_TEMPERATURE_EPS:
                probs = torch.softmax(scores.float() / temperature, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).squeeze(-1)
                prob_rows.append(probs)
            else:
                choice = scores.argmax(dim=-1)
            predecessor = candidate_ids[:, position].gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            path.append(predecessor)
        return (
            torch.stack(path, dim=1),
            candidate_ids,
            torch.stack(prob_rows, dim=1) if prob_rows else None,
        )


def dflash2_rejection_sample(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Accept a prefix of the drafted block and resample the first rejected token.

    Standard speculative-decoding rejection sampling, specialised to a proposal
    supported on the selector's candidate set: ``q(token)`` is read out of
    ``draft_probs`` by matching ``candidate_ids``, and the residual
    ``clamp(p - q, 0)`` is formed by scattering ``-q`` into the full-vocabulary
    target distribution. The accepted tokens plus the resampled one are therefore
    distributed exactly as the target's own samples.

    Args:
        draft_tokens: Long tensor of shape [1, draft]; the selector's path.
        target_probs: Tensor of shape [1, block, vocab]; the verifier's
            next-token distribution at each block position, where
            ``block == draft + 1``.
        draft_probs: Tensor of shape [1, draft, candidates]; the proposal mass on
            each candidate.
        candidate_ids: Long tensor of shape [1, draft, candidates]; the candidate
            token ids the proposal is supported on.

    Returns:
        Tuple ``(accepted, bonus)``: ``accepted`` is the number of drafted tokens
        kept, and ``bonus`` a scalar Long tensor with the token sampled from the
        target (or from the residual) right after them.
    """
    draft_len = draft_tokens.shape[1]
    p = target_probs[:, :draft_len].gather(-1, draft_tokens[..., None])[..., 0]
    q = (draft_probs * (candidate_ids == draft_tokens[..., None])).sum(-1)
    accepted = int((torch.rand_like(q) * q < p).to(torch.int32).cumprod(-1).sum(-1)[0].item())
    if accepted == draft_len:
        return accepted, torch.multinomial(target_probs[0, -1], num_samples=1)[0]

    residual = target_probs[0, accepted].clone()
    residual.scatter_add_(0, candidate_ids[0, accepted], -draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    # A fully-covered position leaves no residual mass; fall back to the target.
    residual = torch.where(
        total > 0, residual / total.clamp_min(torch.finfo(residual.dtype).tiny), target_probs[0, accepted]
    )
    return accepted, torch.multinomial(residual, num_samples=1)[0]


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """A DFlash 2 decoder block: DFlash's block + a two-tap conv around each sublayer."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        dflash_config = getattr(config, "dflash_config", {}) or {}
        kernel_size = int(dflash_config.get("conv_kernel_size", 2))
        group_size = int(dflash_config.get("conv_group_size", 16))
        self.attention_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        conv_block_size: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Run the block, convolving the input and output of each sublayer.

        Args:
            target_hidden: Tensor of shape [batch, context, hidden]; the projected
                target-model context the attention keys/values are extended with.
            hidden_states: Tensor of shape [batch, draft, hidden]; the draft
                (noise-block) positions, ``draft`` being a whole number of
                ``conv_block_size``-long blocks.
            attention_mask: Attention mask over [batch, 1, draft, context + draft],
                a flex ``BlockMask``, or ``None``.
            position_ids: Long tensor of shape [batch, context + draft].
            past_key_value: Draft KV cache, or ``None``.
            use_cache: Whether to write into ``past_key_value``.
            cache_position: Long tensor of shape [draft], or ``None``.
            position_embeddings: Tuple of rotary ``(cos, sin)`` tensors of shape
                [batch, context + draft, head_dim].
            conv_block_size: Draft-block length; the convolutions' predecessor tap
                never crosses a block boundary.
            **kwargs: Forwarded to the attention implementation.

        Returns:
            Tensor of shape [batch, draft, hidden].
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_dynamic = self.attention_conv.prepare(hidden_states, conv_block_size)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(hidden_states, attention_dynamic, conv_block_size)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_dynamic = self.mlp_conv.prepare(hidden_states, conv_block_size)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_dynamic, conv_block_size)
        return residual + hidden_states


class Qwen3DFlash2DraftModel(Qwen3DFlashDraftModel):
    """DFlash 2 draft model: the DFlash stack plus in-block convs and a path selector."""

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]
    decoder_layer_cls = Qwen3DFlash2DecoderLayer

    def __init__(self, config) -> None:
        super().__init__(config)
        if self.projector_type is not None:
            raise ValueError(
                "DFlash 2 replaces the sequential correction head with a pairwise path selector; "
                f"dflash_config.projector_type must be unset, got {self.projector_type!r}."
            )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.candidate_selector = CandidateSelector(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            rank=int(dflash_config.get("selector_rank", 256)),
            top_k=int(dflash_config.get("selector_top_k", 16)),
        )
        self.post_init()
        # Transformers dispatches ``_init_weights`` per module and has just
        # randomised the selector codebooks and the convolutions' projection, so
        # put both back on their identity start. A zero dynamic correction and a
        # zero successor codebook make the freshly built draft numerically equal
        # to plain DFlash: the convolutions pass their input through and every
        # selector score collapses to DFlash's own logit. Both still receive
        # gradient, so training moves them off that starting point.
        for module in self.modules():
            if isinstance(module, (GroupedDynamicCausalConv, CandidateSelector)):
                module.reset_parameters()

    def resolve_conv_block_size(self, query_len: int, conv_block_size: int | None) -> int:
        """Resolve the block length the in-block convolutions must not reach across.

        Args:
            query_len: Number of draft (noise-block) query positions in this call.
            conv_block_size: Explicit block length, or ``None`` to infer it.

        Returns:
            The block length to convolve within. ``None`` resolves to
            ``config.block_size`` when it divides ``query_len`` -- the trainer packs
            ``blocks * block_size`` query positions -- and to ``query_len`` otherwise,
            which is the decode-time case of a single, possibly truncated, block.

        Raises:
            ValueError: If ``conv_block_size`` does not divide ``query_len``.
        """
        if conv_block_size is None:
            return self.block_size if query_len % self.block_size == 0 else query_len
        if conv_block_size < 1 or query_len % conv_block_size != 0:
            raise ValueError(
                f"conv_block_size={conv_block_size} must be >= 1 and divide the {query_len} draft query positions."
            )
        return conv_block_size

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        conv_block_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the DFlash 2 draft stack over ``[context | noise-block]``.

        Args:
            position_ids: Long tensor of shape [batch, context + draft].
            attention_mask: Attention mask over [batch, 1, draft, context + draft],
                a flex ``BlockMask``, or ``None``.
            noise_embedding: Tensor of shape [batch, draft, hidden]; the embedded
                ``[anchor, MASK, ...]`` blocks laid end to end.
            target_hidden: Tensor of shape [batch, context, layers * hidden]; the
                concatenated target-model context features.
            past_key_values: Draft KV cache, or ``None``.
            use_cache: Whether to write into ``past_key_values``.
            conv_block_size: Draft-block length for the in-block convolutions; see
                :meth:`resolve_conv_block_size` for how ``None`` is resolved.
            **kwargs: Forwarded to the attention implementation.

        Returns:
            Tensor of shape [batch, draft, hidden]; the normalised draft hidden
            states, ready for the target's ``lm_head``.
        """
        return super().forward(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=past_key_values,
            use_cache=use_cache,
            conv_block_size=self.resolve_conv_block_size(noise_embedding.shape[1], conv_block_size),
            **kwargs,
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int] | None,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.LongTensor:
        """Block-parallel speculative decoding with pairwise path selection.

        Each cycle drafts one block in a single draft forward, walks the selector
        over the per-position candidates to pick a coherent path, and verifies the
        whole block with one target forward. Below ``GREEDY_TEMPERATURE_EPS`` this
        accepts the longest exact-match prefix, matching ``sample``'s own greedy
        threshold; above it, it accepts via rejection sampling, so the emitted
        tokens follow the target's own distribution.

        Args:
            target: The frozen verifier; must expose ``model.embed_tokens``,
                ``lm_head``, and an HF-style forward with ``output_hidden_states``.
            input_ids: Long tensor of shape [1, prompt].
            max_new_tokens: Maximum number of tokens to generate.
            stop_token_ids: Token ids that end generation, or ``None``.
            temperature: Sampling temperature; ``0`` decodes greedily.
            top_p: Nucleus mass to keep, in ``(0, 1]``; truncates the *target's*
                distribution, which is what the emitted tokens must follow. The
                draft proposes from plain temperature, as the reference does.
            top_k: Candidates to keep, ``0`` for the whole vocabulary.

        Returns:
            Long tensor of shape [1, prompt + generated] containing the prompt
            followed by the accepted tokens.
        """
        self.eval()
        validate_sampling(temperature, top_p, top_k)
        assert_target_supports_rollback(target)
        output_head = resolve_output_head(target)
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size

        output_ids = torch.full(
            (1, max_length + block_size), self.mask_token_id, dtype=torch.long, device=target.device
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        # Build each cache from its own model's config. A bare ``DynamicCache()``
        # gives every layer a plain attention cache, which a hybrid target -- the
        # Qwen3.5 family interleaves ``linear_attention`` with ``full_attention``
        # -- rejects on its first forward (``has_previous_state`` raises). The
        # draft is a uniform Qwen3 stack, but pass its config too so the two are
        # constructed the same way.
        past_key_values_target = DynamicCache(config=getattr(target, "config", None))
        past_key_values_draft = DynamicCache(config=self.config)

        # Prefill the target on the prompt.
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature, top_p, top_k)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        stop_tokens = (
            torch.tensor(stop_token_ids, dtype=output_ids.dtype, device=output_ids.device) if stop_token_ids else None
        )
        start = num_input_tokens
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = self.embed_noise_block(target, block_output_ids)
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                conv_block_size=block_size,
            )[:, -block_size + 1 :, :]
            past_key_values_draft.crop(start)
            # Block position 0 holds the last verified token; it is the predecessor
            # the selector starts its walk from, not something the draft predicts.
            draft_tokens, candidate_ids, draft_probs = self.candidate_selector.walk(
                draft_hidden, self.compute_logits(draft_hidden, output_head), block_output_ids[:, 0], temperature
            )
            block_output_ids[:, 1:] = draft_tokens

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            if temperature >= GREEDY_TEMPERATURE_EPS:
                target_probs = sampling_probs(output.logits, temperature, top_p, top_k)
                acceptance_length, bonus = dflash2_rejection_sample(
                    draft_tokens, target_probs, draft_probs, candidate_ids
                )
            else:
                posterior = sample(output.logits, temperature, top_p, top_k)
                acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
                bonus = posterior[0, acceptance_length]
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = bonus
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]
            if stop_tokens is not None and bool(
                torch.isin(output_ids[0, start - acceptance_length - 1 : start + 1], stop_tokens).any()
            ):
                break

        # ``start`` indexes the last committed token (the bonus token the target
        # produced at the end of the previous block), so the generated sequence is
        # exactly ``[0, start]``; everything past it is still MASK padding.
        output_ids = output_ids[:, : min(start + 1, max_length)]
        if stop_tokens is not None:
            stop_indices = torch.isin(output_ids[0, num_input_tokens:], stop_tokens).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]
        return output_ids
