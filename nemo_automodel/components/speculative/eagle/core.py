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

"""Core EAGLE-3 training logic for the minimal Llama MVP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail."""
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


def _compute_target_distribution(
    target_logits: torch.Tensor,
    selected_token_ids: torch.Tensor,
    selected_token_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project target logits into draft vocabulary space and build supervision mask."""
    target_top_ids = target_logits.argmax(dim=-1)
    position_mask = (selected_token_mask[target_top_ids] & loss_mask.bool()).unsqueeze(-1)
    draft_target_logits = target_logits.index_select(dim=-1, index=selected_token_ids.to(target_logits.device))
    target_probs = torch.softmax(draft_target_logits.float(), dim=-1).detach()
    return target_probs, position_mask


@dataclass
class Eagle3StepMetrics:
    """Aggregated metrics from one EAGLE-3 training step.

    ``step_prefix_hits`` / ``step_valid`` are ``[ttt_steps]`` per-TTT-step
    counts for the simulated accept length (:func:`simulated_accept_length`):
    ``step_prefix_hits[k]`` counts positions whose greedy draft chain is still
    fully correct through depth ``k + 1`` (top-1 hit at every step ``<= k``),
    and ``step_valid[k]`` counts positions supervised at every step ``<= k``
    under the shifted loss/document masks, so numerator and denominator cover
    the same chain population (a chain with an unsupervised earlier depth,
    e.g. from a gappy multi-turn loss mask, can never be a hit and must not
    count as a miss). ``step_valid`` deliberately does NOT exclude positions
    whose target token falls outside the compressed draft vocabulary: serving
    must reject those tokens (the draft cannot emit them), so they count as
    chain breaks instead of dropping out of the denominator. Both are kept
    unreduced so the recipe can accumulate them over a logging window. The
    P-EAGLE trainer, whose depths are drafted in parallel from one anchor
    rather than as a TTT chain, leaves them ``None``.
    """

    loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    step_prefix_hits: torch.Tensor | None = None
    step_valid: torch.Tensor | None = None


def simulated_accept_length(step_prefix_hits: torch.Tensor, step_valid: torch.Tensor) -> torch.Tensor:
    """Expected accepted tokens per speculative round, from prefix-hit counts.

    Models greedy chain drafting: ``step_prefix_hits[k] / step_valid[k]``
    estimates the joint probability that the first ``k + 1`` drafted tokens
    all match the target's greedy choices, i.e. that the chain survives depth
    ``k + 1``, so the expectation is ``1 + sum_k P(survives depth k + 1)``.
    Joint prefix counts keep the correlation between depths that a product of
    per-step marginal accuracies discards (coincident and disjoint hits score
    differently). The leading 1 counts the token the target itself emits on
    every verification round, matching the ``accept_length`` convention of
    the serving benchmarks (``1 + accepted/drafts``). A step with no
    supervised positions contributes zero via the ``clamp_min``. This is a
    training-time proxy for greedy chain decoding; engine tree drafting
    typically accepts more.
    """
    survive = step_prefix_hits.float() / step_valid.float().clamp_min(1.0)
    return 1.0 + survive.sum()


class Eagle3TrainerModule(nn.Module):
    """Draft-side EAGLE-3 trainer module with test-time-training unroll."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        selected_token_ids: torch.Tensor,
        selected_token_mask: torch.Tensor,
        ttt_steps: int,
    ):
        super().__init__()
        # The forward pass weighs each TTT step by ``0.8 ** i`` and divides
        # the running loss by ``sum_{i=0}^{ttt_steps-1} 0.8 ** i``. With
        # ``ttt_steps <= 0`` the loop never runs and the divisor is zero,
        # which would silently produce a NaN loss instead of an actionable
        # error. Catch the misconfiguration here so it surfaces during
        # recipe setup rather than mid-training.
        if not isinstance(ttt_steps, int) or ttt_steps < 1:
            raise ValueError(
                f"Eagle3TrainerModule requires ttt_steps to be an integer >= 1 "
                f"(the draft must run at least one forward step to produce a "
                f"loss), got ttt_steps={ttt_steps!r}."
            )
        self.draft_model = draft_model
        self.register_buffer("selected_token_ids", selected_token_ids, persistent=True)
        self.register_buffer("selected_token_mask", selected_token_mask, persistent=True)
        self.ttt_steps = ttt_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor | None = None,
        *,
        target_probs: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> Eagle3StepMetrics:
        """Run the EAGLE-3 unrolled draft loss for one batch.

        The attention layer is driven through a shared ``cache_hidden``
        list so each TTT step can attend to the K/V branches produced by
        every previous step at the same position. This matches the
        SpecForge ``llama3_eagle.py`` recurrence; without it, multi-step
        TTT would degenerate into ``ttt_steps`` independent single-step
        passes and the draft would never learn the multi-step
        distribution it sees at deployment time.

        ``attention_mask`` is held constant across TTT steps -- only
        ``input_ids`` / ``loss_mask`` / ``position_mask`` /
        ``target_probs`` roll forward by one position per step.

        Packing: ``position_ids`` / ``seq_lens`` make the draft's Block-1 attention
        document-level block-causal, and ``doc_remaining`` gates supervision per
        step (slot ``t`` valid at step ``k`` only while ``k < doc_remaining[t]``),
        masking every cross-document TTT prediction.

        Two supervision sources are accepted: the live path passes the
        target's full-vocab ``target_logits`` and the draft distribution is
        derived here; the offline-cache path (``precompute_eagle3``) passes the
        already-derived ``target_probs`` (over the draft vocab) and
        ``position_mask`` directly, so the full-vocab logits never have to be
        stored. Provide exactly one of the two.
        """
        precomputed = target_probs is not None and position_mask is not None
        if target_logits is not None and precomputed:
            raise ValueError(
                "Eagle3TrainerModule.forward got both target_logits and precomputed "
                "(target_probs, position_mask); pass exactly one supervision source."
            )
        hidden_states = self.draft_model.project_hidden_states(aux_hidden_states)
        if not precomputed:
            if target_logits is None:
                raise ValueError(
                    "Eagle3TrainerModule.forward requires either target_logits (live path) or both "
                    "target_probs and position_mask (offline-cache path); got neither."
                )
            target_probs, position_mask = _compute_target_distribution(
                target_logits=target_logits,
                selected_token_ids=self.selected_token_ids,
                selected_token_mask=self.selected_token_mask,
                loss_mask=loss_mask,
            )

        running_loss = hidden_states.new_zeros(())
        running_correct = hidden_states.new_zeros(())
        running_valid = hidden_states.new_zeros(())
        step_prefix_parts: list[torch.Tensor] = []
        step_valid_parts: list[torch.Tensor] = []

        cur_input_ids = input_ids
        cur_position_mask = position_mask
        cur_target_probs = target_probs
        cur_hidden_states = hidden_states
        # Simulated-accept-length state (see Eagle3StepMetrics). The TTT shift
        # keeps slot ``j`` anchored to the same chain across steps (step ``k``
        # at slot ``j`` drafts depth ``k + 1`` from anchor ``j``), so per-step
        # hits and validity can be ANDed at the same slot index. Numerator and
        # denominator must cover the same chain population: ``prefix_valid``
        # drops a chain as soon as any of its depths is unsupervised (gappy
        # loss masks make the per-step masks non-nested), since
        # ``prefix_correct`` can never score such a chain.
        cur_loss_mask = loss_mask.bool()
        prefix_correct: torch.Tensor | None = None
        prefix_valid: torch.Tensor | None = None

        # EAGLE-3 TTT KV cache: a pair of lists [K_list, V_list] that the
        # attention layer appends to on every step. Re-created per batch.
        cache_hidden: list[list[torch.Tensor]] = [[], []]

        # Weighted average across TTT steps: step ``i`` is weighted by
        # ``0.8 ** i`` and the sum is divided by the total weight. This
        # keeps the EAGLE-3 / SpecForge decay schedule (earlier steps
        # dominate, later steps still contribute a smaller signal) while
        # making the loss magnitude *invariant* to the choice of
        # ``ttt_steps`` and the decay constant -- a proper weighted mean
        # always lands in the same ``~ln(draft_vocab_size)`` range at
        # init, and the optimizer LR does not need to be rescaled when
        # the TTT schedule changes. SpecForge omits this normalization;
        # we keep it deliberately so config knobs stay decoupled from LR.
        weight_sum = sum(0.8**i for i in range(self.ttt_steps))
        for step_idx in range(self.ttt_steps):
            cur_hidden_states = self.draft_model(
                input_ids=cur_input_ids,
                projected_hidden_states=cur_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                seq_lens=seq_lens,
            )
            logits = self.draft_model.compute_logits(cur_hidden_states)

            # Packing: drop supervision whose step_idx-ahead target crosses this
            # slot's document boundary. Gate recomputed per step (depends on step_idx).
            step_position_mask = cur_position_mask
            chain_valid = cur_loss_mask
            if doc_remaining is not None:
                in_doc = step_idx < doc_remaining
                step_position_mask = cur_position_mask & in_doc.unsqueeze(-1)
                chain_valid = chain_valid & in_doc

            step_loss = masked_soft_cross_entropy(
                logits=logits,
                target_probs=cur_target_probs,
                position_mask=step_position_mask,
            )
            running_loss = running_loss + step_loss * (0.8**step_idx)

            valid_mask = step_position_mask.squeeze(-1).bool()
            correct = (logits.argmax(dim=-1) == cur_target_probs.argmax(dim=-1)) & valid_mask
            running_correct = running_correct + correct.sum()
            running_valid = running_valid + valid_mask.sum()

            prefix_correct = correct if prefix_correct is None else prefix_correct & correct
            prefix_valid = chain_valid if prefix_valid is None else prefix_valid & chain_valid
            step_prefix_parts.append(prefix_correct.sum())
            step_valid_parts.append(prefix_valid.sum())

            if step_idx + 1 < self.ttt_steps:
                cur_input_ids = _shift_left_with_zero(cur_input_ids)
                cur_position_mask = _shift_left_with_zero(cur_position_mask)
                cur_target_probs = _shift_left_with_zero(cur_target_probs)
                cur_loss_mask = _shift_left_with_zero(cur_loss_mask)

        avg_loss = running_loss / weight_sum
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            valid_tokens=running_valid,
            step_prefix_hits=torch.stack(step_prefix_parts).detach(),
            step_valid=torch.stack(step_valid_parts).detach(),
        )
