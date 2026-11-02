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

"""EAGLE-4 draft training on top of the EAGLE-3 recipe.

An EAGLE-4 draft is the EAGLE-3.1 Llama-style draft with a sliding-window
draft layer and a low-rank candidate head (see ``draft_llama.py``). It trains
in two stages, both run by ``TrainEagle3Recipe``:

1. **Draft stage** (``recipe_args.eagle4.sliding_window`` only): the usual
   EAGLE-3 TTT loss over the full target vocabulary, with the windowed layer.
2. **Candidate-head stage** (``recipe_args.eagle4.candidate_head`` set and
   ``draft_weights_path`` pointing at the stage-1 export): every stage-1
   parameter is frozen and only the two rank-``r`` factors train, initialized
   from the truncated SVD of the draft ``lm_head``.
   :class:`Eagle4CandidateHeadTrainerModule` runs the frozen draft's TTT unroll
   and minimizes, at every depth, the forward KL from the full head's softmax
   to the low-rank head's softmax. The low-rank head only has to retrieve the
   full head's likely tokens (serving rescores the retrieved candidates
   exactly), and forward KL weights each token by the full head's probability,
   so the limited rank goes to the tokens that matter. The logged ``accuracy``
   is candidate recall: how often the full head's top-1 token is among the
   low-rank head's top ``num_candidates``.

``python -m nemo_automodel.components.speculative.eagle.export_eagle4`` turns
the stage-2 checkpoint into the ``LlamaForCausalLMEagle4`` layout the serving
engines load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy
from nemo_automodel.components.speculative.eagle.core import Eagle3StepMetrics, _shift_left_with_zero
from nemo_automodel.components.speculative.eagle.draft_llama import EAGLE4_ARCHITECTURE, LlamaEagle3DraftModel

# The candidate head's state-dict keys (top-level, outside the ``model.`` backbone).
CANDIDATE_HEAD_KEYS = frozenset({"efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"})


@dataclass(frozen=True)
class Eagle4CandidateHeadConfig:
    """Shape of the EAGLE-4 low-rank candidate head.

    Attributes:
        rank: Rank ``r`` of the factorized vocabulary scores.
        num_candidates: Tokens ``K`` the serving engine keeps from the low-rank
            scores and rescores with the full ``lm_head``; also the ``K`` of the
            logged candidate recall.
    """

    rank: int = 128
    num_candidates: int = 128

    def __post_init__(self) -> None:
        for name in ("rank", "num_candidates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"eagle4.candidate_head.{name} must be a positive integer, got {value!r}.")

    def build(self, draft_model: LlamaEagle3DraftModel, *, ttt_steps: int) -> Eagle4CandidateHeadTrainerModule:
        """Build the stage-2 trainer that fits ``draft_model``'s candidate head."""
        return Eagle4CandidateHeadTrainerModule(draft_model, ttt_steps=ttt_steps, num_candidates=self.num_candidates)


@dataclass(frozen=True)
class Eagle4Config:
    """``recipe_args.eagle4``: turns the EAGLE-3 recipe's draft into an EAGLE-4 draft.

    Attributes:
        sliding_window: Keys each draft query attends to, itself and its TTT
            chain included (so at least ``ttt_steps``), or ``None`` for full
            causal attention.
        candidate_head: Set for the candidate-head stage; ``None`` trains the
            draft itself.
    """

    sliding_window: int | None = None
    candidate_head: Eagle4CandidateHeadConfig | None = None

    def __post_init__(self) -> None:
        window = self.sliding_window
        if window is not None and (not isinstance(window, int) or isinstance(window, bool) or window <= 0):
            raise ValueError(f"eagle4.sliding_window must be a positive integer or null, got {window!r}.")

    def draft_config_updates(self, *, aux_layer_ids: list[int]) -> dict[str, Any]:
        """Return the draft ``config.json`` keys that define this EAGLE-4 draft.

        Args:
            aux_layer_ids: The three target layers whose outputs feed the draft,
                recorded as ``eagle_config.eagle_aux_hidden_state_layer_ids`` so
                the serving engine captures the same layers.
        """
        updates: dict[str, Any] = {
            "architectures": [EAGLE4_ARCHITECTURE],
            # The LlamaForCausalLMEagle4 layer is SwiGLU, whatever the target's MLP.
            "hidden_act": "silu",
            # HF configs require one ``layer_types`` entry per layer, and the
            # draft has a single layer (the cloned target config has the
            # target's count).
            "num_hidden_layers": 1,
            "layer_types": ["sliding_attention" if self.sliding_window is not None else "full_attention"],
            "use_sliding_window": self.sliding_window is not None,
            "sliding_window": self.sliding_window,
            "num_aux_hidden_states": len(aux_layer_ids),
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": list(aux_layer_ids),
                "use_aux_hidden_state": True,
            },
        }
        if self.candidate_head is not None:
            updates["efficient_lm_head_config"] = {
                "type": "low_rank",
                "rank": self.candidate_head.rank,
                "num_candidates": self.candidate_head.num_candidates,
            }
        return updates


class Eagle4CandidateHeadTrainerModule(nn.Module):
    """Stage-2 trainer: fit the low-rank candidate head of a frozen EAGLE-4 draft.

    The draft's TTT unroll (same recurrence, input shift and packing gates as
    ``Eagle3TrainerModule``) runs without gradients; at each depth the full
    ``lm_head`` softmax is the teacher and the low-rank scores the student.
    Depth ``i`` is weighted by ``0.8 ** i`` and the sum normalized by the total
    weight, as in the draft stage.

    Like every EAGLE-3-recipe trainer it carries the draft-vocab mapping buffers
    (``selected_token_ids`` / ``selected_token_mask``) that the recipe saves to and
    restores from the checkpoint metadata; an EAGLE-4 draft uses the full vocabulary.
    """

    def __init__(self, draft_model: LlamaEagle3DraftModel, *, ttt_steps: int, num_candidates: int) -> None:
        super().__init__()
        if not isinstance(ttt_steps, int) or isinstance(ttt_steps, bool) or ttt_steps <= 0:
            raise ValueError(f"ttt_steps must be a positive integer, got {ttt_steps!r}.")
        if draft_model.efficient_lm_head is None:
            raise ValueError("Eagle4CandidateHeadTrainerModule requires a draft with efficient_lm_head_config.")
        vocab_size = draft_model.config.vocab_size
        if num_candidates > vocab_size:
            raise ValueError(f"num_candidates={num_candidates} exceeds the vocabulary size {vocab_size}.")
        self.draft_model = draft_model
        self.ttt_steps = ttt_steps
        self.num_candidates = num_candidates
        self.register_buffer("selected_token_ids", torch.arange(vocab_size), persistent=True)
        self.register_buffer("selected_token_mask", torch.ones(vocab_size, dtype=torch.bool), persistent=True)

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
        """Run the candidate-head loss for one batch.

        Accepts the same inputs as ``Eagle3TrainerModule.forward`` so every
        target backend can feed it. The target supervision (``target_logits``,
        or ``target_probs`` / ``position_mask`` from the offline cache) is not
        used: the teacher is the draft's own full head, and ``loss_mask``
        selects the supervised positions.

        Args:
            input_ids: Tensor of shape ``[batch, sequence]``, token ids.
            attention_mask: Tensor of shape ``[batch, sequence]``, right-padded
                ``1`` / ``0`` validity mask.
            loss_mask: Tensor of shape ``[batch, sequence]``, ``1`` where the
                draft is supervised.
            aux_hidden_states: Tensor of shape ``[batch, sequence, 3 * target_hidden]``,
                the concatenated target hidden states the draft projects.
            target_logits: Unused; tensor of shape ``[batch, sequence, vocab]``.
            target_probs: Unused; tensor of shape ``[batch, sequence, vocab]``.
            position_mask: Unused; tensor of shape ``[batch, sequence, 1]``.
            position_ids: Optional tensor of shape ``[batch, sequence]`` with
                per-document positions (packing).
            seq_lens: Optional tensor of shape ``[batch, max_docs]`` with packed
                document lengths, 0-padded.
            doc_remaining: Optional tensor of shape ``[batch, sequence]``: tokens
                left in each slot's document; depth ``i`` is supervised only
                while ``i < doc_remaining``.

        Returns:
            ``Eagle3StepMetrics`` with the weighted forward KL (0 for a perfect head), ``accuracy`` set to the
            candidate recall and ``valid_tokens`` summed over depths (no
            accept-length counters).
        """
        del target_logits, target_probs, position_mask
        draft = self.draft_model
        cache_hidden: list[list[torch.Tensor]] = [[], []]
        cur_input_ids = input_ids
        cur_valid = loss_mask.bool()
        with torch.no_grad():
            hidden_states = draft.project_hidden_states(aux_hidden_states)

        running_loss = hidden_states.new_zeros((), dtype=torch.float32)
        running_hits = hidden_states.new_zeros((), dtype=torch.float32)
        running_valid = hidden_states.new_zeros((), dtype=torch.float32)
        weight_sum = sum(0.8**i for i in range(self.ttt_steps))
        for step_idx in range(self.ttt_steps):
            with torch.no_grad():
                hidden_states = draft(
                    input_ids=cur_input_ids,
                    projected_hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_hidden=cache_hidden,
                    seq_lens=seq_lens,
                )
                teacher_logits = draft.compute_logits(hidden_states)
                teacher_probs = torch.softmax(teacher_logits, dim=-1, dtype=torch.float32)
            student_logits = draft.compute_candidate_logits(hidden_states)

            step_valid = cur_valid if doc_remaining is None else cur_valid & (step_idx < doc_remaining)
            step_cross_entropy = masked_soft_cross_entropy(
                logits=student_logits, target_probs=teacher_probs, position_mask=step_valid.unsqueeze(-1)
            )
            with torch.no_grad():
                # KL(p_full || q) = H(p_full, q) - H(p_full); the entropy term is
                # constant in the factors, so subtracting it only fixes the
                # logged value (0 for a perfect head).
                valid = step_valid.to(teacher_probs.dtype)
                entropy = torch.special.entr(teacher_probs).sum(dim=-1)
                step_entropy = (entropy * valid).sum() / valid.sum().clamp_min(1.0)
                # The full head's top-1 token is a candidate when at most K
                # low-rank scores reach its score; ties count against it, as a
                # serving top-K may drop a tied token (one pass, no vocab topk).
                top1_score = student_logits.gather(-1, teacher_logits.argmax(dim=-1, keepdim=True))
                hits = ((student_logits >= top1_score).sum(dim=-1) <= self.num_candidates) & step_valid
            running_loss = running_loss + (step_cross_entropy - step_entropy) * (0.8**step_idx)
            running_hits = running_hits + hits.sum()
            running_valid = running_valid + step_valid.sum()

            if step_idx + 1 < self.ttt_steps:
                cur_input_ids = _shift_left_with_zero(cur_input_ids)
                cur_valid = _shift_left_with_zero(cur_valid)

        return Eagle3StepMetrics(
            loss=running_loss / weight_sum,
            accuracy=running_hits / running_valid.clamp_min(1.0),
            valid_tokens=running_valid,
        )
