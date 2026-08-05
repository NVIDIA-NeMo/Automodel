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
from typing import Optional

import torch
from torch import nn

from nemo_automodel.components.speculative.dspark._sampling import sample_tokens


class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert self.markov_rank > 0, f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_token_ids,
                hidden_states=step_hidden,
            )
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkov):
    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(dtype=prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):
    """RNN-based head that maintains recurrent state across positions within a block.

    Unlike the memoryless Markov heads, position k can access the full prefix
    history x_{<k} through a GRU-like recurrent state.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.hidden_size = hidden_size
        # Joint projection: [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]
        # W_g, W_c, W_o in R^{(2r+d) x r} parameterized as one linear layer.
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single RNN step.

        Args:
            state: [*, r] previous recurrent state
            prev_embeddings: [*, r] W1[x_{k-1}]
            hidden_states: [*, d] backbone hidden at step k

        Returns:
            new_state: [*, r]
            bias: [*, vocab_size]
        """
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stateless single-step bias (state initialized to zero).

        This is used for compatibility but does NOT carry state across steps.
        For full RNN behavior, use apply_block_logits or sample_block_tokens.
        """
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply RNN bias during training (teacher-forced, unrolled over block_size).

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids: [B, num_blocks, block_size] - prev token ids for each position
            hidden_states: [B, num_blocks, block_size, d]
        """
        assert hidden_states is not None
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]  # e.g. [B, num_blocks]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            output_logits.append(base_logits[..., k, :] + bias)

        return torch.stack(output_logits, dim=-2)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling with RNN state.

        Args:
            base_logits: [batch, proposal_len, vocab]
            first_prev_token_ids: [batch] - token preceding the first draft position
            hidden_states: [batch, proposal_len, d]
            temperature: sampling temperature

        Returns:
            sampled_tokens: [batch, proposal_len]
            corrected_logits: [batch, proposal_len, vocab]
        """
        assert hidden_states is not None
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)
            h_k = hidden_states[:, step_idx, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)

            step_logits = base_logits[:, step_idx, :] + bias
            corrected_logits.append(step_logits.unsqueeze(1))

            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def build_markov_head(config) -> nn.Module | None:
    markov_rank = int(config.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(config.markov_head_type).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    assert False, f"Unsupported markov_head_type: {markov_head_type!r}"


__all__ = [
    "VanillaMarkov",
    "GatedMarkovHead",
    "RNNHead",
    "build_markov_head",
]
