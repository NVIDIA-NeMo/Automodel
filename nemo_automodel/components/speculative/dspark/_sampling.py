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
from __future__ import annotations

import torch


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, vocab_size = probs.shape
    flat = probs.reshape(-1, vocab_size)
    return torch.multinomial(flat, num_samples=1).reshape(bsz, seq_len)


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)

    bsz, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)


def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    return sample_from_probs(residual.unsqueeze(1)).squeeze(1)


__all__ = [
    "gather_token_probs",
    "logits_to_probs",
    "sample_from_probs",
    "sample_residual",
    "sample_tokens",
]
