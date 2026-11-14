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

"""The gated-LoRA MTP sampler (Samragh et al., 2025; arXiv:2507.11851, Sec. 2.3): a small MLP
that refines a mask position's hidden state using the previously predicted token's embedding,
before it is projected through the shared (frozen) LM head. Trained jointly with the LoRA
adapter; the base model's own projection path is unaffected when this module is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedMTPSampler(nn.Module):
    """Refines a gated-MTP depth's hidden state using the previous token's embedding.

    Implements the paper's non-recurrent sampler: a 2-layer MLP over
    ``[previous_token_embedding; hidden_state]``, added back as a residual. The second linear
    layer is zero-initialized, so the module starts as an exact no-op (``forward`` returns
    ``hidden_state`` unchanged) -- consistent with gated-LoRA MTP's existing "lossless until
    trained" design.

    Attributes:
        mlp: Two-layer MLP producing the residual correction.
    """

    def __init__(self, hidden_size: int, num_depths: int) -> None:
        """Build the sampler MLP for a model with the given hidden size.

        Args:
            hidden_size: The base model's hidden size.
            num_depths: Number of gated-MTP prediction depths. Stored for callers; the module
                itself is depth-agnostic (one shared MLP, called once per depth).
        """
        super().__init__()
        self.num_depths = num_depths
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self._zero_init_residual()

    def _zero_init_residual(self) -> None:
        """Zero the second Linear layer so the sampler starts as a no-op residual."""
        last_linear = self.mlp[3]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def forward(self, hidden_state: torch.Tensor, prev_token_embedding: torch.Tensor) -> torch.Tensor:
        """Refine a mask position's hidden state given the previous token's embedding.

        Args:
            hidden_state: Mask position's hidden state, shape ``[..., hidden_size]``.
            prev_token_embedding: Embedding of the token immediately preceding this depth's
                target, same leading shape as ``hidden_state``.

        Returns:
            Refined hidden state, same shape as ``hidden_state``. The caller projects this
            through the shared LM head, exactly as the un-refined hidden state would be.
        """
        combined = torch.cat([prev_token_embedding, hidden_state], dim=-1)
        return hidden_state + self.mlp(combined)
