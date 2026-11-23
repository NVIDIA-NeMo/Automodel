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

    Implements Section 2.3's two Linear/SiLU/LayerNorm blocks over
    ``[previous_token_embedding; hidden_state]``. The shared LM head projects the
    transformed feature directly: there is no residual connection to the input.
    Both Linear layers use PyTorch's ordinary initialization. Zeroing a Linear
    before LayerNorm would amplify its first gradients and parameter updates.

    Attributes:
        mlp: Two-layer MLP producing the transformed feature.
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

    def forward(self, hidden_state: torch.Tensor, prev_token_embedding: torch.Tensor) -> torch.Tensor:
        """Refine a mask position's hidden state given the previous token's embedding.

        Args:
            hidden_state: Tensor of shape [..., hidden], with arbitrary leading dimensions.
            prev_token_embedding: Tensor of shape [..., hidden], with the same leading
                dimensions; embedding of the preceding sampled token.

        Returns:
            Transformed feature, Tensor of shape [..., hidden]. The caller projects it
            through the shared LM head. Neither input is mutated or aliased.
        """
        combined = torch.cat([prev_token_embedding, hidden_state], dim=-1)
        return self.mlp(combined)
