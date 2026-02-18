# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Combined QKV attention projection for efficient multi-head attention.

This module provides a mixin class that enables combined QKV projection
for any attention module, improving memory efficiency and reducing kernel launch overhead.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def _assert_colwise_parallel(weight: torch.Tensor, name: str) -> None:
    """Verify that a combined-projection weight uses ColwiseParallel (Shard(0)) if TP is active."""
    if isinstance(weight, DTensor) and weight.placements:
        from torch.distributed.tensor.placement_types import Shard

        if weight.placements[0] != Shard(0):
            raise ValueError(
                f"{name} uses an interleaved layout that requires ColwiseParallel "
                f"(Shard(0)) for correct TP sharding, but got placements={weight.placements}. "
                f"Check your TP plan."
            )


class CombinedQKVAttentionMixin:
    """Mixin for combined QKV projection in attention modules.

    This mixin ALWAYS uses combined QKV projections for efficiency.
    Use this with custom transformer attention modules (Llama, Qwen2, etc.).

    Usage:
        class MyAttention(CombinedQKVAttentionMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                # ... other init code ...
                self.setup_qkv_projection(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=self.head_dim,
                    bias=config.attention_bias
                )

            def forward(self, hidden_states, ...):
                query_states, key_states, value_states = self.compute_qkv(hidden_states)
                # ... rest of attention logic ...
    """

    def setup_qkv_projection(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        bias: bool = False,
        use_combined_qkv: bool = True,
    ):
        """Setup combined QKV projection (ALWAYS uses combined format).

        Args:
            hidden_size: Model hidden size
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (for GQA)
            head_dim: Dimension per attention head
            bias: Whether to use bias in projections
            use_combined_qkv: DEPRECATED - always True for custom implementations
        """
        self.use_combined_qkv = True  # Always combined in custom implementations
        self.q_size = num_attention_heads * head_dim
        self.kv_size = num_key_value_heads * head_dim
        self._num_kv_groups = num_attention_heads // num_key_value_heads
        self._head_dim = head_dim
        self._tp_checked = False

        # Combined QKV projection for improved efficiency
        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_attention_heads + 2 * num_key_value_heads) * head_dim,
            bias=bias,
        )

    def compute_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K, V from hidden states using combined projection.

        The QKV weight uses a KV-head-grouped interleaved layout:
            [Q_group_0 | K_0 | V_0 | Q_group_1 | K_1 | V_1 | ...]
        This ensures ColwiseParallel TP sharding gives each rank complete
        KV-head groups. We split within each group (a local operation).

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]

        Returns:
            Tuple of (query, key, value) tensors, each [batch, seq_len, ...]
        """
        if not self._tp_checked:
            _assert_colwise_parallel(self.qkv_proj.weight, "qkv_proj")
            self._tp_checked = True

        qkv = self.qkv_proj(hidden_states)

        group_width = (self._num_kv_groups + 2) * self._head_dim
        qkv = qkv.unflatten(-1, (-1, group_width))
        q, k, v = qkv.split(
            [self._num_kv_groups * self._head_dim, self._head_dim, self._head_dim], dim=-1
        )
        return q.flatten(-2), k.flatten(-2), v.flatten(-2)
