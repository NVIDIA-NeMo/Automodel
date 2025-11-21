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

        # Combined QKV projection for improved efficiency
        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_attention_heads + 2 * num_key_value_heads) * head_dim,
            bias=bias,
        )

    def compute_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K, V from hidden states using combined projection.

        Handles tensor parallelism by dynamically computing split sizes based on actual tensor dimensions.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]

        Returns:
            Tuple of (query, key, value) tensors, each [batch, seq_len, ...]
        """
        # Combined QKV projection and split
        qkv = self.qkv_proj(hidden_states)

        # Compute split sizes based on actual tensor size (handles TP sharding)
        qkv_size = qkv.shape[-1]
        total_size = self.q_size + 2 * self.kv_size
        local_q_size = (self.q_size * qkv_size) // total_size
        local_kv_size = (self.kv_size * qkv_size) // total_size

        q, k, v = qkv.split([local_q_size, local_kv_size, local_kv_size], dim=-1)
        return q, k, v
