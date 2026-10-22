# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Hugging Face deepseek_v32 routing precision compatibility."""

from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer

from nemo_automodel.components.models.common.precision import preserve_fp32_module_compute

if TYPE_CHECKING:
    from transformers.cache_utils import Cache


def apply_hf_precision(model: nn.Module, *, device_type: str) -> nn.Module:
    """Protect upstream HF routing and indexer scores from autocast.

    Args:
        model: Loaded model, including HF modules retained by force_hf or fallback.
        device_type: Device selected by the single-rank compute policy.

    Returns:
        Model with its HF router computation protected and original state intact.
    """
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter

    model = preserve_fp32_module_compute(model, module_type=DeepseekV32TopkRouter, device_type=device_type)
    for module in model.modules():
        if isinstance(module, DeepseekV32Indexer):
            cast(nn.Module, module).forward = MethodType(_indexer_forward, module)
    return model


# Adapted from Transformers 5.18.0 (Apache-2.0). Remove this shim when
# upstream protects both explicit FP32 indexer score matmuls from autocast.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
@wraps(DeepseekV32Indexer.forward, assigned=())
@torch.no_grad()
def _indexer_forward(
    self: DeepseekV32Indexer,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None = None,
) -> torch.Tensor:
    """Select keys using FP32 score formation and the upstream HF indexer layout.

    Args:
        self: Upstream indexer owning learned projections and cache identity.
        hidden_states: Activations [batch, sequence, hidden].
        q_resid: Query residuals [batch, sequence, q_lora_rank].
        position_embeddings: Cosine and sine tensors [batch, sequence, rotary_dim].
        attention_mask: Causal/padding mask broadcastable to [batch, sequence, keys].
        position_ids: Integer positions [batch, sequence], retained for HF call compatibility.
        past_key_values: Optional HF cache; updates retain the upstream order.

    Returns:
        Int32 key indices [batch, sequence, min(index_topk, keys)].
    """
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_pos_emb

    (batch_size, seq_len, _) = hidden_states.shape
    (cos, sin) = position_embeddings
    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
    (q_rot, q_pass) = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
    (k_rot, k_pass) = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    (q_rot, k_rot) = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
    if past_key_values is not None:
        k = past_key_values.update_indexer(k, self.layer_idx)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
    scores = F.relu(scores)
    weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * self.n_heads ** (-0.5)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
    if attention_mask.dtype == torch.bool:
        index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
    else:
        index_scores = index_scores + attention_mask
    topk = min(self.index_topk, index_scores.shape[-1])
    return index_scores.topk(topk, dim=-1).indices.to(torch.int32)
