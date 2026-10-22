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

"""Hugging Face glm5_next routing precision compatibility."""

from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

from nemo_automodel.components.models.common.precision import preserve_fp32_module_compute

if TYPE_CHECKING:
    from transformers.cache_utils import Cache, CacheLayerMixin


def apply_hf_precision(model: nn.Module, *, device_type: str) -> nn.Module:
    """Protect upstream HF routing and indexer scores from autocast.

    Args:
        model: Loaded model, including HF modules retained by force_hf or fallback.
        device_type: Device selected by the single-rank compute policy.

    Returns:
        Model with its HF router computation protected and original state intact.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection, Glm5NextTextTopkRouter

    model = preserve_fp32_module_compute(model, module_type=Glm5NextTextTopkRouter, device_type=device_type)
    model = preserve_fp32_module_compute(model, module_type=Glm5NextTextHyperConnection, device_type=device_type)
    for module in model.modules():
        if isinstance(module, Glm5NextTextIndexer):
            cast(nn.Module, module).forward = MethodType(_indexer_forward, module)
    return model


# Adapted from Transformers 5.18.0 (Apache-2.0). Remove this shim when
# upstream protects both explicit FP32 indexer score matmuls from autocast.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
@wraps(Glm5NextTextIndexer.forward, assigned=())
@torch.no_grad()
def _indexer_forward(
    self: Glm5NextTextIndexer,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Cache | None,
) -> torch.Tensor:
    """Select keys using FP32 score formation and the upstream HF indexer layout.

    Args:
        self: Upstream indexer owning learned projections and cache identity.
        hidden_states: Activations [batch, sequence, hidden].
        q_resid: Query residuals [batch, sequence, q_lora_rank].
        attention_mask: Boolean padding mask [batch, sequence], with True for valid tokens.
        past_key_values: Optional HF cache; updates retain the upstream order.

    Returns:
        Int32 indices [batch, sequence, index_topk + optional_tail], with -1
        for masked entries. The optional tail has index_kpool - 1 entries.
    """
    (batch_size, seq_len) = hidden_states.shape[:2]
    hidden_shape = (batch_size, seq_len, -1, self.head_dim)
    q = self.wq_b(q_resid).view(hidden_shape)
    k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)
    gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
    valid_channel = attention_mask.to(k.dtype)[..., None]
    packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)
    kv_len = seq_len
    current_length = seq_len
    if past_key_values is not None:
        cache_layer = cast("CacheLayerMixin", past_key_values.layers[self.layer_idx])
        packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
        cached_keys = cache_layer.keys
        if cached_keys is None:
            raise ValueError("The HF pooled indexer requires an initialized attention key cache")
        kv_len = cached_keys.shape[-2]
        current_length = cache_layer.get_seq_length()
    valid_keys = cast(torch.BoolTensor, packed_states[..., -1].bool())
    visible_tokens = self.get_visible_tokens(valid_keys=valid_keys, q_length=seq_len, current_length=current_length)
    (pool_keys, pool_indices, pool_valid) = self.get_pooled_states(packed_states=packed_states)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * self.softmax_scale)
    weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * self.n_heads ** (-0.5)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = visible_tokens.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1))
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
    select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]
    topk_indices = selected_indices.flatten(-2)
    topk_indices = topk_indices.masked_fill(~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)
    output_width = self.index_topk
    if self.index_kpool_always_select_tail:
        topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
        output_width += self.index_kpool - 1
    topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
    topk_indices = topk_indices[..., :output_width]
    topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
    return topk_indices.to(torch.int32)
