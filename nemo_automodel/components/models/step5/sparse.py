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

"""CSA proxy compression and learned region selection for Step-5."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.kernels import indexer_score_tile
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding


@dataclass
class RegionSelection:
    """Sparse selection tensors, all local to one CP rank's queries.

    Attributes:
        region_ids: Sorted int32 tensor [batch, groups, local_queries, topk + 1].
            Unused slots contain a sentinel larger than any valid region.
        token_regions: Int64 tensor of shape [batch, global_keys].
        query_documents: Int64 tensor of shape [batch, local_queries].
        key_documents: Int64 tensor of shape [batch, global_keys].
    """

    region_ids: torch.Tensor
    token_regions: torch.Tensor
    query_documents: torch.Tensor
    key_documents: torch.Tensor


class Step5IndexerRMSNorm(Step3p5RMSNorm):
    """Indexer RMSNorm with source-required activation-dtype parameter rounding."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalize [..., proxy_dim] activations and preserve their dtype."""
        raw = hidden.float()
        scale = self.weight.to(hidden.dtype).float() + 1
        return (raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + self.variance_epsilon) * scale).to(hidden.dtype)


class Step5IndexerLayerNorm(nn.LayerNorm):
    """Indexer variance E[x²] - E[x]², evaluated inside the module's FSDP hooks."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalize [..., proxy_dim] with BF16-rounded scale/bias and FP32 math."""
        raw = hidden.float()
        mean = raw.mean(-1, keepdim=True)
        variance = raw.square().mean(-1, keepdim=True) - mean.square()
        scale = self.weight.to(hidden.dtype).float()
        bias = self.bias.to(hidden.dtype).float()
        return ((raw - mean) * torch.rsqrt(variance + self.eps) * scale + bias).to(hidden.dtype)


class Step5IndexerWeightProjection(nn.Linear):
    """Keep FP32 source weights while matching the reference BF16 projection."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project [..., hidden_size] to signed FP32 indexer head weights."""
        return F.linear(hidden, self.weight.to(hidden.dtype)).float()


class Step5SparseIndexer(nn.Module):
    """Checkpoint-backed, discrete CSA selector; LM loss does not train top-k indices."""

    def __init__(self, config: Step5Config, layer_idx: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config.sparse_config
        width = self.config.proxy_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.config.sparse_indexer_num_heads * width, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(config.hidden_size, width, bias=False, dtype=dtype)
        self.z_proj = nn.Linear(config.hidden_size, width, bias=False, dtype=dtype)
        self.w_proj = Step5IndexerWeightProjection(
            config.hidden_size, self.config.sparse_indexer_num_heads, bias=False, dtype=torch.float32
        )
        self.q_norm = Step5IndexerRMSNorm(width, config.rms_norm_eps)
        self.k_norm = Step5IndexerLayerNorm(width, eps=config.rms_norm_eps, dtype=torch.float32)
        self.register_buffer("ssmax_s", torch.zeros(config.num_attention_heads, dtype=torch.float32))
        span = self.config.sparse_indexer_rope_dim if self.config.sparse_indexer_use_rope else 0
        self.rotary_emb = Step5RotaryEmbedding(config, layer_idx, span=span)

    def project_keys(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized index keys and per-dimension compression logits.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden].
            positions: Tensor of shape [batch, sequence].

        Returns:
            Two tensors of shape [batch, sequence, proxy_dim], in the hidden dtype.
        """
        key = self.k_proj(hidden)
        key = self.k_norm(key)
        key = self.rotary_emb(key.unsqueeze(2), positions).squeeze(2)
        return key, self.z_proj(hidden)

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        keys: torch.Tensor,
        compression_logits: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding: torch.Tensor,
        *,
        query_start: int = 0,
    ) -> RegionSelection:
        """Select historical CSA regions independently for each provider group.

        Args:
            hidden: Tensor of shape [batch, local_queries, hidden].
            positions: Tensor of shape [batch, local_queries].
            keys: Tensor of shape [batch, global_keys, proxy_dim].
            compression_logits: Tensor of shape [batch, global_keys, proxy_dim].
            key_positions: Tensor of shape [batch, global_keys], resetting at document starts.
            key_padding: Boolean tensor of shape [batch, global_keys], true for padding.
            query_start: Offset of the contiguous query shard in the global key sequence.

        Returns:
            Compact region ids and document metadata. Scores exist only for a query tile.
        """
        cfg = self.config
        batch, queries = hidden.shape[:2]
        groups = cfg.num_provider_groups
        heads = cfg.sparse_indexer_num_heads // groups
        dim = cfg.proxy_dim
        query = self.q_proj(hidden).view(batch, queries, groups * heads, dim)
        query = self.q_norm(query)
        query = self.rotary_emb(query, positions).view(batch, queries, groups, heads, dim)
        weights = self.w_proj(hidden).view(batch, queries, groups, heads)
        weights = weights * heads**-0.5

        documents = ((key_positions == 0) & ~key_padding).long().cumsum(1)
        region_position = key_positions // cfg.region_block_size
        new_region = torch.ones_like(key_positions, dtype=torch.bool)
        new_region[:, 1:] = (region_position[:, 1:] != region_position[:, :-1]) | (
            documents[:, 1:] != documents[:, :-1]
        )
        token_regions = new_region.long().cumsum(1) - 1
        region_count = int(token_regions.max().item()) + 1
        slot = key_positions % cfg.region_block_size
        rows = torch.arange(batch, device=keys.device).unsqueeze(1)
        packed_keys = keys.new_zeros(batch, region_count, cfg.region_block_size, dim)
        packed_z = torch.full_like(packed_keys, float("-inf"))
        packed_keys[rows, token_regions, slot] = keys
        packed_z[rows, token_regions, slot] = compression_logits.masked_fill(key_padding.unsqueeze(-1), float("-inf"))
        shift = packed_z.float().amax(dim=(-1, -2), keepdim=True)
        probabilities = torch.exp(packed_z.float() - shift).nan_to_num(0.0)
        summary = (probabilities * packed_keys.float()).sum(2) / probabilities.sum(2).clamp_min(1e-20)
        # Both the BF16 materialization and the subsequent E4M3 rounding are
        # part of the released selector, independent of expert-weight precision.
        summary = summary.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
        query = query.to(torch.float8_e4m3fn).to(torch.bfloat16)
        region_documents = torch.zeros(batch, region_count, device=hidden.device, dtype=torch.long)
        region_documents.scatter_reduce_(1, token_regions, documents, reduce="amax")
        current = token_regions[:, query_start : query_start + queries]
        query_documents = documents[:, query_start : query_start + queries]
        budget = min(cfg.topk, region_count)
        region_ids = torch.full(
            (batch, groups, queries, budget + 1), region_count, device=hidden.device, dtype=torch.int32
        )
        region_range = torch.arange(region_count, device=hidden.device)
        # Bound the FP32 score workspace by query tiles; never retain [Q, regions]
        # scores or a [Q, K] token mask for the backward pass.
        for start in range(0, queries, 128):
            stop = min(start + 128, queries)
            scores = indexer_score_tile(query, weights, summary, start, stop)
            candidates = (region_range < current[:, start:stop, None]) & (
                region_documents[:, None] == query_documents[:, start:stop, None]
            )
            scores.masked_fill_(~candidates[:, None], float("-inf"))
            # topk finds a cutoff; an ascending prefix resolves exact ties without
            # perturbing scores or sorting the entire history.
            cutoff = scores.topk(budget, dim=-1, sorted=False).values.amin(-1, keepdim=True)
            greater = (scores > cutoff) & candidates[:, None]
            tied = (scores == cutoff) & candidates[:, None]
            remaining = budget - greater.sum(-1, keepdim=True)
            selected = greater | (tied & (tied.cumsum(-1) <= remaining))
            selected.scatter_(-1, current[:, None, start:stop, None].expand(-1, groups, -1, 1), True)
            # Only selected ids survive beyond this tile. Sorting these bounded
            # slots enables binary membership lookup inside fused attention.
            ids = torch.where(selected, region_range, region_count)
            if budget < region_count:
                ids = ids.topk(budget + 1, dim=-1, largest=False, sorted=True).values
            else:
                ids = torch.cat((ids.sort(dim=-1).values, ids.new_full((*ids.shape[:-1], 1), region_count)), -1)
            region_ids[:, :, start:stop].copy_(ids)
        return RegionSelection(region_ids, token_regions, query_documents, documents)
