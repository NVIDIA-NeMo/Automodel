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

"""Hy4-flash Engram: hashed bigram / trigram memory added to the hyper-connection streams.

The hash, the per-head norms and the gate follow the transformers reference (``HYV4FlashEngramLayer``); the table is
the shared row-owner-sharded embedding of Qwen3.8-Flash-Next (one ~32 GB table per layer, sharded by rows across the
owner group, lookups by all-to-all). Parameters keep the reference's names; the forward stacks the per-head weights
so each norm and projection runs as one kernel.

Precision: the weighted norms are evaluated as the reference's ``nn.RMSNorm``. The gate, and the sum of the gated
signal and the per-stream branch, are evaluated in fp32 and cast back, where the reference stays in BF16; in fp32
the two are identical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig


def draw_hash_multipliers(seed: int, layer_id: int, max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """Odd multipliers an n-gram is mixed with, drawn as the memory was built.

    NumPy's default generator seeded with ``seed + 10007 * layer_id`` (``layer_id`` is the one-based id used in the
    config) draws ``max_ngram_size`` values bounded so a product with any compressed id stays within int64.

    Returns:
        int64 tensor of shape [max_ngram_size].
    """
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
    drawn = np.random.default_rng(seed + 10007 * layer_id).integers(0, half_bound, size=max_ngram_size, dtype=np.int64)
    return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)


def build_ngram_tokens(
    input_ids: torch.Tensor, document_positions: torch.Tensor, max_ngram_size: int, pad_id: int
) -> torch.Tensor:
    """Stack each token with the ``max_ngram_size - 1`` tokens before it.

    Positions that reach before the start of their document read ``pad_id``; the document start is where the
    leading (time) position plane restarts, so n-grams never span two packed documents.

    Args:
        input_ids: Integer tensor of shape [batch, sequence].
        document_positions: Integer tensor of shape [batch, sequence], the time plane of the position ids.
        max_ngram_size: Longest n-gram order.
        pad_id: Id used for missing history.

    Returns:
        Integer tensor of shape [batch, sequence, max_ngram_size], newest token first.
    """
    shifted = [input_ids]
    for offset in range(1, max_ngram_size):
        previous = F.pad(input_ids, (offset, 0), value=pad_id)[:, : input_ids.shape[1]]
        shifted.append(torch.where(document_positions < offset, pad_id, previous))
    return torch.stack(shifted, dim=-1)


class HYV4FlashWeightOnlyNorm(nn.Module):
    """Holder of an RMSNorm weight whose epsilon is the dtype epsilon (``nn.RMSNorm(eps=None)`` semantics)."""

    def __init__(self, size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype))


def weighted_rms_norm(x: torch.Tensor, norms: nn.ModuleList) -> torch.Tensor:
    """Apply one weighted RMSNorm per slice of the second-to-last axis, as ``torch.nn.RMSNorm`` evaluates it.

    The normalization and the multiplication by the weight both run in fp32 and the result is cast once (for a
    BF16 input ``F.rms_norm(x) * weight`` would round before the weight is applied).

    Args:
        x: Tensor of shape [..., slices, features].
        norms: ``slices`` modules, each holding a ``weight`` of shape [features].

    Returns:
        Tensor of the shape and dtype of ``x``.
    """
    weight = torch.stack([norm.weight for norm in norms])
    return (F.rms_norm(x.float(), (x.shape[-1],)) * weight.float()).to(x.dtype)


class HYV4FlashEngramLayer(nn.Module):
    """One Engram block (decoder layer ``layer_idx``, one-based id ``layer_idx + 1``).

    Args:
        config: Text configuration.
        layer_idx: Zero-based decoder layer index; must satisfy ``layer_idx + 1 in config.engram_layer_ids``.
        process_group: Row-owner group of the table; ``None`` builds the whole table locally (tests, tiny configs).
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_idx: int,
        *,
        process_group: dist.ProcessGroup | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if config.engram_fp8_embedding:
            raise NotImplementedError(
                "engram_fp8_embedding=True (per-vector fp8 round trip on lookup) is not implemented; "
                "the released checkpoint sets it to false"
            )
        self.layer_id = layer_idx + 1
        if self.layer_id not in config.engram_layer_ids:
            raise ValueError(
                f"decoder layer {layer_idx} (one-based id {self.layer_id}) has no Engram block; "
                f"engram_layer_ids is {config.engram_layer_ids}"
            )
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.max_ngram_size = config.engram_max_ngram_size
        self.heads_per_order = config.engram_n_head_per_ngram
        self.seed = config.engram_seed
        self.pad_id = config.engram_pad_id
        self.initializer_range = config.initializer_range
        sizes = [size for order in config.engram_layer_head_vocab_sizes[str(self.layer_id)] for size in order]
        self.num_heads = len(sizes)
        self._head_vocab_sizes = tuple(sizes)
        self.num_rows = sum(sizes)
        head_dim = config.engram_n_embed_per_ngram // self.heads_per_order
        self.head_dim = head_dim
        engram_size = (self.max_ngram_size - 1) * config.engram_n_embed_per_ngram
        owner = 1 if process_group is None else dist.get_world_size(process_group)
        # Named like the checkpoint: ``multi_head_embedding.embedding.weight`` is the flat table of all hash heads.
        self.multi_head_embedding = nn.Module()
        self.multi_head_embedding.embedding = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=-(-self.num_rows // owner) * owner,
            embedding_dim=head_dim,
            initializer_range=config.initializer_range,
            strict_validation=False,
        ).build(process_group=process_group, dtype=dtype)
        self.lookup_table = nn.Buffer(torch.arange(config.engram_lookup_table_size, dtype=torch.int32))
        self.head_vocab_sizes = nn.Buffer(torch.tensor(sizes, dtype=torch.int64), persistent=False)
        self.head_offsets = nn.Buffer(torch.tensor([0, *sizes[:-1]], dtype=torch.int64).cumsum(0), persistent=False)
        self.hash_multipliers = nn.Buffer(
            draw_hash_multipliers(self.seed, self.layer_id, self.max_ngram_size, config.engram_lookup_table_size),
            persistent=False,
        )
        # (storage address, version) of the lookup table the multipliers were drawn for; see ``_lookup_table_state``.
        self._multipliers_drawn_for: tuple[int, int] | None = None
        self.lookup_per_head_norms = nn.ModuleList(
            HYV4FlashWeightOnlyNorm(head_dim, dtype) for _ in range(self.num_heads)
        )
        self.value_proj = nn.Linear(engram_size, self.hidden_size, dtype=dtype)
        self.key_projs = nn.ModuleList(
            nn.Linear(engram_size, self.hidden_size, dtype=dtype) for _ in range(self.hc_mult)
        )
        self.norm_key = nn.ModuleList(HYV4FlashWeightOnlyNorm(self.hidden_size, dtype) for _ in range(self.hc_mult))
        self.norm_query = nn.ModuleList(HYV4FlashWeightOnlyNorm(self.hidden_size, dtype) for _ in range(self.hc_mult))
        self.unified_pre_gamma = nn.Parameter(torch.ones(self.hidden_size, dtype=dtype))
        self.unified_per_hc_scales = nn.ParameterList(
            nn.Parameter(torch.empty(self.hidden_size, dtype=dtype)) for _ in range(self.hc_mult)
        )

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize as the reference does and refresh the hash state.

        ``unified_pre_gamma`` is 1 and the per-stream scales are drawn from N(0, 0.006).
        """
        self.multi_head_embedding.embedding.reset_parameters()
        std = self.initializer_range
        for linear in (self.value_proj, *self.key_projs):
            nn.init.normal_(linear.weight, std=std)
            nn.init.zeros_(linear.bias)
        for norm in (*self.lookup_per_head_norms, *self.norm_key, *self.norm_query):
            nn.init.ones_(norm.weight)
        nn.init.ones_(self.unified_pre_gamma)
        for scale in self.unified_per_hc_scales:
            nn.init.normal_(scale, std=0.006)
        self.head_vocab_sizes.copy_(torch.tensor(self._head_vocab_sizes, dtype=torch.int64))
        self.head_offsets.copy_(torch.tensor([0, *self._head_vocab_sizes[:-1]], dtype=torch.int64).cumsum(0))
        self.lookup_table.copy_(torch.arange(self.lookup_table.numel(), dtype=torch.int32))
        self.refresh_hash_state()

    @torch.no_grad()
    def refresh_hash_state(self) -> None:
        """Re-draw the hash multipliers from the *loaded* lookup table.

        The multipliers are bounded by the number of distinct compressed ids, which is ``max(lookup_table) + 1``
        of the checkpoint's frozen vocabulary map (the reference draws them after loading for the same reason).
        """
        compressed = int(self.lookup_table.max()) + 1
        self.hash_multipliers.copy_(draw_hash_multipliers(self.seed, self.layer_id, self.max_ngram_size, compressed))
        self._multipliers_drawn_for = self._lookup_table_state()

    def _lookup_table_state(self) -> tuple[int, int]:
        """Identity of the current lookup-table contents without reading them: storage address and version counter.

        Any way a checkpoint can reach the buffer changes one of the two: an in-place write (``load_state_dict``,
        ``copy_``) bumps the version, a replaced or moved buffer has a new address. Weight initialization runs
        before the checkpoint is loaded, so multipliers drawn at initialization must not be kept once the frozen
        vocabulary map arrives; comparing this state on every forward costs no device read.
        """
        return (self.lookup_table.data_ptr(), self.lookup_table._version)

    def hash_ngrams(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """Mix shifted token ids into one global table row per hash head.

        Args:
            ngram_tokens: Integer tensor of shape [batch, sequence, max_ngram_size], newest token first.

        Returns:
            int64 tensor of shape [batch, sequence, num_heads] of rows of the layer's flat table, n-gram-order major.
        """
        table_size = self.lookup_table.shape[0]
        compressed = self.lookup_table[ngram_tokens.clamp_max(table_size - 1)]
        compressed = torch.where(ngram_tokens < table_size, compressed, self.lookup_table[self.pad_id])
        mixed = compressed.long() * self.hash_multipliers
        ids = []
        for order in range(2, self.max_ngram_size + 1):
            mix = mixed[..., 0]
            for offset in range(1, order):
                mix = torch.bitwise_xor(mix, mixed[..., offset])
            start = (order - 2) * self.heads_per_order
            ids.append(mix.unsqueeze(-1) % self.head_vocab_sizes[start : start + self.heads_per_order])
        return torch.cat(ids, dim=-1) + self.head_offsets

    def forward(self, streams: torch.Tensor, ngram_tokens: torch.Tensor, engram_mask: torch.Tensor) -> torch.Tensor:
        """Return the update to add to ``streams`` [batch, sequence, hc_mult, hidden] (same shape and dtype).

        ``ngram_tokens`` is an integer tensor [batch, sequence, max_ngram_size]; positions where ``engram_mask``
        [batch, sequence] is False (image / padding) get zero.
        """
        if self._multipliers_drawn_for != self._lookup_table_state():
            self.refresh_hash_state()
        dtype = streams.dtype
        rows = self.hash_ngrams(ngram_tokens)
        emb = self.multi_head_embedding.embedding(rows).to(dtype)  # [B, S, heads, head_dim]
        emb = weighted_rms_norm(emb, self.lookup_per_head_norms)
        emb = emb.flatten(-2)
        batch, seq = emb.shape[:2]
        key_weight = torch.cat([m.weight for m in self.key_projs]).to(dtype)
        key_bias = torch.cat([m.bias for m in self.key_projs]).to(dtype)
        key = F.linear(emb, key_weight, key_bias).view(batch, seq, self.hc_mult, -1)
        key = weighted_rms_norm(key, self.norm_key)
        query = weighted_rms_norm(streams, self.norm_query)
        logit = (key.float() * query.float()).sum(-1) * self.hidden_size**-0.5
        # Signed square root as the reference writes it: ``sign(0) = 0``, so a logit of exactly zero gates at 0.5
        # (``copysign`` would turn the clamp floor into +-1e-3 there).
        gate = torch.sigmoid(logit.abs().clamp_min(1e-6).sqrt() * logit.sign()).unsqueeze(-1)
        main = self.unified_pre_gamma.to(dtype) * F.rms_norm(self.value_proj(emb), (self.hidden_size,))
        main = main.float().unsqueeze(2)
        out = gate * main + main * torch.stack(list(self.unified_per_hc_scales)).float()
        return out.masked_fill(~engram_mask[:, :, None, None], 0.0).to(dtype)
