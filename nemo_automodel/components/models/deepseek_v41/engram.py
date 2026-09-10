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

"""Engram conditional memory for DeepSeek V4.1.

Ported from the released ``inference/engram.py`` / ``inference/model.py``
(``Engram`` and ``NgramHashState``).  Each position is hashed as the
``max_ngram_size - 1`` n-grams ending there (2-gram .. ``max_ngram_size``-gram),
each split over ``n_heads`` hash heads.  Every (n-gram size, head) pair owns a
prime-sized bucket range in the layer's embedding table.  The looked-up rows
are projected to one key per hyper-connection stream plus a shared value, and
written into the residual streams through a normalized dot-product gate.

Hashing runs over a *compressed* token id space where tokens that normalize
alike (case, accents, whitespace) collapse together.  The mapping is derived
from the tokenizer (:func:`build_compressed_token_map`); the hash multipliers
are derived from the compressed vocabulary size, so the mapping has to match
what training used (``config.engram_compressed_vocab_size``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

_MULTIPLIER_SEED = 10007
_DEAD_TOKEN = -1


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for every ``n < 3.3e24`` (covers the 16M bucket range)."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small_primes:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small_primes:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer: PreTrainedTokenizerFast) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    Returns the lookup (``token_id -> compressed_id``) and the compressed vocab size.
    Mirrors the released reference exactly; the size feeds every hash multiplier.
    """
    from tokenizers import Regex, normalizers

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("DeepSeek V4.1 Engram requires the checkpoint's original fast tokenizer")

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback), drawn from a per-layer RNG.

    Bounded so that ``token_id * multiplier`` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(_MULTIPLIER_SEED * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables (see ``inference/engram.py``)."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head]
    n_heads: int
    head_dim: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, config: DeepseekV41Config) -> EngramLayout | None:
        layer_ids = tuple(int(i) for i in config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = int(config.engram_max_ngram_size), int(config.engram_n_heads)
        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], int(config.engram_vocab_size) - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(int(n) for n in config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=int(config.engram_head_dim),
        )

    def bucket_span(self, layer_hash_index: int) -> int:
        """Total number of buckets addressed by one layer's hash columns."""
        return sum(sum(per_ngram) for per_ngram in self.primes[layer_hash_index])


class DeepseekV41EngramHasher(nn.Module):
    """Map each position to the hash ids of the n-grams ending there.

    Training is stateless: the whole sequence is visible, so look-back is a
    shift along the sequence axis. A non-increasing position starts a new
    segment. Look-back stops at segment boundaries and at masked tokens, so
    an n-gram never spans a document boundary or a padding/image token.

    The complete token map is derived from the required tokenizer during
    construction. Its immutable integer values are retained for restoration
    after meta-device materialization, without retaining the tokenizer object.
    """

    def __init__(self, config: DeepseekV41Config, layout: EngramLayout, tokenizer: PreTrainedTokenizerFast) -> None:
        """Construct hashes from the checkpoint tokenizer's complete vocabulary.

        Args:
            config: Engram dimensions, compressed vocabulary size, and raw pad ID.
            layout: Per-layer prime buckets and logical table capacities.
            tokenizer: Required fast tokenizer used to normalize every raw token.

        Raises:
            TypeError: The tokenizer is not a fast tokenizer.
            ValueError: The compressed vocabulary differs from the configuration,
                the pad ID lies outside the tokenizer, or a table cannot hold its buckets.
        """
        super().__init__()
        self.layout = layout
        lookup, self.compressed_vocab_size = build_compressed_token_map(tokenizer)
        self._token_map_values = tuple(lookup)
        if self.compressed_vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                "Engram compressed tokenizer vocabulary mismatch: "
                f"got {self.compressed_vocab_size}, expected {config.engram_compressed_vocab_size}; "
                "hash multipliers depend on this size"
            )
        if not 0 <= config.engram_pad_token_id < len(self._token_map_values):
            raise ValueError(f"Engram pad token ID {config.engram_pad_token_id} lies outside the tokenizer vocabulary")
        self.pad_id = self._token_map_values[config.engram_pad_token_id]
        for layer_hash_index, layer_id in enumerate(layout.layer_ids):
            bucket_span = layout.bucket_span(layer_hash_index)
            num_embeddings = layout.num_embeddings[layer_hash_index]
            if bucket_span > num_embeddings:
                raise ValueError(
                    f"Engram layer {layer_id} requires {bucket_span} rows for its hash buckets, "
                    f"but engram_num_embeddings specifies {num_embeddings}"
                )
        # Keep deterministic source values outside device buffers, so meta ->
        # materialized construction cannot erase the hash identities.
        with torch.device("cpu"):
            multipliers = compute_hash_multipliers(
                layout.layer_ids, layout.max_ngram_size, self.compressed_vocab_size
            ).tolist()
        self._multiplier_values = tuple(tuple(row) for row in multipliers)
        primes = torch.tensor(layout.primes, dtype=torch.int64)  # [n_layers, n_ngrams, n_heads]
        flat_primes = primes.flatten(1)
        self.register_buffer("primes", primes, persistent=False)
        # Exclusive prefix sum: each (n-gram size, head) column owns its own bucket range.
        self.register_buffer("offsets", flat_primes.cumsum(1) - flat_primes, persistent=False)
        self.register_buffer(
            "multipliers",
            torch.tensor(self._multiplier_values, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("token_map", torch.tensor(self._token_map_values, dtype=torch.int64), persistent=False)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Restore deterministic integer buffers after meta-device materialization.

        Args:
            buffer_device: Destination device, or the current primes device.
                Restores int64 primes [layers, ngram_orders, heads], offsets
                [layers, hash_columns], multipliers [layers, max_ngram_size],
                and token_map [complete tokenizer vocabulary].
                Matching buffer storage is updated in place; otherwise it is replaced.
        """
        device = self.primes.device if buffer_device is None else buffer_device
        primes = torch.tensor(self.layout.primes, dtype=torch.int64, device=device)
        flat_primes = primes.flatten(1)
        for name, value in (
            ("primes", primes),
            ("offsets", flat_primes.cumsum(1) - flat_primes),
            ("multipliers", torch.tensor(self._multiplier_values, dtype=torch.int64, device=device)),
            ("token_map", torch.tensor(self._token_map_values, dtype=torch.int64, device=device)),
        ):
            buffer = self.get_buffer(name)
            if buffer.shape == value.shape and buffer.device == value.device:
                buffer.copy_(value)
            else:
                setattr(self, name, value)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return hash ids shaped ``[B, L, n_engram_layers, n_hash_cols]``.

        Args:
            input_ids: Int32 or int64 raw token IDs ``[B, L]``, each within the
                token-map vocabulary. The input tensor is not modified.
            position_ids: Int32 or int64 positions ``[B, L]`` matching input_ids.
                A position less than or equal to its predecessor starts a new
                segment. The input tensor is not modified.
            token_mask: ``[B, L]`` bool, ``False`` for tokens that take no part in an n-gram.

        Returns:
            Int64 hash IDs ``[B, L, n_engram_layers, n_hash_cols]`` in the
            corresponding layers' logical table ranges.

        Raises:
            ValueError: Raw token IDs have an invalid shape, dtype, or range,
                or positions have an invalid shape or dtype.
        """
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram input_ids must be an int32/int64 tensor of shape [batch, sequence]")
        if input_ids.numel() and bool(((input_ids < 0) | (input_ids >= self.token_map.numel())).any()):
            raise ValueError("Engram input_ids contains a token ID outside the tokenizer vocabulary")
        if position_ids.shape != input_ids.shape or position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram position_ids must be int32/int64 with the same shape as input_ids")
        batch, sequence = input_ids.shape
        compressed = self.token_map[input_ids.long()]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, _DEAD_TOKEN))

        positions = torch.arange(sequence, device=input_ids.device).expand(batch, sequence)
        starts = torch.cat(
            (torch.ones_like(position_ids[:, :1], dtype=torch.bool), position_ids[:, 1:] <= position_ids[:, :-1]),
            dim=1,
        )
        segment_start = torch.cummax(torch.where(starts, positions, 0), dim=1).values
        tokens = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source_positions = positions - shift
            source = compressed.gather(1, source_positions.clamp_min(0))
            blocked = blocked | (source_positions < segment_start) | (source == _DEAD_TOKEN)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

        # XOR the multiplied ids together one lookback at a time, so the running value after
        # step i is the hash of the (i+1)-gram; each lands in its own prime-sized bucket range
        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_engram_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class DeepseekV41Engram(nn.Module):
    """Write an n-gram lookup into the residual streams, gated by how well it matches them.

    ``x``: ``[B, L, hc_mult, dim]``.  The hash ids fetch ``n_hash_cols`` rows of
    ``head_dim``; ``wkv`` turns them into one key per hyper-connection stream plus
    a shared value.  The gate is a normalized dot product of stream against key
    passed through a signed square root and a sigmoid (matching the training kernel).
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        layout: EngramLayout,
        backend: BackendConfig,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Construct the projections and a trainable contiguous row-owner table.

        Args:
            config: Model dimensions and parameter dtype.
            layer_idx: Decoder layer containing this Engram.
            layout: Logical hash-table row ranges for all Engram layers.
            backend: Projection backend selected for the enclosing model.
            engram_process_group: Runtime row-owner group. None keeps the
                complete table local, even when distributed execution is
                initialized. Physical rows are padded evenly across explicit
                owners without changing the logical hash ranges.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_hash_index = layout.layer_ids.index(layer_idx)
        self.dim = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.n_hash_cols = layout.n_hash_cols
        self.eps = float(config.rms_norm_eps)
        self.initializer_range = config.initializer_range
        self.clamp_value = 1e-6
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.num_embeddings = layout.num_embeddings[self.layer_hash_index]
        owner_world_size = dist.get_world_size(engram_process_group) if engram_process_group is not None else 1
        padded_rows = (self.num_embeddings + owner_world_size - 1) // owner_world_size * owner_world_size
        table_config = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=padded_rows,
            embedding_dim=layout.head_dim,
            initializer_range=config.initializer_range,
        )
        self.embed = table_config.build(process_group=engram_process_group, dtype=model_dtype)
        self.wkv = initialize_linear_module(
            backend.linear,
            layout.n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            dtype=model_dtype,
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=model_dtype))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=model_dtype))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize local table storage and projections, leaving padded rows zero."""
        self.embed.reset_parameters()
        self._zero_padding_rows()
        nn.init.normal_(self.wkv.weight, mean=0.0, std=self.initializer_range)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)
        self.embed.mark_sharding_contract()

    @torch.no_grad()
    def _zero_padding_rows(self) -> None:
        """Clear physical rows beyond the logical checkpoint on their local owner."""
        local_weight = self.embed.weight.to_local() if isinstance(self.embed.weight, DTensor) else self.embed.weight
        valid_local_rows = max(0, min(local_weight.shape[0], self.num_embeddings - self.embed.global_row_start))
        local_weight[valid_local_rows:].zero_()

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Inject memory rows through the normalized residual gate.

        Args:
            x: Floating-point residual streams [batch, sequence, hc_mult, dim].
            hash_ids: Int32 or int64 logical row IDs [batch, sequence, n_hash_cols].
                Physical padding rows beyond num_embeddings are not valid IDs.
            token_mask: Optional boolean mask [batch, sequence]; false keeps the
                corresponding residual streams unchanged. All inputs must be on
                the same device as the owner table.

        Returns:
            Tensor with the shape and dtype of x, without modifying its storage.

        Raises:
            ValueError: Local residual or hash shapes are invalid, any owner
                supplies invalid hash dtypes or logical row IDs, or the local
                token mask has an invalid shape or dtype.
        """
        if x.ndim != 4 or x.shape[-2:] != (self.hc_mult, self.dim):
            raise ValueError("Engram x must have shape [batch, sequence, hc_mult, dim]")
        if hash_ids.shape != (*x.shape[:2], self.n_hash_cols):
            raise ValueError("Engram hash_ids must have shape [batch, sequence, n_hash_cols] matching x")
        valid = hash_ids.dtype in (torch.int32, torch.int64)
        if valid and hash_ids.numel():
            valid = bool(((hash_ids >= 0) & (hash_ids < self.num_embeddings)).all())
        validity = torch.tensor(int(valid), device=hash_ids.device, dtype=torch.int32)
        if self.embed.process_group is not None:
            dist.all_reduce(validity, op=dist.ReduceOp.MIN, group=self.embed.process_group)
        if not bool(validity):
            raise ValueError(
                f"Engram hash_ids must be integer logical row IDs in [0, {self.num_embeddings}) on every rank"
            )
        if token_mask is not None and (token_mask.shape != x.shape[:2] or token_mask.dtype != torch.bool):
            raise ValueError("Engram token_mask must be bool with shape [batch, sequence]")
        rows = self.embed(hash_ids)  # [B, L, n_hash_cols, head_dim]
        kv = self.wkv(rows.flatten(-2).to(x.dtype))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h = x.float()
        # normalized per (token, hc copy) over ``dim``, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
