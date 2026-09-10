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
import torch.nn.functional as F
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
        if not layer_ids or not config.engram_enabled:
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
    shift along the sequence axis.  Look-back stops at the start of each
    document (``position_ids == 0``) and at masked tokens, so an n-gram never
    spans a document boundary or a padding/image token.

    The token map is tokenizer-derived and has to be attached before the first
    forward (:meth:`set_token_map` / :meth:`set_tokenizer`).  When the
    compressed vocabulary equals the model vocabulary (tiny test configs) an
    identity map is used automatically.
    """

    def __init__(self, config: DeepseekV41Config, layout: EngramLayout):
        super().__init__()
        self.layout = layout
        for layer_hash_index, layer_id in enumerate(layout.layer_ids):
            bucket_span = layout.bucket_span(layer_hash_index)
            num_embeddings = layout.num_embeddings[layer_hash_index]
            if bucket_span > num_embeddings:
                raise ValueError(
                    f"Engram layer {layer_id} requires {bucket_span} rows for its hash buckets, "
                    f"but engram_num_embeddings specifies {num_embeddings}"
                )
        self.vocab_size = int(config.vocab_size)
        self.compressed_vocab_size = int(config.engram_compressed_vocab_size) or self.vocab_size
        self.raw_pad_token_id = int(config.engram_pad_token_id)
        self._token_map_values: tuple[int, ...] = ()
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
        self.register_buffer("token_map", torch.empty(0, dtype=torch.int64), persistent=False)
        if self.compressed_vocab_size == self.vocab_size:
            self.set_token_map(list(range(self.vocab_size)), self.vocab_size)

    @property
    def has_token_map(self) -> bool:
        return self.token_map.numel() > 0

    def set_token_map(self, lookup: list[int] | torch.Tensor, compressed_vocab_size: int) -> None:
        """Attach the compressed token IDs without retaining a tokenizer object.

        Args:
            lookup: Integer list or materialized tensor [vocabulary], containing
                IDs in [0, compressed_vocab_size). Its input storage is not mutated.
            compressed_vocab_size: Number of compressed IDs; must match the hash configuration.
        """
        if int(compressed_vocab_size) != self.compressed_vocab_size:
            raise ValueError(
                "Engram compressed vocabulary mismatch: tokenizer yields "
                f"{compressed_vocab_size} ids but config.engram_compressed_vocab_size is "
                f"{self.compressed_vocab_size}. Every hash multiplier derives from this size."
            )
        if isinstance(lookup, torch.Tensor) and lookup.is_meta:
            raise ValueError("Engram token-map setup requires materialized integer values")
        lookup = torch.as_tensor(lookup, dtype=torch.int64, device="cpu")
        if lookup.ndim != 1:
            raise ValueError("Engram token map must be a one-dimensional lookup")
        if lookup.numel() < self.vocab_size:
            raise ValueError(f"Engram token map covers {lookup.numel()} ids, expected at least {self.vocab_size}")
        selected = lookup[: self.vocab_size]
        if torch.any((selected < 0) | (selected >= self.compressed_vocab_size)):
            raise ValueError("Engram token-map IDs must be within the compressed vocabulary")
        self._token_map_values = tuple(selected.tolist())
        self.token_map = selected.to(self.primes.device)
        self.pad_id = self._token_map_values[self.raw_pad_token_id]

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerFast) -> None:
        """Derive and attach the compressed token map from a HuggingFace tokenizer."""
        lookup, size = build_compressed_token_map(tokenizer)
        self.set_token_map(lookup, size)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Restore deterministic integer buffers after meta-device materialization.

        Args:
            buffer_device: Destination device, or the current primes device.
                Restores int64 primes [layers, ngram_orders, heads], offsets
                [layers, hash_columns], multipliers [layers, max_ngram_size],
                and token_map [vocabulary] (empty until explicit tokenizer setup).
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
            position_ids: ``[B, L]`` document-relative positions (``0`` starts a document).
            token_mask: ``[B, L]`` bool, ``False`` for tokens that take no part in an n-gram.

        Returns:
            Int64 hash IDs ``[B, L, n_engram_layers, n_hash_cols]`` in the
            corresponding layers' logical table ranges.

        Raises:
            RuntimeError: The compressed token map has not been attached.
            ValueError: Raw token IDs have an invalid shape, dtype, or range.
        """
        if not self.has_token_map:
            raise RuntimeError(
                "DeepseekV41EngramHasher has no token map. Call model.set_engram_tokenizer(tokenizer) "
                "or set config.engram_compressed_vocab_size to the model vocab size for identity hashing."
            )
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram input_ids must be an int32/int64 tensor of shape [batch, sequence]")
        if input_ids.numel() and bool(((input_ids < 0) | (input_ids >= self.token_map.numel())).any()):
            raise ValueError("Engram input_ids contains a token ID outside the tokenizer vocabulary")
        compressed = self.token_map[input_ids.long()]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, _DEAD_TOKEN))

        tokens = []
        blocked = torch.zeros_like(compressed, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            if shift == 0:
                source = compressed
            else:
                source = F.pad(compressed, (shift, 0), value=_DEAD_TOKEN)[:, : compressed.shape[1]]
            blocked = blocked | (position_ids < shift) | (source == _DEAD_TOKEN)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
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
        """Construct the projections and a contiguous row-owner table.

        Args:
            config: Model dimensions and the existing table trainability setting.
            layer_idx: Decoder layer containing this Engram.
            layout: Logical hash-table row ranges for all Engram layers.
            backend: Projection backend selected for the enclosing model.
            engram_process_group: Runtime row owners, defaulting to WORLD when
                distributed world size exceeds one. Without distributed owners,
                the complete table is local. Physical rows are padded evenly
                across owners without changing the logical hash ranges.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_hash_index = layout.layer_ids.index(layer_idx)
        self.dim = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.n_hash_cols = layout.n_hash_cols
        self.eps = float(config.rms_norm_eps)
        self.clamp_value = 1e-6
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.num_embeddings = layout.num_embeddings[self.layer_hash_index]
        if engram_process_group is None and dist.is_initialized() and dist.get_world_size() > 1:
            engram_process_group = dist.group.WORLD
        owner_world_size = dist.get_world_size(engram_process_group) if engram_process_group is not None else 1
        padded_rows = (self.num_embeddings + owner_world_size - 1) // owner_world_size * owner_world_size
        table_config = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=padded_rows,
            embedding_dim=layout.head_dim,
            # Preserve nn.Embedding's constructor initialization; model-level
            # init_weights supplies config.initializer_range afterwards.
            initializer_range=1.0,
        )
        self.embed = table_config.build(process_group=engram_process_group, dtype=model_dtype)
        self.embed.weight.requires_grad_(bool(config.engram_trainable))
        self._zero_padding_rows()
        self.wkv = initialize_linear_module(
            backend.linear,
            layout.n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            dtype=model_dtype,
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=model_dtype))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=model_dtype))

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize local table storage and projections, leaving padded rows zero."""
        local_weight = self.embed.weight.to_local() if isinstance(self.embed.weight, DTensor) else self.embed.weight
        nn.init.normal_(local_weight, mean=0.0, std=init_std)
        self._zero_padding_rows()
        self.embed.mark_sharding_contract()
        nn.init.trunc_normal_(self.wkv.weight, mean=0.0, std=init_std)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    @torch.no_grad()
    def _zero_padding_rows(self) -> None:
        """Clear physical rows beyond the logical checkpoint on their local owner."""
        local_weight = self.embed.weight.to_local() if isinstance(self.embed.weight, DTensor) else self.embed.weight
        valid_local_rows = max(0, self.num_embeddings - self.embed.global_row_start)
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
            ValueError: Any owner rank supplies an invalid shape, dtype, device,
                or logical row ID. Every owner validates before entering lookup.
        """
        owner_device = self.embed.weight.device
        x_valid = (
            isinstance(x, torch.Tensor)
            and x.ndim == 4
            and x.shape[-2:] == (self.hc_mult, self.dim)
            and x.is_floating_point()
            and x.device == owner_device
        )
        hashes_valid = (
            x_valid
            and isinstance(hash_ids, torch.Tensor)
            and hash_ids.shape == (*x.shape[:2], self.n_hash_cols)
            and hash_ids.dtype in (torch.int32, torch.int64)
            and hash_ids.device == owner_device
        )
        if hashes_valid and hash_ids.numel():
            hashes_valid = bool(((hash_ids >= 0) & (hash_ids < self.num_embeddings)).all())
        mask_valid = token_mask is None or (
            x_valid
            and isinstance(token_mask, torch.Tensor)
            and token_mask.shape == x.shape[:2]
            and token_mask.dtype == torch.bool
            and token_mask.device == owner_device
        )
        # The shared table validates physical IDs, but knows neither the model's
        # logical row count nor its residual/mask shapes. Combine those checks
        # into one collective so a local error cannot strand another requester.
        validity = torch.tensor((x_valid, hashes_valid, mask_valid), device=owner_device, dtype=torch.int32)
        if self.embed.process_group is not None:
            dist.all_reduce(validity, op=dist.ReduceOp.MIN, group=self.embed.process_group)
        x_valid, hashes_valid, mask_valid = validity.tolist()
        if not x_valid:
            raise ValueError(
                "Engram x must be floating-point [batch, sequence, hc_mult, dim] on the table device on every owner rank"
            )
        if not hashes_valid:
            raise ValueError(
                "Engram hash_ids must be int32/int64 [batch, sequence, n_hash_cols] on the table device, "
                f"with logical row IDs in [0, {self.num_embeddings}) on every owner rank"
            )
        if not mask_valid:
            raise ValueError("Engram token_mask must be bool [batch, sequence] on the table device on every owner rank")
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
