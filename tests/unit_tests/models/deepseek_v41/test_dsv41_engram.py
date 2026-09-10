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

"""CPU unit tests for Engram hashing and the Engram residual-stream module."""

import numpy as np
import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.engram import (
    DeepseekV41Engram,
    DeepseekV41EngramHasher,
    EngramLayout,
    _is_prime,
    compute_hash_multipliers,
    find_next_prime,
)
from tests.unit_tests.models.deepseek_v41.conftest import tiny_config


def _reference_hash(hasher: DeepseekV41EngramHasher, input_ids: torch.Tensor) -> torch.Tensor:
    """Port of the released ``NgramHashState.forward`` for one document starting at position 0."""
    layout = hasher.layout
    batch, seqlen = input_ids.shape
    compressed = hasher.token_map[input_ids]
    positions = torch.arange(seqlen).expand(batch, seqlen)
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(layout.max_ngram_size):
        source = compressed.gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift)
        tokens.append(torch.where(blocked, torch.full_like(source, hasher.pad_id), source))
    tokens = torch.stack(tokens, dim=-1)
    products = tokens.unsqueeze(2) * hasher.multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, layout.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % hasher.primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + hasher.offsets


class TestLayout:
    def test_primes_are_distinct_and_above_bucket_size(self):
        config = tiny_config()
        layout = EngramLayout.from_config(config)
        flat = [p for layer in layout.primes for per_ngram in layer for p in per_ngram]
        assert len(flat) == len(set(flat)) == (config.engram_max_ngram_size - 1) * config.engram_n_heads
        assert all(_is_prime(p) and p >= config.engram_vocab_size for p in flat)
        assert layout.n_hash_cols == 4
        assert layout.bucket_span(0) == sum(flat)

    def test_disabled_engram_has_no_layout(self):
        assert EngramLayout.from_config(tiny_config(engram_enabled=False)) is None
        assert EngramLayout.from_config(tiny_config(engram_layer_ids=[], engram_num_embeddings=[])) is None

    def test_prime_helpers(self):
        def trial_division(n: int) -> bool:
            return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))

        assert [n for n in range(2, 30) if _is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert all(_is_prime(n) == trial_division(n) for n in range(16_000_000, 16_000_300))
        first = find_next_prime(16_000_000 - 1, set())
        second = find_next_prime(16_000_000 - 1, {first})
        assert trial_division(first) and trial_division(second) and second > first
        assert all(not trial_division(n) for n in range(16_000_000, first))

    def test_multipliers_are_odd_and_bounded(self):
        mult = compute_hash_multipliers((1, 14), 4, 99092)
        assert mult.shape == (2, 4) and mult.dtype == torch.int64
        assert (mult % 2 == 1).all()
        assert (mult < np.iinfo(np.int64).max // 99092).all()
        assert not torch.equal(mult[0], mult[1])


class TestHasher:
    def test_identity_map_matches_reference_for_single_document(self):
        config = tiny_config()
        hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
        assert hasher.has_token_map
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 7))
        position_ids = torch.arange(7).expand(2, -1)
        hashes = hasher(input_ids, position_ids)
        assert hashes.shape == (2, 7, 1, hasher.layout.n_hash_cols)
        assert torch.equal(hashes, _reference_hash(hasher, input_ids))
        assert (hashes >= 0).all() and (hashes < hasher.layout.bucket_span(0)).all()

    def test_document_boundaries_reset_lookback(self):
        config = tiny_config()
        hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
        doc_a = torch.tensor([[5, 6, 7]])
        doc_b = torch.tensor([[9, 10]])
        packed = torch.cat([doc_a, doc_b], dim=1)
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        packed_hashes = hasher(packed, position_ids)
        separate = torch.cat(
            [hasher(doc_a, torch.arange(3).unsqueeze(0)), hasher(doc_b, torch.arange(2).unsqueeze(0))], dim=1
        )
        assert torch.equal(packed_hashes, separate)

    def test_token_mask_blocks_ngrams(self):
        config = tiny_config()
        hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
        ids = torch.tensor([[3, 4, 5, 6]])
        pos = torch.arange(4).unsqueeze(0)
        mask = torch.tensor([[True, False, True, True]])
        masked = hasher(ids, pos, mask)
        plain = hasher(ids, pos)
        assert torch.equal(masked[:, 0], plain[:, 0])
        assert not torch.equal(masked[:, 2], plain[:, 2])  # the 2-gram at pos 2 looks back into the dead token

    def test_token_map_validation(self):
        config = tiny_config(engram_compressed_vocab_size=100)
        hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
        assert not hasher.has_token_map
        with pytest.raises(RuntimeError, match="token map"):
            hasher(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long))
        with pytest.raises(ValueError, match="compressed vocabulary mismatch"):
            hasher.set_token_map(list(range(config.vocab_size)), 99)
        with pytest.raises(ValueError, match="covers"):
            hasher.set_token_map(list(range(10)), 100)
        hasher.set_token_map([i % 100 for i in range(config.vocab_size)], 100)
        assert hasher.has_token_map and hasher.pad_id == 2


class TestEngramModule:
    def test_forward_shape_and_gate_mask(self):
        config = tiny_config()
        layout = EngramLayout.from_config(config)
        engram = DeepseekV41Engram(config, layer_idx=1, layout=layout).float()
        assert not engram.embed.weight.requires_grad
        torch.manual_seed(0)
        with torch.no_grad():
            engram.embed.weight.normal_()
            engram.wkv.weight.normal_(std=0.1)
        x = torch.randn(2, 3, config.hc_mult, config.hidden_size)
        hash_ids = torch.randint(0, layout.num_embeddings[0], (2, 3, layout.n_hash_cols))
        out = engram(x, hash_ids)
        assert out.shape == x.shape and not torch.allclose(out, x)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        masked = engram(x, hash_ids, mask)
        assert torch.allclose(masked[1], x[1])
        assert torch.allclose(masked[0, 1], x[0, 1]) and not torch.allclose(masked[0, 0], x[0, 0])

    def test_trainable_flag(self):
        config = tiny_config(engram_trainable=True)
        engram = DeepseekV41Engram(config, layer_idx=1, layout=EngramLayout.from_config(config))
        assert engram.embed.weight.requires_grad
