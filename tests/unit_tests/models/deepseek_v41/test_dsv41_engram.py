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

from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tokenizers import Tokenizer, models
from torch.distributed.device_mesh import DeviceMesh
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
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


@pytest.mark.parametrize("max_ngram", [3, 4])
@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_short_hashes_match_reference(max_ngram, length):
    config = tiny_config(engram_max_ngram_size=max_ngram)
    hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
    tokens = torch.arange(5, 5 + length).expand(2, -1)
    positions = torch.arange(length).expand(2, -1)
    actual = hasher(tokens, positions)
    expected = _reference_hash(hasher, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _reference_residual(
    hidden: torch.Tensor,
    hash_ids: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute branch-local normalized memory injection without a fused norm product.

    Args:
        hidden: Tensor of shape [batch, sequence, hc_mult, hidden].
        hash_ids: Integer tensor of shape [batch, sequence, hash_heads].
        parameters: Reference tensors with keys embed.weight [rows, head_dim],
            wkv.weight [(hc_mult + 1) * hidden, hash_heads * head_dim], and
            q_weight/k_weight [hc_mult, hidden].
        token_mask: Boolean tensor of shape [batch, sequence].
        eps: RMS normalization epsilon.

    Returns:
        Tensor of shape [batch, sequence, hc_mult, hidden].
    """
    hidden_size = hidden.shape[-1]
    projected = F.linear(F.embedding(hash_ids, parameters["embed.weight"]).flatten(-2), parameters["wkv.weight"])
    branches = []
    for branch in range(hidden.shape[-2]):
        key = projected[..., branch * hidden_size : (branch + 1) * hidden_size]
        query = F.rms_norm(hidden[:, :, branch], (hidden_size,), parameters["q_weight"][branch], eps)
        key = F.rms_norm(key, (hidden_size,), parameters["k_weight"][branch], eps)
        score = (query * key).sum(-1) / hidden_size**0.5
        magnitude = score.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(score >= 0, magnitude, -magnitude)) * token_mask
        branches.append(hidden[:, :, branch] + gate.unsqueeze(-1) * projected[..., -hidden_size:])
    return torch.stack(branches, dim=2)


def test_hash_matches_official_fixture_with_compressed_tokens_and_image_barriers() -> None:
    config = DeepseekV41Config(
        vocab_size=9,
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[113, 253],
        engram_max_ngram_size=4,
        engram_vocab_size=11,
        engram_n_heads=2,
        engram_head_dim=4,
        engram_compressed_vocab_size=6,
    )
    vocab = {"[UNK]": 0, " The": 1, "the": 2, "THE": 3, "é": 4, "E": 5, " ": 6, "x": 7, "\ufffd": 8}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")), unk_token="[UNK]"
    )
    hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
    hasher.set_tokenizer(tokenizer)
    input_ids = torch.tensor([[7, 1, 4, 3, 8, 7], [2, 3, 1, 7, 5, 6]])
    mask = torch.tensor([[True, True, False, False, True, True], [True] * 6])
    # Frozen from the downloaded DeepSeek-V4.1-Flash inference/engram.py.
    # Different-cased/accented strings merge; the exact one-space token survives.
    torch.testing.assert_close(hasher.token_map, torch.tensor([0, 1, 1, 1, 2, 2, 3, 4, 5]))
    expected = torch.tensor(
        [
            [
                [[10, 13, 39, 43, 71, 100], [3, 34, 100, 141, 191, 245]],
                [[9, 11, 32, 43, 72, 88], [11, 65, 95, 111, 164, 244]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[1, 19, 37, 51, 80, 83], [25, 60, 102, 116, 153, 214]],
                [[6, 12, 32, 56, 61, 93], [25, 58, 85, 135, 163, 214]],
            ],
            [
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[10, 13, 39, 43, 71, 100], [3, 34, 100, 141, 191, 245]],
                [[3, 22, 34, 48, 78, 97], [28, 64, 84, 121, 184, 200]],
                [[4, 19, 31, 55, 76, 94], [5, 42, 106, 115, 181, 220]],
            ],
        ]
    )
    torch.testing.assert_close(
        hasher(input_ids, torch.arange(input_ids.shape[1]).expand_as(input_ids), token_mask=mask),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("all_masked", [False, True])
def test_engram_residual_and_all_parameter_gradients_match_independent_norms(all_masked):
    torch.manual_seed(17)
    config = tiny_config(engram_trainable=True)
    model = DeepseekV41Engram(config, 1, EngramLayout.from_config(config))
    with torch.no_grad():
        model.q_weight.normal_()
        model.k_weight.normal_()
    parameters = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    hidden = torch.randn(2, 4, config.hc_mult, config.hidden_size, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    ids = torch.randint(0, 3, (2, 4, model.n_hash_cols))
    mask = torch.zeros(2, 4, dtype=torch.bool) if all_masked else torch.tensor([[True, False, False, True]] * 2)
    actual = model(hidden, ids, token_mask=mask)
    expected = _reference_residual(reference_hidden, ids, parameters, mask, config.rms_norm_eps)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[~mask], hidden[~mask], rtol=0, atol=0)


def _invalid_owner_input_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        config = tiny_config(engram_num_embeddings=[17], engram_vocab_size=2, engram_n_heads=1)
        model = DeepseekV41Engram(config, 1, EngramLayout.from_config(config))
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp_shard_cp",))
        model.embed.parallelize_weight(mesh)
        assert model.num_embeddings == 17 and model.embed.num_embeddings == 18
        cases = {
            "hidden_shape": "Engram x",
            "hidden_dtype": "Engram x",
            "hidden_device": "Engram x",
            "hash_shape": "hash_ids",
            "hash_dtype": "hash_ids",
            "hash_device": "hash_ids",
            "negative": "logical row IDs",
            "logical_padding": "logical row IDs",
            "mask_shape": "token_mask",
            "mask_dtype": "token_mask",
            "mask_device": "token_mask",
        }
        lookup_calls = []
        handle = model.embed.register_forward_pre_hook(lambda *_: lookup_calls.append(True))
        try:
            for case, message in cases.items():
                hidden = torch.randn(1, 2, config.hc_mult, config.hidden_size)
                hashes = torch.zeros(1, 2, model.n_hash_cols, dtype=torch.long)
                mask = torch.ones(1, 2, dtype=torch.bool)
                # Exactly one rank is invalid; rank 0 must fail before lookup too.
                if rank == 1:
                    if case == "hidden_shape":
                        hidden = hidden.squeeze(0)
                    elif case == "hidden_dtype":
                        hidden = hidden.long()
                    elif case == "hidden_device":
                        hidden = hidden.to("meta")
                    elif case == "hash_shape":
                        hashes = hashes[:, :, :1]
                    elif case == "hash_dtype":
                        hashes = hashes.float()
                    elif case == "hash_device":
                        hashes = hashes.to("meta")
                    elif case == "negative":
                        hashes[0, 0, 0] = -1
                    elif case == "logical_padding":
                        hashes[0, 0, 0] = 17
                    elif case == "mask_shape":
                        mask = mask[:, :1]
                    elif case == "mask_dtype":
                        mask = mask.long()
                    elif case == "mask_device":
                        mask = mask.to("meta")
                with pytest.raises(ValueError, match=message):
                    model(hidden, hashes, mask)
                assert not lookup_calls
                dist.barrier()
        finally:
            handle.remove()
    finally:
        dist.destroy_process_group()


def test_invalid_inputs_fail_on_all_owners_before_lookup(tmp_path):
    mp.spawn(_invalid_owner_input_worker, args=(str(tmp_path / "invalid-owner-input"),), nprocs=2, join=True)
