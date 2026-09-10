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

"""Frozen official hash fixtures and CPU Engram residual/owner-gradient parity."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tokenizers import Tokenizer, models
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram, DeepseekV41NgramHash


def _tiny_config() -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        hidden_size=8,
        hc_mult=2,
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[113, 253],
        engram_max_ngram_size=4,
        engram_vocab_size=11,
        engram_n_heads=2,
        engram_head_dim=4,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=6,
        dtype="float32",
        initializer_range=0.1,
    )


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, " The": 1, "the": 2, "THE": 3, "é": 4, "E": 5, " ": 6, "x": 7, "\ufffd": 8}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


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
    hasher = DeepseekV41NgramHash(_tiny_config(), _tokenizer())
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
    torch.testing.assert_close(hasher(input_ids, token_mask=mask), expected, rtol=0, atol=0)


def test_packed_hashes_equal_individual_documents() -> None:
    hasher = DeepseekV41NgramHash(_tiny_config(), _tokenizer())
    first, second = torch.tensor([[7, 4, 8]]), torch.tensor([[1, 7, 6, 5]])
    packed_ids = torch.cat((first, second), dim=1)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    expected = torch.cat((hasher(first), hasher(second)), dim=1)
    torch.testing.assert_close(hasher(packed_ids, position_ids=positions), expected, rtol=0, atol=0)
    assert not torch.equal(hasher(packed_ids), expected)


def test_hash_metadata_restored_after_meta_materialization() -> None:
    tokenizer = _tokenizer()
    reference = DeepseekV41NgramHash(_tiny_config(), tokenizer)
    with torch.device("meta"):
        actual = DeepseekV41NgramHash(_tiny_config(), tokenizer)
    actual.to_empty(device="cpu")
    actual.init_weights()
    for name, value in reference.named_buffers():
        torch.testing.assert_close(actual.get_buffer(name), value, rtol=0, atol=0)
    ids = torch.tensor([[7, 5, 3]])
    torch.testing.assert_close(actual(ids), reference(ids), rtol=0, atol=0)


def test_hash_rejects_mismatched_compressed_vocabulary() -> None:
    config = _tiny_config()
    config.engram_compressed_vocab_size = 7
    with pytest.raises(ValueError, match="compressed tokenizer vocabulary mismatch"):
        DeepseekV41NgramHash(config, _tokenizer())


@pytest.mark.parametrize("all_masked", [False, True])
def test_residual_and_parameter_gradients_match_reference(all_masked: bool) -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    model = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    with torch.no_grad():
        model.q_weight.normal_()
        model.k_weight.normal_()
    parameters = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    hidden = torch.randn(2, 4, config.hc_mult, config.hidden_size, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    ids = torch.randint(0, 3, (2, 4, 6))  # Repeated rows exercise embedding gradient accumulation.
    mask = torch.zeros(2, 4, dtype=torch.bool) if all_masked else torch.tensor([[True, False, False, True]] * 2)
    actual = model(hidden, ids, token_mask=mask)
    expected = _reference_residual(reference_hidden, ids, parameters, mask, config.rms_norm_eps)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[~mask], hidden[~mask], rtol=0, atol=0)


def test_engram_meta_initialization_is_finite_and_trainable() -> None:
    config = _tiny_config()
    with torch.device("meta"):
        model = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    model.to_empty(device="cpu")
    model.init_weights()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    hidden = torch.randn(1, 3, config.hc_mult, config.hidden_size, requires_grad=True)
    ids = torch.zeros(1, 3, 6, dtype=torch.long)
    output = model(hidden, ids)
    output.backward(torch.randn_like(output))
    assert torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def _owner_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(22)
        config = _tiny_config()
        reference = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
        model = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"), process_group=dist.group.WORLD)
        assert model.num_embeddings == 113 and model.embed.num_embeddings == 114
        valid_rows = 57 if rank == 0 else 56
        assert torch.count_nonzero(model.embed.weight[valid_rows:]) == 0
        assert torch.count_nonzero(model.embed.weight[:valid_rows]) > 0
        with torch.device("meta"):
            materialized = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"), process_group=dist.group.WORLD)
        materialized.to_empty(device="cpu")
        materialized.init_weights()
        assert torch.count_nonzero(materialized.embed.weight[valid_rows:]) == 0
        assert torch.count_nonzero(materialized.embed.weight[:valid_rows]) > 0
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp_shard_cp",))
        owner_parameter = model.embed.parallelize_weight(mesh)
        assert isinstance(owner_parameter, DTensor) and owner_parameter.placements == (Shard(0),)
        model.init_weights()
        assert torch.count_nonzero(owner_parameter.to_local()[valid_rows:]) == 0
        assert torch.count_nonzero(owner_parameter.to_local()[:valid_rows]) > 0
        with torch.no_grad():
            padded_weight = F.pad(reference.embed.weight, (0, 0, 0, 1))
            owner_parameter.to_local().copy_(padded_weight[rank * 57 : (rank + 1) * 57])
            for name in ("wkv.weight", "q_weight", "k_weight"):
                model.get_parameter(name).copy_(reference.get_parameter(name))
        torch.manual_seed(71)
        hidden = torch.randn(2, 3, config.hc_mult, config.hidden_size)
        upstream = torch.randn_like(hidden)
        ids = torch.tensor([[[0, 0, 1, 2, 56, 57]] * 3, [[57, 57, 90, 112, 0, 1]] * 3])
        mask = torch.tensor([[True, False, True], [True, True, False]])
        expected = reference(hidden, ids, token_mask=mask)
        actual = model(hidden[rank : rank + 1], ids[rank : rank + 1], token_mask=mask[rank : rank + 1])
        torch.testing.assert_close(actual, expected[rank : rank + 1], rtol=1e-5, atol=1e-6)
        expected.backward(upstream)
        actual.backward(upstream[rank : rank + 1])
        expected_table_grad = F.pad(reference.embed.weight.grad, (0, 0, 0, 1))[rank * 57 : (rank + 1) * 57]
        torch.testing.assert_close(owner_parameter.grad.to_local(), expected_table_grad, rtol=1e-5, atol=1e-6)
        for name in ("wkv.weight", "q_weight", "k_weight"):
            dist.all_reduce(model.get_parameter(name).grad)
            torch.testing.assert_close(model.get_parameter(name).grad, reference.get_parameter(name).grad)
        torch.optim.SGD(model.parameters(), lr=0.03).step()
        torch.optim.SGD(reference.parameters(), lr=0.03).step()
        expected_weight = F.pad(reference.embed.weight, (0, 0, 0, 1))[rank * 57 : (rank + 1) * 57]
        torch.testing.assert_close(owner_parameter.to_local(), expected_weight, rtol=1e-5, atol=1e-6)
        # One rank requests the physical padding row: both ranks must reject
        # before either enters the table's ID/value collectives.
        invalid = ids[rank : rank + 1].clone()
        if rank == 1:
            invalid[0, 0, 0] = 113
        with pytest.raises(ValueError, match="logical row IDs"):
            model(hidden[rank : rank + 1], invalid)
    finally:
        dist.destroy_process_group()


def test_two_rank_owner_lookup_gradients_update_and_logical_bounds(tmp_path: Path) -> None:
    mp.spawn(_owner_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
