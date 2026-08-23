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

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.loss import vocab_parallel_entropy, vocab_parallel_log_probs


def _run_vocab_parallel_parity(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("tp",))
        temperature = 0.7
        cases = (((6,), 6), ((2, 3), 5))

        for case_index, (leading_shape, vocab_size) in enumerate(cases):
            torch.manual_seed(1234 + case_index)
            full_logits = torch.randn(*leading_shape, vocab_size, dtype=torch.float32)
            targets = (torch.arange(math.prod(leading_shape), dtype=torch.long) % vocab_size).reshape(leading_shape)
            targets.reshape(-1)[-1] = vocab_size - 1
            chunk_size = (vocab_size + world_size - 1) // world_size
            shard_offset = min(rank * chunk_size, vocab_size)
            shard_size = min(chunk_size, vocab_size - shard_offset)
            full_stride = full_logits.stride()

            local_log_prob_logits = (
                full_logits[..., shard_offset : shard_offset + shard_size].detach().clone().requires_grad_()
            )
            distributed_log_prob_logits = DTensor.from_local(
                local_log_prob_logits,
                mesh,
                [Shard(-1)],
                run_check=False,
                shape=full_logits.shape,
                stride=full_stride,
            )
            actual_log_probs = vocab_parallel_log_probs(
                distributed_log_prob_logits,
                targets,
                temperature=temperature,
            )

            reference_log_prob_logits = full_logits.detach().clone().requires_grad_()
            reference_log_probs = torch.log_softmax(reference_log_prob_logits / temperature, dim=-1)
            reference_log_probs = reference_log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            assert actual_log_probs.dtype == torch.float32
            torch.testing.assert_close(actual_log_probs, reference_log_probs, rtol=1e-6, atol=1e-6)

            upstream = torch.linspace(0.25, 1.25, targets.numel(), dtype=torch.float32).reshape(leading_shape)
            (actual_log_probs * upstream).sum().backward()
            (reference_log_probs * upstream).sum().backward()
            assert local_log_prob_logits.grad is not None
            torch.testing.assert_close(
                local_log_prob_logits.grad,
                reference_log_prob_logits.grad[..., shard_offset : shard_offset + shard_size],
                rtol=2e-6,
                atol=2e-6,
            )

            local_entropy_logits = (
                full_logits[..., shard_offset : shard_offset + shard_size].detach().clone().requires_grad_()
            )
            distributed_entropy_logits = DTensor.from_local(
                local_entropy_logits,
                mesh,
                [Shard(-1)],
                run_check=False,
                shape=full_logits.shape,
                stride=full_stride,
            )
            actual_entropy = vocab_parallel_entropy(distributed_entropy_logits, temperature=temperature)

            reference_entropy_logits = full_logits.detach().clone().requires_grad_()
            reference_log_distribution = torch.log_softmax(reference_entropy_logits / temperature, dim=-1)
            reference_distribution = reference_log_distribution.exp()
            reference_entropy = -(reference_distribution * reference_log_distribution).sum(dim=-1)
            assert actual_entropy.dtype == torch.float32
            torch.testing.assert_close(actual_entropy, reference_entropy, rtol=2e-6, atol=2e-6)

            (actual_entropy * upstream).sum().backward()
            (reference_entropy * upstream).sum().backward()
            assert local_entropy_logits.grad is not None
            torch.testing.assert_close(
                local_entropy_logits.grad,
                reference_entropy_logits.grad[..., shard_offset : shard_offset + shard_size],
                rtol=3e-6,
                atol=3e-6,
            )

            with torch.no_grad():
                no_grad_log_probs = vocab_parallel_log_probs(
                    distributed_log_prob_logits,
                    targets,
                    temperature=temperature,
                )
                no_grad_entropy = vocab_parallel_entropy(distributed_entropy_logits, temperature=temperature)
            assert not no_grad_log_probs.requires_grad
            assert not no_grad_entropy.requires_grad
    finally:
        dist.destroy_process_group()


def test_vocab_parallel_forward_and_backward_match_dense_reference(tmp_path) -> None:
    mp.spawn(
        _run_vocab_parallel_parity,
        args=(2, str(tmp_path / "vocab_parallel_pg")),
        nprocs=2,
        join=True,
    )


@pytest.fixture
def one_rank_mesh():
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield DeviceMesh("cpu", [0], mesh_dim_names=("tp",))
    finally:
        dist.destroy_process_group()


def test_vocab_parallel_rejects_invalid_placements(one_rank_mesh) -> None:
    replicated = DTensor.from_local(torch.randn(2, 3), one_rank_mesh, [Replicate()], run_check=False)
    token_sharded = DTensor.from_local(torch.randn(2, 3), one_rank_mesh, [Shard(0)], run_check=False)
    targets = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="exactly one Shard placement"):
        vocab_parallel_log_probs(replicated, targets)
    with pytest.raises(ValueError, match="last vocabulary axis"):
        vocab_parallel_entropy(token_sharded)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_vocab_parallel_rejects_invalid_temperature(one_rank_mesh, temperature) -> None:
    logits = DTensor.from_local(torch.randn(2, 3), one_rank_mesh, [Shard(-1)], run_check=False)

    with pytest.raises(ValueError, match="positive and finite"):
        vocab_parallel_entropy(logits, temperature=temperature)


@pytest.mark.parametrize(
    ("targets", "error_type", "message"),
    [
        (torch.tensor([[0, 1]]), ValueError, "targets shape"),
        (torch.tensor([0.0, 1.0]), TypeError, "torch.int64"),
    ],
)
def test_vocab_parallel_rejects_invalid_targets(one_rank_mesh, targets, error_type, message) -> None:
    logits = DTensor.from_local(torch.randn(2, 3), one_rank_mesh, [Shard(-1)], run_check=False)

    with pytest.raises(error_type, match=message):
        vocab_parallel_log_probs(logits, targets)


@pytest.mark.parametrize(
    ("targets", "invalid_index"),
    [(torch.tensor([-1, 1]), 0), (torch.tensor([0, 3]), 1)],
)
def test_vocab_parallel_marks_out_of_range_targets_nan(one_rank_mesh, targets, invalid_index) -> None:
    logits = DTensor.from_local(torch.randn(2, 3), one_rank_mesh, [Shard(-1)], run_check=False)

    result = vocab_parallel_log_probs(logits, targets)

    assert torch.isnan(result[invalid_index])
    assert torch.isfinite(result[1 - invalid_index])


def test_vocab_parallel_requires_dtensor_logits() -> None:
    logits = torch.randn(2, 3)

    with pytest.raises(TypeError, match="logits must be a DTensor"):
        vocab_parallel_log_probs(logits, torch.tensor([0, 1]))
    with pytest.raises(TypeError, match="logits must be a DTensor"):
        vocab_parallel_entropy(logits)
