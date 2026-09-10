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

"""Real FSDP numerical coverage for independently sharded embedding tables."""

from __future__ import annotations

import copy
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer import _fully_shard_untied_input_output_embeddings


class _ToyLM(nn.Module):
    def __init__(self, *, tied: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.embed_tokens = nn.Embedding(32, 8)
        self.body = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute token logits.

        Args:
            token_ids: Token IDs of shape [batch, sequence].

        Returns:
            Logits of shape [batch, sequence, vocab].
        """
        return self.lm_head(torch.tanh(self.body(self.embed_tokens(token_ids))))


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a distributed tensor for numerical comparison.

    Args:
        tensor: Tensor of arbitrary shape, optionally a DTensor sharded on its
            first dimension.

    Returns:
        Replicated tensor with the same global shape and values as ``tensor``.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_case(mesh: DeviceMesh, *, tied: bool) -> None:
    torch.manual_seed(2026)
    model = _ToyLM(tied=tied)
    reference = copy.deepcopy(model)
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)

    _fully_shard_untied_input_output_embeddings(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=None,
        input_reshard_after_forward=True,
        fully_shard_fn=fully_shard,
    )
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)

    assert isinstance(model, FSDPModule)
    if tied:
        assert not isinstance(model.embed_tokens, FSDPModule)
        assert not isinstance(model.lm_head, FSDPModule)
    else:
        assert isinstance(model.embed_tokens, FSDPModule)
        assert isinstance(model.lm_head, FSDPModule)

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    actual = model(token_ids)
    expected = reference(token_ids)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()

    actual_parameters = dict(model.named_parameters())
    actual_grad_norms = []
    expected_grad_norms = []
    for name, expected_parameter in reference.named_parameters():
        actual_gradient = _full_tensor(actual_parameters[name].grad)
        expected_gradient = expected_parameter.grad
        torch.testing.assert_close(actual_gradient, expected_gradient)
        actual_grad_norms.append(actual_gradient.float().norm())
        expected_grad_norms.append(expected_gradient.float().norm())
    actual_global_norm = torch.stack(actual_grad_norms).norm()
    expected_global_norm = torch.stack(expected_grad_norms).norm()
    torch.testing.assert_close(actual_global_norm, expected_global_norm)

    model_optimizer.step()
    reference_optimizer.step()
    actual_state = model.state_dict()
    expected_state = reference.state_dict()
    for name, expected_parameter in expected_state.items():
        torch.testing.assert_close(_full_tensor(actual_state[name]), expected_parameter)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        _run_case(mesh, tied=False)
        _run_case(mesh, tied=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_split_embedding_fsdp_matches_unsharded_reference(world_size: int) -> None:
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
