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

"""Two-rank Gloo checks of the FP32-logits head under tensor parallelism and FSDP2.

Each case compares FP32 logits, input and weight gradients, and the parameters after
one SGD step against an unsharded single-process reference that runs the same head.
Tensor-parallel input gradients are summed from per-rank BF16 partials, so they are
compared with the BF16 ``nn.Linear`` head under the same layout, bit for bit.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from nemo_automodel.components.models.common.fp32_logits import Float32LogitsLinear

pytestmark = pytest.mark.cpu_only

_WORLD_SIZE = 2
_HIDDEN = 32
_VOCAB = 96
_LR = 0.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a DTensor of any placement, or return a regular tensor unchanged.

    Args:
        tensor: Tensor or DTensor of any shape.

    Returns:
        Regular tensor with the global shape.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _head(fp32_logits: bool, weight: torch.Tensor) -> nn.Linear:
    """Build a bias-free head owning a copy of ``weight``.

    Args:
        fp32_logits: Return a ``Float32LogitsLinear`` instead of an ``nn.Linear``.
        weight: Tensor of shape [vocab, hidden]; its dtype becomes the parameter dtype.

    Returns:
        Head with weight of shape [vocab, hidden].
    """
    head = nn.Linear(_HIDDEN, _VOCAB, bias=False, dtype=weight.dtype)
    with torch.no_grad():
        head.weight.copy_(weight)
    return Float32LogitsLinear.from_linear(head) if fp32_logits else head


def _check_tensor_parallel(hidden: torch.Tensor, weight: torch.Tensor, upstream: torch.Tensor) -> None:
    """Compare column-parallel heads with an unsharded reference.

    Args:
        hidden: BF16 tensor of shape [batch, sequence, hidden], identical on every rank.
        weight: BF16 tensor of shape [vocab, hidden], identical on every rank.
        upstream: BF16-valued FP32 tensor of shape [batch, sequence, vocab].
    """
    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("tp",))
    rank = dist.get_rank()
    reference_head = _head(True, weight)
    reference_hidden = hidden.clone().requires_grad_()
    reference_logits = reference_head(reference_hidden)
    reference_logits.backward(upstream)
    reference_step = reference_head.weight.detach() - _LR * reference_head.weight.grad

    styles = {
        "colwise Shard(-1)": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        "colwise Replicate": ColwiseParallel(output_layouts=Replicate()),
        "sequence parallel": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    }
    for name, style in styles.items():
        grads = {}
        for fp32_logits in (True, False):
            head = parallelize_module(_head(fp32_logits, weight), mesh, style)
            optimizer = torch.optim.SGD(head.parameters(), lr=_LR)
            if name == "sequence parallel":
                local_hidden = hidden.chunk(_WORLD_SIZE, dim=1)[rank].clone().requires_grad_()
                logits = head(DTensor.from_local(local_hidden, mesh, [Shard(1)], run_check=False))
                _full(logits).backward(upstream.to(logits.dtype))
                parts = [torch.empty_like(local_hidden) for _ in range(_WORLD_SIZE)]
                dist.all_gather(parts, local_hidden.grad)
                hidden_grad = torch.cat(parts, dim=1)
            else:
                local_hidden = hidden.clone().requires_grad_()
                logits = head(local_hidden)
                _full(logits).backward(upstream.to(logits.dtype))
                hidden_grad = local_hidden.grad
            grads[fp32_logits] = hidden_grad
            if fp32_logits:
                full_logits = _full(logits)
                assert full_logits.dtype == torch.float32, name
                torch.testing.assert_close(full_logits, reference_logits, rtol=1e-6, atol=1e-6, msg=name)
                torch.testing.assert_close(_full(head.weight.grad), reference_head.weight.grad, msg=name)
                # Summing per-rank BF16 partials rounds differently from one unsharded GEMM.
                error = (hidden_grad.float() - reference_hidden.grad.float()).norm() / reference_hidden.grad.norm()
                assert error < 1e-2, (name, error)
                optimizer.step()
                torch.testing.assert_close(_full(head.weight).detach(), reference_step, msg=name)
        torch.testing.assert_close(grads[True], grads[False], rtol=0, atol=0, msg=name)


class _TinyLM(nn.Module):
    """Embedding followed by an FP32-logits head, optionally sharing one weight."""

    def __init__(self, weight: torch.Tensor, *, tied: bool) -> None:
        """Build the model.

        Args:
            weight: FP32 tensor of shape [vocab, hidden] copied into the embedding and the head.
            tied: Share one Parameter between the embedding and the head.
        """
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _HIDDEN)
        self.lm_head = _head(True, weight)
        with torch.no_grad():
            self.embed.weight.copy_(weight)
        if tied:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Project embedded tokens into logits.

        Args:
            input_ids: Token ids of shape [batch, sequence].

        Returns:
            FP32 logits of shape [batch, sequence, vocab].
        """
        return self.lm_head(self.embed(input_ids))


def _check_fsdp(weight: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
    """Compare FSDP2 with FP32 storage and BF16 compute against the same policy on one process.

    Args:
        weight: FP32 tensor of shape [vocab, hidden].
        input_ids: Token ids of shape [batch, sequence], identical on every rank.
        labels: Targets of shape [batch, sequence], identical on every rank.
    """
    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32)
    for tied in (False, True):
        master = _TinyLM(weight, tied=tied)
        compute = _TinyLM(weight, tied=tied).to(torch.bfloat16)
        reference_logits = compute(input_ids)
        F.cross_entropy(reference_logits.flatten(0, 1), labels.flatten()).backward()
        reference = {
            name: (param.detach() - _LR * dict(compute.named_parameters())[name].grad.float())
            for name, param in master.named_parameters()
        }

        model = _TinyLM(weight, tied=tied)
        if not tied:
            fully_shard(model.lm_head, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        assert (model.lm_head.weight is model.embed.weight) == tied
        optimizer = torch.optim.SGD(model.parameters(), lr=_LR)
        logits = model(input_ids)
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
        optimizer.step()

        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
        for name, param in model.named_parameters():
            torch.testing.assert_close(_full(param).detach(), reference[name], rtol=0, atol=0, msg=f"{tied=} {name}")


def _worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        generator = torch.Generator().manual_seed(0)
        weight = torch.randn(_VOCAB, _HIDDEN, generator=generator) * 0.1
        hidden = torch.randn(2, 8, _HIDDEN, generator=generator).bfloat16()
        upstream = torch.randn(2, 8, _VOCAB, generator=generator).bfloat16().float()
        input_ids = torch.randint(0, _VOCAB, (2, 8), generator=generator)
        labels = torch.randint(0, _VOCAB, (2, 8), generator=generator)
        _check_tensor_parallel(hidden, weight.bfloat16(), upstream)
        _check_fsdp(weight, input_ids, labels)
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_and_fsdp2_match_the_unsharded_head():
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
