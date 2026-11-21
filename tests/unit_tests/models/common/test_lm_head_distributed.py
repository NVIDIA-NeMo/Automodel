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

"""Two-rank Gloo checks of an ``LMHead`` with FP32 logits under tensor parallelism and FSDP2.

Each case compares FP32 logits, input and weight gradients, and the parameters after
one SGD step against an unsharded single-process reference that runs the same head.
Tensor-parallel input gradients are summed from per-rank BF16 partials, so they are
compared with the BF16 ``nn.Linear`` head under the same layout, bit for bit.
Tensor parallelism also covers a vocabulary the mesh does not divide evenly and weight
layouts that DTensor must redistribute.
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
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from nemo_automodel.components.models.common.lm_head import LMHead

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
        fp32_logits: Give the ``LMHead`` FP32 logits instead of logits in its weight dtype.
        weight: Tensor of shape [vocab, hidden]; its dtype becomes the parameter dtype.

    Returns:
        Head with weight of shape [vocab, hidden].
    """
    # The plain head is an LMHead too: tensor parallelism does not swap its class to TPLinear.
    logits_dtype = torch.float32 if fp32_logits else None
    head = LMHead(weight.shape[1], weight.shape[0], logits_dtype=logits_dtype, dtype=weight.dtype)
    with torch.no_grad():
        head.weight.copy_(weight)
    return head


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


def _check_weight_layouts(hidden: torch.Tensor, weight: torch.Tensor, upstream: torch.Tensor) -> None:
    """Check FP32 logits and input gradients for each weight layout, which DTensor redistributes as needed.

    Args:
        hidden: BF16 tensor of shape [batch, sequence, hidden], identical on every rank.
        weight: BF16 tensor of shape [vocab, hidden], identical on every rank.
        upstream: BF16-valued FP32 tensor of shape [batch, sequence, vocab].
    """
    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("tp",))
    mesh_2d = init_device_mesh("cpu", (1, _WORLD_SIZE), mesh_dim_names=("dp", "tp"))
    reference_hidden = hidden.clone().requires_grad_()
    reference_logits = LMHead.linear(reference_hidden, weight, logits_dtype=torch.float32)
    reference_logits.backward(upstream)
    layouts = {
        "vocabulary-sharded": (mesh, [Shard(0)]),
        "replicated": (mesh, [Replicate()]),
        "hidden-sharded": (mesh, [Shard(1)]),
        "2-D mesh": (mesh_2d, [Replicate(), Shard(0)]),
    }
    for name, (layout_mesh, placements) in layouts.items():
        local_hidden = hidden.clone().requires_grad_()
        replicated_hidden = DTensor.from_local(local_hidden, layout_mesh, [Replicate()] * layout_mesh.ndim)
        sharded_weight = distribute_tensor(weight, layout_mesh, placements)
        logits = LMHead.linear(replicated_hidden, sharded_weight, logits_dtype=torch.float32)
        assert logits.dtype == torch.float32, name
        torch.testing.assert_close(logits.full_tensor(), reference_logits, rtol=1e-6, atol=1e-6, msg=name)
        logits.full_tensor().backward(upstream)
        error = (local_hidden.grad.float() - reference_hidden.grad.float()).norm() / reference_hidden.grad.norm()
        assert error < 1e-2, (name, error)


def _check_biased_tensor_parallel(hidden: torch.Tensor, weight: torch.Tensor, upstream: torch.Tensor) -> None:
    """Compare a column-parallel head that has a bias with the unsharded head.

    Args:
        hidden: BF16 tensor of shape [batch, sequence, hidden], identical on every rank.
        weight: BF16 tensor of shape [vocab, hidden], identical on every rank.
        upstream: BF16-valued FP32 tensor of shape [batch, sequence, vocab].
    """
    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("tp",))
    bias = weight[:, 0].contiguous()

    def biased_head() -> LMHead:
        head = LMHead(weight.shape[1], weight.shape[0], bias=True, logits_dtype=torch.float32, dtype=weight.dtype)
        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)
        return head

    reference = biased_head()
    reference_logits = reference(hidden)
    reference_logits.backward(upstream)
    head = parallelize_module(biased_head(), mesh, ColwiseParallel(output_layouts=Shard(-1), use_local_output=False))
    logits = head(hidden)
    logits.full_tensor().backward(upstream)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits.full_tensor(), reference_logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(head.weight.grad.full_tensor(), reference.weight.grad)
    torch.testing.assert_close(head.bias.grad.full_tensor(), reference.bias.grad)


def _check_gemma4_tied_tensor_parallel(hidden: torch.Tensor, weight: torch.Tensor) -> None:
    """Check that Gemma 4's E-series tensor parallelism projects its vocabulary-sharded head with FP32 logits.

    Args:
        hidden: BF16 tensor of shape [batch, sequence, hidden], identical on every rank.
        weight: BF16 tensor of shape [vocab, hidden], identical on every rank.
    """
    from types import SimpleNamespace

    from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("tp",))
    head = _head(True, weight)
    head.weight = nn.Parameter(distribute_tensor(head.weight.detach(), mesh, [Shard(0)]))
    # The E-series plan leaves the tied head's module hooks unused and calls it through this method.
    model = SimpleNamespace(lm_head=head, _gemma4_tp_enabled=True, _gemma4_tp_mesh=mesh)
    logits = Gemma4ForConditionalGeneration._dense_parallel_lm_head(model, hidden, 0)
    assert logits.dtype == torch.float32 and logits.placements == (Shard(2),)
    torch.testing.assert_close(logits.full_tensor(), _head(True, weight)(hidden), rtol=1e-6, atol=1e-6)


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


def _check_fsdp_head_casts_its_own_input(weight: torch.Tensor, hidden: torch.Tensor) -> None:
    """Check a standalone FSDP2 head with FP32 storage and BF16 compute on FP32 hidden states.

    FSDP2 does not cast the input under this policy and the sharded weight shows the FP32 storage dtype,
    so only the head knows its compute dtype, inside forward.

    Args:
        weight: FP32 tensor of shape [vocab, hidden].
        hidden: FP32 tensor of shape [batch, sequence, hidden], identical on every rank.
    """
    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
    )
    bf16_hidden, bf16_weight = hidden.bfloat16(), weight.bfloat16()
    references = {
        None: F.linear(bf16_hidden, bf16_weight),
        torch.float32: F.linear(bf16_hidden.float(), bf16_weight.float()),
    }
    for logits_dtype, reference in references.items():
        head = LMHead(_HIDDEN, _VOCAB, logits_dtype=logits_dtype)
        head.weight = nn.Parameter(weight.clone())
        fully_shard(head, mesh=mesh, mp_policy=policy)
        assert head.weight.dtype == torch.float32
        logits = head(hidden)
        assert logits.dtype == reference.dtype == (logits_dtype or torch.bfloat16)
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)

    # Only an LMHead casts its input: a plain linear head must receive the compute dtype.
    plain = nn.Linear(_HIDDEN, _VOCAB, bias=False)
    fully_shard(plain, mesh=mesh, mp_policy=policy)
    with pytest.raises(RuntimeError, match="same dtype"):
        plain(hidden)


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
        # The mesh splits this vocabulary unevenly, into 49 and 48 rows.
        uneven_weight = torch.randn(_VOCAB + 1, _HIDDEN, generator=generator) * 0.1
        uneven_upstream = torch.randn(2, 8, _VOCAB + 1, generator=generator).bfloat16().float()
        _check_tensor_parallel(hidden, weight.bfloat16(), upstream)
        _check_tensor_parallel(hidden, uneven_weight.bfloat16(), uneven_upstream)
        _check_weight_layouts(hidden, weight.bfloat16(), upstream)
        _check_biased_tensor_parallel(hidden, weight.bfloat16(), upstream)
        _check_gemma4_tied_tensor_parallel(hidden, weight.bfloat16())
        _check_fsdp(weight, input_ids, labels)
        _check_fsdp_head_casts_its_own_input(weight, hidden.float())
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_and_fsdp2_match_the_unsharded_head():
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
