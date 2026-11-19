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

"""Two-GPU NCCL checks that FusedLinearCrossEntropy(impl="cce") streams a sharded LM head.

The cut_cross_entropy kernels need CUDA, so the CPU/Gloo unit tests cover the
streaming machinery with the PyTorch chunk kernel, and this test covers the
cut_cross_entropy chunk kernel and the memory bound that motivated streaming.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Shard, distribute_tensor


def _forbid_full_tensor(*args, **kwargs):
    raise AssertionError("DTensor.full_tensor was called")


def _check_parity(rank: int, world: int, device: torch.device, mesh, *, softcap: float, chunk_bytes: int) -> None:
    """Compare the streamed loss and gradients with dense cut_cross_entropy and an FP32 loss.

    The vocabulary of 1001 rows shards unevenly over two ranks, and the last
    rank has no supervised tokens but still enters every collective.
    """
    import nemo_automodel.components.loss.linear_ce_base as linear_ce_base
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

    linear_ce_base._STREAM_CHUNK_BYTES = chunk_bytes
    vocab, hidden_dim = 1001, 256
    torch.manual_seed(5)
    weight = (torch.randn(vocab, hidden_dim, device=device) / 8).bfloat16()
    hidden = torch.randn(world, 2, 64, hidden_dim, device=device).bfloat16()
    labels = torch.randint(vocab, (world, 2, 64), device=device)
    labels[0, 0, :7] = -100
    labels[-1] = -100
    normalizer = int((labels != -100).sum())
    loss_fn = FusedLinearCrossEntropy(logit_softcapping=softcap)

    sharded = torch.nn.Parameter(distribute_tensor(weight, mesh, [Shard(0)]))
    local_hidden = hidden[rank].clone().requires_grad_()
    original = DTensor.full_tensor
    DTensor.full_tensor = _forbid_full_tensor
    try:
        loss = loss_fn(
            local_hidden, labels[rank], sharded, num_label_tokens=normalizer, grad_reduce_group=dist.group.WORLD
        )
        # Match the recipe's compensation for FSDP's averaged gradients.
        (loss * world).backward()
    finally:
        DTensor.full_tensor = original

    dense_weight = weight.clone().requires_grad_()
    dense_hidden = hidden.clone().requires_grad_()
    dense = loss_fn(dense_hidden.flatten(0, 1), labels.flatten(0, 1), dense_weight, num_label_tokens=normalizer)
    dense.backward()
    logits = F.linear(hidden.float(), weight.float())
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    fp32 = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="sum") / normalizer

    total = loss.detach().clone()
    dist.all_reduce(total)
    torch.testing.assert_close(total, dense.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(total, fp32, rtol=1e-4, atol=1e-4)
    # Both gradients are bf16; streaming adds at most one bf16 rounding per chunk sum.
    torch.testing.assert_close(sharded.grad.full_tensor(), dense_weight.grad, rtol=2e-2, atol=2e-4)
    torch.testing.assert_close(local_hidden.grad / world, dense_hidden.grad[rank], rtol=2e-2, atol=2e-4)


def _peak_loss_memory(fn) -> int:
    """Return the peak bytes allocated beyond the starting point while running ``fn``."""
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - start


def _check_memory(world: int, device: torch.device, mesh) -> None:
    """Streaming keeps the head's transient memory below one full LM-head copy.

    Gathering the head instead keeps the full weight and its full-size gradients
    live, about twice the full head here.
    """
    import nemo_automodel.components.loss.linear_ce_base as linear_ce_base
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

    linear_ce_base._STREAM_CHUNK_BYTES = 16 * 1024**2
    vocab, hidden_dim, tokens = 131072, 2048, 1024
    full_bytes = vocab * hidden_dim * 2
    torch.manual_seed(0)
    sharded = torch.nn.Parameter(
        distribute_tensor(torch.randn(vocab, hidden_dim, device=device, dtype=torch.bfloat16) / 32, mesh, [Shard(0)])
    )
    hidden = torch.randn(1, tokens, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(vocab, (1, tokens), device=device)
    loss_fn = FusedLinearCrossEntropy()

    def streamed() -> None:
        loss = loss_fn(hidden, labels, sharded, num_label_tokens=tokens, grad_reduce_group=dist.group.WORLD)
        (loss * world).backward()

    def gathered() -> None:
        full = sharded.full_tensor(grad_placements=(Partial(),))
        full.register_hook(lambda grad: grad / world)
        (loss_fn(hidden, labels, full, num_label_tokens=tokens) * world).backward()

    gathered()  # warm up the kernels outside the measurements
    sharded.grad = None
    hidden.grad = None
    gathered_peak = _peak_loss_memory(gathered)
    sharded.grad = None
    hidden.grad = None
    streamed_peak = _peak_loss_memory(streamed)
    # The weight-gradient shard itself (full_bytes / world) is needed by training either way.
    assert streamed_peak < full_bytes, (streamed_peak, full_bytes)
    assert gathered_peak > 1.5 * full_bytes, (gathered_peak, full_bytes)


def _worker() -> None:
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_cp",))
        # 2 * 61 * 256 * 2 bytes: 61 rows per rank per chunk, so nine chunks with a short last one.
        for softcap in (0.0, 30.0):
            _check_parity(rank, world, device, mesh, softcap=softcap, chunk_bytes=2 * 61 * 256 * 2)
        _check_parity(rank, world, device, mesh, softcap=0.0, chunk_bytes=1 << 30)
        _check_memory(world, device, mesh)
        if rank == 0:
            print("LINEAR_CE_STREAMED_DISTRIBUTED_PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs for sharded gradients")
def test_fused_linear_ce_streams_sharded_lm_head():
    pytest.importorskip("cut_cross_entropy")
    repo_root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", __file__, "--worker"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout
    assert "LINEAR_CE_STREAMED_DISTRIBUTED_PASS" in result.stdout


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
