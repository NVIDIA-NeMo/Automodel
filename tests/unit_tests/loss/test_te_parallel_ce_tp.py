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

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from nemo_automodel.components.loss.te_parallel_ce import (
    HAVE_TE_PARALLEL_CE,
    MISSING_TE_PARALLEL_CE_MSG,
    TEParallelCrossEntropy,
)


def _run_tp_softcapped_cross_entropy(rank: int, init_file: str, dtype: torch.dtype) -> None:
    """Compare a TP2 soft-capped projection and CE with a dense autograd reference."""
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(1234)
        device = torch.device("cuda", rank)
        tp_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        batch_size, seq_len, hidden_size, vocab_size = 2, 8, 32, 258
        reference_head = torch.nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            reference_head.weight.normal_(std=2.0)
        parallel_head = parallelize_module(
            copy.deepcopy(reference_head),
            tp_mesh,
            ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        )
        reference_hidden = (
            torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype) * 16
        ).requires_grad_()
        parallel_hidden = reference_hidden.detach().clone().requires_grad_()
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        # Exercise both vocabulary shards, their boundaries, and ignored tokens.
        labels[0, :4] = torch.tensor([0, vocab_size // 2 - 1, vocab_size // 2, vocab_size - 1], device=device)
        labels[0, 4] = labels[1, 1] = labels[1, -1] = -100
        num_label_tokens = int((labels != -100).sum().item())

        reference_projection = reference_head(reference_hidden)
        parallel_projection = parallel_head(parallel_hidden)
        reference_projection.retain_grad()
        parallel_projection.retain_grad()
        assert isinstance(parallel_projection, DTensor)
        assert parallel_projection.placements == (Shard(2),)
        assert parallel_projection.to_local().shape == (batch_size, seq_len, vocab_size // 2)

        # MuseGlimmer applies scaling and soft-capping after the TP projection.
        output_multiplier, softcap = 0.19611613513818404, 20.0
        assert (reference_projection.detach() * output_multiplier).abs().max() > softcap
        reference_logits = torch.tanh(reference_projection * output_multiplier / softcap) * softcap
        parallel_logits = torch.tanh(parallel_projection * output_multiplier / softcap) * softcap
        expected = (
            F.cross_entropy(reference_logits.float().reshape(-1, vocab_size), labels.reshape(-1), reduction="sum")
            / num_label_tokens
        )
        # Infer the TP group from the DTensor, as the recipe does.
        actual = TEParallelCrossEntropy()(parallel_logits, labels, num_label_tokens=num_label_tokens)
        # A non-unit upstream gradient also exercises normalization in the Triton backward.
        (expected * 0.375).backward()
        (actual * 0.375).backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        # BF16 rounds each local CE gradient before TP combines the hidden-state gradients.
        rtol, atol = (1e-4, 1e-6) if dtype == torch.float32 else (3e-2, 1e-4)
        torch.testing.assert_close(
            parallel_projection.grad.to_local(),
            reference_projection.grad.chunk(2, dim=-1)[rank],
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            parallel_head.weight.grad.to_local(),
            reference_head.weight.grad.chunk(2, dim=0)[rank],
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(parallel_hidden.grad, reference_hidden.grad, rtol=rtol, atol=atol)
        assert torch.count_nonzero(parallel_hidden.grad[labels == -100]).item() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not HAVE_TE_PARALLEL_CE, reason=MISSING_TE_PARALLEL_CE_MSG)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs for TP2 CE")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_te_parallel_ce_tp2_softcapping_matches_dense_loss_and_gradients(tmp_path: Path, dtype: torch.dtype) -> None:
    """Match dense CE loss, pre-softcap logits, LM-head and hidden-state gradients on two GPUs."""
    mp.spawn(_run_tp_softcapped_cross_entropy, args=(str(tmp_path / "tp_ce"), dtype), nprocs=2, join=True)
