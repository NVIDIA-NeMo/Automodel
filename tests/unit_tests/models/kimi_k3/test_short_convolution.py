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


"""Numerical regression for the K3 training short convolution (FLA kernels)."""

import os
import socket
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

if TYPE_CHECKING:
    from nemo_automodel.components.models.kimi_k3.model import KimiTrainingShortConvolution

pytest.importorskip("fla")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA short convolution requires CUDA")

_CHANNELS = 16
_WIDTH = 4


def _reference(x: torch.Tensor, weight: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Independent per-document causal depthwise convolution followed by SiLU.

    Args:
        x: FP32 tensor of shape [1, sequence, channels], sequence=sum(lengths).
        weight: FP32 tensor of shape [channels, 1, width].
        lengths: Document lengths along the packed sequence.

    Returns:
        FP32 tensor of shape [1, sequence, channels].
    """
    outputs = []
    for document in x.split(lengths, dim=1):
        padded = F.pad(document.transpose(1, 2), (weight.shape[-1] - 1, 0))
        outputs.append(F.silu(F.conv1d(padded, weight, groups=weight.shape[0])).transpose(1, 2))
    return torch.cat(outputs, dim=1)


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    """Return cumulative document offsets of shape [documents + 1]."""
    return F.pad(torch.tensor(lengths, device=device).cumsum(0), (1, 0)).to(torch.int32)


def _convolution(device: torch.device) -> "KimiTrainingShortConvolution":
    """Return a seeded K3 short convolution with FP32 weight [channels, 1, width]."""
    from nemo_automodel.components.models.kimi_k3.model import KimiTrainingShortConvolution

    torch.manual_seed(31)
    conv = KimiTrainingShortConvolution(_CHANNELS, _WIDTH, dtype=torch.float32).to(device)
    with torch.no_grad():
        conv.weight.normal_(std=0.5)
    return conv


@pytest.mark.parametrize("cotangent", ["random", "expanded"])
@pytest.mark.parametrize("lengths", [[1, 1], [1, 3, 1], [5, 2, 6]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_short_convolution_matches_per_document_reference(dtype, lengths, cotangent):
    device = torch.device("cuda")
    conv = _convolution(device)
    torch.manual_seed(32)
    x = torch.randn(1, sum(lengths), _CHANNELS, device=device).to(dtype).requires_grad_()
    ref_x = x.detach().float().requires_grad_()
    ref_weight = conv.weight.detach().clone().requires_grad_()

    output, final_state = conv(x, cu_seqlens=_cu_seqlens(lengths, device))
    reference = _reference(ref_x, ref_weight, lengths)
    assert final_state is None
    assert output.dtype == dtype and conv.weight.dtype == torch.float32
    if cotangent == "random":
        upstream = torch.randn_like(reference)
        (output.float() * upstream).sum().backward()
        (reference * upstream).sum().backward()
    else:
        # sum() hands the kernel an expanded, zero-stride cotangent.
        output.sum().backward()
        reference.sum().backward()

    # FP32 differs only by summation order. In BF16, the output and its
    # cotangent are each rounded once (relative 2**-8), and a weight gradient
    # sums up to ``sequence`` such products per channel, hence 2e-2 for unit-scale data.
    tolerance = {"atol": 1e-5, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}
    torch.testing.assert_close(output.float(), reference.detach(), **tolerance)
    torch.testing.assert_close(x.grad.float(), ref_x.grad, **tolerance)
    torch.testing.assert_close(conv.weight.grad, ref_weight.grad, **tolerance)
    # Singleton documents must still produce and train nonzero values.
    first = lengths[0]
    assert output[0, :first].abs().sum() > 0
    assert x.grad[0, :first].abs().sum() > 0


def _free_port() -> int:
    """Return an unused local TCP port for the process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _cp_rank(rank: int, port: int, dtype: torch.dtype) -> None:
    """Compare one rank's CP2 shard of a 16-token packed row against CP1.

    Args:
        rank: CP rank owning tokens [8 * rank, 8 * rank + 8).
        port: Rendezvous port.
        dtype: Input dtype; weights stay FP32.
    """
    from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext, build_fla_cp_context

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        device = torch.device("cuda", rank)
        # The second document spans the CP cut at token 8.
        lengths = [3, 9, 4]
        doc_ids = torch.repeat_interleave(torch.tensor([1, 2, 3], device=device), torch.tensor(lengths, device=device))
        conv = _convolution(device)
        torch.manual_seed(33)
        full_x = torch.randn(1, sum(lengths), _CHANNELS, device=device).to(dtype)
        upstream = torch.randn(1, sum(lengths), _CHANNELS, device=device)

        cp1_x = full_x.clone().requires_grad_()
        cp1_out, _ = conv(cp1_x, cu_seqlens=_cu_seqlens(lengths, device))
        (cp1_out.float() * upstream).sum().backward()
        cp1_weight_grad = conv.weight.grad.clone()
        conv.zero_grad(set_to_none=True)

        local = slice(rank * 8, rank * 8 + 8)
        context = KimiPackedContext(doc_ids=doc_ids[None], seq_start=rank * 8, cp_size=2)
        cp_context = build_fla_cp_context(context, 0, dist.group.WORLD, _WIDTH)
        cp2_x = full_x[:, local].clone().requires_grad_()
        cp2_out, _ = conv(cp2_x, cp_context=cp_context)
        (cp2_out.float() * upstream[:, local]).sum().backward()
        weight_grad = conv.weight.grad.clone()
        dist.all_reduce(weight_grad)

        tolerance = {"atol": 1e-5, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}
        torch.testing.assert_close(cp2_out, cp1_out[:, local].detach(), **tolerance)
        torch.testing.assert_close(cp2_x.grad, cp1_x.grad[:, local], **tolerance)
        torch.testing.assert_close(weight_grad, cp1_weight_grad, **tolerance)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="CP short convolution needs 2 GPUs")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_short_convolution_cp2_matches_cp1_across_document_cut(dtype):
    mp.spawn(_cp_rank, args=(_free_port(), dtype), nprocs=2, join=True)
