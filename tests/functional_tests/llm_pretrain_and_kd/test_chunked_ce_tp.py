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
"""TP2 projection parity; run with four ranks to also exercise FSDP2 x TP2."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import parallelize_module
from transformers import PretrainedConfig

from nemo_automodel.components.distributed.parallel_styles import ColwiseParallelLora
from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.utils import calculate_loss, prepare_lm_weight


def _check_memory(tp_mesh, device):
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    vocab, hidden_dim, chunk = 8192, 128, 128
    weight = torch.nn.Parameter(
        distribute_tensor(torch.randn(vocab, hidden_dim, device=device, dtype=torch.bfloat16), tp_mesh, [Shard(0)])
    )
    loss_fn = ChunkedCrossEntropy(chunk, compile=False)

    def measure(tokens, chunked):
        hidden = torch.randn(tokens, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(vocab, (tokens,), device=device)

        def step():
            if chunked:
                loss = loss_fn(hidden, labels, weight, logit_softcap=20.0, logit_scale=0.196)
            else:
                hidden_tp = DTensor.from_local(hidden, tp_mesh, [Replicate()], run_check=False)
                logits = F.linear(hidden_tp, weight) * 0.196
                logits = torch.tanh(logits / 20.0) * 20.0
                loss = MaskedCrossEntropy()(logits, labels)
            loss.backward()

        step()
        hidden.grad = weight.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        hidden.grad = weight.grad = None
        return peak

    small, large, dense = measure(1024, True), measure(2048, True), measure(2048, False)
    local_logits_bytes = 2048 * (vocab // 2) * 2
    assert large < local_logits_bytes
    assert large < dense * 0.5
    assert large - small < local_logits_bytes * 0.1
    if tp_mesh.get_local_rank() == 0:
        print(f"TP2_MEMORY chunked_1k={small} chunked_2k={large} dense_2k={dense}", flush=True)


def _worker():
    dist.init_process_group("nccl")
    try:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dp_size = dist.get_world_size() // 2
        mesh = init_device_mesh("cuda", (dp_size, 2), mesh_dim_names=("dp", "tp"))
        dp_rank = mesh["dp"].get_local_rank()
        dp_group = mesh["dp"].get_group()
        for dtype in (torch.float32, torch.bfloat16):
            for softcap in (None, 2.0):
                for empty_peer in (False, True):
                    torch.manual_seed(72)
                    model = torch.nn.Module()
                    model.config = PretrainedConfig(final_logit_softcapping=softcap)
                    model.lm_head = torch.nn.Linear(16, 32, bias=False, device=device, dtype=dtype)
                    reference_weight = model.lm_head.weight.detach().clone().requires_grad_()
                    parallelize_module(model.lm_head, mesh["tp"], ColwiseParallelLora(output_layouts=Shard(-1)))
                    if dp_size > 1:
                        fully_shard(model.lm_head, mesh=mesh["dp"])
                    loss_fn = ChunkedCrossEntropy(4, compile=os.environ.get("CHUNKED_CE_TEST_COMPILE") == "1")
                    if dp_size > 1:
                        with pytest.raises(ValueError, match="grad_reduce_group to match"):
                            prepare_lm_weight(loss_fn, model, grad_reduce_group=mesh["tp"].get_group())
                    total = torch.zeros((), device=device)
                    reference_total = torch.zeros((), device=device)
                    for microbatch in range(2):
                        hidden = torch.randn(dp_size, 11, 16, device=device, dtype=dtype)
                        labels = torch.randint(32, (dp_size, 11), device=device)
                        labels[:, :3] = -100
                        if empty_peer:
                            labels[-1] = -100
                        normalizer = max(1, int((labels != -100).sum())) * 2
                        local_hidden = hidden[dp_rank].clone().requires_grad_()
                        reference_hidden = hidden.clone().requires_grad_()
                        weight = prepare_lm_weight(loss_fn, model, grad_reduce_group=dp_group)
                        loss = calculate_loss(
                            loss_fn,
                            model=model,
                            hidden_states=local_hidden,
                            labels=labels[dp_rank],
                            lm_weight=weight,
                            grad_reduce_group=dp_group,
                            num_label_tokens=normalizer,
                        )
                        reference_logits = F.linear(reference_hidden, reference_weight)
                        if softcap is not None:
                            reference_logits = torch.tanh(reference_logits / softcap) * softcap
                        reference = (
                            F.cross_entropy(reference_logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
                            / normalizer
                        )
                        (loss * dp_size).backward()
                        reference.backward()
                        total += loss.detach()
                        reference_total += reference.detach()
                        rtol, atol = (3e-5, 3e-6) if dtype == torch.float32 else (3e-2, 3e-4)
                        torch.testing.assert_close(
                            local_hidden.grad / dp_size, reference_hidden.grad[dp_rank], rtol=rtol, atol=atol
                        )
                    dist.all_reduce(total, group=dp_group)
                    torch.testing.assert_close(total, reference_total, rtol=rtol, atol=atol)
                    full_grad = model.lm_head.weight.grad.full_tensor()
                    torch.testing.assert_close(full_grad, reference_weight.grad, rtol=rtol, atol=atol)
                    torch.testing.assert_close(
                        full_grad.float().norm(), reference_weight.grad.float().norm(), rtol=rtol, atol=atol
                    )
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    optimizer.step()
                    with torch.no_grad():
                        reference_weight.add_(reference_weight.grad, alpha=-0.1)
                    torch.testing.assert_close(
                        model.lm_head.weight.full_tensor(), reference_weight, rtol=rtol, atol=atol
                    )
                    if dist.get_rank() == 0:
                        print(f"PASS: DP{dp_size} TP2 {dtype} {softcap=} {empty_peer=}", flush=True)
        _check_memory(mesh["tp"], device)
        if dist.get_rank() == 0:
            print("CHUNKED_CE_TP_PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("compiled", [False, True])
def test_tp_loss_gradient_and_update_parity(compiled):
    root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["CHUNKED_CE_TEST_COMPILE"] = "1" if compiled else "0"
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", __file__, "--worker"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert "CHUNKED_CE_TP_PASS" in result.stdout


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
