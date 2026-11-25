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

"""Packed context parallelism (2 Gloo ranks) must equal the single-rank packed forward, forward and backward."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashForCausalLM

from .tiny import CPU_BACKEND, tiny_config

DOC_LENGTHS = (23, 17, 8)  # 23 > index_topk 8 and > window 8 (sparse selection and SWA both bite); 48 tokens in total


def _model():
    torch.manual_seed(0)
    model = HY4FlashForCausalLM(tiny_config(vision={"num_hidden_layers": 0}), backend=CPU_BACKEND).float()
    model.initialize_weights(dtype=torch.float32)
    return model


def _inputs():
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 280, (1, sum(DOC_LENGTHS)), generator=generator)
    ids[0, 30:33] = ids[0, 4:7]  # repeated n-grams
    doc = torch.cat([torch.full((n,), i + 1) for i, n in enumerate(DOC_LENGTHS)])[None]
    positions = torch.cat([torch.arange(n) for n in DOC_LENGTHS])[None]
    return ids, doc, positions


def _reference():
    model = _model()
    ids, doc, positions = _inputs()
    out = model(ids, position_ids=positions, packed_seq_ids=doc, attention_mask=torch.ones_like(ids))
    out.logits.square().sum().backward()
    return out.logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _worker(rank, rendezvous, queue):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("cp",))
        model = _model()
        for layer in model.model.layers:
            layer.self_attn.setup_cp_attention(mesh)
        ids, doc, positions = _inputs()
        half = ids.shape[1] // 2
        sl = slice(rank * half, (rank + 1) * half)
        out = model(
            ids[:, sl],
            position_ids=positions[:, sl],
            packed_seq_ids=doc[:, sl],
            attention_mask=torch.ones_like(ids[:, sl]),
            cp_group=mesh.get_group(),
        )
        # sum of squares over all ranks' logits equals the single-rank objective
        out.logits.square().sum().backward()
        grads = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad = parameter.grad.clone()
                dist.all_reduce(grad)
                grads[name] = grad
        queue.put((rank, out.logits.detach(), grads))
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(300)
def test_cp2_matches_single_rank_forward_and_gradients(tmp_path):
    reference_logits, reference_grads = _reference()
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_worker, args=(rank, str(tmp_path / "init"), queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = {}
    for _ in range(2):
        rank, logits, grads = queue.get(timeout=240)
        results[rank] = (logits, grads)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    logits = torch.cat([results[0][0], results[1][0]], dim=1)
    torch.testing.assert_close(logits, reference_logits, rtol=2e-4, atol=2e-4)
    for name, grad in reference_grads.items():
        torch.testing.assert_close(results[0][1][name], grad, rtol=2e-3, atol=2e-4, msg=lambda m, name=name: f"grad {name}: {m}")
