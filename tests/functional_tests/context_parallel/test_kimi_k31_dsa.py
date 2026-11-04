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

"""Two-rank CUDA DSA forward/backward parity with genuinely sparse selection."""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext, build_document_causal_mask
from nemo_automodel.components.models.kimi_k3.model import KimiMLAAttention
from tests.unit_tests.models.kimi_k3.test_dsa import _backend, _tiny_dsa_config, _tiny_text_config


def _run_sparse_cp(rank: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=5))
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
        config = _tiny_text_config(_tiny_dsa_config(index_topk=4))
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 8
        config.v_head_dim = 16
        torch.manual_seed(4645)
        reference = KimiMLAAttention(config, 1, _backend()).cuda()
        reference.init_weights(torch.device("cuda"), 0.2)
        parallel = copy.deepcopy(reference)
        parallel.setup_cp_attention(mesh)
        x = torch.randn(1, 16, config.hidden_size, device="cuda", requires_grad=True)
        upstream = torch.randn_like(x)
        docs = torch.ones(1, 16, dtype=torch.long, device="cuda")
        mask = build_document_causal_mask(docs, docs, q_global_start=0, dtype=x.dtype)
        expected = reference(x, attention_mask=mask)
        expected.backward(upstream)
        local_x = x.detach()[:, rank * 8 : (rank + 1) * 8].clone().requires_grad_()
        context = KimiPackedContext(doc_ids=docs, seq_start=rank * 8, cp_size=2)
        actual = parallel(local_x, packed_context=context)
        torch.testing.assert_close(actual, expected.detach()[:, rank * 8 : (rank + 1) * 8], rtol=2e-5, atol=2e-6)
        actual.backward(upstream[:, rank * 8 : (rank + 1) * 8])
        torch.testing.assert_close(local_x.grad, x.grad[:, rank * 8 : (rank + 1) * 8], rtol=2e-4, atol=3e-6)
        expected_params = dict(reference.named_parameters())
        for name, parameter in parallel.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
                continue
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, expected_params[name].grad, rtol=2e-4, atol=3e-6)
        # Negative control: dense keys must change the output after the top-k boundary.
        with torch.no_grad():
            reference.indexer.index_topk = 16
            dense = reference(x.detach(), attention_mask=mask)
        assert (dense[:, 4:] - expected.detach()[:, 4:]).abs().max().item() > 1e-4
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs for real CP collectives")
def test_sparse_dsa_cp2_matches_eager_forward_and_gradients() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_run_sparse_cp, args=(port,), nprocs=2, join=True)
