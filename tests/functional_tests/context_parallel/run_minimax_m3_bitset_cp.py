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

"""Two-GPU packed-bitset CP output and gradient parity.

Run with torchrun --standalone --nproc-per-node=2. Both ranks compare their
load-balanced query shards to an independently enumerated full-sequence oracle.
"""

import copy
import json
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import (
    MiniMaxM3CPSparseAttention,
    cp_load_balanced_global_slots,
)
from tests.functional_tests.context_parallel.run_minimax_m3_sparse_cp import _FLEX_CFG
from tests.functional_tests.models.minimax_m3_vl.test_local_flex import _reference


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype in (torch.float32, torch.bfloat16):
        for packed in (False, True):
            torch.manual_seed(412)
            cfg = MiniMaxM3VLTextConfig(**_FLEX_CFG)
            backend = BackendConfig(
                attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False, experts="torch", dispatcher="torch"
            )
            attn = MiniMaxM3CPSparseAttention(cfg, backend).cuda().to(dtype)
            with torch.no_grad():
                for p in attn.parameters():
                    p.uniform_(-0.1, 0.1)
            ref = copy.deepcopy(attn)
            attn._cp_mesh = mesh
            length = 512
            x = torch.randn(1, length, cfg.hidden_size, device="cuda", dtype=dtype)
            dy = torch.randn_like(x)
            slots = cp_load_balanced_global_slots(2, length // 2, x.device, rank=rank)
            positions = torch.arange(length, device="cuda")[None]
            if packed:
                positions = positions % 256
            docs = torch.arange(length, device="cuda") // 256 if packed else torch.zeros(length, device="cuda")
            padding = torch.zeros(1, length, device="cuda", dtype=torch.bool)
            padding[:, 125:129] = True
            padding[:, -4:] = True
            keep = (docs[:, None] == docs[None, :])[None, None] & ~padding[:, None, None, :]
            keep |= torch.eye(length, device="cuda", dtype=torch.bool)[None, None]
            angles = positions[..., None] / (10000 ** (torch.arange(0, 32, 2, device="cuda") / 32))
            freqs = torch.cat((angles.cos(), angles.sin()), -1)
            xr = x.detach().clone().requires_grad_()
            xl = x[:, slots].detach().clone().requires_grad_()
            expected, _, _ = _reference(ref, xr, freqs, keep)
            actual = attn._cp_forward(
                xl, freqs_cis=freqs[:, slots], padding_mask=padding[:, slots], position_ids=positions[:, slots]
            )
            tol = 2e-5 if dtype == torch.float32 else 0.02
            torch.testing.assert_close(actual, expected[:, slots], rtol=tol, atol=tol)
            actual.backward(dy[:, slots])
            expected.backward(dy)
            metrics = {}
            pairs = [("input", xl.grad, xr.grad[:, slots])]
            for (name, p), (_, pr) in zip(attn.named_parameters(), ref.named_parameters()):
                assert (p.grad is None) == (pr.grad is None), name
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    pairs.append((name, p.grad, pr.grad))
            for name, a, b in pairs:
                relative = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
                metrics[name] = relative
                if dtype == torch.float32:
                    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
                else:
                    assert relative < 0.006, (name, relative)
            if rank == 0:
                print(json.dumps({"dtype": str(dtype), "packed": packed, "grad_relative_l2": metrics}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
