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

"""Check CP state adjoints against independent head-specific decay arithmetic.

Run with torchrun --standalone --nproc_per_node=2 and K3_RUN set to an artifact directory.
The first head is an unaffected control; the second exposes missing head offsets.
"""

import hashlib
import json
import os
from pathlib import Path

import fla
import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu_pre_process
from torch.distributed.device_mesh import init_device_mesh


def main() -> None:
    """Exercise actual CUDA kernels and fail if another head's decay is used."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert dist.get_world_size() == 2
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError("Disable the cuBLAS TF32 override and set TRITON_F32_DEFAULT=ieee.")
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
    cases = []
    for head_dim in (64, 128):
        q = torch.ones(1, 128, 2, head_dim, device="cuda")
        k = torch.zeros_like(q)
        w = torch.zeros_like(q)
        do = torch.ones_like(q)
        dv = torch.zeros_like(q)
        gk = torch.zeros_like(q)
        gk[:, :, 0].fill_(-5.0)
        gk[:, :, 1].fill_(-0.01)
        context = build_cp_context(torch.tensor([0, 256], dtype=torch.long, device="cuda"), mesh.get_group())
        actual, _ = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q,
            k,
            w,
            do,
            dv,
            gk=gk,
            scale=1.0,
            cu_seqlens=context.cu_seqlens,
            use_exp2=True,
            context=context,
            transpose_state_layout=True,
        )
        if rank == 0:
            # k=w=0 removes all delta-rule corrections. Each 64-token chunk
            # contributes 64; the later contribution decays by this head's factor.
            decay = torch.tensor([-5.0, -0.01], dtype=torch.float64, device="cuda").exp2()
            expected = (64.0 + 64.0 * decay)[None, :, None, None].expand_as(actual)
            error = (actual.double() - expected).abs().flatten(2).amax(-1)[0]
            cases.append(
                {
                    "head_dim": head_dim,
                    "per_head_max_abs_error": error.tolist(),
                    "observed": actual[0, :, 0, 0].tolist(),
                    "expected": expected[0, :, 0, 0].tolist(),
                    "passed": bool((error < 1e-4).all()),
                }
            )
    if rank == 0:
        source = Path(fla.__file__).parent / "ops/cp/chunk_delta_h.py"
        report = {
            "fla_version": fla.__version__,
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cases": cases,
        }
        (Path(os.environ["K3_RUN"]) / "cp_decay_gradient.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        assert all(case["passed"] for case in cases), cases


if __name__ == "__main__":
    main()
