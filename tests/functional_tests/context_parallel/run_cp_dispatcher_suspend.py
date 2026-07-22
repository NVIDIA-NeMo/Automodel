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

"""Direct GPU check for cp_dispatcher_suspended (L1, 2 GPUs).

The sunk VLM path embeds and splices vision in-forward, so a VLM vision/audio
tower's non-causal attention now runs inside the ring-SDPA context_parallel
context while CP is active. This driver confirms the contract of
``cp_dispatcher_suspended`` used by every sunk VLM (minimax/qwen3_5/qwen3_5_moe/
nemotron_omni): a vision-tower-like non-causal SDPA inside a real ring context is
rejected/mis-gathered without the suspend, runs as a plain local SDPA with it,
and the ring is restored for the sharded (causal) text decoder afterward.

    torchrun --standalone --nproc-per-node=2 run_cp_dispatcher_suspend.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.context_parallel import ContextParallelismSharder
    from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    # Enter a real ring context the way the recipe does (aux-only shard installs it).
    seqlen = 16
    batch = {
        "input_ids": torch.randint(2, 50, (1, seqlen), device=device),
        "labels": torch.randint(2, 50, (1, seqlen), device=device),
        "position_ids": torch.arange(seqlen, device=device).unsqueeze(0),
    }
    train_ctx = ContextParallelismSharder.sdpa_aux().shard_batch(cp_mesh, None, batch).context

    def vision_sdpa():
        # A vision-tower-like non-causal attention: [batch, heads, patches, dim].
        q = torch.randn(1, 4, 8, 16, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)

    with train_ctx:
        # (a) inside the ring without suspend -> the CP ring intercepts the
        #     non-causal SDPA (load-balancing rejection or seq all-gather).
        without_suspend_failed = False
        try:
            out_bad = vision_sdpa()
            without_suspend_failed = out_bad.shape[2] != 8  # seq doubled by the ring gather
        except RuntimeError:
            without_suspend_failed = True

        # (b) with suspend -> plain local SDPA, output seq unchanged.
        with cp_dispatcher_suspended(cp_mesh):
            out_good = vision_sdpa()
        suspend_ok = out_good.shape == (1, 4, 8, 16) and torch.isfinite(out_good.float()).all().item()

        # (c) the ring is restored afterwards: a causal SDPA still routes through CP.
        after_restored = True
        try:
            F.scaled_dot_product_attention(
                torch.randn(1, 4, 8, 16, device=device, dtype=torch.bfloat16),
                torch.randn(1, 4, 8, 16, device=device, dtype=torch.bfloat16),
                torch.randn(1, 4, 8, 16, device=device, dtype=torch.bfloat16),
                is_causal=True,
            )
        except RuntimeError:
            after_restored = False

    rc = 0
    if rank == 0:
        print(
            f"without_suspend_failed_or_gathered={without_suspend_failed}  "
            f"suspend_ok={suspend_ok}  ring_restored_after={after_restored}"
        )
        ok = without_suspend_failed and suspend_ok and after_restored
        print("RESULT:", "PASS" if ok else "FAIL")
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
