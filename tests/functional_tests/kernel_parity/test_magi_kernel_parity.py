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

"""MagiAttention CUDA parity against SDPA, including cold kernel compilation."""

import pytest
import torch

import nemo_automodel.components.distributed.context_parallel.magi as mu

pytestmark = pytest.mark.timeout(300)


def _init_single_rank_cp_group() -> torch.distributed.ProcessGroup:
    import os

    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    return dist.new_group([0], backend="nccl")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_make_magi_attn_func_matches_sdpa_causal() -> None:
    """The custom-factory attn_func (cp=1 causal self-key) matches SDPA causal."""
    pytest.importorskip("magi_attention")
    import torch.distributed as dist
    import torch.nn.functional as F
    from einops import rearrange
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.cuda.set_device(0)
    cp_group = _init_single_rank_cp_group()
    try:
        mu.set_active_cp_group(cp_group)
        mu.set_active_attn_spec(None)
        nh_q, nh_kv, hd, S = 16, 4, 128, 384
        torch.manual_seed(0)
        q = torch.randn(S, nh_q, hd, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(S, nh_kv, hd, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(S, nh_kv, hd, device="cuda", dtype=torch.bfloat16)

        attn = mu.make_magi_attn_func(softmax_scale=hd**-0.5)
        out = attn(q, k, v)  # thd layout [S, nh, hd]
        torch.cuda.synchronize()  # Attribute any asynchronous failure to Magi, before running the oracle.

        qh = rearrange(q, "s nh hd -> 1 nh s hd")
        kh = rearrange(k, "s nh hd -> 1 nh s hd").repeat_interleave(nh_q // nh_kv, 1)
        vh = rearrange(v, "s nh hd -> 1 nh s hd").repeat_interleave(nh_q // nh_kv, 1)
        # Use the independent dense-math oracle rather than compiling/selecting another fused backend.
        with sdpa_kernel(SDPBackend.MATH):
            ref = rearrange(F.scaled_dot_product_attention(qh, kh, vh, is_causal=True), "1 nh s hd -> s nh hd")

        rel = (out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6)
        assert rel.item() < 5e-3, f"magi vs sdpa rel={rel.item()}"
        assert not torch.isnan(out).any()
    finally:
        mu.set_active_cp_group(None)
        if dist.is_initialized():
            dist.destroy_process_group()
