#!/usr/bin/env python
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

"""Real multi-rank (cp_size>=2) forward-equivalence check for MiniMax M3 sparse CP attention.

The existing unit test (test_cp_forward_equivalence) only covers the cp_size==1 DEGENERATE
case (all-gather = identity, slots = arange). This script runs the CP-aware sparse attention
across `world_size` ranks with a genuinely SHARDED sequence and checks that the
CP-reconstructed per-token output matches the eager full-sequence sparse forward.

A MATCH => CP attention (gather/reorder + global block selection + causal mask) is correct.
A MISMATCH, especially where the CP output is "easier"/smoother => future-token leakage or a
wrong global-slot/causal predicate -- the suspected cause of the low cp-loss + tiny gradients.

Usage (we have up to 8 GPUs locally):
    torchrun --nproc_per_node=2 --master_port=29531 \
        tests/functional_tests/context_parallel/run_minimax_m3_sparse_cp.py
    torchrun --nproc_per_node=4 ... run_minimax_m3_sparse_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist

# Flex-valid tiny config (block_size 128, head_dim 64) -- the block_size-4 conftest config
# trips FlexAttention's BLOCK_M>=16 requirement. Mirrors test_cp_forward_equivalence._FLEX_CFG.
_FLEX_CFG = dict(
    hidden_size=256,
    intermediate_size=64,
    dense_intermediate_size=128,
    shared_intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    rotary_dim=32,
    partial_rotary_factor=0.5,
    vocab_size=128,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    num_local_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    moe_layer_freq=[0, 1, 1],
    use_gemma_norm=True,
    use_qk_norm=True,
    qk_norm_type="per_head",
    scoring_func="sigmoid",
    use_routing_bias=True,
    routed_scaling_factor=2.0,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    num_mtp_modules=0,
    sparse_attention_config=dict(
        use_sparse_attention=True,
        sparse_index_dim=64,
        sparse_num_index_heads=2,
        sparse_topk_blocks=2,
        sparse_block_size=128,
        sparse_score_type="max",
        sparse_init_block=0,
        sparse_local_block=1,
        sparse_attention_freq=[0, 1, 1],
        sparse_disable_index_value=[0, 1, 1],
    ),
)


def main():
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        print("ERROR: launch with torchrun (needs RANK/WORLD_SIZE/LOCAL_RANK).", file=sys.stderr)
        sys.exit(1)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    cp_size = world_size

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
    from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import (
        MiniMaxM3CPSparseAttention,
        cp_load_balanced_global_slots,
    )
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM

    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )

    bf16 = bool(os.environ.get("NEMO_M3_CP_BF16"))
    dtype = torch.bfloat16 if bf16 else torch.float32
    torch.manual_seed(0)
    cfg = MiniMaxM3VLTextConfig(torch_dtype=("bfloat16" if bf16 else "float32"), **_FLEX_CFG)
    model = MiniMaxM3SparseForCausalLM(cfg, backend=backend).eval().to(device)
    model.initialize_weights(dtype=dtype)
    model.to(dtype)
    text = model.model

    # Keep weights bit-identical across ranks (initialize_weights uses ranked RNG).
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    attn = next(
        layer.self_attn for layer in text.layers.values() if isinstance(layer.self_attn, MiniMaxM3CPSparseAttention)
    )

    # Global sequence: divisible by 2*cp; 2*cp_size blocks of block_size=128 so topk=2 is
    # genuinely sparse (selection matters -> a wrong causal/slot mapping changes the output).
    bsz = 1
    seqlen = 256 * cp_size
    hidden = cfg.hidden_size
    t_local = seqlen // cp_size

    x_full = torch.randn(bsz, seqlen, hidden, device=device, dtype=dtype)
    dist.broadcast(x_full, src=0)
    pos_full = torch.arange(seqlen, device=device).unsqueeze(0)
    freqs_full = text.make_freqs_cis(pos_full)

    with torch.no_grad():
        # Reference: eager full-sequence sparse attention (correct causal block-sparse).
        attn._cp_mesh = None
        eager_full = attn(x_full, freqs_cis=freqs_full, attention_mask=None)  # [B, T, H]

        # CP: shard the sequence the way the model expects (load-balanced 2*cp slots),
        # run the CP forward, all-gather rank-major, then scatter back to global order.
        cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
        attn._cp_mesh = cp_mesh

        slots_r = cp_load_balanced_global_slots(cp_size, t_local, device, rank=rank)  # [t_local]
        x_local = x_full.index_select(1, slots_r).contiguous()
        freqs_local = text.make_freqs_cis(slots_r.unsqueeze(0))
        cp_local = attn._cp_forward(x_local, freqs_cis=freqs_local)  # [B, t_local, H]

        gathered = [torch.zeros_like(cp_local) for _ in range(cp_size)]
        dist.all_gather(gathered, cp_local.contiguous())
        gathered_concat = torch.cat(gathered, dim=1)  # [B, T, H] rank-major
        gathered_slots = cp_load_balanced_global_slots(cp_size, t_local, device)  # [T] rank-major -> global
        cp_full = torch.empty_like(eager_full)
        cp_full[:, gathered_slots, :] = gathered_concat

    if rank == 0:
        cp_full = cp_full.float()
        eager_full = eager_full.float()
        diff = (cp_full - eager_full).abs()
        mean_abs = diff.mean().item()
        max_abs = diff.max().item()
        # Per-position mean diff: leakage typically concentrates in early positions (they
        # gain illegal access to later tokens), so print where the largest drift sits.
        per_pos = diff.mean(dim=(0, 2))  # [T]
        worst = torch.topk(per_pos, k=min(5, seqlen)).indices.tolist()
        print(f"\n{'=' * 70}")
        print(f"MiniMax M3 sparse CP forward equivalence  cp_size={cp_size}  seqlen={seqlen}")
        print(f"{'=' * 70}")
        print(f"mean_abs={mean_abs:.3e}  max_abs={max_abs:.3e}")
        print(f"worst positions (highest per-token drift): {worst}")
        pp = per_pos.tolist()
        print("per-pos diff [0..9]:", [round(v, 4) for v in pp[:10]])
        print("eager[0,0,:6]:", [round(v, 4) for v in eager_full[0, 0, :6].tolist()])
        print("cp   [0,0,:6]:", [round(v, 4) for v in cp_full[0, 0, :6].tolist()])
        print("eager[0,1,:6]:", [round(v, 4) for v in eager_full[0, 1, :6].tolist()])
        print("cp   [0,1,:6]:", [round(v, 4) for v in cp_full[0, 1, :6].tolist()])
        # fp32: same q/k/v + same selection + same causal => only kernel-rounding drift.
        ok = mean_abs < 1e-3 and max_abs < 5e-2
        print("RESULT:", "PASS (CP == eager full-seq)" if ok else "FAIL (CP diverges -> forward/causality bug)")
        print(f"{'=' * 70}\n")
        rc = 0 if ok else 1
    else:
        rc = 0

    rc_t = torch.tensor(rc, device=device)
    dist.all_reduce(rc_t, op=dist.ReduceOp.MAX)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(rc_t.item()))


if __name__ == "__main__":
    main()
