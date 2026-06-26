# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Correctness check for the cp=1 prefix-tree mask on the magi attention backend.

Runs the MagiAttention FFA kernel with a shared-prefix prefix-tree spec and
compares its output, position by position, against an independently constructed
dense-mask SDPA reference. The reference mask is built directly from the rollout
structure (each completion attends FULL to the prompt and CAUSAL to itself), NOT
from the spec, so a wrong spec or a wrong kernel application is caught.

Requires one GPU and a source-built ``magi_attention`` (not on PyPI). Run with::

    torchrun --nproc-per-node=1 tests/functional_tests/attention/prefix_tree_magi_parity.py

Exit code 0 means parity holds within the bf16 tolerance.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from _prefix_tree_reference import build_reference_mask

from nemo_automodel.components.datasets.llm.prefix_tree import fold_shared_prefix_rollouts
from nemo_automodel.components.distributed.magi_attn_utils import (
    AttnMaskSpec,
    is_magi_available,
    make_magi_attn_func,
    set_active_attn_spec,
    set_active_cp_group,
)

# bf16 vs fp32-reference tolerance (the FFA kernel runs in bf16).
MAX_DIFF_TOL = 2e-2
COS_SIM_TOL = 0.999


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device.")
        return 0
    if not is_magi_available():
        print("SKIP: magi_attention not importable (source CUDA build required).")
        return 0

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    # cp=1: a size-1 group makes the magi attn_func take its cp=1 flex-key path
    # (it short-circuits to SDPA only when the active cp group is None).
    set_active_cp_group(dist.group.WORLD)

    # One shared-prefix rollout group: prompt + 3 completions of differing lengths.
    torch.manual_seed(0)
    prompt_ids = list(range(1, 49))  # P = 48
    completion_lens = [16, 24, 8]
    completions = [list(range(100 + 1000 * i, 100 + 1000 * i + n)) for i, n in enumerate(completion_lens)]
    folded = fold_shared_prefix_rollouts(prompt_ids, completions)
    spec, sample_token_ranges = AttnMaskSpec.prefix_tree(folded.node_lengths, folded.sample_paths)
    prompt_len = len(prompt_ids)
    total = len(folded.input_ids)

    num_heads, head_dim = 8, 64
    scale = head_dim**-0.5
    q = torch.randn(total, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # --- magi (under test): FFA with the prefix-tree spec, THD [T, H, D] ---
    attn_func = make_magi_attn_func(softmax_scale=scale)
    set_active_attn_spec(spec)
    out_magi = attn_func(q, k, v)  # [T, H, D]
    set_active_attn_spec(None)

    # --- reference: exact SDPA with the independently built dense mask, fp32 ---
    allowed = build_reference_mask(prompt_len, completion_lens, total).to(device)
    q4, k4, v4 = (t.float().transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # [1, H, T, D]
    out_ref = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=allowed, scale=scale)
    out_ref = out_ref.squeeze(0).transpose(0, 1)  # [T, H, D]

    a = out_magi.float()
    b = out_ref.float()
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    cos_sim = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"overall: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cosine_sim={cos_sim:.8f}")

    # Per-completion slice diffs, to localize any divergence to a specific branch.
    ok = True
    for i, (start, end) in enumerate([rng[-1] for rng in sample_token_ranges]):
        cd = (a[start:end] - b[start:end]).abs().max().item()
        print(f"  completion[{i}] tokens [{start}:{end}] max_diff={cd:.6e}")
        ok = ok and cd < MAX_DIFF_TOL

    passed = ok and max_diff < MAX_DIFF_TOL and cos_sim > COS_SIM_TOL
    print("PARITY PASSED" if passed else "PARITY FAILED")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
