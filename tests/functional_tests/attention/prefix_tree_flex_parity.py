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

"""Backend-agnostic correctness check for the cp=1 prefix-tree mask.

Realizes the folded :class:`AttnMaskSpec` as a ``flex_attention`` mask (pure
torch, runs on Ampere/A100 and Hopper alike, no magi/FA4 build) and compares its
output, position by position, against an independently constructed dense-mask
SDPA reference. The reference is built straight from the rollout structure (each
completion attends FULL to the prompt and CAUSAL to itself), NOT from the spec,
so a wrong spec or a wrong realization of it is caught.

This validates the part we added (fold rollouts -> AttnMaskSpec, realize the
mask); the magi FFA kernel that consumes the same spec on Hopper is checked
separately by ``prefix_tree_magi_parity.py``. Run with::

    python tests/functional_tests/attention/prefix_tree_flex_parity.py

Exit code 0 means parity holds.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo_automodel.components.datasets.llm.prefix_tree import fold_shared_prefix_rollouts

# fp32 both sides: a tight tolerance that flags any mask error rather than noise.
MAX_DIFF_TOL = 1e-3
COS_SIM_TOL = 0.9999


def _build_reference_mask(prompt_len: int, completion_lens: list[int], total: int) -> torch.Tensor:
    """Dense boolean attend-mask, built straight from the rollout structure.

    ``allowed[q, k]`` is True iff query ``q`` may attend to key ``k`` under the
    "prompt ++ completion as an independent causal sequence" semantics:
      * prompt query: causal within the prompt only;
      * completion query: FULL to the prompt, CAUSAL within its own completion.
    """
    node_of = [-1] * prompt_len
    for c, n in enumerate(completion_lens):
        node_of.extend([c] * n)
    assert len(node_of) == total

    allowed = torch.zeros(total, total, dtype=torch.bool)
    for q in range(total):
        qc = node_of[q]
        for k in range(q + 1):  # causal upper bound: never attend to the future
            kc = node_of[k]
            if kc == -1 or kc == qc:
                # prompt key (full for completions, causal for prompt) or same completion.
                allowed[q, k] = True
    return allowed


def _spec_mask_mod(spec):
    """Turn an :class:`AttnMaskSpec` (AttnSlice rectangles) into a flex mask_mod."""
    rects = list(zip(spec.q_ranges, spec.k_ranges, spec.mask_types))

    def mask_mod(b, h, q_idx, kv_idx):
        keep = q_idx < 0  # all-False seed of the right broadcast shape/dtype
        for (qs, qe), (ks, ke), mt in rects:
            rect = (q_idx >= qs) & (q_idx < qe) & (kv_idx >= ks) & (kv_idx < ke)
            if mt == "causal":
                rect = rect & (kv_idx <= q_idx)
            keep = keep | rect
        return keep

    return mask_mod


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device.")
        return 0
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    # One shared-prefix rollout group: prompt + 3 completions of differing lengths.
    torch.manual_seed(0)
    prompt_ids = list(range(1, 49))  # P = 48
    completion_lens = [16, 24, 8]
    completions = [list(range(100 + 1000 * i, 100 + 1000 * i + n)) for i, n in enumerate(completion_lens)]
    folded = fold_shared_prefix_rollouts(prompt_ids, completions)
    prompt_len = len(prompt_ids)
    total = len(folded.input_ids)

    num_heads, head_dim = 8, 64
    scale = head_dim**-0.5
    q = torch.randn(1, num_heads, total, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(1, num_heads, total, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(1, num_heads, total, head_dim, device=device, dtype=torch.float32)

    # --- under test: flex_attention with the spec realized as a block mask ---
    block_mask = create_block_mask(_spec_mask_mod(folded.spec), B=1, H=1, Q_LEN=total, KV_LEN=total, device=device)
    out_flex = flex_attention(q, k, v, block_mask=block_mask, scale=scale)  # [1, H, T, D]

    # --- reference: exact SDPA with the independently built dense mask ---
    allowed = _build_reference_mask(prompt_len, completion_lens, total).to(device)
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scale)

    a = out_flex.float()
    b = out_ref.float()
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    cos_sim = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"overall: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cosine_sim={cos_sim:.8f}")

    # Per-completion slice diffs (output laid out as [1, H, T, D] -> slice on T).
    ok = True
    for i, (start, end) in enumerate([rng[-1] for rng in folded.sample_token_ranges]):
        cd = (a[:, :, start:end] - b[:, :, start:end]).abs().max().item()
        print(f"  completion[{i}] tokens [{start}:{end}] max_diff={cd:.6e}")
        ok = ok and cd < MAX_DIFF_TOL

    passed = ok and max_diff < MAX_DIFF_TOL and cos_sim > COS_SIM_TOL
    print("PARITY PASSED" if passed else "PARITY FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
