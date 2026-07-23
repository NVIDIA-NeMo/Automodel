#!/usr/bin/env python
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

r"""2-GPU regression guard for the ``GroupedExpertsTE`` down-projection-bias fix.

Guards PR #2591 / Linear AM-487 -- DO NOT REMOVE.

Background
----------
``GroupedExpertsTE.forward`` (``nemo_automodel/components/moe/experts.py``) runs the
MoE down projection through TE's ``GroupedLinear``, which adds the per-expert down
bias *inside* the grouped GEMM -- i.e. UNWEIGHTED. But the per-token routing
probability is applied at the activation (``permuted_probs``), so the down bias must
also be weighted by that probability. Without the correction every one of a token's
top-k expert contributions carries a full, prob-independent down bias, and the
combine step sums them (~k x bias instead of ~prob x bias). This is a large, silent
systematic offset that only manifests with ``experts: te`` (``GroupedExpertsTE``)
under expert parallelism; the non-EP ``GroupedExperts`` and the ``gmm``
``GroupedExpertsDeepEP`` paths weight it correctly. On gpt-oss-20b it pushed step-0
training loss to ~8.2 instead of the correct ~4.5.

The fix (PR #2591) adds the missing ``(permuted_probs - 1.0) * down_bias`` term via
``_apply_bias`` right after ``output2 = self.down_linear(output1, m_splits)``, so the
net down-bias contribution becomes ``permuted_probs * down_bias``.

What this test does
-------------------
On 2 ranks with an ``ep_size=2`` mesh it builds ``GroupedExpertsTE`` (8 experts
sharded 4+4 across the ranks, DeepEP token dispatcher, ``expert_bias=True``,
``quick_geglu`` activation, deterministic weights with a NON-ZERO down bias). It
feeds identical (seeded, broadcast) hidden states / router indices / router probs so
both ranks agree on routing, runs the EP forward, gathers every rank's local-token
output into the full output, and compares against a single-GPU reference computed
with plain matmuls that applies the *correct* prob-weighted down bias
(``act(x @ Wgu^T + bgu, p) @ Wdown^T + p * bdown`` summed over each token's top-k
experts).

* WITH the fix    -> EP output matches the prob-weighted reference -> PASS.
* WITHOUT the fix -> the down-bias term is ``1.0 * bdown`` instead of ``p * bdown``
  -> the gathered output differs from the reference by ``sum_e (1 - p_e) * bdown_e``
  per token -> the assertion FAILS.

The reference is also evaluated against an intentionally *buggy* (unweighted) variant
so the test additionally asserts that the chosen tolerance is actually able to tell
the two apart (guards against a tolerance so loose the guard is vacuous).

Usage::

    torchrun --nproc_per_node=2 tests/functional_tests/moe/run_te_down_bias_parity.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExpertsTE,
    get_expert_activation_for_deepep,
)

# --- problem sizes (tiny, gpt-oss-style) ---
N_ROUTED_EXPERTS = 8
N_ACTIVATED_EXPERTS = 2  # topk > 1 is required for the bug to manifest
DIM = 64
MOE_INTER_DIM = 64
TOKENS_PER_RANK = 16
WEIGHT_SEED = 20259
INPUT_SEED = 81471


def _build_config(dtype: torch.dtype) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=N_ROUTED_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=N_ACTIVATED_EXPERTS,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=DIM,
        inter_dim=MOE_INTER_DIM,
        moe_inter_dim=MOE_INTER_DIM,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=True,
        expert_activation="quick_geglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=dtype,
    )


def _global_expert_tensors(cfg: MoEConfig, device: torch.device, dtype: torch.dtype):
    """Deterministic weights/biases for ALL experts (identical on every rank).

    Returns stacked tensors in TE ``GroupedLinear`` orientation ``[n, out, in]`` plus
    per-expert biases ``[n, out]``.  A non-zero, per-expert-distinct down bias is
    essential: the bug lives entirely in how the down bias is weighted.
    """
    gen = torch.Generator(device="cpu").manual_seed(WEIGHT_SEED)
    gate_up_out = MOE_INTER_DIM * 2  # gated activation
    # weight{i} layout is [out_features, in_features].
    gate_up_w = torch.randn(N_ROUTED_EXPERTS, gate_up_out, DIM, generator=gen) * 0.08
    down_w = torch.randn(N_ROUTED_EXPERTS, DIM, MOE_INTER_DIM, generator=gen) * 0.08
    gate_up_b = torch.randn(N_ROUTED_EXPERTS, gate_up_out, generator=gen) * 0.1
    # Down bias deliberately O(1) so the (prob vs. unweighted) difference is large.
    down_b = torch.randn(N_ROUTED_EXPERTS, DIM, generator=gen) * 0.5 + 0.25
    return (
        gate_up_w.to(device=device, dtype=dtype),
        down_w.to(device=device, dtype=dtype),
        gate_up_b.to(device=device, dtype=dtype),
        down_b.to(device=device, dtype=dtype),
    )


def _build_ep_experts(cfg, ep_mesh, gate_up_w, down_w, gate_up_b, down_b, device, dtype):
    """Construct GroupedExpertsTE on the EP mesh and load this rank's expert slice."""
    experts = GroupedExpertsTE(cfg, dispatcher_backend="deepep")
    experts.init_token_dispatcher(ep_mesh=ep_mesh, moe_mesh=ep_mesh)

    ep_rank = ep_mesh.get_local_rank()
    n_local = N_ROUTED_EXPERTS // ep_mesh.size()
    start = ep_rank * n_local

    # Materialize the (meta-device) GroupedLinear params onto the real device, then
    # copy in this rank's deterministic expert slice (weight{i} / bias{i}).
    experts.gate_up_linear.reset_parameters()
    experts.down_linear.reset_parameters()
    experts = experts.to(device)

    with torch.no_grad():
        for i in range(n_local):
            g = start + i
            getattr(experts.gate_up_linear, f"weight{i}").copy_(gate_up_w[g])
            getattr(experts.down_linear, f"weight{i}").copy_(down_w[g])
            getattr(experts.gate_up_linear, f"bias{i}").copy_(gate_up_b[g])
            getattr(experts.down_linear, f"bias{i}").copy_(down_b[g])
    return experts


def _reference_full_output(cfg, x_full, indices_full, probs_full, gate_up_w, down_w, gate_up_b, down_b, dtype):
    """Single-GPU oracle: prob-weighted down bias, plain matmuls.

    For each token and each of its top-k experts ``e`` with routing prob ``p``::

        gu  = x @ Wgu_e^T + bgu_e
        act = activation(gu, p)                 # multiplies by p internally
        out_e = act @ Wdown_e^T + bias_scale * p * bdown_e

    ``bias_scale`` selects the correct (=1.0, prob-weighted) or the buggy (=1/p so the
    bias becomes the unweighted ``bdown_e``) variant; the latter is only used to prove
    the tolerance can separate the two.
    """
    activation_fn = get_expert_activation_for_deepep(cfg)

    def _compute(bias_weighting: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x_full, dtype=torch.float32)
        n_tokens, topk = indices_full.shape
        for t in range(n_tokens):
            for k in range(topk):
                e = int(indices_full[t, k].item())
                p = probs_full[t, k]
                xt = x_full[t : t + 1]  # [1, dim]
                gu = xt @ gate_up_w[e].t() + gate_up_b[e]
                p_col = p.view(1, 1).to(gu.dtype)
                act = activation_fn(gu, p_col)
                out_e = act @ down_w[e].t() + bias_weighting[t, k].to(act.dtype) * down_b[e]
                out[t] += out_e[0].float()
        return out

    probs_f = probs_full.float()
    correct = _compute(probs_f)  # prob-weighted: p * bdown  (matches the fix)
    buggy = _compute(torch.ones_like(probs_f))  # unweighted: 1 * bdown (the bug)
    return correct.to(dtype), buggy.to(dtype)


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if world != 2:
        if rank == 0:
            print(f"ERROR: this guard requires world_size=2 (ep=2), got {world}", file=sys.stderr)
        dist.destroy_process_group()
        return 1

    dtype = torch.bfloat16
    cfg = _build_config(dtype)
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))

    # --- deterministic, identical-across-ranks expert tensors ---
    gate_up_w, down_w, gate_up_b, down_b = _global_expert_tensors(cfg, device, dtype)

    experts = _build_ep_experts(cfg, ep_mesh, gate_up_w, down_w, gate_up_b, down_b, device, dtype)

    # --- deterministic, identical-across-ranks routing + inputs (full token set) ---
    total_tokens = TOKENS_PER_RANK * world
    gen = torch.Generator(device="cpu").manual_seed(INPUT_SEED)
    x_full = (torch.randn(total_tokens, DIM, generator=gen) * 1.0).to(device=device, dtype=dtype)
    # Distinct top-k experts per token, with softmax-normalized probs.
    indices_full = torch.stack(
        [torch.randperm(N_ROUTED_EXPERTS, generator=gen)[:N_ACTIVATED_EXPERTS] for _ in range(total_tokens)]
    ).to(device)
    logits = (torch.randn(total_tokens, N_ACTIVATED_EXPERTS, generator=gen)).to(device=device, dtype=torch.float32)
    probs_full = torch.softmax(logits, dim=-1)

    # Each rank processes a distinct, contiguous shard of the global token set.
    s = rank * TOKENS_PER_RANK
    e = s + TOKENS_PER_RANK
    x_local = x_full[s:e].contiguous()
    indices_local = indices_full[s:e].contiguous()
    probs_local = probs_full[s:e].contiguous().to(dtype)
    token_mask = torch.ones(TOKENS_PER_RANK, dtype=torch.bool, device=device)

    # --- EP forward; each rank returns its local tokens' output ---
    y_local = experts(x_local, token_mask, probs_local, indices_local)  # [TOKENS_PER_RANK, dim]

    # Gather every rank's local output into the full [total_tokens, dim] output.
    gathered = [torch.empty_like(y_local) for _ in range(world)]
    dist.all_gather(gathered, y_local.contiguous())
    y_full = torch.cat(gathered, dim=0)

    # --- reference (computed identically on each rank) ---
    ref_correct, ref_buggy = _reference_full_output(
        cfg, x_full, indices_full, probs_full, gate_up_w, down_w, gate_up_b, down_b, dtype
    )

    # bf16 grouped-GEMM vs. plain-matmul reference: keep tolerance tight enough that
    # the ~prob-weighted vs. unweighted down-bias gap (O(0.5) per expert) cannot hide,
    # but loose enough for legitimate bf16 reduction-order drift.
    atol, rtol = 3e-2, 3e-2

    diff_correct = (y_full.float() - ref_correct.float()).abs()
    # Sanity: how far the *buggy* reference is from the correct one (the signal we guard).
    bug_gap = (ref_buggy.float() - ref_correct.float()).abs()

    max_err = diff_correct.max().item()
    mean_err = diff_correct.mean().item()
    max_bug_gap = bug_gap.max().item()
    mean_bug_gap = bug_gap.mean().item()

    matches = torch.allclose(y_full.float(), ref_correct.float(), atol=atol, rtol=rtol)
    # The guard is only meaningful if the buggy output would actually be rejected.
    buggy_would_fail = not torch.allclose(ref_buggy.float(), ref_correct.float(), atol=atol, rtol=rtol)

    ok_tensor = torch.tensor([1 if matches else 0, 1 if buggy_would_fail else 0], device=device, dtype=torch.int)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    all_match = bool(ok_tensor[0].item())
    all_buggy_fail = bool(ok_tensor[1].item())

    if rank == 0:
        print(
            f"[GroupedExpertsTE down-bias guard] ep_size={world} "
            f"experts={N_ROUTED_EXPERTS} topk={N_ACTIVATED_EXPERTS} dim={DIM} "
            f"inter={MOE_INTER_DIM} dtype={dtype}",
            flush=True,
        )
        print(
            f"  EP-vs-reference   : max_err={max_err:.4e} mean_err={mean_err:.4e} (atol={atol}, rtol={rtol})",
            flush=True,
        )
        print(
            f"  bug signal (gap)  : max={max_bug_gap:.4e} mean={mean_bug_gap:.4e} "
            f"(unweighted vs prob-weighted down bias)",
            flush=True,
        )
        print(f"  tolerance separates buggy from correct: {all_buggy_fail}", flush=True)

    success = all_match and all_buggy_fail
    if rank == 0:
        if success:
            print("PASS: GroupedExpertsTE EP-vs-single-GPU down-bias parity", flush=True)
        else:
            if not all_buggy_fail:
                print(
                    "FAIL: tolerance does not separate buggy/correct -- guard would be vacuous",
                    file=sys.stderr,
                    flush=True,
                )
            if not all_match:
                print(
                    "FAIL: GroupedExpertsTE EP output does NOT match the prob-weighted "
                    f"reference (max_err={max_err:.4e} > tol). The down-projection bias is "
                    "almost certainly applied UNWEIGHTED -- the PR #2591 / AM-487 fix is "
                    "missing or was reverted.",
                    file=sys.stderr,
                    flush=True,
                )

    dist.barrier()
    dist.destroy_process_group()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
