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

"""TP logit parity test for GPT-2 with a YAML-style TP plan.

Validates that a minified GPT-2 model produces the same logit
distribution under TP=2 as under TP=1 when the TP plan is specified
entirely as a dict of plain strings — exactly as a user would write in a
YAML config file::

    distributed:
      tp_plan:
        "wte":                "rowwise_rep"
        "h.*.attn.qkv_proj":  "fused_qkv_colwise"
        "h.*.attn.out_proj":  "rowwise"
        "h.*.mlp.fc1":        "colwise"
        "h.*.mlp.fc2":        "rowwise"
        "lm_head":            "colwise_rep"

Three plan delivery paths are exercised:

1. **YAML string-dict** — a plain ``dict[str, str]`` (the primary path
   this test targets).  ``_get_parallel_plan`` auto-translates string
   values to ``ParallelStyle`` objects.
2. **Registered plan** — ``GPT2LMHeadModel`` is found in
   ``PARALLELIZE_FUNCTIONS`` (no explicit plan needed).
3. **String-import plan** — ``tp_shard_plan`` is the dotted import path
   of a function that returns the plan dict.

All three must produce KL divergence ≤ threshold vs the TP=1 baseline.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_tp_gpt2_yaml_plan.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan
from nemo_automodel.components.models.gpt2 import GPT2LMHeadModel


# ---------------------------------------------------------------------------
# YAML-style plan: plain dict of strings (no Python imports needed by user)
# ---------------------------------------------------------------------------
GPT2_YAML_PLAN: Dict[str, str] = {
    "wte": "rowwise_rep",
    "h.*.attn.qkv_proj": "fused_qkv_colwise",
    "h.*.attn.out_proj": "rowwise",
    "h.*.mlp.fc1": "colwise",
    "h.*.mlp.fc2": "rowwise",
    "lm_head": "colwise_rep",
}


# ---------------------------------------------------------------------------
# Importable plan function — used by the string-import test path
# ---------------------------------------------------------------------------
def gpt2_tp_plan_fn() -> Dict[str, str]:
    """Return the GPT-2 TP plan as a string-valued dict.

    Importable as ``tests.functional_tests.llm_pretrain_and_kd
    .run_tp_gpt2_yaml_plan.gpt2_tp_plan_fn``.
    """
    return dict(GPT2_YAML_PLAN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _init_distributed():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(backend="gloo")


def _device() -> torch.device:
    if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cpu")


def _device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_gpt2(
    seed: int, device: torch.device, *, tie_word_embeddings: bool = True,
) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    if _device_type() == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = GPT2LMHeadModel(
        vocab_size=128,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        tie_word_embeddings=tie_word_embeddings,
    ).to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _gather_logits(logits: torch.Tensor, tp_mesh: DeviceMesh) -> torch.Tensor:
    if isinstance(logits, DTensor):
        logits = logits.redistribute(device_mesh=tp_mesh, placements=[Replicate()]).to_local()
    return cast(torch.Tensor, logits)


def _kl_divergence_from_logits(
    *, reference_logits: torch.Tensor, candidate_logits: torch.Tensor,
) -> torch.Tensor:
    assert reference_logits.shape == candidate_logits.shape
    vocab_size = reference_logits.shape[-1]
    ref_lp = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab_size)
    cand_lp = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab_size)
    return F.kl_div(cand_lp, ref_lp, reduction="none", log_target=True).sum(-1)


def _run_case(
    label: str,
    model: GPT2LMHeadModel,
    tp_mesh: DeviceMesh,
    ref_logits: torch.Tensor,
    input_ids: torch.Tensor,
    kl_threshold: float,
    device: torch.device,
) -> bool:
    """Run forward on *model* (already TP-parallelised), compute KL vs ref."""
    with torch.no_grad():
        tp_logits = model(input_ids)
    tp_logits_full = _gather_logits(tp_logits, tp_mesh)

    kl = _kl_divergence_from_logits(reference_logits=ref_logits, candidate_logits=tp_logits_full)
    max_kl = kl.max().item()
    ok = max_kl <= kl_threshold

    ok_t = torch.tensor(1 if ok else 0, device=device, dtype=torch.int)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    all_ok = bool(ok_t.item())

    kl_t = torch.tensor(max_kl, device=device, dtype=torch.float32)
    dist.all_reduce(kl_t, op=dist.ReduceOp.MAX)

    if _rank() == 0:
        status = "PASS" if all_ok else "FAIL"
        print(f"{status}: {label}  max_kl={kl_t.item():.6g}  (threshold={kl_threshold:g})")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl_threshold", type=float, default=1e-6)
    args = parser.parse_args(list(argv) if argv is not None else None)

    _init_distributed()
    device = _device()
    dt = _device_type()
    rank = _rank()
    world_size = _world_size()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: requires world_size=2, got {world_size}", file=sys.stderr)
        return 1

    seed = 42

    # ------------------------------------------------------------------
    # 1. TP=1 reference logits (untied weights, same as TP cases)
    # ------------------------------------------------------------------
    ref_model = _build_gpt2(seed, device, tie_word_embeddings=False)

    torch.manual_seed(999)
    if dt == "cuda":
        torch.cuda.manual_seed_all(999)
    input_ids = torch.randint(1, 128, (2, 64), dtype=torch.long, device=device)

    with torch.no_grad():
        ref_logits = ref_model(input_ids)
    del ref_model

    all_ok = True

    # ------------------------------------------------------------------
    # 2. TP=2 — YAML string-dict plan (the primary test case)
    #    This is exactly what a user writes in YAML:
    #      distributed:
    #        tp_plan:
    #          "wte": "rowwise_rep"
    #          "h.*.attn.qkv_proj": "fused_qkv_colwise"
    #          ...
    # ------------------------------------------------------------------
    tp_mesh = DeviceMesh(dt, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
    model_yaml = _build_gpt2(seed, device, tie_word_embeddings=False)
    plan_yaml = _get_parallel_plan(model_yaml, sequence_parallel=False, tp_shard_plan=GPT2_YAML_PLAN)
    # Verify that string translation happened: values should be ParallelStyle, not str
    for k, v in plan_yaml.items():
        assert isinstance(v, ParallelStyle), (
            f"Expected ParallelStyle for key {k!r}, got {type(v).__name__}={v!r}. "
            "String-to-ParallelStyle translation failed."
        )
    parallelize_module(model_yaml, tp_mesh, plan_yaml)
    ok = _run_case("YAML string-dict plan", model_yaml, tp_mesh, ref_logits, input_ids, args.kl_threshold, device)
    all_ok &= ok
    del model_yaml

    # ------------------------------------------------------------------
    # 3. TP=2 — registered plan (PARALLELIZE_FUNCTIONS lookup, no
    #    explicit tp_shard_plan)
    # ------------------------------------------------------------------
    tp_mesh2 = DeviceMesh(dt, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
    model_reg = _build_gpt2(seed, device, tie_word_embeddings=False)
    plan_reg = _get_parallel_plan(model_reg, sequence_parallel=False, tp_shard_plan=None)
    parallelize_module(model_reg, tp_mesh2, plan_reg)
    ok = _run_case("registered plan", model_reg, tp_mesh2, ref_logits, input_ids, args.kl_threshold, device)
    all_ok &= ok
    del model_reg

    # ------------------------------------------------------------------
    # 4. TP=2 — string-import plan (function returning string-dict)
    # ------------------------------------------------------------------
    tp_mesh3 = DeviceMesh(dt, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
    model_imp = _build_gpt2(seed, device, tie_word_embeddings=False)
    plan_imp = _get_parallel_plan(
        model_imp,
        sequence_parallel=False,
        tp_shard_plan="tests.functional_tests.llm_pretrain_and_kd.run_tp_gpt2_yaml_plan.gpt2_tp_plan_fn",
    )
    parallelize_module(model_imp, tp_mesh3, plan_imp)
    ok = _run_case("string-import plan (function)", model_imp, tp_mesh3, ref_logits, input_ids, args.kl_threshold, device)
    all_ok &= ok
    del model_imp

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    dist.destroy_process_group()

    if rank == 0:
        if all_ok:
            print("PASS: all GPT-2 TP plan paths passed")
        else:
            print("FAIL: one or more GPT-2 TP plan paths failed")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
