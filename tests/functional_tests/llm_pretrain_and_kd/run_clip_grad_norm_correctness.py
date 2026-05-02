#!/usr/bin/env python3
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

"""Correctness tests for _clip_grad_norm_impl.

Each test builds distributed DTensor params whose fully-materialized grads are
deterministic and identical across ranks (seeded CUDA RNG), computes a reference
norm via per-grad DTensor.full_tensor() + torch.linalg.vector_norm, and asserts
the distributed implementation returns the same value (within tolerance).

Focus scenarios covered: pure EP with stacked torch_mm-layout experts
([n_experts, dim_in, dim_out] sharded on dim 0), pure FSDP2 (DP Shard(0)),
EP+FSDP2 on separate meshes, EP+FSDP2 on a 2D mesh, Replicate-only, and
inf-norm. Tests that need more ranks than available are skipped.
"""

import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from nemo_automodel.components.training.utils import _clip_grad_norm_impl

# Large max_norm so clipping never fires — we only want the returned norm value.
_MAX_NORM = 1e10
_ATOL = 1e-4
_RTOL = 1e-4


def _setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return rank, world, device


def _make_dtensor_param(full_shape, mesh, placements, device, seed, scale=1.0):
    """Build an nn.Parameter whose full param and full grad are deterministic across ranks.

    Uses a seeded torch.Generator so torch.randn produces the same values on every rank,
    then distribute_tensor slices to the local shard. Same approach for the grad.
    """
    gp = torch.Generator(device=device).manual_seed(seed)
    full_param = torch.randn(full_shape, generator=gp, device=device)
    gg = torch.Generator(device=device).manual_seed(seed + 10_000)
    full_grad = torch.randn(full_shape, generator=gg, device=device) * scale

    param = torch.nn.Parameter(distribute_tensor(full_param, mesh, placements))
    param.grad = distribute_tensor(full_grad, mesh, placements)
    return param


def _reference_norm(params, norm_type):
    """Oracle: materialize every grad in full on every rank and compute a straight vector norm."""
    flats = []
    for p in params:
        g = p.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()
        flats.append(g.detach().float().flatten())
    all_grads = torch.cat(flats)
    if norm_type == float("inf"):
        return all_grads.abs().max().item()
    return torch.linalg.vector_norm(all_grads, ord=norm_type).item()


def _assert_close(actual, expected, name):
    diff = abs(actual - expected)
    tol = _ATOL + _RTOL * abs(expected)
    if diff > tol:
        raise AssertionError(
            f"{name}: distributed={actual:.6f}, reference={expected:.6f}, |diff|={diff:.6g}, tol={tol:.6g}"
        )


class _Skip(Exception):
    pass


def _run(rank, name, params, norm_type=2.0, pp_mesh=None):
    ref = _reference_norm(params, norm_type)
    out = _clip_grad_norm_impl(params, max_norm=_MAX_NORM, norm_type=norm_type, pp_mesh=pp_mesh)
    got = out.item() if isinstance(out, torch.Tensor) else float(out)
    _assert_close(got, ref, name)
    if rank == 0:
        print(f"[PASS] {name}: norm={got:.6f} (ref={ref:.6f})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ep_stacked_experts_2norm(rank, world, device):
    """Pure EP with stacked MoE experts: [n_experts, dim_in, dim_out] sharded on expert dim.

    This is the torch_mm / grouped_mm layout. Under EP, dim 0 (expert dim) is
    split across ranks — the scenario at the heart of the PR's fix.
    """
    if world < 2:
        raise _Skip("needs >=2 ranks")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    n_experts = world * 2  # >1 expert per rank, evenly divisible
    params = [
        _make_dtensor_param((n_experts, 64, 128), ep_mesh, (Shard(0),), device, seed=100, scale=1.5),
        _make_dtensor_param((n_experts, 128, 64), ep_mesh, (Shard(0),), device, seed=101, scale=2.0),
    ]
    _run(rank, "ep_stacked_experts_2norm", params, norm_type=2.0)


def test_ep_stacked_experts_inf_norm(rank, world, device):
    """Same EP layout as above but using inf-norm."""
    if world < 2:
        raise _Skip("needs >=2 ranks")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    n_experts = world * 2
    params = [
        _make_dtensor_param((n_experts, 32, 64), ep_mesh, (Shard(0),), device, seed=200, scale=3.0),
        _make_dtensor_param((n_experts, 64, 32), ep_mesh, (Shard(0),), device, seed=201, scale=1.2),
    ]
    _run(rank, "ep_stacked_experts_inf_norm", params, norm_type=float("inf"))


def test_ep_uneven_param_counts(rank, world, device):
    """EP with different stacked tensors in the same group but different local param counts in
    a follow-up group. Guards against the failure mode described in the PR where the stacked
    per-param scalar-DTensor length could differ across ranks.
    """
    if world < 2:
        raise _Skip("needs >=2 ranks")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    n_experts = world * 2
    # Several params with the same placements — all land in one sharding group.
    params = [
        _make_dtensor_param((n_experts, 16, 32), ep_mesh, (Shard(0),), device, seed=300 + i, scale=1.0 + 0.1 * i)
        for i in range(5)
    ]
    _run(rank, "ep_uneven_param_counts", params, norm_type=2.0)


def test_fsdp2_only(rank, world, device):
    """Pure FSDP2 (DP Shard(0)) — non-expert params."""
    if world < 2:
        raise _Skip("needs >=2 ranks")
    dp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    params = [
        _make_dtensor_param((world * 4, 128), dp_mesh, (Shard(0),), device, seed=400, scale=1.0),
        _make_dtensor_param((world * 8, 64), dp_mesh, (Shard(0),), device, seed=401, scale=2.5),
    ]
    _run(rank, "fsdp2_only", params, norm_type=2.0)


def test_ep_plus_fsdp2_separate_meshes(rank, world, device):
    """Realistic EP+FSDP2 layout: expert params on EP mesh, non-expert params on DP mesh.

    This matches how Automodel's MoE models set up parallelism: experts are
    Shard(0) on the ep mesh while the rest is Shard(0) on the dp mesh.
    """
    if world < 2:
        raise _Skip("needs >=2 ranks")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    n_experts = world * 2
    params = [
        # Stacked experts (torch_mm layout) on EP
        _make_dtensor_param((n_experts, 64, 128), ep_mesh, (Shard(0),), device, seed=500, scale=2.0),
        _make_dtensor_param((n_experts, 128, 64), ep_mesh, (Shard(0),), device, seed=501, scale=1.5),
        # Non-expert params on DP (FSDP2 shards dim 0)
        _make_dtensor_param((world * 4, 256), dp_mesh, (Shard(0),), device, seed=502, scale=1.0),
        _make_dtensor_param((world * 8, 128), dp_mesh, (Shard(0),), device, seed=503, scale=3.0),
    ]
    _run(rank, "ep_plus_fsdp2_separate_meshes", params, norm_type=2.0)


def test_ep_plus_fsdp2_2d_mesh(rank, world, device):
    """EP+DP as a single 2D mesh.

    Experts: Shard(0) on EP dim (expert dim of the stacked tensor) AND Shard(0) on DP
    dim (outer/FSDP dim) — double-sharded.
    Non-experts: Replicate on EP, Shard(0) on DP — FSDP-only.
    """
    if world < 4:
        raise _Skip("needs >=4 ranks")
    dp_size = 2
    ep_size = world // dp_size
    if ep_size < 2:
        raise _Skip("needs ep_size>=2")
    mesh_2d = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    n_experts = ep_size * 2
    # For a tensor of shape [n_experts, outer, inner]:
    #   placements = (Shard(1), Shard(0)) means DP shards dim 1 (outer), EP shards dim 0 (experts).
    outer = dp_size * 4
    params = [
        _make_dtensor_param((n_experts, outer, 64), mesh_2d, (Shard(1), Shard(0)), device, seed=600, scale=2.0),
        # Non-experts: DP-sharded, EP-replicated
        _make_dtensor_param((dp_size * 8, 128), mesh_2d, (Shard(0), Replicate()), device, seed=601, scale=1.5),
        # Fully replicated (e.g. scalar-ish params like norms/biases)
        _make_dtensor_param((64, 64), mesh_2d, (Replicate(), Replicate()), device, seed=602, scale=0.8),
    ]
    _run(rank, "ep_plus_fsdp2_2d_mesh", params, norm_type=2.0)


def test_replicate_only(rank, world, device):
    """DTensor with Replicate placement — must NOT allreduce (would multiply norm^p by world)."""
    if world < 2:
        raise _Skip("needs >=2 ranks")
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("m",))
    params = [
        _make_dtensor_param((64, 128), mesh, (Replicate(),), device, seed=700, scale=1.5),
        _make_dtensor_param((128, 64), mesh, (Replicate(),), device, seed=701, scale=2.0),
    ]
    _run(rank, "replicate_only", params, norm_type=2.0)


def test_partial_grad_placement(rank, world, device):
    """DTensor grad with Partial placement — full value is the sum of per-rank contributions.

    This is the `has_partial=True` branch in the fixed impl: it calls full_tensor() to
    materialize the grad before computing the norm. The test gives each rank an equal
    contribution so the summed full grad is deterministic.
    """
    if world < 2:
        raise _Skip("needs >=2 ranks")
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("m",))

    # Construct the "full" grad deterministically on every rank, then give each rank an
    # equal share as its local Partial contribution.
    shape = (64, 128)
    gp = torch.Generator(device=device).manual_seed(900)
    full_param = torch.randn(shape, generator=gp, device=device)
    gg = torch.Generator(device=device).manual_seed(901)
    full_grad = torch.randn(shape, generator=gg, device=device)

    local_grad = full_grad / world  # sum across ranks == full_grad
    param = torch.nn.Parameter(
        DTensor.from_local(full_param / world, device_mesh=mesh, placements=(Partial(),), run_check=False)
    )
    param.grad = DTensor.from_local(local_grad, device_mesh=mesh, placements=(Partial(),), run_check=False)
    _run(rank, "partial_grad_placement", [param], norm_type=2.0)


def test_2d_mesh_trivial_dp(rank, world, device):
    """2D mesh with a size-1 DP dim and size-world EP dim — exercises the multi-dim code
    path without needing 4 GPUs.
    """
    if world < 2:
        raise _Skip("needs >=2 ranks")
    mesh_2d = init_device_mesh("cuda", (1, world), mesh_dim_names=("dp", "ep"))
    n_experts = world * 2
    params = [
        _make_dtensor_param((n_experts, 64, 128), mesh_2d, (Replicate(), Shard(0)), device, seed=1000, scale=2.0),
        # A trivially-sharded (size-1) dim with a Shard on the other
        _make_dtensor_param((32, world * 4), mesh_2d, (Replicate(), Shard(1)), device, seed=1001, scale=1.5),
    ]
    _run(rank, "2d_mesh_trivial_dp", params, norm_type=2.0)


def test_many_separate_meshes(rank, world, device):
    """Three separate 1D meshes on all ranks — exercises the multi-group combination path."""
    if world < 2:
        raise _Skip("needs >=2 ranks")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    tp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("tp",))
    n_experts = world * 2
    params = [
        _make_dtensor_param((n_experts, 64, 64), ep_mesh, (Shard(0),), device, seed=1100, scale=1.0),
        _make_dtensor_param((world * 4, 128), dp_mesh, (Shard(0),), device, seed=1101, scale=2.0),
        _make_dtensor_param((64, world * 4), tp_mesh, (Shard(1),), device, seed=1102, scale=1.5),
        _make_dtensor_param((32, 32), dp_mesh, (Replicate(),), device, seed=1103, scale=0.8),
    ]
    _run(rank, "many_separate_meshes", params, norm_type=2.0)


def test_ep_plus_fsdp2_inf_norm(rank, world, device):
    """EP+FSDP2 separate meshes but using inf-norm — exercises the MAX allreduce path."""
    if world < 2:
        raise _Skip("needs >=2 ranks")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    n_experts = world * 2
    params = [
        _make_dtensor_param((n_experts, 64, 128), ep_mesh, (Shard(0),), device, seed=1200, scale=3.0),
        _make_dtensor_param((world * 4, 256), dp_mesh, (Shard(0),), device, seed=1201, scale=1.0),
        _make_dtensor_param((64, 64), dp_mesh, (Replicate(),), device, seed=1202, scale=2.0),
    ]
    _run(rank, "ep_plus_fsdp2_inf_norm", params, norm_type=float("inf"))


def test_small_tensors_edge_cases(rank, world, device):
    """Tiny tensors (1-element, 1-D) to shake out shape-edge assumptions."""
    if world < 2:
        raise _Skip("needs >=2 ranks")
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("m",))
    params = [
        # 1-D tensor sharded across world
        _make_dtensor_param((world * 4,), mesh, (Shard(0),), device, seed=1300, scale=1.5),
        # 1-D replicated
        _make_dtensor_param((16,), mesh, (Replicate(),), device, seed=1301, scale=2.0),
    ]
    _run(rank, "small_tensors_edge_cases", params, norm_type=2.0)


def test_mixed_placements_same_mesh(rank, world, device):
    """Several placements on the same 2D mesh — exercises the sharding_groups grouping."""
    if world < 4:
        raise _Skip("needs >=4 ranks")
    tp_size = 2
    dp_size = world // tp_size
    if dp_size < 2:
        raise _Skip("needs dp_size>=2")
    mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    params = [
        _make_dtensor_param((dp_size * 4, 256), mesh, (Shard(0), Replicate()), device, seed=800, scale=1.0),
        _make_dtensor_param((256, tp_size * 4), mesh, (Replicate(), Shard(1)), device, seed=801, scale=2.0),
        _make_dtensor_param((dp_size * 8, tp_size * 4), mesh, (Shard(0), Shard(1)), device, seed=802, scale=1.5),
        _make_dtensor_param((128, 128), mesh, (Replicate(), Replicate()), device, seed=803, scale=0.5),
    ]
    _run(rank, "mixed_placements_same_mesh", params, norm_type=2.0)


def main():
    rank, world, device = _setup()
    if rank == 0:
        print("=" * 80)
        print(f"Running clip_grad_norm correctness tests with world_size={world}")
        print("=" * 80)

    tests = [
        test_ep_stacked_experts_2norm,
        test_ep_stacked_experts_inf_norm,
        test_ep_uneven_param_counts,
        test_fsdp2_only,
        test_ep_plus_fsdp2_separate_meshes,
        test_ep_plus_fsdp2_2d_mesh,
        test_replicate_only,
        test_partial_grad_placement,
        test_2d_mesh_trivial_dp,
        test_many_separate_meshes,
        test_ep_plus_fsdp2_inf_norm,
        test_small_tensors_edge_cases,
        test_mixed_placements_same_mesh,
    ]

    failures = []
    skipped = []
    passed = 0
    for t in tests:
        try:
            t(rank, world, device)
            passed += 1
        except _Skip as s:
            skipped.append((t.__name__, str(s)))
            if rank == 0:
                print(f"[SKIP] {t.__name__}: {s}")
        except Exception as e:
            failures.append((t.__name__, str(e)))
            if rank == 0:
                print(f"[FAIL] {t.__name__}: {e}")
                traceback.print_exc()
        dist.barrier()

    if rank == 0:
        print()
        print(f"Summary: {passed} passed, {len(skipped)} skipped, {len(failures)} failed (of {len(tests)} total)")
        if failures:
            print("Failures:")
            for n, err in failures:
                print(f"  - {n}: {err}")

    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
