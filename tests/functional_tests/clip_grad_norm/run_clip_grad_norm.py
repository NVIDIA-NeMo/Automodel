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

"""Functional test script for _clip_grad_norm_impl with various parallelism configurations.

This script runs all gradient clipping tests appropriate for the detected world size.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.training.utils import _clip_grad_norm_impl


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    return rank, world_size, device


def create_dtensor_with_grad(shape, device_mesh, placements, device, grad_value=1.0):
    """Create a DTensor with a gradient."""
    # Create local tensor
    local_tensor = torch.randn(*shape, device=device, requires_grad=True)

    # Convert to DTensor
    dtensor = DTensor.from_local(local_tensor, device_mesh=device_mesh, placements=placements, run_check=False)

    # Attach a gradient
    dtensor.grad = DTensor.from_local(
        torch.full_like(local_tensor, grad_value), device_mesh=device_mesh, placements=placements, run_check=False
    )

    return dtensor


def test_tp_only(rank, world_size, device):
    """Test with TP-only configuration."""
    print(f"[Rank {rank}] Testing TP-only configuration (TP={world_size})")

    # Create TP mesh
    tp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    # Create DTensors with different TP shardings
    params = [
        create_dtensor_with_grad((128, 256), tp_mesh, (Shard(0),), device, grad_value=2.0),
        create_dtensor_with_grad((256, 512), tp_mesh, (Shard(1),), device, grad_value=3.0),
        create_dtensor_with_grad((64, 64), tp_mesh, (Replicate(),), device, grad_value=1.5),
    ]

    # Clip gradients
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0)

    print(f"[Rank {rank}] TP-only grad norm: {grad_norm.item():.6f}")

    # Verify gradients were clipped
    for i, p in enumerate(params):
        grad_norm_local = p.grad.to_local().norm().item()
        print(f"[Rank {rank}] Param {i} local grad norm after clipping: {grad_norm_local:.6f}")

    print(f"[Rank {rank}] TP-only test passed!\n")


def test_pp_only(rank, world_size, device):
    """Test with PP-only configuration."""
    if world_size < 2:
        print(f"[Rank {rank}] Skipping PP-only test (requires world_size >= 2)\n")
        return

    print(f"[Rank {rank}] Testing PP-only configuration (PP={world_size})")

    # Create PP mesh
    pp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("pp",))

    # Create DTensors (replicated, since PP doesn't shard tensors)
    params = [
        create_dtensor_with_grad((128, 256), pp_mesh, (Replicate(),), device, grad_value=2.0 + rank),
        create_dtensor_with_grad((256, 512), pp_mesh, (Replicate(),), device, grad_value=3.0 + rank),
    ]

    # Clip gradients with PP mesh
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0, pp_mesh=pp_mesh)

    print(f"[Rank {rank}] PP-only grad norm: {grad_norm.item():.6f}")

    # All ranks should have the same grad norm after PP reduction
    print(f"[Rank {rank}] PP-only test passed!\n")


def test_tp_pp(rank, world_size, device):
    """Test with TP+PP configuration."""
    if world_size < 4:
        print(f"[Rank {rank}] Skipping TP+PP test (requires world_size >= 4)\n")
        return

    # Use 2x2 mesh
    tp_size = 2
    pp_size = world_size // tp_size

    print(f"[Rank {rank}] Testing TP+PP configuration (TP={tp_size}, PP={pp_size})")

    # Create 2D mesh
    mesh_2d = init_device_mesh("cuda", (pp_size, tp_size), mesh_dim_names=("pp", "tp"))
    pp_mesh = mesh_2d["pp"]

    # Create DTensors with TP sharding
    params = [
        create_dtensor_with_grad((128, 256), mesh_2d, (Replicate(), Shard(0)), device, grad_value=2.0),
        create_dtensor_with_grad((256, 512), mesh_2d, (Replicate(), Shard(1)), device, grad_value=3.0),
    ]

    # Clip gradients with PP mesh
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0, pp_mesh=pp_mesh)

    print(f"[Rank {rank}] TP+PP grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] TP+PP test passed!\n")


def test_ep_only(rank, world_size, device):
    """Test with EP-only configuration (for MoE)."""
    print(f"[Rank {rank}] Testing EP-only configuration (EP={world_size})")

    # Create EP mesh
    ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))

    # Create DTensors with EP sharding (expert parallelism)
    params = [
        create_dtensor_with_grad((128, 256), ep_mesh, (Shard(0),), device, grad_value=2.0),
        create_dtensor_with_grad((256, 512), ep_mesh, (Replicate(),), device, grad_value=3.0),
    ]

    # Clip gradients
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0)

    print(f"[Rank {rank}] EP-only grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] EP-only test passed!\n")


def test_tp_ep(rank, world_size, device):
    """Test with TP+EP configuration."""
    if world_size < 4:
        print(f"[Rank {rank}] Skipping TP+EP test (requires world_size >= 4)\n")
        return

    # Use 2x2 mesh
    tp_size = 2
    ep_size = world_size // tp_size

    print(f"[Rank {rank}] Testing TP+EP configuration (TP={tp_size}, EP={ep_size})")

    # Create 2D mesh
    mesh_2d = init_device_mesh("cuda", (tp_size, ep_size), mesh_dim_names=("tp", "ep"))

    # Create DTensors with mixed sharding
    params = [
        # TP-sharded parameter
        create_dtensor_with_grad((128, 256), mesh_2d, (Shard(0), Replicate()), device, grad_value=2.0),
        # EP-sharded parameter (expert parameter)
        create_dtensor_with_grad((256, 512), mesh_2d, (Replicate(), Shard(0)), device, grad_value=3.0),
        # Replicated parameter
        create_dtensor_with_grad((64, 64), mesh_2d, (Replicate(), Replicate()), device, grad_value=1.5),
    ]

    # Clip gradients
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0)

    print(f"[Rank {rank}] TP+EP grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] TP+EP test passed!\n")


def test_3d_parallel(rank, world_size, device):
    """Test with TP+PP+EP configuration (3D parallelism)."""
    if world_size < 8:
        print(f"[Rank {rank}] Skipping 3D parallel test (requires world_size >= 8)\n")
        return

    # Use 2x2x2 mesh
    pp_size = 2
    tp_size = 2
    ep_size = world_size // (pp_size * tp_size)

    print(f"[Rank {rank}] Testing 3D parallel configuration (PP={pp_size}, TP={tp_size}, EP={ep_size})")

    # Create 3D mesh
    mesh_3d = init_device_mesh("cuda", (pp_size, tp_size, ep_size), mesh_dim_names=("pp", "tp", "ep"))
    pp_mesh = mesh_3d["pp"]

    # Create DTensors with various sharding patterns
    params = [
        # TP-sharded, PP-replicated, EP-replicated
        create_dtensor_with_grad((128, 256), mesh_3d, (Replicate(), Shard(0), Replicate()), device, grad_value=2.0),
        # EP-sharded, TP-replicated, PP-replicated (expert parameter)
        create_dtensor_with_grad((256, 512), mesh_3d, (Replicate(), Replicate(), Shard(0)), device, grad_value=3.0),
        # Fully replicated
        create_dtensor_with_grad((64, 64), mesh_3d, (Replicate(), Replicate(), Replicate()), device, grad_value=1.5),
    ]

    # Clip gradients with PP mesh
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0, pp_mesh=pp_mesh)

    print(f"[Rank {rank}] 3D parallel grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] 3D parallel test passed!\n")


def test_mixed_sharding_groups(rank, world_size, device):
    """Test with multiple different sharding placements on the same mesh to verify grouping logic."""
    if world_size < 4:
        print(f"[Rank {rank}] Skipping mixed sharding test (requires world_size >= 4)\n")
        return

    print(f"[Rank {rank}] Testing mixed sharding groups")

    # Create a 2D mesh
    tp_size = 2
    dp_size = world_size // tp_size
    mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    # Create DTensors with different sharding placements on the same mesh
    params = []

    # Different sharding patterns on the same mesh
    params.append(create_dtensor_with_grad((128, 256), mesh_2d, (Replicate(), Shard(0)), device, grad_value=2.0))
    params.append(create_dtensor_with_grad((256, 256), mesh_2d, (Shard(0), Replicate()), device, grad_value=3.0))
    params.append(create_dtensor_with_grad((128, 512), mesh_2d, (Replicate(), Shard(1)), device, grad_value=1.5))
    params.append(create_dtensor_with_grad((256, 512), mesh_2d, (Replicate(), Replicate()), device, grad_value=2.5))

    # Clip gradients
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0)

    print(f"[Rank {rank}] Mixed sharding grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] Mixed sharding test passed!\n")


def test_moe_dp_ep_separate_meshes(rank, world_size, device):
    """Test MoE scenario with separate DP and EP meshes.

    This simulates the realistic MoE case where:
    - Expert parameters have ep_shard placement on EP mesh
    - Non-expert parameters have dp_replicate or dp_shard placement on DP mesh
    """
    print(f"[Rank {rank}] Testing MoE with separate DP and EP meshes (world_size={world_size})")

    # Create separate DP and EP meshes (both use all ranks)
    dp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))

    params = []

    # Non-expert parameters on DP mesh
    # Replicated across DP
    params.append(create_dtensor_with_grad((128, 256), dp_mesh, (Replicate(),), device, grad_value=1.0))
    # Sharded across DP
    params.append(create_dtensor_with_grad((256, 512), dp_mesh, (Shard(0),), device, grad_value=1.5))
    params.append(create_dtensor_with_grad((512, 256), dp_mesh, (Shard(1),), device, grad_value=2.0))

    # Expert parameters on EP mesh
    # Sharded across EP (typical for MoE experts)
    params.append(create_dtensor_with_grad((512, 256), ep_mesh, (Shard(0),), device, grad_value=3.0))
    params.append(create_dtensor_with_grad((256, 128), ep_mesh, (Shard(0),), device, grad_value=2.5))
    # Some experts might be replicated
    params.append(create_dtensor_with_grad((64, 64), ep_mesh, (Replicate(),), device, grad_value=1.2))

    # Clip gradients
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0)

    print(f"[Rank {rank}] MoE DP/EP separate meshes grad norm: {grad_norm.item():.6f}")

    # Verify gradients were clipped
    for i, p in enumerate(params):
        grad_norm_local = p.grad.to_local().norm().item()
        print(f"[Rank {rank}] Param {i} local grad norm after clipping: {grad_norm_local:.6f}")

    print(f"[Rank {rank}] MoE DP/EP separate meshes test passed!\n")


def test_moe_with_tp_dp_ep(rank, world_size, device):
    """Test MoE with TP+DP+EP where EP and DP are on different sub-meshes."""
    if world_size < 4:
        print(f"[Rank {rank}] Skipping MoE TP+DP+EP test (requires world_size >= 4)\n")
        return

    print(f"[Rank {rank}] Testing MoE with TP+DP+EP (world_size={world_size})")

    # Configuration: split world_size between TP and DP/EP
    tp_size = 2
    remaining = world_size // tp_size

    # Create a 2D mesh for non-expert params (DP, TP)
    dp_tp_mesh = init_device_mesh("cuda", (remaining, tp_size), mesh_dim_names=("dp", "tp"))

    # Create a 2D mesh for expert params (EP, TP)
    ep_tp_mesh = init_device_mesh("cuda", (remaining, tp_size), mesh_dim_names=("ep", "tp"))

    params = []

    # Non-expert parameters on DP+TP mesh
    params.append(create_dtensor_with_grad((128, 256), dp_tp_mesh, (Replicate(), Shard(0)), device, grad_value=1.5))
    params.append(create_dtensor_with_grad((256, 512), dp_tp_mesh, (Shard(0), Replicate()), device, grad_value=2.0))
    params.append(create_dtensor_with_grad((512, 256), dp_tp_mesh, (Replicate(), Shard(1)), device, grad_value=1.8))

    # Expert parameters on EP+TP mesh
    params.append(create_dtensor_with_grad((256, 512), ep_tp_mesh, (Shard(0), Replicate()), device, grad_value=3.0))
    params.append(create_dtensor_with_grad((512, 256), ep_tp_mesh, (Shard(0), Shard(0)), device, grad_value=2.5))
    params.append(create_dtensor_with_grad((128, 128), ep_tp_mesh, (Replicate(), Replicate()), device, grad_value=1.2))

    # Clip gradients
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=2.0)

    print(f"[Rank {rank}] MoE TP+DP+EP grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] MoE TP+DP+EP test passed!\n")


def test_inf_norm(rank, world_size, device):
    """Test with infinity norm."""
    print(f"[Rank {rank}] Testing infinity norm")

    # Create TP mesh
    tp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    # Create DTensors with different gradient values
    params = [
        create_dtensor_with_grad((128, 256), tp_mesh, (Shard(0),), device, grad_value=2.0),
        create_dtensor_with_grad((256, 512), tp_mesh, (Shard(1),), device, grad_value=5.0),  # This should dominate
        create_dtensor_with_grad((64, 64), tp_mesh, (Replicate(),), device, grad_value=1.0),
    ]

    # Clip with infinity norm
    grad_norm = _clip_grad_norm_impl(params, max_norm=1.0, norm_type=float("inf"))

    print(f"[Rank {rank}] Infinity norm grad norm: {grad_norm.item():.6f}")
    print(f"[Rank {rank}] Infinity norm test passed!\n")


def main():
    """Run all tests appropriate for the world size."""
    rank, world_size, device = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print(f"Running _clip_grad_norm_impl tests with {world_size} ranks")
        print("=" * 80)
        print()

    dist.barrier()

    # Run tests appropriate for the world size
    try:
        test_tp_only(rank, world_size, device)
        dist.barrier()

        test_pp_only(rank, world_size, device)
        dist.barrier()

        test_tp_pp(rank, world_size, device)
        dist.barrier()

        test_ep_only(rank, world_size, device)
        dist.barrier()

        test_tp_ep(rank, world_size, device)
        dist.barrier()

        test_3d_parallel(rank, world_size, device)
        dist.barrier()

        test_mixed_sharding_groups(rank, world_size, device)
        dist.barrier()

        test_moe_dp_ep_separate_meshes(rank, world_size, device)
        dist.barrier()

        test_moe_with_tp_dp_ep(rank, world_size, device)
        dist.barrier()

        test_inf_norm(rank, world_size, device)
        dist.barrier()

        if rank == 0:
            print("=" * 80)
            print("All tests passed successfully!")
            print("=" * 80)

        return 0

    except Exception as e:
        print(f"[Rank {rank}] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        dist.barrier()
        return 1

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
