#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check GLM sequence-gather autograd against a single-process mathematical oracle.

Default launch uses two GPUs; explicitly requested larger groups are development
coverage, not a requirement for the two-GPU PR test tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.glm5_next.cp import all_gather_backward_anchor, all_gather_sequence


def strided_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor into a view with stride two on its final axis.

    Args:
        tensor: Tensor of arbitrary nonempty shape on the test device.

    Returns:
        Independent tensor with the same values, shape and dtype, with a
        noncontiguous final axis.
    """
    storage = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] * 2))
    view = storage[..., ::2]
    view.copy_(tensor)
    return view


def coefficient(shape: tuple[int, ...], rank: int, device: torch.device) -> torch.Tensor:
    """Build exact binary fractions with position-dependent gradients."""
    count = 1
    for size in shape:
        count *= size
    return ((torch.arange(count, device=device).reshape(shape) % 7).float() - 3) * ((rank + 1) / 32)


def check_case(
    group: dist.ProcessGroup,
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
    *,
    strided: bool,
    empty_query_rank: bool,
    use_checkpoint: bool,
) -> dict[str, int | float | str | bool]:
    """Compare gather, input gradients, norm and a local SGD update to an oracle."""
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    # Sequence-first construction gives a known global tensor independent of
    # the production gather implementation, including batch=2 and last-axis CP.
    full = (torch.arange(size * 3 * 2 * 4, device=device).reshape(size * 3, 2, 4) % 3).float() / 4
    full = full.movedim(0, dim).to(dtype)
    axis = dim % full.ndim
    local_initial = full.narrow(axis, rank * 3, 3).clone()
    local = torch.nn.Parameter(strided_copy(local_initial) if strided else local_initial.clone())
    optimizer = torch.optim.SGD([local], lr=0.125)
    weights = coefficient(tuple(full.shape), rank, device).to(dtype)
    if strided:
        weights = strided_copy(weights)
    weights_before = weights.clone()

    def gather(value: torch.Tensor) -> torch.Tensor:
        """Gather local sequence states.

        Args:
            value: Local tensor with three sequence positions on axis dim.

        Returns:
            Global tensor with size*3 sequence positions on the same axis.
        """
        result = all_gather_sequence(value, group, dim=dim)
        return result.square() if use_checkpoint else result

    gathered = checkpoint(gather, local, use_reentrant=False) if use_checkpoint else gather(local)
    torch.testing.assert_close(gathered, full.square() if use_checkpoint else full, rtol=0, atol=0)
    objective = (
        all_gather_backward_anchor(gathered).sum()
        if empty_query_rank and rank == size - 1
        else (gathered * weights).sum()
    )
    # The local branch also checks that collective backward doesn't lose a
    # second contribution to the same leaf.
    objective = objective + local.square().sum() * 0.125
    objective.backward()
    torch.testing.assert_close(weights, weights_before, rtol=0, atol=0)

    oracle_full = full.detach().double().requires_grad_()
    oracle_objective = oracle_full.sum() * 0
    for other in range(size):
        if not (empty_query_rank and other == size - 1):
            oracle_objective = (
                oracle_objective
                + (
                    (oracle_full.square() if use_checkpoint else oracle_full)
                    * coefficient(tuple(full.shape), other, device).to(dtype).double()
                ).sum()
            )
    oracle_objective.backward()
    collective_gradient = oracle_full.grad.narrow(axis, rank * 3, 3).to(dtype)
    expected = collective_gradient + (local_initial * 0.25)
    torch.testing.assert_close(local.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(local.grad.float().norm(), expected.float().norm(), rtol=0, atol=0)
    before = local.detach().clone()
    optimizer.step()
    expected_parameter = torch.nn.Parameter(before)
    expected_parameter.grad = expected
    torch.optim.SGD([expected_parameter], lr=0.125).step()
    torch.testing.assert_close(local, expected_parameter, rtol=0, atol=0)
    # This oracle must reject a one-token shift inside the sequence shard.
    if size > 1 and not (empty_query_rank and size == 2):
        wrong = collective_gradient.roll(1, dims=axis)
        assert not torch.equal(collective_gradient, wrong), "Oracle cannot detect a wrong sequence position"
    return dict(
        cp=size,
        rank=rank,
        dtype=str(dtype),
        dim=dim,
        strided=strided,
        empty_query_rank=empty_query_rank,
        checkpoint=use_checkpoint,
        gradient_max_abs=0.0,
        optimizer_max_abs=0.0,
    )


def check_borrowed_gradient(
    group: dist.ProcessGroup, device: torch.device, dtype: torch.dtype
) -> dict[str, int | str | bool]:
    """Require gather backward to preserve the caller's upstream-gradient buffer."""
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    local = torch.zeros((2, 3, 4), device=device, dtype=dtype, requires_grad=True)
    gathered = all_gather_sequence(local, group, dim=1)
    upstream = coefficient(tuple(gathered.shape), rank, device).to(dtype)
    before = upstream.clone()
    gathered.backward(upstream)
    expected = (
        sum(coefficient(tuple(gathered.shape), other, device).double() for other in range(size))
        .to(dtype)
        .narrow(1, rank * 3, 3)
    )
    torch.testing.assert_close(local.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(upstream, before, rtol=0, atol=0)
    return dict(cp=size, rank=rank, dtype=str(dtype), borrowed_gradient_preserved=True)


def check_live_gather_buffers(
    group: dist.ProcessGroup, device: torch.device, *, batch_size: int, width: int, separate_stream: bool
) -> dict[str, int | str | bool]:
    """Keep eight gathered microbatches live while exercising allocator reuse."""
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    local_length = 4096
    default_stream = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device) if separate_stream else default_stream
    stream.wait_stream(default_stream)
    inputs = []
    outputs = []
    with torch.cuda.stream(stream):
        rows = torch.arange(local_length, device=device) + rank * local_length
        columns = torch.arange(width, device=device)
        batches = torch.arange(batch_size, device=device)
        for microbatch in range(8):
            storage = torch.empty((batch_size, local_length, width * 2), device=device, dtype=torch.bfloat16)
            local = storage[..., ::2]
            values = ((batches[:, None, None] * 5 + rows[None, :, None] + columns + microbatch * 3) % 17 - 8) / 16
            local.copy_(values)
            local.requires_grad_()
            gathered = all_gather_sequence(local, group, dim=1)
            inputs.append(local)
            outputs.append(gathered)
            # Request the contiguous input-temporary allocation size after the
            # gather call, while its GPU work may still be in flight.
            for _ in range(2):
                pressure = torch.empty((batch_size, local_length, width), device=device, dtype=torch.bfloat16)
                pressure.fill_(float("nan"))
                del pressure
    default_stream.wait_stream(stream)
    rows = torch.arange(local_length * size, device=device)
    columns = torch.arange(width, device=device)
    batches = torch.arange(batch_size, device=device)
    for microbatch in reversed(range(8)):
        expected = ((batches[:, None, None] * 5 + rows[None, :, None] + columns + microbatch * 3) % 17 - 8) / 16
        torch.testing.assert_close(outputs[microbatch], expected.to(torch.bfloat16), rtol=0, atol=0)
        upstream = torch.full_like(outputs[microbatch], (rank + 1) / 32)
        gradient = torch.autograd.grad(outputs[microbatch], inputs[microbatch], upstream)[0]
        torch.testing.assert_close(gradient, torch.full_like(gradient, size * (size + 1) / 64), rtol=0, atol=0)
        torch.testing.assert_close(upstream, torch.full_like(upstream, (rank + 1) / 32), rtol=0, atol=0)
    return dict(
        cp=size,
        rank=rank,
        local_length=local_length,
        width=width,
        batch_size=batch_size,
        microbatches=8,
        separate_stream=separate_stream,
        live_buffers_preserved=True,
    )


def main() -> None:
    """Run the real Gloo/NCCL process-group matrix and save bounded evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", required=True)
    parser.add_argument("--buffer-stress", action="store_true", help="Exercise eight live 4K microbatches on CUDA")
    args = parser.parse_args()
    if args.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    records = []
    for size in args.group_sizes:
        if world % size:
            raise ValueError(f"CP size {size} does not divide world size {world}")
        own_group = None
        for start in range(0, world, size):
            ranks = list(range(start, start + size))
            group = dist.new_group(ranks, backend=args.backend)
            if rank in ranks:
                own_group = group
        assert own_group is not None
        dtypes = [torch.float32] if args.backend == "gloo" else [torch.bfloat16, torch.float32]
        for dtype in dtypes:
            records.append(check_borrowed_gradient(own_group, device, dtype))
            for dim in [0, 1, -1]:
                for strided, empty, recompute in [
                    (False, False, False),
                    (True, False, False),
                    (True, True, False),
                    (True, True, True),
                ]:
                    records.append(
                        check_case(
                            own_group,
                            device,
                            dtype,
                            dim,
                            strided=strided,
                            empty_query_rank=empty,
                            use_checkpoint=recompute,
                        )
                    )
        if args.buffer_stress:
            if args.backend != "nccl":
                raise ValueError("--buffer-stress requires NCCL/CUDA")
            for batch_size in (1, 2):
                for width in (256, 512):
                    for separate_stream in (False, True):
                        records.append(
                            check_live_gather_buffers(
                                own_group, device, batch_size=batch_size, width=width, separate_stream=separate_stream
                            )
                        )
        dist.barrier()
    from nemo_automodel.components.models.glm5_next import cp

    source = Path(cp.__file__).resolve()
    output = Path(args.output + f".rank{rank}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                status="PASS",
                backend=args.backend,
                rank=rank,
                world_size=world,
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )
    if rank == 0:
        print(f"PASS {args.backend}: {len(records)} cases/rank; groups={args.group_sizes}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
