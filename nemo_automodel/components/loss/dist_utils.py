# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import torch
import torch.distributed as dist


class _AllGatherWithGrad(torch.autograd.Function):
    """All-gather on dim-0 with autograd support across distributed ranks."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not dist.is_available() or not dist.is_initialized():
            ctx.world_size = 1
            return (x,)
        world_size = dist.get_world_size()
        ctx.world_size = world_size
        out = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        if ctx.world_size == 1:
            return grads[0]
        rank = dist.get_rank()
        stacked = torch.stack(grads, dim=0).contiguous()
        dist.all_reduce(stacked)
        return stacked[rank]


def all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """Gather ``x`` across ranks on dim-0 while preserving autograd."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    gathered = _AllGatherWithGrad.apply(x)
    return torch.cat(gathered, dim=0)


def all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """Gather ``x`` across ranks on dim-0 without autograd."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    with torch.no_grad():
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)
