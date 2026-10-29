#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU Muse SWA forward/backward parity at the released attention dimensions.

Run with TE >= 2.19:
    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_muse_spark_long_document_cp.py

Uses hidden=8192, GQA=128/16, head_dim=64, window=2048. Packed document
lengths straddle the window and CP padding boundaries. No checkpoint required.
Use --nproc-per-node=8 for additional single-node CP8 coverage.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.muse_spark.configuration_muse_spark import MuseSparkConfig
from nemo_automodel.components.models.muse_spark.layers import MuseSparkAttention, MuseSparkRotaryEmbedding


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Accumulate error in chunks without materializing FP64 copies of large weights."""
    error = torch.zeros((), dtype=torch.float64, device=actual.device)
    norm = torch.zeros_like(error)
    for a, b in zip(actual.reshape(-1).split(1_000_000), reference.reshape(-1).split(1_000_000)):
        a, b = a.double(), b.to(actual.device).double()
        error += (a - b).square().sum()
        norm += b.square().sum()
    return (error / norm.clamp_min(1e-30)).sqrt().item()


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    cp_size = dist.get_world_size()
    assert cp_size in (2, 8)
    rank = dist.get_rank()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    config = MuseSparkConfig(dtype="bfloat16")
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", compile_norm=False)
    attention = MuseSparkAttention(config, 0, backend).to(device)
    # TE initializes its extension module while constructing the attention backend.
    import transformer_engine_torch as tex  # noqa: PLC0415

    rotary = MuseSparkRotaryEmbedding(config.head_dim, theta=config.rope_theta).to(device)
    lengths = [2047, 2048, 2049, 4097]
    multiple = 2 * cp_size
    spans = [(length + multiple - 1) // multiple * multiple for length in lengths]
    padded_cu = torch.tensor([0] + spans, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    real_cu = torch.tensor([0] + lengths, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    positions = torch.cat([torch.cat((torch.arange(n), torch.zeros(p - n))) for n, p in zip(lengths, spans)])
    positions = positions.to(device=device, dtype=torch.long)
    valid = torch.cat([torch.arange(p) < n for n, p in zip(lengths, spans)]).to(device)
    hidden = torch.randn(sum(spans), config.hidden_size, dtype=torch.bfloat16, device=device)
    upstream = torch.randn_like(hidden) * valid[:, None]
    references = {}
    result = {"document_lengths": lengths, "padded_lengths": spans, "window": 2048, "cp_size": cp_size}
    for cp in (1, cp_size):
        attention.zero_grad(set_to_none=True)
        if cp > 1:
            attention.attn_module.set_context_parallel_group(
                dist.group.WORLD, list(range(cp_size)), torch.cuda.Stream(), cp_comm_type="p2p"
            )
            indices = tex.thd_get_partitioned_indices(padded_cu, hidden.shape[0], cp_size, rank).long()
        else:
            indices = torch.arange(hidden.shape[0], device=device)
        x = hidden[indices].detach().requires_grad_(True)
        # Match production: CP partitions padded per-document spans. Padding is
        # causal tail padding, so its outputs cannot affect any valid token.
        kwargs = {
            "qkv_format": "thd",
            "cu_seqlens": real_cu if cp == 1 else padded_cu,
            "max_seqlen": max(lengths) if cp == 1 else max(spans),
            "_muse_spark_thd_max_real_seqlen": max(lengths),
        }
        if cp == 1:
            kwargs["cu_seqlens_padded"] = padded_cu
        output = attention(x, rotary(x, positions[indices]), **kwargs)
        output.backward(upstream[indices])
        assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
        if cp == 1:
            references["output"] = output.detach().cpu()
            references["input_grad"] = x.grad.cpu()
            references["parameters"] = {name: p.grad.cpu() for name, p in attention.named_parameters()}
        else:
            assert attention.attn_module.cp_comm_type == "all_gather"
            selected = valid[indices]
            result["output_relative_l2"] = relative_l2(
                output.detach()[selected], references["output"][indices.cpu()][selected.cpu()]
            )
            result["input_gradient_relative_l2"] = relative_l2(x.grad, references["input_grad"][indices.cpu()])
            parameter_errors = {}
            for name, parameter in attention.named_parameters():
                # CP replicates weights: sum local contributions, do not average.
                gradient = parameter.grad.float()
                dist.all_reduce(gradient)
                parameter_errors[name] = relative_l2(gradient, references["parameters"][name])
            result["parameter_gradient_relative_l2"] = parameter_errors
            assert result["output_relative_l2"] < 1e-3, result
            assert result["input_gradient_relative_l2"] < 2e-2, result
            assert max(parameter_errors.values()) < 2e-2, result
            # Changing prior documents and all padding must leave the last
            # document's valid outputs unchanged, including CP chunk boundaries.
            changed = hidden.clone()
            last_start = sum(spans[:-1])
            changed[:last_start] = -changed[:last_start]
            changed[~valid] = 3
            with torch.no_grad():
                y = changed[indices]
                perturbed = attention(y, rotary(y, positions[indices]), **kwargs)
            last = selected & (indices >= last_start)
            leak = (perturbed[last] - output.detach()[last]).abs().max().float()
            dist.all_reduce(leak, op=dist.ReduceOp.MAX)
            result["document_and_padding_leak"] = leak.item()
            assert leak.item() == 0, result
            # The same document may exceed 2K, but attention must still forget
            # its first token once that token falls outside the 2048-token window.
            changed = hidden.clone()
            changed[last_start] = -changed[last_start]
            with torch.no_grad():
                y = changed[indices]
                perturbed = attention(y, rotary(y, positions[indices]), **kwargs)
            far = last & (indices >= last_start + config.sliding_window)
            far_delta = (perturbed[far] - output.detach()[far]).abs().max().float()
            dist.all_reduce(far_delta, op=dist.ReduceOp.MAX)
            result["outside_sliding_window_leak"] = far_delta.item()
            assert far_delta.item() == 0, result
            first = indices == last_start
            near_delta = torch.zeros((), device=device)
            if first.any():
                near_delta = (perturbed[first] - output.detach()[first]).abs().max().float()
            dist.all_reduce(near_delta, op=dist.ReduceOp.MAX)
            result["changed_token_output_delta"] = near_delta.item()
            assert near_delta.item() > 0, "Perturbation must be observable inside the window"
        del output, x
    result["passed"] = True
    if rank == 0:
        print(json.dumps(result, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
