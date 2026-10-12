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

"""Opt-in CSA full-block FA4 component gates; native 4K keeps its causal fastpath."""

import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.sparse import RegionSelection, region_metadata


def _reference_vjp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute an independent FP32 mathematical VJP with bounded query chunks.

    Args:
        query: BF16 tensor [1, queries, providers * 16, 192].
        key: BF16 tensor [1, keys, providers, 192], possibly a strided KV-unbind view.
        value: BF16 tensor with the key layout.
        upstream: BF16 output gradient with the query layout.

    Returns:
        FP32 output/dQ with query layout and dK/dV with key layout. No dense
        full-sequence query-key mask or score allocation is constructed.
    """
    q = query.detach().float().requires_grad_()
    k = key.detach().float().requires_grad_()
    v = value.detach().float().requires_grad_()
    output, dq = torch.empty_like(q), torch.empty_like(q)
    dk, dv = torch.zeros_like(k), torch.zeros_like(v)
    keys = torch.arange(key.shape[1], device=query.device)[None]
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, query.shape[1], 256):
            stop = min(start + 256, query.shape[1])
            positions = torch.arange(start, stop, device=query.device)[:, None]
            # Each provider chooses a different legal 512-region interval,
            # plus the current region. At 8K this catches head/group mixups.
            providers = key.shape[2]
            group = torch.arange(providers, device=query.device)[:, None, None]
            current = positions // 8
            lower = (current - 512).clamp_min(0)[None] * group // max(providers - 1, 1)
            selected = ((keys // 8 >= lower) & (keys // 8 < lower + 512)) | (keys // 8 == current)
            allowed = ((keys <= positions)[None] & selected).repeat_interleave(16, dim=0)[None]
            chunk = q[:, start:stop]
            result = F.scaled_dot_product_attention(
                chunk.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=allowed,
                enable_gqa=True,
            ).transpose(1, 2)
            grads = torch.autograd.grad(result, (chunk, k, v), upstream[:, start:stop].float())
            output[:, start:stop] = result.detach()
            dq[:, start:stop] = grads[0]
            dk += grads[1]
            dv += grads[2]
    return output, dq, dk, dv


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compare floating tensors without retaining their graph.

    Args:
        actual: Floating tensor of arbitrary shape [...].
        expected: Nonzero reference with the same shape.

    Returns:
        Scalar relative root-mean-square error.
    """
    difference = actual.float() - expected.float()
    return float(difference.square().sum().sqrt() / expected.float().square().sum().sqrt())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [4096, 8192])
def test_forward_full_blocks_match_fixed_routes_and_independent_vjp(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
) -> None:
    """Exercise real D192/GQA16 FA4 with noncontiguous interleaved KV.

    The 4K component intentionally omits causal_layout to test custom FA4.
    Native all-selected 4K dispatch remains covered by the existing CP runner.
    """
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Forward full-block option is restricted to SM90")
    torch.manual_seed(583)
    providers = 4 if length == 8192 else 1
    query = torch.randn(1, length, providers * 16, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, length, 2, providers, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key, value = kv.unbind(2)
    assert not key.is_contiguous() and not value.is_contiguous()
    positions = torch.arange(length, device="cuda")[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions,
        padding,
        query_start=0,
        queries=length,
        region_size=8,
        topk=512,
        csa_forward_full_blocks=True,
    )
    current = positions // 8
    lower = (current - 512).clamp_min(0)[:, None] * torch.arange(providers, device="cuda")[None, :, None]
    lower = lower // max(providers - 1, 1)
    historical = lower[..., None] + torch.arange(512, device="cuda")
    historical = torch.where(historical < current[:, None, :, None], historical, metadata.region_count)
    ids = torch.cat((historical, current[:, None, :, None].expand(-1, providers, -1, -1)), dim=-1)
    ids = ids.sort(dim=-1).values.int()
    selection = RegionSelection(
        ids,
        metadata.token_regions,
        metadata.query_documents,
        metadata.documents,
        metadata.region_count,
    )
    upstream = torch.randn_like(query)
    full_block_counts = []
    original = fa4._split_forward_full_blocks

    def record_split(
        counts: torch.Tensor,
        indices: torch.Tensor,
        *,
        prefix_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Observe forward metadata while preserving the real kernel call.

        Args:
            counts: Int32 counts [provider_batches, heads, query_tiles];
                either [groups, 1, query_tiles] or [1, query_heads, query_tiles].
            indices: Int32 indices [provider_batches, heads, query_tiles, key_tiles].
            prefix_length: Proven initial all-selected query count.

        Returns:
            Partial counts, partial indices and full counts in their original
            semantic layouts, unchanged from the production helper.
        """
        result = original(counts, indices, prefix_length=prefix_length)
        full_block_counts.append(result[2].detach().cpu())
        return result

    monkeypatch.setattr(fa4, "_split_forward_full_blocks", record_split)
    runs = []
    memory = []
    for enabled in (False, True):
        chosen = replace(selection, forward_full_block_prefix=metadata.forward_full_block_prefix if enabled else 0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        output = fa4.fa4_attention(
            query,
            key,
            value,
            positions,
            positions,
            padding,
            metadata.documents,
            query_start=0,
            window=None,
            selection=chosen,
        )
        forward_peak = torch.cuda.max_memory_allocated()
        gradients = torch.autograd.grad(output, (query, kv), upstream)
        torch.cuda.synchronize()
        memory.append(
            {
                "allocated_before_bytes": allocated_before,
                "forward_peak_allocated_bytes": forward_peak,
                "forward_backward_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        runs.append((output.detach(), gradients[0], gradients[1][:, :, 0], gradients[1][:, :, 1]))
    assert len(full_block_counts) == 1 and full_block_counts[0].sum() > 0
    assert full_block_counts[0].shape[:2] == ((1, providers * 16) if providers > 1 else (1, 1))
    # Identical block traversal is a bitwise-forward hypothesis under test,
    # not an excuse to relax a mismatch.
    assert torch.equal(runs[0][0].view(torch.uint8), runs[1][0].view(torch.uint8))
    torch.cuda.reset_peak_memory_stats()
    expected = _reference_vjp(query, key, value, upstream)
    oracle_peak = torch.cuda.max_memory_allocated()
    pair_metrics = {}
    for name, baseline, candidate in zip(("dQ", "dK", "dV"), runs[0][1:], runs[1][1:], strict=True):
        relative = _relative_rms(candidate, baseline)
        maximum = float((candidate.float() - baseline.float()).abs().max())
        assert relative < 0.012, (name, relative, maximum)
        pair_metrics[name] = {"relative_rms": relative, "maxabs": maximum}
    metrics = []
    for result in runs:
        torch.testing.assert_close(result[0].float(), expected[0], atol=0.015, rtol=0.015)
        errors = [_relative_rms(a, e) for a, e in zip(result[1:], expected[1:], strict=True)]
        assert all(error < 0.012 for error in errors), errors
        assert all(torch.isfinite(tensor).all() for tensor in result)
        metrics.append(errors)
    bucket_errors = {}
    pair_buckets = {}
    for label, start, stop in (("prefix", 0, min(length, 4104)), ("sparse", 4104, length)):
        if start < stop:
            torch.testing.assert_close(
                runs[1][0][:, start:stop].float(), expected[0][:, start:stop], atol=0.015, rtol=0.015
            )
            bucket_errors[label] = _relative_rms(runs[1][1][:, start:stop], expected[1][:, start:stop])
            assert bucket_errors[label] < 0.012
            pair_buckets[label] = {
                "relative_rms": _relative_rms(runs[1][1][:, start:stop], runs[0][1][:, start:stop]),
                "maxabs": float((runs[1][1][:, start:stop].float() - runs[0][1][:, start:stop].float()).abs().max()),
            }
            assert pair_buckets[label]["relative_rms"] < 0.012
    print(
        json.dumps(
            {
                "length": length,
                "head_dim": 192,
                "gqa": 16,
                "providers": providers,
                "q_stride": query.stride(),
                "kv_stride": kv.stride(),
                "key_stride": key.stride(),
                "value_stride": value.stride(),
                "full_blocks": int(full_block_counts[0].sum()),
                "forward_bitwise": True,
                "gradient_relative_rms_each_vs_fp32": metrics,
                "candidate_vs_baseline_gradients": pair_metrics,
                "candidate_vs_baseline_dq_buckets": pair_buckets,
                "memory_baseline_candidate": memory,
                "oracle_peak_allocated_bytes": oracle_peak,
                "query_gradient_buckets": bucket_errors,
                "truly_sparse_queries": max(0, length - 4104),
                "scope": "Correctness only: fixed legal routes, synthetic nonzero upstream, and synchronized metadata observation. No performance or checkpoint/native model parity claim.",
            }
        )
    )
