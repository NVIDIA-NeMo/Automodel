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

"""Real-kernel forward/backward parity for HY V4's cuDNN DSA path.

This exercises the same split used by the model: cuDNN Frontend for indexer
scores/top-k, FlashMLA for sparse-attention forward, and cuDNN for its backward.
The numerical oracle is explicit PyTorch math over the same selected indices.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch

from nemo_automodel.components.models.common.cudnn_sparse_attention import (
    _sparse_attention_delta_fp32,
    _sparse_attention_delta_fp32_compiled,
)
from nemo_automodel.components.models.hy_v4.kernels.cudnn_dsa import (
    cudnn_indexer_topk,
    cudnn_sparse_attention,
)

FLASH_MLA_REFERENCE_COMMIT = "b7643bd54521f563b839b98289b5cd048c062ba2"


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Return scalar metrics for two equal-shaped tensors.

    Args:
        actual: Optimized-kernel tensor with arbitrary layout.
        reference: Explicit-math tensor with the same layout and shape.

    Returns:
        Scalar absolute-error, relative-L2, and cosine metrics.
    """
    if actual.shape != reference.shape:
        raise AssertionError(f"Shape mismatch: {tuple(actual.shape)} != {tuple(reference.shape)}")
    actual = actual.detach().double().flatten().cpu()
    reference = reference.detach().double().flatten().cpu()
    difference = actual - reference
    actual_norm = torch.linalg.vector_norm(actual)
    reference_norm = torch.linalg.vector_norm(reference)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-30)).item(),
        "cosine": (torch.dot(actual, reference) / (actual_norm * reference_norm).clamp_min(1e-30)).item(),
    }


def _torch_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    head_weights: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Compute causal HY4 index selection with explicit FP32 operations.

    Args:
        q: BF16 queries ``[tokens, index_heads, 128]``.
        k: BF16 keys ``[tokens, 128]``.
        head_weights: Already-scaled weights ``[tokens, index_heads]``.
        topk: Fixed sparse selection width.

    Returns:
        New contiguous int32 indices ``[tokens, 1, topk]``.
    """
    scores = torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu()
    # ``cudnn_indexer_topk`` converts the already-scaled projection to BF16
    # before invoking the cuDNN frontend wrapper.
    scores = (scores * head_weights.to(torch.bfloat16).float().unsqueeze(-1)).sum(dim=1)
    causal = torch.ones_like(scores, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    values, selected = scores.topk(topk, dim=-1)
    valid = torch.isfinite(values)
    sort_keys = torch.where(valid, selected, torch.full_like(selected, scores.shape[-1]))
    order = sort_keys.argsort(dim=-1)
    selected = selected.gather(-1, order)
    valid = valid.gather(-1, order)
    selected = selected.masked_fill(~valid, -1)
    return selected.to(torch.int32).unsqueeze(1).contiguous()


def _torch_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate sparse MQA and its learned sink with explicit FP32 math.

    Args:
        q: Absorbed BF16 queries ``[tokens, heads, 576]``.
        kv: Shared BF16 latent K/V ``[tokens, 1, 576]``.
        indices: Int32 sparse indices ``[tokens, 1, sparse_width]``.
        sink: FP32 sink logits ``[heads]``.
        scale: Query/key score scale.

    Returns:
        New BF16 latent values ``[tokens, heads, 512]``.
    """
    keys = kv.squeeze(1)
    rows: list[torch.Tensor] = []
    for query_idx in range(q.shape[0]):
        selected = indices[query_idx, 0].long()
        selected = selected[selected >= 0]
        selected_kv = keys.index_select(0, selected)
        scores = torch.matmul(q[query_idx].float(), selected_kv.float().T) * scale
        probabilities = torch.softmax(torch.cat((scores, sink.float().unsqueeze(-1)), dim=-1), dim=-1)[..., :-1]
        rows.append(torch.matmul(probabilities, selected_kv[..., :512].float()))
    return torch.stack(rows).to(torch.bfloat16)


def run(output: Path | None) -> dict:
    """Run real-kernel parity, assert thresholds, and optionally persist evidence."""
    import cudnn

    flash_version = version("flash-mla")
    if FLASH_MLA_REFERENCE_COMMIT[:7] not in flash_version:
        raise RuntimeError(f"Expected FlashMLA build from {FLASH_MLA_REFERENCE_COMMIT}, got version {flash_version!r}.")
    if str(getattr(cudnn, "__version__", "")) != "1.27.0":
        raise RuntimeError(f"Expected cuDNN Frontend 1.27.0, got {getattr(cudnn, '__version__', None)!r}.")

    torch.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = 64
    index_heads = 32
    index_topk = 16
    index_q = torch.randn(tokens, index_heads, 128, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(tokens, 128, device=device, dtype=torch.bfloat16)
    # Positive, nonuniform weights keep the random top-k margins well-conditioned.
    head_weights = torch.rand(tokens, index_heads, device=device, dtype=torch.float32).add_(0.25)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    actual_indices = cudnn_indexer_topk(
        index_q,
        index_k,
        head_weights,
        cu_seqlens,
        index_topk,
    )
    reference_indices = _torch_indexer(index_q, index_k, head_weights, index_topk)
    index_match = actual_indices.eq(reference_indices)
    index_report = {
        "element_match_fraction": index_match.float().mean().item(),
        "row_match_fraction": index_match.flatten(1).all(dim=1).float().mean().item(),
    }

    attention_heads = 64
    attention_topk = 512
    q = torch.randn(tokens, attention_heads, 576, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(tokens, 1, 576, device=device, dtype=torch.bfloat16, requires_grad=True)
    sink = torch.linspace(-0.75, 0.75, attention_heads, device=device, dtype=torch.float32).requires_grad_()
    causal_indices = torch.full((tokens, 1, attention_topk), -1, device=device, dtype=torch.int32)
    for query_idx in range(tokens):
        causal_indices[query_idx, 0, : query_idx + 1] = torch.arange(query_idx + 1, device=device, dtype=torch.int32)
    topk_length = torch.arange(1, tokens + 1, device=device, dtype=torch.int32)
    scale = 256**-0.5
    upstream = torch.randn(tokens, attention_heads, 512, device=device, dtype=torch.bfloat16)

    actual_output = cudnn_sparse_attention(
        q,
        kv,
        causal_indices,
        scale,
        topk_length=topk_length,
        all_rows_nonempty=True,
        attn_sink=sink,
    )
    (actual_output * upstream).sum().backward()
    actual_gradients = (q.grad.detach().clone(), kv.grad.detach().clone(), sink.grad.detach().clone())

    q_reference = q.detach().clone().requires_grad_()
    kv_reference = kv.detach().clone().requires_grad_()
    sink_reference = sink.detach().clone().requires_grad_()
    reference_output = _torch_sparse_attention(q_reference, kv_reference, causal_indices, sink_reference, scale)
    (reference_output * upstream).sum().backward()

    attention_report = {
        "forward": _metrics(actual_output, reference_output),
        "grad_q": _metrics(actual_gradients[0], q_reference.grad),
        "grad_kv": _metrics(actual_gradients[1], kv_reference.grad),
        "grad_sink": _metrics(actual_gradients[2], sink_reference.grad),
    }

    # HY4-preview's production shape is large enough that the eager FP32
    # product in the sink-gradient delta consumes 512 MiB. Check the compiled
    # reduction at that exact shape and record its steady-state peak memory.
    full_grad_output = torch.randn(4096, attention_heads, 512, device=device, dtype=torch.bfloat16)
    full_output = torch.randn_like(full_grad_output)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    delta_reference = _sparse_attention_delta_fp32(full_grad_output, full_output)
    torch.cuda.synchronize()
    delta_reference_peak = torch.cuda.max_memory_allocated() - baseline
    _sparse_attention_delta_fp32_compiled(full_grad_output[:32], full_output[:32])
    torch.cuda.synchronize()
    del delta_reference
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    delta_actual = _sparse_attention_delta_fp32_compiled(full_grad_output, full_output)
    torch.cuda.synchronize()
    delta_compiled_peak = torch.cuda.max_memory_allocated() - baseline
    delta_reference = _sparse_attention_delta_fp32(full_grad_output, full_output)
    delta_report = {
        "parity": _metrics(delta_actual, delta_reference),
        "reference_incremental_peak_bytes": delta_reference_peak,
        "compiled_incremental_peak_bytes": delta_compiled_peak,
    }
    report = {
        "attention": attention_report,
        "cudnn_frontend_version": cudnn.__version__,
        "flash_mla_reference_commit": FLASH_MLA_REFERENCE_COMMIT,
        "flash_mla_version": flash_version,
        "indexer": index_report,
        "sink_delta": delta_report,
        "torch_version": torch.__version__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if index_report["row_match_fraction"] < 0.98:
        raise AssertionError(f"cuDNN indexer top-k parity failed: {index_report}")
    for name, metrics in attention_report.items():
        relative_l2_limit = 0.025 if name == "forward" else 0.08
        cosine_limit = 0.999 if name == "forward" else 0.995
        if metrics["relative_l2"] >= relative_l2_limit or metrics["cosine"] <= cosine_limit:
            raise AssertionError(f"Sparse-attention {name} parity failed: {metrics}")
    if delta_report["parity"]["relative_l2"] >= 1.0e-3 or delta_report["parity"]["cosine"] <= 0.999999:
        raise AssertionError(f"Sparse-attention sink delta parity failed: {delta_report}")
    if delta_compiled_peak >= delta_reference_peak:
        raise AssertionError(f"Compiled sparse-attention sink delta did not reduce peak memory: {delta_report}")

    if output is not None:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite kernel parity report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
