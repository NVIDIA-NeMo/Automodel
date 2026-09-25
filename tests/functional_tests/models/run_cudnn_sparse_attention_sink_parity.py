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

"""Real-kernel parity for the shared learnable-sink sparse-attention path."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch

from nemo_automodel.components.models.common.cudnn_sparse_attention import (
    _sparse_attention_delta_fp32,
    _sparse_attention_delta_fp32_compiled,
    cudnn_sparse_attention,
)

FLASH_MLA_REFERENCE_COMMIT = "b7643bd54521f563b839b98289b5cd048c062ba2"


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Return scalar accuracy metrics for equal-shaped tensors."""
    if actual.shape != reference.shape:
        raise AssertionError(f"Shape mismatch: {tuple(actual.shape)} != {tuple(reference.shape)}")
    actual_flat = actual.detach().double().flatten().cpu()
    reference_flat = reference.detach().double().flatten().cpu()
    difference = actual_flat - reference_flat
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference_flat).clamp_min(1e-30)
        ).item(),
        "cosine": (
            torch.dot(actual_flat, reference_flat)
            / (torch.linalg.vector_norm(actual_flat) * torch.linalg.vector_norm(reference_flat)).clamp_min(1e-30)
        ).item(),
    }


def _reference_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate sparse MQA plus a zero-valued sink with explicit FP32 math."""
    keys = kv.squeeze(1)
    rows = []
    for query_idx in range(q.shape[0]):
        selected = indices[query_idx, 0].long()
        selected_kv = keys.index_select(0, selected[selected >= 0])
        scores = torch.matmul(q[query_idx].float(), selected_kv.float().T) * scale
        probabilities = torch.softmax(torch.cat((scores, sink.unsqueeze(-1)), dim=-1), dim=-1)[..., :-1]
        rows.append(torch.matmul(probabilities, selected_kv[..., :512].float()))
    return torch.stack(rows).to(torch.bfloat16)


def run(output: Path | None = None) -> dict:
    """Run forward/backward and production-shape sink-delta parity."""
    import cudnn

    flash_version = version("flash-mla")
    if FLASH_MLA_REFERENCE_COMMIT[:7] not in flash_version:
        raise RuntimeError(f"Expected FlashMLA build from {FLASH_MLA_REFERENCE_COMMIT}, got {flash_version!r}.")
    if str(getattr(cudnn, "__version__", "")) != "1.27.0":
        raise RuntimeError(f"Expected cuDNN Frontend 1.27.0, got {getattr(cudnn, '__version__', None)!r}.")

    torch.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = 64
    heads = 64
    sparse_width = 512
    scale = 256**-0.5
    q = torch.randn(tokens, heads, 576, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(tokens, 1, 576, device=device, dtype=torch.bfloat16, requires_grad=True)
    sink = torch.linspace(-0.75, 0.75, heads, device=device, dtype=torch.float32).requires_grad_()
    indices = torch.full((tokens, 1, sparse_width), -1, device=device, dtype=torch.int32)
    for query_idx in range(tokens):
        indices[query_idx, 0, : query_idx + 1] = torch.arange(query_idx + 1, device=device, dtype=torch.int32)
    topk_length = torch.arange(1, tokens + 1, device=device, dtype=torch.int32)
    upstream = torch.randn(tokens, heads, 512, device=device, dtype=torch.bfloat16)

    actual = cudnn_sparse_attention(
        q,
        kv,
        indices,
        scale,
        topk_length=topk_length,
        all_rows_nonempty=True,
        attn_sink=sink,
    )
    (actual * upstream).sum().backward()
    actual_gradients = (q.grad.detach().clone(), kv.grad.detach().clone(), sink.grad.detach().clone())

    q_reference = q.detach().clone().requires_grad_()
    kv_reference = kv.detach().clone().requires_grad_()
    sink_reference = sink.detach().clone().requires_grad_()
    reference = _reference_sparse_attention(q_reference, kv_reference, indices, sink_reference, scale)
    (reference * upstream).sum().backward()
    attention_report = {
        "forward": _metrics(actual, reference),
        "grad_q": _metrics(actual_gradients[0], q_reference.grad),
        "grad_kv": _metrics(actual_gradients[1], kv_reference.grad),
        "grad_sink": _metrics(actual_gradients[2], sink_reference.grad),
    }

    # At [4096, 64, 512], the eager FP32 product alone is 512 MiB. Verify that
    # the compiled reduction preserves parity while lowering steady-state peak.
    full_grad_output = torch.randn(4096, heads, 512, device=device, dtype=torch.bfloat16)
    full_output = torch.randn_like(full_grad_output)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    delta_reference = _sparse_attention_delta_fp32(full_grad_output, full_output)
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated() - baseline
    _sparse_attention_delta_fp32_compiled(full_grad_output[:32], full_output[:32])
    torch.cuda.synchronize()
    del delta_reference
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    delta_actual = _sparse_attention_delta_fp32_compiled(full_grad_output, full_output)
    torch.cuda.synchronize()
    compiled_peak = torch.cuda.max_memory_allocated() - baseline
    delta_reference = _sparse_attention_delta_fp32(full_grad_output, full_output)
    delta_report = {
        "parity": _metrics(delta_actual, delta_reference),
        "reference_incremental_peak_bytes": reference_peak,
        "compiled_incremental_peak_bytes": compiled_peak,
    }

    report = {
        "attention": attention_report,
        "cudnn_frontend_version": cudnn.__version__,
        "flash_mla_reference_commit": FLASH_MLA_REFERENCE_COMMIT,
        "flash_mla_version": flash_version,
        "sink_delta": delta_report,
        "torch_version": torch.__version__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    for name, metrics in attention_report.items():
        relative_l2_limit = 0.025 if name == "forward" else 0.08
        cosine_limit = 0.999 if name == "forward" else 0.995
        if metrics["relative_l2"] >= relative_l2_limit or metrics["cosine"] <= cosine_limit:
            raise AssertionError(f"Sparse-attention {name} parity failed: {metrics}")
    if delta_report["parity"]["relative_l2"] >= 1.0e-3 or delta_report["parity"]["cosine"] <= 0.999999:
        raise AssertionError(f"Sparse-attention sink delta parity failed: {delta_report}")
    if compiled_peak >= reference_peak:
        raise AssertionError(f"Compiled sink delta did not reduce peak memory: {delta_report}")

    if output is not None:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite parity report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
