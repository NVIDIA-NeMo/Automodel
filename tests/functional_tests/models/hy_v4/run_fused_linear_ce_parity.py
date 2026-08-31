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

"""Compare HY V4's materialized FP32-logit CE with cut cross entropy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.models.hy_v4.model import HyV4LMHead


def _tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Return scale-aware difference metrics for two CPU tensors."""
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    return {
        "max_abs": delta.abs().max().item(),
        "rel_l2": (delta.norm() / expected.norm().clamp_min(1.0e-30)).item(),
        # Large FP32 reductions may round a mathematically unit-bounded cosine
        # a few ulps above one.
        "cosine": min(1.0, max(-1.0, cosine)),
    }


def _reference(
    hidden_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor, int]:
    """Run HY V4's FP32-accumulating LM head followed by PyTorch CE."""
    device = torch.device("cuda")
    hidden = hidden_cpu.to(device).requires_grad_(True)
    labels = labels_cpu.to(device)
    head = HyV4LMHead(
        hidden_cpu.shape[-1],
        weight_cpu.shape[0],
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        head.weight.copy_(weight_cpu)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    logits = head(hidden)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    loss.backward()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    return loss.item(), hidden.grad.cpu(), head.weight.grad.cpu(), peak_delta


def _fused(
    hidden_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor, int]:
    """Run the shared cut-cross-entropy kernel used by the production recipe."""
    device = torch.device("cuda")
    hidden = hidden_cpu.to(device).requires_grad_(True)
    weight = weight_cpu.to(device).requires_grad_(True)
    labels = labels_cpu.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    loss = FusedLinearCrossEntropy(ignore_index=-100, reduction="sum")(
        hidden,
        labels,
        weight,
    )
    loss.backward()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    return loss.item(), hidden.grad.cpu(), weight.grad.cpu(), peak_delta


def main() -> None:
    """Run loss, gradient, and peak-memory parity and write a JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("HY V4 fused linear CE parity requires CUDA")

    torch.manual_seed(args.seed)
    hidden = torch.randn(1, args.tokens, args.hidden, dtype=torch.bfloat16)
    weight = torch.randn(args.vocab, args.hidden, dtype=torch.bfloat16) / args.hidden**0.5
    labels = torch.randint(0, args.vocab, (1, args.tokens), dtype=torch.long)
    labels[:, ::17] = -100

    reference_loss, reference_hidden_grad, reference_weight_grad, reference_peak = _reference(
        hidden,
        weight,
        labels,
    )
    fused_loss, fused_hidden_grad, fused_weight_grad, fused_peak = _fused(hidden, weight, labels)

    hidden_grad_metrics = _tensor_metrics(fused_hidden_grad, reference_hidden_grad)
    weight_grad_metrics = _tensor_metrics(fused_weight_grad, reference_weight_grad)
    loss_abs = abs(fused_loss - reference_loss)
    loss_rel = loss_abs / max(abs(reference_loss), 1.0e-30)
    passed = (
        loss_rel <= 1.0e-6
        and hidden_grad_metrics["rel_l2"] <= 2.0e-2
        and hidden_grad_metrics["cosine"] >= 0.9998
        and weight_grad_metrics["rel_l2"] <= 5.0e-3
        and weight_grad_metrics["cosine"] >= 0.99999
        and fused_peak < reference_peak
    )
    result = {
        "passed": passed,
        "seed": args.seed,
        "shape": {"tokens": args.tokens, "hidden": args.hidden, "vocab": args.vocab},
        "loss": {
            "reference": reference_loss,
            "fused": fused_loss,
            "abs": loss_abs,
            "rel": loss_rel,
        },
        "hidden_grad": hidden_grad_metrics,
        "weight_grad": weight_grad_metrics,
        "peak_memory_bytes": {
            "reference_delta": reference_peak,
            "fused_delta": fused_peak,
            "saved": reference_peak - fused_peak,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
