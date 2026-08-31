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

"""Compare AutoModel HY V4 against same-weight outputs from pinned vLLM.

The reference artifact must be produced by ``run_vllm_reference.py`` from the
unmodified public checkpoint tensors. The AutoModel side intentionally uses
the production ``NeMoAutoModelForCausalLM.from_pretrained`` path so this test
also audits custom-model registry resolution and checkpoint loading.

Usage::

    python tests/functional_tests/models/hy_v4/run_automodel_vllm_parity.py \
        --checkpoint /path/to/Hy4-preview-1l-reference \
        --reference /path/to/vllm-b2f685-1l.safetensors \
        --output /path/to/automodel-vllm-b2f685-1l.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig

VLLM_REFERENCE_COMMIT = "b2f685834a6456197e7033966fdef52a23f1abcd"
DEFAULT_MAX_MEAN_KL = 1.0e-2
DEFAULT_MIN_LOGITS_COSINE = 0.999
DEFAULT_MIN_TOP1_AGREEMENT = 0.95


def _tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Return FP64 error metrics for two equal-shaped tensors.

    Args:
        actual: AutoModel output tensor of arbitrary shape.
        reference: vLLM output tensor with the same shape as ``actual``.

    Returns:
        Scalar absolute-error, relative-L2, and cosine metrics.
    """
    if actual.shape != reference.shape:
        raise AssertionError(f"Tensor shape mismatch: AutoModel={tuple(actual.shape)}, vLLM={tuple(reference.shape)}")
    actual64 = actual.detach().double().flatten().cpu()
    reference64 = reference.detach().double().flatten().cpu()
    difference = actual64 - reference64
    reference_norm = torch.linalg.vector_norm(reference64)
    actual_norm = torch.linalg.vector_norm(actual64)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-30)).item(),
        "cosine": (torch.dot(actual64, reference64) / (actual_norm * reference_norm).clamp_min(1e-30)).item(),
    }


def _logit_distribution_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, float | int]:
    """Compare full-vocabulary logits without materializing full softmax copies.

    Args:
        actual: AutoModel FP32 logits shaped ``[tokens, vocab]`` on CUDA.
        reference: Pinned-vLLM FP32 logits shaped ``[tokens, vocab]`` on CPU.
        chunk_size: Number of token rows processed per metric chunk.

    Returns:
        PR-style forward-KL, JSD, cosine, top-1, and absolute-error metrics.
    """
    if actual.shape != reference.shape:
        raise AssertionError(f"Logit shape mismatch: AutoModel={tuple(actual.shape)}, vLLM={tuple(reference.shape)}")
    if actual.ndim != 2:
        raise AssertionError(f"Expected [tokens, vocab] logits, got {tuple(actual.shape)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    kl_chunks: list[torch.Tensor] = []
    jsd_chunks: list[torch.Tensor] = []
    dot = actual_square = reference_square = difference_square = 0.0
    absolute_sum = 0.0
    maximum_absolute = 0.0
    top1_matches = 0
    element_count = actual.numel()

    for start in range(0, actual.shape[0], chunk_size):
        end = min(start + chunk_size, actual.shape[0])
        student_logits = actual[start:end].detach().float()
        teacher_logits = reference[start:end].to(device=actual.device, dtype=torch.float32)
        difference = student_logits - teacher_logits

        dot += torch.sum(student_logits * teacher_logits, dtype=torch.float64).item()
        actual_square += torch.sum(student_logits.square(), dtype=torch.float64).item()
        reference_square += torch.sum(teacher_logits.square(), dtype=torch.float64).item()
        difference_square += torch.sum(difference.square(), dtype=torch.float64).item()
        absolute_sum += torch.sum(difference.abs(), dtype=torch.float64).item()
        maximum_absolute = max(maximum_absolute, difference.abs().max().item())
        top1_matches += int((student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).sum().item())

        student_log_prob = F.log_softmax(student_logits, dim=-1)
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_prob = student_log_prob.exp()
        teacher_prob = teacher_log_prob.exp()
        kl_chunks.append(torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1).double().cpu())
        log_mean_prob = torch.logaddexp(teacher_log_prob, student_log_prob) - math.log(2.0)
        jsd_chunks.append(
            (
                0.5
                * (
                    torch.sum(teacher_prob * (teacher_log_prob - log_mean_prob), dim=-1)
                    + torch.sum(student_prob * (student_log_prob - log_mean_prob), dim=-1)
                )
            )
            .double()
            .cpu()
        )

    per_token_kl = torch.cat(kl_chunks)
    per_token_jsd = torch.cat(jsd_chunks)
    token_count = actual.shape[0]
    return {
        "mean_kl_vllm_to_automodel": per_token_kl.mean().item(),
        "p95_kl_vllm_to_automodel": torch.quantile(per_token_kl, 0.95).item(),
        "max_kl_vllm_to_automodel": per_token_kl.max().item(),
        "mean_jsd": per_token_jsd.mean().item(),
        "logits_cosine_similarity": dot / max(math.sqrt(actual_square * reference_square), 1.0e-30),
        "top1_token_agreement": top1_matches / token_count,
        "top1_token_agreement_percent": 100.0 * top1_matches / token_count,
        "mean_absolute_logit_difference": absolute_sum / element_count,
        "max_absolute_logit_difference": maximum_absolute,
        "relative_l2": math.sqrt(difference_square / max(reference_square, 1.0e-30)),
        "token_count": token_count,
        "vocab_size": actual.shape[1],
    }


def _validate_reference(reference_path: Path) -> dict[str, Any]:
    """Validate the pinned vLLM provenance companion file."""
    metadata_path = reference_path.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("vllm_reference_commit") != VLLM_REFERENCE_COMMIT:
        raise RuntimeError(
            f"Reference commit mismatch: expected {VLLM_REFERENCE_COMMIT}, got {metadata.get('vllm_reference_commit')}."
        )
    checkpoint = metadata.get("checkpoint_provenance", {})
    if checkpoint.get("vllm_reference_commit") != VLLM_REFERENCE_COMMIT:
        raise RuntimeError("Reference checkpoint provenance is not tied to the pinned vLLM commit.")
    return metadata


def _assert_parameter_contract(model: torch.nn.Module) -> dict[str, Any]:
    """Audit that checkpoint loading left no meta/nonfinite model parameters."""
    meta_parameters: list[str] = []
    nonfinite_parameters: list[str] = []
    dtype_counts: dict[str, int] = {}
    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += parameter.numel()
        dtype_name = str(parameter.dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + parameter.numel()
        if parameter.device.type == "meta":
            meta_parameters.append(name)
        else:
            flat_parameter = parameter.detach().view(-1)
            for chunk in flat_parameter.split(16 * 1024 * 1024):
                if not torch.isfinite(chunk).all().item():
                    nonfinite_parameters.append(name)
                    break
    if meta_parameters:
        raise AssertionError(f"Checkpoint load left meta parameters: {meta_parameters[:10]}")
    if nonfinite_parameters:
        raise AssertionError(f"Checkpoint load produced nonfinite parameters: {nonfinite_parameters[:10]}")
    return {
        "parameter_count": parameter_count,
        "parameter_dtype_counts": dtype_counts,
        "meta_parameters": meta_parameters,
        "nonfinite_parameters": nonfinite_parameters,
    }


def run_parity(
    checkpoint: Path,
    reference_path: Path,
    output_path: Path | None,
    *,
    metric_chunk_size: int = 16,
    max_mean_kl: float = DEFAULT_MAX_MEAN_KL,
    min_logits_cosine: float = DEFAULT_MIN_LOGITS_COSINE,
    min_top1_agreement: float = DEFAULT_MIN_TOP1_AGREEMENT,
) -> dict[str, Any]:
    """Load AutoModel through its production path and compare pinned outputs.

    Args:
        checkpoint: Local few-layer checkpoint with exact public HY V4 tensors.
        reference_path: Safetensors artifact produced by the pinned vLLM runner.
        output_path: Optional JSON destination for comparison metrics.
        metric_chunk_size: Token rows per full-vocabulary metric chunk.
        max_mean_kl: Largest accepted mean ``KL(vLLM || AutoModel)``.
        min_logits_cosine: Smallest accepted global full-logit cosine similarity.
        min_top1_agreement: Smallest accepted fraction of matching top-1 tokens.

    Returns:
        JSON-serializable parity report.
    """
    reference_metadata = _validate_reference(reference_path)
    reference = load_file(reference_path)
    device = torch.device("cuda", torch.cuda.current_device())
    backend = BackendConfig(
        attn="cudnn",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        backend=backend,
        use_liger_kernel=False,
        use_sdpa_patching=False,
    )
    model.eval()
    load_audit = _assert_parameter_contract(model)

    input_ids = reference["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
    token_count = input_ids.shape[1]
    reference_logits = reference["logits"]
    all_token_logits = reference_logits.ndim == 2 and reference_logits.shape[0] == token_count
    position_ids = torch.arange(token_count, dtype=torch.long, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([[0, token_count]], dtype=torch.int32, device=device)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            qkv_format="thd",
            cu_seqlens=cu_seqlens,
            logits_to_keep=0 if all_token_logits else 1,
            output_hidden_states=True,
        )

    full_hidden = output.hidden_states
    if full_hidden is None:
        raise AssertionError("AutoModel did not return hidden states for parity.")
    full_hidden = full_hidden.squeeze(0)
    sample_hidden = full_hidden[-1:].contiguous()
    logits = output.logits.squeeze(0)

    metrics: dict[str, Any] = {
        "full_hidden_states": _tensor_metrics(full_hidden, reference["full_hidden_states"]),
        "sample_hidden_states": _tensor_metrics(sample_hidden, reference["sample_hidden_states"]),
    }
    failures: list[str] = []
    if all_token_logits:
        distribution_metrics = _logit_distribution_metrics(
            logits,
            reference_logits,
            chunk_size=metric_chunk_size,
        )
        metrics["logits"] = distribution_metrics
        if distribution_metrics["mean_kl_vllm_to_automodel"] >= max_mean_kl:
            failures.append(f"mean KL {distribution_metrics['mean_kl_vllm_to_automodel']:.8g} >= {max_mean_kl:.8g}")
        if distribution_metrics["logits_cosine_similarity"] <= min_logits_cosine:
            failures.append(
                f"logits cosine {distribution_metrics['logits_cosine_similarity']:.8g} <= {min_logits_cosine:.8g}"
            )
        if distribution_metrics["top1_token_agreement"] <= min_top1_agreement:
            failures.append(
                f"top-1 agreement {distribution_metrics['top1_token_agreement']:.8g} <= {min_top1_agreement:.8g}"
            )
        top1_report: dict[str, Any] = {}
    else:
        metrics["logits"] = _tensor_metrics(logits, reference_logits)
        automodel_top1 = int(logits.argmax(dim=-1).item())
        vllm_top1 = int(reference_logits.argmax(dim=-1).item())
        top1_report = {
            "automodel_top1": automodel_top1,
            "top1_match": automodel_top1 == vllm_top1,
            "vllm_top1": vllm_top1,
        }
        if not top1_report["top1_match"]:
            failures.append(f"top-1 mismatch: AutoModel={automodel_top1}, vLLM={vllm_top1}")
        failed_tensor_metrics = {
            name: values
            for name, values in metrics.items()
            if values["relative_l2"] >= 1e-2 or values["cosine"] <= 0.9999
        }
        if failed_tensor_metrics:
            failures.append(f"tensor metrics outside tolerance: {failed_tensor_metrics}")
    report = {
        "all_token_logits": all_token_logits,
        "checkpoint": str(checkpoint.resolve()),
        "cuda_version": torch.version.cuda,
        "load_audit": load_audit,
        "metrics": metrics,
        "parity_failures": failures,
        "parity_passed": not failures,
        "parity_thresholds": {
            "max_mean_kl_vllm_to_automodel": max_mean_kl,
            "min_logits_cosine_similarity": min_logits_cosine,
            "min_top1_token_agreement": min_top1_agreement,
        },
        "reference": str(reference_path.resolve()),
        "reference_metadata": reference_metadata,
        "torch_version": torch.__version__,
        "vllm_reference_commit": VLLM_REFERENCE_COMMIT,
        **top1_report,
    }
    if output_path is not None:
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite parity report: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise AssertionError(f"AutoModel/vLLM parity failed: {failures}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metric-chunk-size", type=int, default=16)
    parser.add_argument("--max-mean-kl", type=float, default=DEFAULT_MAX_MEAN_KL)
    parser.add_argument("--min-logits-cosine", type=float, default=DEFAULT_MIN_LOGITS_COSINE)
    parser.add_argument("--min-top1-agreement", type=float, default=DEFAULT_MIN_TOP1_AGREEMENT)
    args = parser.parse_args()
    run_parity(
        args.checkpoint,
        args.reference,
        args.output,
        metric_chunk_size=args.metric_chunk_size,
        max_mean_kl=args.max_mean_kl,
        min_logits_cosine=args.min_logits_cosine,
        min_top1_agreement=args.min_top1_agreement,
    )


if __name__ == "__main__":
    main()
