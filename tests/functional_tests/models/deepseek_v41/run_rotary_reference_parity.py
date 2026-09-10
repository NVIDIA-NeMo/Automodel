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

"""Isolate V4.1 RoPE phase, rotation and post-quantization differences.

The unchanged pinned official phase and rotation functions are the oracle.
Candidate complex multiplication is diagnostic code only. CPU runs compare
rotations; CUDA runs also compare unchanged official cache-QAT kernels against
native QAT. Real released head/channel dimensions are retained throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import torch
from run_reference_parity import REFERENCE_HASHES, REVISION, _import_reference, _tensor_metrics

from nemo_automodel.components.models.deepseek_v41.attention import _apply_rope, _RotaryEmbedding
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache


def _metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    """Compare identically shaped real-valued tensors, including sparse BF16 changes."""
    metrics = asdict(_tensor_metrics(expected, actual))
    changed = expected != actual
    return {
        **metrics,
        "exact": not changed.any().item(),
        "changed_elements": changed.sum().item(),
        "changed_fraction": changed.float().mean().item(),
    }


def _complex_rotation(values: torch.Tensor, frequencies: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    """Candidate differentiable rotation preserving the source's complex multiply.

    Args:
        values: BF16 or FP32 [batch, sequence, (heads), channels] tensor.
        frequencies: Complex64 [batch, sequence, rotary_pairs] tensor.
        inverse: Conjugate frequencies for attention-output rotation.

    Returns:
        Tensor with original layout/dtype and independent storage.
    """
    rotary_dim = 2 * frequencies.shape[-1]
    pairs = torch.view_as_complex(values[..., -rotary_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    if values.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).to(values.dtype)
    return torch.cat((values[..., :-rotary_dim], rotated), dim=-1)


def _source_order_phases(rotary: _RotaryEmbedding, positions: torch.Tensor) -> torch.Tensor:
    """Candidate phase ordering, using the original smooth-then-complement arithmetic."""
    import math

    frequencies = 1.0 / (
        rotary.theta ** (torch.arange(0, rotary.dim, 2, device=positions.device, dtype=torch.float32) / rotary.dim)
    )
    if rotary.original_length > 0:
        low = max(
            math.floor(
                rotary.dim
                * math.log(rotary.original_length / (rotary.beta_fast * 2 * math.pi))
                / (2 * math.log(rotary.theta))
            ),
            0,
        )
        high = min(
            math.ceil(
                rotary.dim
                * math.log(rotary.original_length / (rotary.beta_slow * 2 * math.pi))
                / (2 * math.log(rotary.theta))
            ),
            rotary.dim - 1,
        )
        ramp = (
            (torch.arange(rotary.dim // 2, device=positions.device, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / rotary.factor * (1 - smooth) + frequencies * smooth
    return positions.float().unsqueeze(-1) * frequencies


def _official_quantize(reference: ModuleType, values: torch.Tensor, format: str, block_size: int) -> torch.Tensor:
    """Call an unchanged official in-place quantizer on a private input copy."""
    output = values.clone()
    if format == "fp8":
        reference.act_quant(output, block_size, reference.scale_fmt, reference.scale_dtype, True)
    elif format == "nvfp4":
        reference.fp4_act_quant(output, block_size, True, scale_dtype=torch.float8_e4m3fn)
    else:
        reference.fp4_act_quant(output, block_size, True)
    return output


def main() -> int:
    """Write CPU or CUDA phase/rotation/QAT component evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    if args.sequence_length <= 0:
        raise ValueError("Sequence length must be positive")
    device = torch.device("cuda", 0) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one available GPU for this component diagnostic")
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    config = DeepseekV41Config.from_pretrained(args.checkpoint, local_files_only=True).text_config
    reference = _import_reference(args.reference_dir)
    # Explicit device context is required by the original frequency factories.
    report = {
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "native_attention_sha256": hashlib.sha256(
            Path(__import__(_apply_rope.__module__, fromlist=["__file__"]).__file__).read_bytes()
        ).hexdigest(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "scope": "Unchanged official RoPE/QAT component reference; complex/source-order variants are diagnostic candidates only",
        "cases": {},
    }
    layouts = (
        ("query_and_inverse", config.num_attention_heads, config.head_dim, None, 0),
        ("window_kv", None, config.head_dim, "fp8", 32),
        ("compressed_kv", None, config.head_dim, "nvfp4", 16),
        ("index_query", config.index_n_heads, config.index_head_dim, "mxfp4", 32),
        ("index_key", None, config.index_head_dim, "mxfp4", 32),
    )
    for compressed in (False, True):
        rotary = _RotaryEmbedding(config, compressed=compressed)
        for start in (0, 65536):
            positions = torch.arange(start, start + args.sequence_length, device=device).unsqueeze(0)
            with torch.device(device):
                original_frequencies = reference.precompute_freqs_cis(
                    rotary.dim,
                    start + args.sequence_length,
                    rotary.original_length,
                    rotary.theta,
                    rotary.factor,
                    rotary.beta_fast,
                    rotary.beta_slow,
                )[start:]
            native_angles = rotary(positions)
            candidate_angles = _source_order_phases(rotary, positions)
            native_frequencies = torch.polar(torch.ones_like(native_angles), native_angles)
            candidate_frequencies = torch.polar(torch.ones_like(candidate_angles), candidate_angles)
            name = f"{'compressed_yarn' if compressed else 'swa'}_position_{start}"
            case = {
                "phases": {
                    "current_native": _metrics(
                        torch.view_as_real(original_frequencies), torch.view_as_real(native_frequencies[0])
                    ),
                    "candidate_source_order": _metrics(
                        torch.view_as_real(original_frequencies), torch.view_as_real(candidate_frequencies[0])
                    ),
                },
                "layouts": {},
            }
            for layout_name, heads, channels, format, block_size in layouts:
                shape = (
                    (1, args.sequence_length, channels) if heads is None else (1, args.sequence_length, heads, channels)
                )
                # BF16 inputs match the actual projection/attention output boundary.
                values = torch.randn(shape, device=device).bfloat16()
                for inverse in (False, True) if heads is not None else (False,):
                    suffix = "inverse" if inverse else "forward"
                    original = values.clone()
                    reference.apply_rotary_emb(original[..., -rotary.dim :], original_frequencies, inverse)
                    variants = {
                        "current_native": _apply_rope(values, native_angles, inverse=inverse),
                        "complex_native_phase": _complex_rotation(values, native_frequencies, inverse=inverse),
                        "complex_source_order_phase": _complex_rotation(values, candidate_frequencies, inverse=inverse),
                        "complex_reference_phase": _complex_rotation(
                            values, original_frequencies.unsqueeze(0), inverse=inverse
                        ),
                    }
                    measurements = {
                        label: {"after_bf16_rotation": _metrics(original, value)} for label, value in variants.items()
                    }
                    if format is not None and device.type == "cuda":
                        quantized_reference = _official_quantize(reference, original, format, block_size)
                        same_rotated_native = quantize_cache(original, format=format, block_size=block_size)
                        for label, value in variants.items():
                            measurements[label]["after_qat"] = _metrics(
                                quantized_reference, quantize_cache(value, format=format, block_size=block_size)
                            )
                        measurements["qat_on_exact_official_rotation"] = _metrics(
                            quantized_reference, same_rotated_native
                        )
                    case["layouts"][f"{layout_name}_{suffix}"] = measurements
                    print(name, layout_name, suffix, json.dumps(measurements), flush=True)
            report["cases"][name] = case
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
