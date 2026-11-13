# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Dequantization of the Inkling 2 Small routed-expert checkpoint formats.

The released checkpoint (``hf_quant_config.json``, ``quant_method: inkling2``) stores only the routed experts
quantized, ``model.llm.layers.N.mlp.experts.{w13,w2}_weight`` in ``[experts, out, in]`` (``nn.Linear``)
orientation: ``w13`` is ``[experts, 2 * intermediate, hidden]`` with gate and up rows interleaved (row ``2j`` is
gate row ``j``, row ``2j + 1`` is up row ``j``) and ``w2`` is ``[experts, hidden, intermediate]``. The functions
here decode element-wise and keep that row order; the state-dict adapter reorders rows afterwards.

NVFP4 (layers 0-39, ``weight_scale_format: amax_denominator``): ``weight`` is ``uint8 [E, out, in / 2]`` with two
E2M1 codes per byte (column ``2k`` in the low nibble, ``2k + 1`` in the high nibble; bit 3 of a code is the sign),
``weight_scale`` is ``float8_e4m3fn [E, out, in / 16]`` (one scale per 16 consecutive input columns),
``weight_amax`` is ``float32 [E]`` and ``weight_outer_scale_denominator`` is ``float32 [1]`` (2688 = 6 * 448 in
the release). A weight decodes as::

    W[e, o, i] = (e2m1(code[e, o, i]) * e4m3(scale[e, o, i // 16])) * (amax[e] / denominator)

The serving GEMM applies the same outer scale to its output as ``(x_amax * amax[e]) / (1536 * denominator)``,
where ``x_amax / 1536`` is the per-token activation outer scale (lightseekorg/tokenspeed@c485a46a,
``tokenspeed_kernel/ops/moe/block_scaled.py``, ``torch_block_scaled_moe_routes``); splitting off the activation
factor leaves the weight term above.

MXFP8 (layers 40-47): ``weight`` is ``float8_e4m3fn [E, out, in]`` and ``weight_scale`` is ``uint8 [E, out, in / 32]``
holding UE8M0 exponents, ``W[e, o, i] = e4m3(weight[e, o, i]) * 2 ** (scale[e, o, i // 32] - 127)``, the
decoding of huggingface/transformers@109a7d19bf (``Fp8Dequantize`` with ``scale_fmt: ue8m0``). The manifest declares
``weight_block_shape [32, 32]`` but stores one exponent per row and 32 columns, and in the release those exponents
differ between the rows of a 32-row block, so they are decoded per row.

Numerics: the E2M1 code times its E4M3 scale is exact in FP32 and the outer scale ``amax / denominator`` is one
correctly rounded FP32 division, so each NVFP4 weight carries a single FP32 rounding of that product. MXFP8
decoding is exact in FP32. These weights reproduce the checkpoint's weight values only: the release's GEMMs also
quantize activations (per-token NVFP4 with outer denominator 1536 and a 4-or-6 block-scale choice by mean
absolute error, MXFP8 per 32 values) before multiplying, which BF16 training on dequantized weights does not
emulate.
"""

import re
from typing import Literal, TypedDict

import torch

from nemo_automodel.components.quantization.mxfp4 import _FP4_BYTE_TABLE
from nemo_automodel.components.quantization.nvfp4 import NVFP4_BLOCK_SIZE

MXFP8_BLOCK_SIZE = 32
# UE8M0 byte 0xFF is NaN in the OCP MX format; no finite weight can use it.
_UE8M0_NAN = 255
# Elements decoded per step. Bounds the FP32 transients (gather index, values) to about 0.5 GiB however many
# experts a call receives; one 4608 x 4608 expert projection is 21M elements.
_DEQUANT_CHUNK_ELEMS = 2**26

QUANT_METHOD = "inkling2"
QUANTIZATION_MANIFEST = "hf_quant_config.json"
ExpertFormat = Literal["nvfp4", "mxfp8"]
_EXPERT_WEIGHT_RE = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.w(?:13|2)_weight")
# Weight-side fields of the manifest's quantized parameter groups that decide how the stored bytes decode.
_NVFP4_WEIGHT_FIELDS = {
    "output_dtype": "uint8",
    "weight_transform": "nvfp4_nearest",
    "weight_block_shape": [1, 16],
    "weight_scale_format": "amax_denominator",
}
_MXFP8_WEIGHT_FIELDS = {"output_dtype": "float8_e4m3fn", "weight_transform": "mxfp8_nearest"}
_MXFP8_BLOCK_SHAPES = ([1, 32], [32, 32])
_UNQUANTIZED_COMPUTE_DTYPES = ("float32", "bfloat16")


class ManifestParameterGroup(TypedDict, total=False):
    """One ``parameter_groups`` entry of ``hf_quant_config.json``: tensor names and their precision."""

    tensors: list[str]
    compute_dtype: str
    output_dtype: str
    weight_transform: str
    weight_block_shape: list[int]
    weight_scale_format: str


class ManifestSection(TypedDict):
    """The ``model`` (or ``mtp``) section of ``hf_quant_config.json``."""

    parameter_groups: list[ManifestParameterGroup]


class QuantizationManifest(TypedDict, total=False):
    """The released ``hf_quant_config.json`` (fields read here; activation settings are not used)."""

    quant_method: str
    model: ManifestSection


class Inkling2QuantizationConfig(TypedDict):
    """``quantization_config`` of a checkpoint with quantized experts.

    Attributes:
        quant_method: Always ``"inkling2"``.
        expert_formats: Released expert weight key (``model.llm.layers.N.mlp.experts.{w13,w2}_weight``) to its
            stored format. Experts not listed are unquantized.
    """

    quant_method: str
    expert_formats: dict[str, ExpertFormat]


def quantization_config_from_manifest(manifest: QuantizationManifest) -> Inkling2QuantizationConfig:
    """Read the stored format of every quantized expert tensor from ``hf_quant_config.json``.

    Only the weight-side fields decide how the stored bytes decode, so only they are validated; the manifest's
    activation-quantization settings describe the serving GEMMs, which training does not emulate.

    Args:
        manifest: Parsed ``hf_quant_config.json``.

    Returns:
        The ``quantization_config`` to attach to the model config.

    Raises:
        ValueError: If the manifest is not ``inkling2``, has an unknown compute dtype or unsupported weight fields,
            quantizes a tensor that is not a routed-expert weight, or lists a tensor twice.
    """
    if manifest.get("quant_method") != QUANT_METHOD or "model" not in manifest:
        raise ValueError(f"Expected an inkling2 {QUANTIZATION_MANIFEST} with a model section")
    expert_formats: dict[str, ExpertFormat] = {}
    for group in manifest["model"]["parameter_groups"]:
        compute = group.get("compute_dtype")
        if compute in _UNQUANTIZED_COMPUTE_DTYPES:
            continue
        if compute == "nvfp4":
            fmt: ExpertFormat = "nvfp4"
            supported = all(group.get(field) == value for field, value in _NVFP4_WEIGHT_FIELDS.items())
        elif compute == "mxfp8":
            fmt = "mxfp8"
            supported = all(group.get(field) == value for field, value in _MXFP8_WEIGHT_FIELDS.items())
            supported = supported and group.get("weight_block_shape") in _MXFP8_BLOCK_SHAPES
        else:
            raise ValueError(f"Unsupported Inkling 2 compute dtype {compute!r}")
        if not supported:
            raise ValueError(f"Unsupported {compute} weight precision fields in {QUANTIZATION_MANIFEST}")
        for name in group.get("tensors", []):
            if not _EXPERT_WEIGHT_RE.fullmatch(name):
                raise ValueError(f"{compute} is only supported for routed-expert weights, got {name}")
            if name in expert_formats:
                raise ValueError(f"{name} is listed twice in {QUANTIZATION_MANIFEST}")
            expert_formats[name] = fmt
    return {"quant_method": QUANT_METHOD, "expert_formats": expert_formats}


def checkpoint_expert_formats(quantization_config: Inkling2QuantizationConfig | None) -> dict[str, ExpertFormat]:
    """Return the stored expert formats a model config declares (empty for an unquantized checkpoint).

    Args:
        quantization_config: The config's ``quantization_config``, or None.

    Returns:
        Released expert weight key to ``"nvfp4"`` or ``"mxfp8"``.

    Raises:
        ValueError: For a quantization config of another method or an unknown format.
    """
    if quantization_config is None:
        return {}
    if quantization_config.get("quant_method") != QUANT_METHOD:
        raise ValueError(f"Inkling 2 checkpoints use quant_method {QUANT_METHOD!r}, got {quantization_config!r}")
    formats = dict(quantization_config.get("expert_formats", {}))
    if unknown := sorted({fmt for fmt in formats.values() if fmt not in ("nvfp4", "mxfp8")}):
        raise ValueError(f"Unknown Inkling 2 expert formats {unknown}")
    return formats


def _target_device(weight: torch.Tensor, device: torch.device | str | None) -> torch.device:
    """Return the device to decode on: ``device`` when given, else the device holding ``weight``.

    Args:
        weight: Checkpoint tensor of arbitrary shape.
        device: Requested decode device, or None.

    Returns:
        The decode and output device.
    """
    return weight.device if device is None else torch.device(device)


def _row_step(columns: int, dtype: torch.dtype) -> int:
    """Return how many ``columns``-wide rows to decode per step within ``_DEQUANT_CHUNK_ELEMS``.

    Raises:
        ValueError: When the requested output ``dtype`` is not floating point.
    """
    if not dtype.is_floating_point:
        raise ValueError(f"Expert dequantization needs a floating-point output dtype, got {dtype}")
    return max(1, _DEQUANT_CHUNK_ELEMS // columns)


def dequantize_nvfp4_experts(
    weight: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    outer_scale_denominator: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode Inkling 2 NVFP4 expert weights (``amax_denominator`` outer scale) to floating point.

    ``W[e, o, i] = (e2m1(code) * e4m3(scale[e, o, i // 16])) * (amax[e] / outer_scale_denominator)``: the code times
    its block scale is exact in FP32, the outer scale is one FP32 division, and their product is rounded once.
    Rows are decoded in chunks of ``_DEQUANT_CHUNK_ELEMS`` elements, so the transients stay bounded and only the
    output is full size. Inputs are not modified.

    Args:
        weight: ``uint8`` tensor of shape [experts, out, in / 2]: two E2M1 codes per byte, input column ``2k`` in
            the low nibble and ``2k + 1`` in the high nibble, sign in bit 3 of each code.
        scale: ``float8_e4m3fn`` tensor of shape [experts, out, in / 16]: one block scale per 16 consecutive input
            columns of a row.
        amax: ``float32`` tensor of shape [experts]: each expert's finite, non-negative weight maximum.
        outer_scale_denominator: ``float32`` tensor of shape [1], finite and positive, shared by all experts.
        device: Device to decode on and return the output on; defaults to ``weight``'s device. Inputs on another
            device are copied over one chunk at a time.
        dtype: Floating-point output dtype. The value is computed in FP32 and rounded to ``dtype`` once, so a BF16
            output equals the FP32 output cast to BF16 at half the output memory.

    Returns:
        Tensor of shape [experts, out, in] in ``dtype`` on the decode device, with independent storage and the
        input's row order (``w13`` rows stay gate/up interleaved).

    Raises:
        ValueError: On a dtype or shape that is not this layout, a negative or non-finite ``amax``, a non-positive or
            non-finite denominator, or a NaN block scale.
    """
    if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"NVFP4 experts need uint8 weight and float8_e4m3fn scale, got {weight.dtype}, {scale.dtype}")
    if amax.dtype != torch.float32 or outer_scale_denominator.dtype != torch.float32:
        raise ValueError(
            f"NVFP4 amax and outer_scale_denominator must be float32, got {amax.dtype}, {outer_scale_denominator.dtype}"
        )
    if weight.ndim != 3 or weight.shape[-1] % (NVFP4_BLOCK_SIZE // 2):
        raise ValueError(f"NVFP4 weight must be [experts, out, in / 2] with in divisible by 16, got {weight.shape}")
    experts, rows, packed_columns = weight.shape
    columns = packed_columns * 2
    blocks = columns // NVFP4_BLOCK_SIZE
    if scale.shape != (experts, rows, blocks):
        raise ValueError(f"NVFP4 scale shape {tuple(scale.shape)} must be {(experts, rows, blocks)}")
    if amax.shape != (experts,) or outer_scale_denominator.shape != (1,):
        raise ValueError(
            f"NVFP4 amax must be [{experts}] and denominator [1], got {tuple(amax.shape)}, "
            f"{tuple(outer_scale_denominator.shape)}"
        )
    target = _target_device(weight, device)
    amax_f32 = amax.to(target)
    denominator = outer_scale_denominator.to(target)
    if not bool(torch.isfinite(amax_f32).all()) or bool((amax_f32 < 0).any()):
        raise ValueError("NVFP4 amax must be finite and non-negative")
    if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
        raise ValueError("NVFP4 outer_scale_denominator must be finite and positive")
    outer = amax_f32 / denominator  # [experts], one correctly rounded FP32 division each

    # Row b of the byte table is (value(low nibble), value(high nibble)), so one gather per byte yields the codes
    # of columns 2k and 2k + 1 in column order.
    byte_table = _FP4_BYTE_TABLE.to(target)
    flat_weight = weight.reshape(experts * rows, packed_columns)
    flat_scale = scale.reshape(experts * rows, blocks)
    step = _row_step(columns, dtype)
    output = torch.empty((experts, rows, columns), dtype=dtype, device=target)
    flat_output = output.view(experts * rows, columns)
    for start in range(0, experts * rows, step):
        stop = min(start + step, experts * rows)
        codes = flat_weight[start:stop].to(target)
        values = torch.index_select(byte_table, 0, codes.reshape(-1).to(torch.int32))
        values = values.view(stop - start, blocks, NVFP4_BLOCK_SIZE)
        block_scale = flat_scale[start:stop].to(target).to(torch.float32)
        if bool(torch.isnan(block_scale).any()):
            raise ValueError("NVFP4 scale holds NaN (e4m3 0x7F/0xFF)")
        values.mul_(block_scale.unsqueeze(-1))  # exact: E2M1 has 2 significant bits, E4M3 has 4
        row_outer = outer[torch.arange(start, stop, device=target) // rows]
        values.mul_(row_outer.view(-1, 1, 1))  # the only FP32 rounding
        flat_output[start:stop] = values.view(stop - start, columns)
    return output


def dequantize_mxfp8_experts(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode Inkling 2 MXFP8 expert weights (E4M3 values, UE8M0 exponents per 32 input columns).

    ``W[e, o, i] = e4m3(weight[e, o, i]) * 2 ** (scale[e, o, i // 32] - 127)``, exact in FP32. Rows are decoded in
    chunks of ``_DEQUANT_CHUNK_ELEMS`` elements, so the transients stay bounded and only the output is full size.
    Inputs are not modified.

    Args:
        weight: ``float8_e4m3fn`` tensor of shape [experts, out, in] with ``in`` divisible by 32.
        scale: ``uint8`` tensor of shape [experts, out, in / 32] of UE8M0 exponents (bias 127), one per row and 32
            consecutive input columns, decoded per row (the release's exponents differ between the rows of a 32-row
            block).
        device: Device to decode on and return the output on; defaults to ``weight``'s device. Inputs on another
            device are copied over one chunk at a time.
        dtype: Floating-point output dtype. FP32 holds every finite value exactly; a BF16 output is the FP32 value
            rounded once, which is exact while the value stays in BF16's normal range.

    Returns:
        Tensor of shape [experts, out, in] in ``dtype`` on the decode device, with independent storage and the
        input's row order.

    Raises:
        ValueError: On a dtype or shape that is not this layout, or a scale byte of 255 (NaN in UE8M0).
    """
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
        raise ValueError(f"MXFP8 experts need float8_e4m3fn weight and uint8 scale, got {weight.dtype}, {scale.dtype}")
    if weight.ndim != 3 or weight.shape[-1] % MXFP8_BLOCK_SIZE:
        raise ValueError(f"MXFP8 weight must be [experts, out, in] with in divisible by 32, got {weight.shape}")
    experts, rows, columns = weight.shape
    blocks = columns // MXFP8_BLOCK_SIZE
    if scale.shape != (experts, rows, blocks):
        raise ValueError(f"MXFP8 scale shape {tuple(scale.shape)} must be {(experts, rows, blocks)} (one per row)")
    target = _target_device(weight, device)
    flat_weight = weight.reshape(experts * rows, columns)
    flat_scale = scale.reshape(experts * rows, blocks)
    step = _row_step(columns, dtype)
    output = torch.empty((experts, rows, columns), dtype=dtype, device=target)
    flat_output = output.view(experts * rows, columns)
    for start in range(0, experts * rows, step):
        stop = min(start + step, experts * rows)
        exponents = flat_scale[start:stop].to(target)
        if bool((exponents == _UE8M0_NAN).any()):
            raise ValueError("MXFP8 scale holds 255, which is NaN in UE8M0")
        block_scale = exponents.view(torch.float8_e8m0fnu).to(torch.float32)  # exactly 2 ** (byte - 127)
        values = flat_weight[start:stop].to(target).to(torch.float32).view(stop - start, blocks, MXFP8_BLOCK_SIZE)
        values.mul_(block_scale.unsqueeze(-1))  # exact: a power of two times a 4-significant-bit value
        flat_output[start:stop] = values.view(stop - start, columns)
    return output
