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

"""Stateless merged-weight QAT in canonical [..., out, in] layout.

FP8 uses finite E4M3 (maximum 448); MXFP4 uses E2M1 (maximum 6).
Both default to ceil-power-of-two block scales encoded as E8M0 bytes;
FP8 also supports unrounded float32 block scales. This is a
numerical reference core, not a fused kernel, distributed wrapper, or exporter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.distributed.tensor import DTensor

_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class WeightQuantizationConfig:
    """Declarative configuration for canonical merged-weight quantization.

    Args:
        format: Exactly ``fp8`` (E4M3FN) or ``mxfp4`` (E2M1).
        block_size: Tuple (output rows, input columns); YAML lists normalize to
            immutable tuples. FP8 supports only
            (128, 128) and (32, 32); MXFP4 supports only (1, 32). None resolves
            at build time to (128, 128) for FP8 or (1, 32) for MXFP4, without
            changing this config. Every weight dimension must be positive and
            its final two dimensions must be divisible by the block size.
        scale_format: ``e8m0`` preserves the power-of-two byte encoding.
            ``float32`` uses max(block_amax, 1e-12) / 448 without power-of-two
            rounding and is supported only for FP8.
    """

    format: Literal["fp8", "mxfp4"] = "fp8"
    block_size: tuple[int, int] | None = None
    scale_format: Literal["e8m0", "float32"] = "e8m0"

    def __post_init__(self) -> None:
        """Reject unsupported formats and noncanonical block specifications."""
        if self.format not in ("fp8", "mxfp4"):
            raise ValueError("format must be exactly 'fp8' or 'mxfp4'")
        if self.scale_format not in ("e8m0", "float32"):
            raise ValueError("scale_format must be exactly 'e8m0' or 'float32'")
        if self.format == "mxfp4" and self.scale_format != "e8m0":
            raise ValueError("mxfp4 requires scale_format='e8m0'")
        if self.block_size is None:
            return
        if (
            not isinstance(self.block_size, (tuple, list))
            or len(self.block_size) != 2
            or any(type(size) is not int for size in self.block_size)
        ):
            raise TypeError("block_size must be a tuple/list of two integers or None")
        object.__setattr__(self, "block_size", tuple(self.block_size))
        allowed = ((128, 128), (32, 32)) if self.format == "fp8" else ((1, 32),)
        if self.block_size not in allowed:
            raise ValueError(f"unsupported block_size {self.block_size} for {self.format}; expected one of {allowed}")

    def build(self) -> WeightFakeQuantizer:
        """Construct an independent stateless quantizer with resolved defaults.

        Returns:
            WeightFakeQuantizer holding a resolved, frozen copy of this config.
        """
        return WeightFakeQuantizer(self)


def _validate_tensor(tensor: torch.Tensor, *, name: str, dtypes: tuple[torch.dtype, ...]) -> None:
    """Validate storage without operating on distributed or unsupported tensors.

    Args:
        tensor: Tensor of arbitrary shape; semantic axes are checked by the caller.
            Must be local, strided, CPU or CUDA storage, with dtype in dtypes.
            Noncontiguous strides are supported; the tensor is not mutated.
        name: Input name for errors.
        dtypes: Permitted storage dtypes.
    """
    if isinstance(tensor, DTensor):
        raise TypeError(f"{name} must be a local tensor, not a DTensor")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise TypeError(f"{name} must have strided layout")
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on CPU or CUDA, got {tensor.device}")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {tensor.dtype}; expected one of {dtypes}")


def _validate_shape(shape: tuple[int, ...], block_size: tuple[int, int]) -> None:
    """Check a positive canonical [..., out, in] shape and exact block divisibility."""
    if not isinstance(shape, tuple) or len(shape) < 2 or any(type(size) is not int for size in shape):
        raise TypeError("shape must be a tuple of at least two integers in [..., out, in] order")
    if any(size <= 0 for size in shape):
        raise ValueError("shape dimensions must all be positive")
    if shape[-2] % block_size[0] or shape[-1] % block_size[1]:
        raise ValueError(f"shape {shape} is not divisible by block_size {block_size}; padding is not supported")


@dataclass(frozen=True)
class QuantizedWeight:
    """Encoded local weight with explicit logical shape and immutable metadata.

    Args:
        payload: Strided uint8 CPU/CUDA tensor. FP8 shape is [..., out, in],
            storing actual E4M3FN bytes (0x7f and 0xff are invalid). MXFP4 shape
            is [..., out, in / 2]; even input columns occupy the low nibble and
            odd columns the high nibble. Each nibble has a sign bit and E2M1
            magnitude codes 0..7 for [0, .5, 1, 1.5, 2, 3, 4, 6]. All leading
            axes are preserved independently. Noncontiguous storage is accepted.
        scales: Strided tensor of shape [..., out / block_rows, in / block_cols]
            on payload's device. For scale_format=e8m0, uint8 values 0..254
            mean 2**(byte - 127); 255 is invalid. For scale_format=float32,
            values are finite, strictly positive float32 multipliers.
        config: Frozen configuration with an explicitly resolved block_size.
        shape: Logical shape [..., out, in], including arbitrary leading axes.

    Construction retains caller-owned tensors, without copying or mutating them.
    The metadata is frozen but tensor storage is mutable; dequantize revalidates
    it. Quantizer-created instances have storage independent of the source
    weight. This dataclass is not a module and quantizers register no scale
    buffers. Unlike E8M0 bytes, float32 scales would lose precision if a caller
    registered them as buffers and applied nn.Module.to(dtype).
    """

    payload: torch.Tensor
    scales: torch.Tensor
    config: WeightQuantizationConfig
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the encoded tensor contract documented on QuantizedWeight."""
        self._validate()

    def _validate(self) -> None:
        """Validate the payload/scales layouts and encoded values documented on QuantizedWeight."""
        if not isinstance(self.config, WeightQuantizationConfig) or self.config.block_size is None:
            raise TypeError("config must be a WeightQuantizationConfig with resolved block_size")
        _validate_shape(self.shape, self.config.block_size)
        _validate_tensor(self.payload, name="payload", dtypes=(torch.uint8,))
        scale_dtype = torch.uint8 if self.config.scale_format == "e8m0" else torch.float32
        _validate_tensor(self.scales, name="scales", dtypes=(scale_dtype,))
        if self.payload.device != self.scales.device:
            raise ValueError("payload and scales must be on the same device")
        rows, cols = self.config.block_size
        payload_shape = self.shape if self.config.format == "fp8" else (*self.shape[:-1], self.shape[-1] // 2)
        scale_shape = (*self.shape[:-2], self.shape[-2] // rows, self.shape[-1] // cols)
        if tuple(self.payload.shape) != payload_shape:
            raise ValueError(f"payload shape must be {payload_shape}, got {tuple(self.payload.shape)}")
        if tuple(self.scales.shape) != scale_shape:
            raise ValueError(f"scales shape must be {scale_shape}, got {tuple(self.scales.shape)}")
        if self.config.scale_format == "e8m0":
            if bool((self.scales == 255).any()):
                raise ValueError("scales contain invalid E8M0 byte 255")
        elif not bool((torch.isfinite(self.scales) & (self.scales > 0)).all()):
            raise ValueError("float32 scales must be finite and strictly positive")
        if self.config.format == "fp8" and bool(((self.payload & 0x7F) == 0x7F).any()):
            raise ValueError("payload contains nonfinite E4M3FN byte 0x7f or 0xff")

    def dequantize(self, *, dtype: torch.dtype) -> torch.Tensor:
        """Decode bytes and multiply scales in fp32, then cast to dtype.

        Args:
            dtype: Output dtype, exactly float32, bfloat16, or float16. The
                payload/scales layouts are those documented on QuantizedWeight.
                Scale multiplication uses FP32 (exact for E8M0 unless it
                underflows). Lower precision follows PyTorch casting, including
                subnormal rounding and underflow; a lossy cast need not
                requantize to the same bytes.

        Returns:
            Independent tensor of shape [..., out, in] on payload's CPU/CUDA
            device, in dtype. Signed zeros are preserved. Neither stored tensor
            is mutated or aliased by the output.

        Raises:
            TypeError: Unsupported dtype or encoded tensor storage.
            ValueError: Invalid shapes/codes, or decoded values overflow fp32 or
                the requested dtype. Overflow is never silently clamped.
        """
        if dtype not in _FLOAT_DTYPES:
            raise TypeError("dtype must be torch.float32, torch.bfloat16, or torch.float16")
        self._validate()
        if self.config.format == "fp8":
            values = self.payload.contiguous().view(torch.float8_e4m3fn).float()
        else:
            codes = torch.stack((self.payload & 0x0F, self.payload >> 4), dim=-1).reshape(self.shape)
            levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=self.payload.device)
            values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
        # Validation above guarantees a resolved block size.
        block_size = self.config.block_size
        assert block_size is not None
        rows, cols = block_size
        blocked = values.reshape(*self.shape[:-2], self.shape[-2] // rows, rows, self.shape[-1] // cols, cols)
        if self.config.scale_format == "e8m0":
            exponents = self.scales.to(torch.int32) - 127
            scale = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
        else:
            scale = self.scales
        decoded = (blocked.transpose(-3, -2) * scale[..., None, None]).transpose(-3, -2).reshape(self.shape)
        output = decoded.to(dtype)
        if not bool(torch.isfinite(output).all()):
            raise ValueError(f"decoded weight overflows requested dtype {dtype}")
        return output


class WeightFakeQuantizer(nn.Module):
    """Stateless canonical weight QDQ with an exact identity-gradient STE.

    Args:
        config: Validated settings. A resolved frozen copy is retained; the
            caller's config is unchanged. Prefer config.build() for construction.

    No parameters, buffers, calibration history, or persistent fake-quant state
    are created. Config metadata is outside state_dict; serialization belongs
    at the controller's shared dataclass/JSON boundary.
    """

    _config: WeightQuantizationConfig

    def __init__(self, config: WeightQuantizationConfig) -> None:
        super().__init__()
        if not isinstance(config, WeightQuantizationConfig):
            raise TypeError("config must be a WeightQuantizationConfig")
        default = (128, 128) if config.format == "fp8" else (1, 32)
        self._config = replace(config, block_size=config.block_size if config.block_size is not None else default)

    @property
    def config(self) -> WeightQuantizationConfig:
        """Resolved immutable settings, independent of the construction config."""
        return self._config

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor) -> QuantizedWeight:
        """Encode canonical blocks, preserving all leading axes independently.

        Args:
            weight: Finite local strided CPU/CUDA tensor of shape [..., out, in],
                with arbitrary leading axes and dtype float32, bfloat16, or
                float16. Noncontiguous input is supported. Dimensions must be
                positive, and out/in must be exactly divisible by config's
                block_size. DTensor, padding, and implicit layout conversion
                from any noncanonical semantic order are unsupported.

        Returns:
            QuantizedWeight with independent uint8 payload of shape [..., out,
            in] for FP8 or [..., out, in / 2] for MXFP4, and scales of shape
            [..., out / block_rows, in / block_cols], on weight's device.
            Scales are uint8 for e8m0 or float32 for float32 scale_format.
            Packing and byte meanings follow QuantizedWeight. No autograd graph
            is retained and weight is neither mutated nor aliased.

        For float32 scales, scale = max(block_amax, 1e-12) / 448 in FP32,
        with no power-of-two rounding, including for zero and subnormal blocks.
        For E8M0, the scale is the smallest power of two at least block_amax / format_max,
        saturated below at 2**-127; zero blocks use scale 1 (byte 127). Exponents
        above 127 are rejected. Division and round-to-nearest-ties-even encoding
        use fp32. Scale selection uses frexp on amax, not rounded log2 or an
        underflow-prone division by format_max. Dequantization can reject values
        whose reconstruction overflows the requested output dtype.
        """
        _validate_tensor(weight, name="weight", dtypes=_FLOAT_DTYPES)
        shape = tuple(weight.shape)
        block_size = self.config.block_size
        assert block_size is not None
        _validate_shape(shape, block_size)
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("weight must contain only finite values")
        rows, cols = block_size
        blocked = weight.float().reshape(*shape[:-2], shape[-2] // rows, rows, shape[-1] // cols, cols)
        blocked = blocked.transpose(-3, -2)
        amax = blocked.abs().amax(dim=(-2, -1))
        if self.config.scale_format == "float32":
            scales = amax.clamp_min(1e-12) / 448
            scale = scales
        else:
            mantissa, exponent = torch.frexp(amax)
            # 448 = .875 * 2**9 and 6 = .75 * 2**3. Comparing these exact
            # mantissas distinguishes even nextafter neighbors of scale boundaries.
            max_mantissa, max_exponent = (0.875, 9) if self.config.format == "fp8" else (0.75, 3)
            exponent = exponent - max_exponent + (mantissa > max_mantissa).to(exponent.dtype)
            exponent = torch.where(amax == 0, 0, exponent).clamp_min(-127)
            if bool((exponent > 127).any()):
                raise ValueError("weight requires an E8M0 scale exponent greater than 127")
            scales = (exponent + 127).to(torch.uint8)
            scale = torch.ldexp(torch.ones_like(amax), exponent)
        normalized = (blocked / scale[..., None, None]).transpose(-3, -2).reshape(shape)
        if self.config.format == "fp8":
            payload = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            thresholds = normalized.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
            magnitude = normalized.abs().contiguous()
            codes = torch.bucketize(magnitude, thresholds)
            # bucketize selects the lower code at a midpoint; move up only
            # when that lower code is odd. Temporary storage is O(weight),
            # never a [..., out, in, 8] distance table.
            ties = magnitude == thresholds[codes.clamp_max(6)]
            codes = (codes + (ties & ((codes & 1) != 0))).to(torch.uint8)
            codes = codes | (torch.signbit(normalized).to(torch.uint8) << 3)
            payload = codes[..., 0::2] | (codes[..., 1::2] << 4)
        return QuantizedWeight(payload=payload, scales=scales, config=self.config, shape=shape)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return exact QDQ values with an identity derivative, without cancellation.

        Args:
            weight: Tensor of shape [..., out, in] obeying quantize's finite,
                local, strided CPU/CUDA, dtype, and block-divisibility contract.

        Returns:
            Independent tensor of shape [..., out, in], on weight's device and
            in its original dtype, exactly equal to quantize(weight).dequantize
            with that dtype (including signed zero). Backward passes the upstream
            gradient unchanged. No gradients are computed for scales or codes.
        """
        return _WeightSTE.apply(weight, self)


class _WeightSTE(torch.autograd.Function):
    """Direct QDQ forward with no saved state and an identity backward."""

    @staticmethod
    def forward(ctx: FunctionCtx, weight: torch.Tensor, quantizer: WeightFakeQuantizer) -> torch.Tensor:
        """Evaluate QDQ directly instead of weight + detached rounding error.

        Args:
            ctx: Autograd context; no tensors or mutable aliases are saved.
            weight: Tensor of shape [..., out, in], as documented on quantize.
            quantizer: Stateless quantizer holding the resolved configuration.

        Returns:
            Independent QDQ tensor of shape [..., out, in], with weight's
            dtype/device and signed zeros, as documented on dequantize.
        """
        return quantizer.quantize(weight).dequantize(dtype=weight.dtype)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor | None) -> tuple[torch.Tensor | None, None]:
        """Pass the upstream gradient through unchanged.

        Args:
            ctx: Unused autograd context.
            grad_output: Optional gradient tensor of shape [..., out, in], with
                the forward weight's dtype/device; arbitrary strides are accepted.

        Returns:
            The same [..., out, in] gradient tensor (an unmodified alias), or
            None, and None for the non-tensor quantizer argument.
        """
        return grad_output, None
