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

"""True FP8 storage helpers for DeepSeek V4 MLA KV tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import HAS_TILELANG
from nemo_automodel.shared.import_utils import safe_import_from

_HAS_FP8_KV_QUANTIZER, _tilelang_quantize_fp8_ds_mla = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_fp8_kv",
    "quantize_fp8_ds_mla_interface",
    msg="DeepSeek V4 FP8 KV quantizer is unavailable. Install TileLang to use kv_cache_dtype='fp8_ds_mla'.",
)
_HAS_FP8_INDEXER_QUANTIZER, _tilelang_quantize_fp8_indexer = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_fp8_indexer",
    "quantize_fp8_indexer_interface",
    msg="DeepSeek V4 FP8 indexer quantizer is unavailable. Install TileLang to use the vLLM indexer boundary.",
)

DSV4_KV_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_KV_GROUP_SIZE = 64
DSV4_KV_SCALE_GROUPS = 7
DSV4_KV_SCALE_STRIDE = 8
DSV4_E4M3_MAX = 448.0
DSV4_AMAX_FLOOR = 1.0e-4
DSV4_INDEXER_DIM = 128


@dataclass(frozen=True)
class Dsv4Fp8KV:
    """Mixed DSV4 KV representation matching vLLM's ``fp8_ds_mla`` values.

    ``anchor`` is the differentiable BF16 tensor that receives the straight-
    through gradient.  Attention kernels consume ``nope``, ``rope`` and
    ``scales`` directly and never materialize a full dequantized KV tensor.
    """

    anchor: torch.Tensor
    nope: torch.Tensor
    rope: torch.Tensor
    scales: torch.Tensor

    def __post_init__(self) -> None:
        batch, seq_len, dim = self.anchor.shape
        if dim != DSV4_KV_DIM:
            raise ValueError(f"DSV4 FP8 KV expects head_dim={DSV4_KV_DIM}, got {dim}")
        expected = {
            "nope": (batch, seq_len, DSV4_NOPE_DIM),
            "rope": (batch, seq_len, DSV4_ROPE_DIM),
            "scales": (batch, seq_len, DSV4_KV_SCALE_STRIDE),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} shape must be {shape}, got {tuple(getattr(self, name).shape)}")
        if self.nope.dtype != torch.float8_e4m3fn:
            raise TypeError(f"nope must be E4M3, got {self.nope.dtype}")
        if self.rope.dtype != torch.bfloat16:
            raise TypeError(f"rope must be BF16, got {self.rope.dtype}")
        if self.scales.dtype != torch.uint8:
            raise TypeError(f"scales must be UE8M0 bytes, got {self.scales.dtype}")

    @property
    def shape(self) -> torch.Size:
        return self.anchor.shape

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def storage_nbytes(self) -> int:
        return self.nope.nbytes + self.rope.nbytes + self.scales.nbytes

    def dequantize(self) -> torch.Tensor:
        """Torch numerical oracle; production TileLang attention does not call this."""
        descales = torch.exp2(self.scales[..., :DSV4_KV_SCALE_GROUPS].float() - 127.0)
        descales = descales.repeat_interleave(DSV4_KV_GROUP_SIZE, dim=-1)
        nope = self.nope.float() * descales
        return torch.cat((nope.to(torch.bfloat16), self.rope), dim=-1)

    def vllm_token_data(self) -> torch.Tensor:
        """Return vLLM's 576 data bytes per logical token (scales are separate)."""
        nope_bytes = self.nope.contiguous().view(torch.uint8)
        rope_bytes = self.rope.contiguous().view(torch.uint8)
        return torch.cat((nope_bytes, rope_bytes), dim=-1)


@dataclass(frozen=True)
class Dsv4Fp8Indexer:
    """Actual E4M3 indexer activation plus vLLM's per-row FP32 descale."""

    anchor: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor

    def __post_init__(self) -> None:
        if self.anchor.shape[-1] != DSV4_INDEXER_DIM:
            raise ValueError(f"DSV4 FP8 indexer expects dim={DSV4_INDEXER_DIM}, got {self.anchor.shape[-1]}")
        if self.values.shape != self.anchor.shape:
            raise ValueError(f"values shape must be {tuple(self.anchor.shape)}, got {tuple(self.values.shape)}")
        expected_scale_shape = (*self.anchor.shape[:-1], 1)
        if tuple(self.scales.shape) != expected_scale_shape:
            raise ValueError(f"scales shape must be {expected_scale_shape}, got {tuple(self.scales.shape)}")
        if self.values.dtype != torch.float8_e4m3fn:
            raise TypeError(f"values must be E4M3, got {self.values.dtype}")
        if self.scales.dtype != torch.float32:
            raise TypeError(f"scales must be FP32, got {self.scales.dtype}")

    def dequantize(self) -> torch.Tensor:
        return (self.values.float() * self.scales).to(self.anchor.dtype)

    @property
    def shape(self) -> torch.Size:
        return self.anchor.shape

    @property
    def device(self) -> torch.device:
        return self.anchor.device


def _quantize_fp8_ds_mla_torch(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nope = kv[..., :DSV4_NOPE_DIM].float().view(*kv.shape[:-1], DSV4_KV_SCALE_GROUPS, DSV4_KV_GROUP_SIZE)
    amax = nope.abs().amax(dim=-1).clamp_min(DSV4_AMAX_FLOOR)
    exponent = torch.ceil(torch.log2(amax / DSV4_E4M3_MAX)).clamp(min=-127, max=127)
    descales = torch.exp2(exponent)
    quantized = (nope / descales.unsqueeze(-1)).clamp(-DSV4_E4M3_MAX, DSV4_E4M3_MAX)
    quantized = quantized.reshape(*kv.shape[:-1], DSV4_NOPE_DIM).to(torch.float8_e4m3fn)
    scales = torch.zeros(*kv.shape[:-1], DSV4_KV_SCALE_STRIDE, dtype=torch.uint8, device=kv.device)
    scales[..., :DSV4_KV_SCALE_GROUPS] = (exponent + 127).to(torch.uint8)
    return quantized, kv[..., DSV4_NOPE_DIM:].contiguous(), scales


def quantize_dsv4_kv(
    kv: torch.Tensor,
    *,
    backend: Literal["torch", "tilelang"] = "tilelang",
) -> Dsv4Fp8KV:
    """Quantize post-RoPE BF16 KV into vLLM-compatible mixed FP8/BF16 storage."""
    if kv.ndim != 3 or kv.shape[-1] != DSV4_KV_DIM:
        raise ValueError(f"Expected [B, S, {DSV4_KV_DIM}] KV, got {tuple(kv.shape)}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"DSV4 FP8 KV requires BF16 input, got {kv.dtype}")

    anchor = kv.contiguous()
    if backend == "tilelang":
        if not (HAS_TILELANG and _HAS_FP8_KV_QUANTIZER and kv.is_cuda):
            raise RuntimeError("kv_cache_dtype='fp8_ds_mla' requires the TileLang CUDA quantizer")
        flat = anchor.view(-1, DSV4_KV_DIM)
        nope, rope, scales = _tilelang_quantize_fp8_ds_mla(flat)
        batch, seq_len, _ = anchor.shape
        nope = nope.view(batch, seq_len, DSV4_NOPE_DIM)
        rope = rope.view(batch, seq_len, DSV4_ROPE_DIM)
        scales = scales.view(batch, seq_len, DSV4_KV_SCALE_STRIDE)
    elif backend == "torch":
        nope, rope, scales = _quantize_fp8_ds_mla_torch(anchor)
    else:
        raise ValueError(f"Unsupported DSV4 FP8 KV backend: {backend}")
    return Dsv4Fp8KV(anchor=anchor, nope=nope, rope=rope, scales=scales)


def quantize_dsv4_indexer(
    tensor: torch.Tensor,
    *,
    backend: Literal["torch", "tilelang"] = "tilelang",
) -> Dsv4Fp8Indexer:
    """Quantize post-RoPE indexer Q/K rows exactly like vLLM 0.21.

    Both paths use one E4M3 block of 128 elements and a power-of-two FP32
    descale. Production TileLang indexer kernels consume ``values`` and
    ``scales`` directly; ``anchor`` only records the differentiable source.
    """
    if tensor.shape[-1] != DSV4_INDEXER_DIM:
        raise ValueError(f"Expected indexer dim={DSV4_INDEXER_DIM}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"DSV4 FP8 indexer requires BF16 input, got {tensor.dtype}")

    anchor = tensor.contiguous()
    flat = anchor.view(-1, DSV4_INDEXER_DIM)
    if backend == "tilelang":
        if not (HAS_TILELANG and _HAS_FP8_INDEXER_QUANTIZER and tensor.is_cuda):
            raise RuntimeError("DSV4 FP8 indexer requires the TileLang CUDA quantizer")
        values, scales = _tilelang_quantize_fp8_indexer(flat)
    elif backend == "torch":
        rows = flat.float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(DSV4_AMAX_FLOOR)
        scales = torch.exp2(torch.ceil(torch.log2(amax / DSV4_E4M3_MAX)))
        values = (rows / scales).clamp(-DSV4_E4M3_MAX, DSV4_E4M3_MAX).to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported DSV4 FP8 indexer backend: {backend}")

    return Dsv4Fp8Indexer(
        anchor=anchor,
        values=values.view_as(anchor),
        scales=scales.view(*anchor.shape[:-1], 1),
    )


def cat_dsv4_fp8_kv(parts: tuple[Dsv4Fp8KV, ...] | list[Dsv4Fp8KV], *, dim: int = 1) -> Dsv4Fp8KV:
    """Concatenate mixed KV payloads without dequantizing them."""
    if not parts:
        raise ValueError("cat_dsv4_fp8_kv requires at least one payload")
    return Dsv4Fp8KV(
        anchor=torch.cat([part.anchor for part in parts], dim=dim),
        nope=torch.cat([part.nope for part in parts], dim=dim),
        rope=torch.cat([part.rope for part in parts], dim=dim),
        scales=torch.cat([part.scales for part in parts], dim=dim),
    )


__all__ = [
    "DSV4_AMAX_FLOOR",
    "DSV4_E4M3_MAX",
    "DSV4_INDEXER_DIM",
    "DSV4_KV_DIM",
    "DSV4_KV_GROUP_SIZE",
    "DSV4_KV_SCALE_GROUPS",
    "DSV4_KV_SCALE_STRIDE",
    "DSV4_NOPE_DIM",
    "DSV4_ROPE_DIM",
    "Dsv4Fp8KV",
    "Dsv4Fp8Indexer",
    "cat_dsv4_fp8_kv",
    "quantize_dsv4_kv",
    "quantize_dsv4_indexer",
]
