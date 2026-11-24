# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""AutoMFU: Automatic Model FLOPs Utilization calculator.

Similar interface to HuggingFace AutoModel, this module provides automatic
MFU calculation for various model architectures.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from nemo_automodel.components.utils.flops_utils import (
    calculate_mfu,
    get_flops_formula_for_hf_config,
    transformer_flops,
)

logger = logging.getLogger(__name__)

_Precision = Literal["fp32", "tf32", "fp16", "fp8", "fp4"]

# Dense, per-device TFLOPs/s. FP16 and BF16 share the same Tensor Core peak.
# Additional precision entries are from NVIDIA's published specifications:
# https://www.nvidia.com/en-us/data-center/h100/ and the Hopper architecture whitepaper (PCIe)
# https://www.nvidia.com/en-us/data-center/h200/
# https://www.nvidia.com/en-us/data-center/a100/
# https://www.nvidia.com/en-us/data-center/gb200-nvl72/
# https://www.nvidia.com/en-us/data-center/gb300-nvl72/
# https://www.nvidia.com/en-us/data-center/hgx/
# https://www.nvidia.com/en-us/data-center/l40s/ and the Ada architecture whitepaper (L40)
# Other 16-bit entries retain the existing lookup adapted from verl's flops_counter.py.
_DEVICE_TFLOPS: dict[str, dict[_Precision, float]] = {
    "CPU": {"fp16": 0.448},
    # The preliminary Vera Rubin NVL72 spec lists 288 dense FP16/BF16 PFLOPs/s per rack.
    "VR NVL72": {"fp16": 4000},
    "GB200": {"fp32": 80, "tf32": 1250, "fp16": 2500, "fp8": 5000, "fp4": 10000},
    "GB300": {"fp32": 80, "tf32": 1250, "fp16": 2500, "fp8": 5000, "fp4": 15000},
    "B200": {"fp32": 75, "tf32": 1125, "fp16": 2250, "fp8": 4500, "fp4": 9000},
    "B300": {"fp32": 75, "tf32": 1125, "fp16": 2250, "fp8": 4500, "fp4": 13500},
    "MI300X": {"fp16": 1336.0},
    "H100": {"fp32": 67, "tf32": 494.5, "fp16": 989, "fp8": 1979},
    "H800": {"fp16": 989.0},
    "H200": {"fp32": 67, "tf32": 494.5, "fp16": 989, "fp8": 1979},
    "A100": {"fp32": 19.5, "tf32": 156, "fp16": 312},
    "A800": {"fp16": 312.0},
    "L40S": {"fp32": 91.6, "tf32": 183, "fp16": 362.05, "fp8": 733},
    "L40": {"fp32": 90.5, "tf32": 90.5, "fp16": 181.05, "fp8": 362},
    "A40": {"fp16": 149.7},
    "L20": {"fp16": 119.5},
    "H20": {"fp16": 148.0},
    "910B": {"fp16": 354.0},
    "Ascend910": {"fp16": 354.0},
    "RTX 3070 Ti": {"fp16": 21.75},
    "RTX PRO 6000 Blackwell Max-Q Workstation Edition": {"fp16": 438.9},
    "RTX PRO 6000 Blackwell Workstation Edition": {"fp16": 503.8},
    "H100 PCIe": {"fp32": 51, "tf32": 378, "fp16": 756, "fp8": 1513},
    "H100 NVL": {"fp32": 60, "tf32": 417.5, "fp16": 835.5, "fp8": 1670.5},
    "H200 NVL": {"fp32": 60, "tf32": 417.5, "fp16": 835.5, "fp8": 1670.5},
}

_DTYPE_PRECISIONS: dict[torch.dtype, _Precision] = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "fp16",
    torch.float8_e4m3fn: "fp8",
    torch.float8_e5m2: "fp8",
    torch.float4_e2m1fn_x2: "fp4",
}

_UNIT_TO_SCALE = {
    "B": 1e9,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
}

_UNWRAP_ATTRS = ("module", "_orig_mod", "_fsdp_wrapped_module", "model")
_CONFIG_ALIAS_ATTRS = (
    ("n_embd", "hidden_size"),
    ("n_layer", "num_hidden_layers"),
    ("n_head", "num_attention_heads"),
    ("n_positions", "max_position_embeddings"),
    ("n_inner", "intermediate_size"),
)


def get_device_flops(unit: str = "T", *, dtype: torch.dtype = torch.bfloat16) -> float:
    """Get the current device's theoretical FLOPS in a requested unit.

    Args:
        unit: One of ``B/K/M/G/T/P``. Default ``T`` (TFLOPs/s).
        dtype: Training compute dtype. FP32 follows the CUDA matmul TF32 setting.

    Returns:
        Theoretical FLOPS in requested unit. Returns ``float("inf")`` for
        unknown device and precision combinations.
    """
    unit = unit.upper()
    if unit not in _UNIT_TO_SCALE:
        supported = ", ".join(_UNIT_TO_SCALE.keys())
        raise ValueError(f"Unsupported unit '{unit}'. Supported units: {supported}")

    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"

    precision = _DTYPE_PRECISIONS.get(dtype)
    if dtype == torch.float32 and torch.backends.cuda.matmul.fp32_precision == "tf32":
        precision = "tf32"

    normalized_device = device_name.lower()
    if " mig " in normalized_device or precision is None:
        return math.inf
    for key in sorted(_DEVICE_TFLOPS, key=len, reverse=True):
        if key.lower() in normalized_device:
            return _DEVICE_TFLOPS[key].get(precision, math.inf) * (1e12 / _UNIT_TO_SCALE[unit])
    return math.inf


class AutoMFU:
    """Auto MFU calculator - provides MFU calculation for various model architectures.

    This class provides a HuggingFace AutoModel-like interface for calculating
    Model FLOPs Utilization (MFU) during training. Without a usable formula,
    FLOPs and MFU calls return None.
    """

    def __init__(
        self,
        config: PretrainedConfig | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize AutoMFU with a model config.

        Args:
            config: Hugging Face configuration. Omit it for a null calculator.
            compute_dtype: Training compute dtype. Defaults to the model config dtype, then BF16.
        """
        self.config = config
        self.flops_formula = None
        self.reference_mfu = math.inf
        if config is None:
            return

        self.flops_formula = get_flops_formula_for_hf_config(config)
        if self.flops_formula is transformer_flops and not all(
            hasattr(config, attr)
            for attr in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size")
        ):
            self.flops_formula = None
        dtype = compute_dtype or getattr(config, "dtype", None) or torch.bfloat16
        self.reference_mfu = get_device_flops(dtype=dtype)
        if not math.isfinite(self.reference_mfu):
            logger.warning(
                "Unable to determine the theoretical peak for the current device with %s; MFU will be reported as 0%%.",
                dtype,
            )

    @classmethod
    def from_config(
        cls,
        config_or_path_or_model: PretrainedConfig | str | PathLike[str] | object,
        *,
        compute_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> AutoMFU:
        """Create AutoMFU from a config object, model object, or model path/ID.

        Args:
            config_or_path_or_model: Either a PretrainedConfig object, a model object
                (the .config attribute will be extracted), or a model ID/local path.
            compute_dtype: Training compute dtype. When omitted for a model, infer its predominant
                floating-point matrix parameter dtype. Fall back to the config dtype, then BF16.
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained
                when loading from model ID/path.

        Returns:
            AutoMFU instance
        """
        if kwargs.keys() & {"device", "peak_tflops"}:
            raise TypeError("AutoMFU uses the detected device and built-in peak FLOPS table")

        config = config_or_path_or_model
        if isinstance(config, (str, PathLike)):
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(str(config), **kwargs)
        else:
            config = cls._unwrap_config(config)
            cls._ensure_common_config_aliases(config)
            if compute_dtype is None and isinstance(config_or_path_or_model, torch.nn.Module):
                elements_by_dtype: Counter[torch.dtype] = Counter()
                for param in config_or_path_or_model.parameters():
                    if param.ndim >= 2 and param.is_floating_point():
                        elements_by_dtype[param.dtype] += param.numel()
                compute_dtype = max(elements_by_dtype, key=elements_by_dtype.__getitem__, default=None)
        return cls(config, compute_dtype=compute_dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_local_path_or_model: str | PathLike[str] | object,
        *,
        compute_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> AutoMFU:
        """Create AutoMFU from model ID, local path, or a model object.

        Args:
            model_id_or_local_path_or_model: Model ID (e.g., "meta-llama/llama-3-70b"),
                local path, or model object (the .config attribute will be extracted)
            compute_dtype: Training compute dtype. When omitted for a model, infer its predominant
                floating-point matrix parameter dtype. Fall back to the config dtype, then BF16.
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained

        Returns:
            AutoMFU instance
        """
        return cls.from_config(
            model_id_or_local_path_or_model,
            compute_dtype=compute_dtype,
            **kwargs,
        )

    def __call__(
        self,
        input_ids_or_tensor: torch.Tensor | tuple[int, int],
        time_delta: float,
        world_size: int,
    ) -> float | None:
        """Calculate MFU percentage.

        Args:
            input_ids_or_tensor: Either a tensor (batch_size, seq_len) or
                a tuple of (batch_size, seq_len)
            time_delta: Time taken for forward/backward pass in seconds
            world_size: Number of GPUs used for training

        Returns:
            MFU as a percentage, or None if model not supported
        """
        flops = self.get_flops(input_ids_or_tensor)
        if flops is None:
            return None
        tflops = flops / 1e12
        return calculate_mfu(tflops, world_size, time_delta, reference_mfu=self.reference_mfu)

    def get_flops(
        self,
        input_ids_or_tensor: torch.Tensor | tuple[int, int],
    ) -> float | None:
        """Calculate FLOPs for given input shape.

        Args:
            input_ids_or_tensor: Either a tensor (batch_size, seq_len) or
                a tuple of (batch_size, seq_len)

        Returns:
            FLOPs as a float, or None if model not supported
        """
        if self.flops_formula is None:
            return None

        if isinstance(input_ids_or_tensor, torch.Tensor):
            batch_size, seq_len = input_ids_or_tensor.shape[:2]
        else:
            batch_size, seq_len = input_ids_or_tensor

        try:
            return self.flops_formula(self.config, gbs=batch_size, seq_len=seq_len)
        except Exception as e:
            logger.debug("Unable to compute FLOPs for config %s: %s", type(self.config).__name__, e)
            return None

    @staticmethod
    def _unwrap_config(config_or_model: object):
        cur = config_or_model
        visited = set()
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))

            if hasattr(cur, "config"):
                next_obj = getattr(cur, "config")
                if next_obj is not None and next_obj is not cur:
                    cur = next_obj
                    continue

            moved = False
            for attr in _UNWRAP_ATTRS:
                if hasattr(cur, attr):
                    next_obj = getattr(cur, attr)
                    if next_obj is not None and next_obj is not cur:
                        cur = next_obj
                        moved = True
                        break
            if not moved:
                break
        return cur

    @staticmethod
    def _ensure_common_config_aliases(config: object) -> None:
        for src, dst in _CONFIG_ALIAS_ATTRS:
            if not hasattr(config, dst) and hasattr(config, src):
                try:
                    setattr(config, dst, getattr(config, src))
                except Exception:
                    # Some config objects may block dynamic attrs; fallback
                    # behavior will return None if attrs remain unavailable.
                    pass
