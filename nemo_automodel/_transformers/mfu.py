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

import logging
from os import PathLike
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from nemo_automodel.components.utils.flops_utils import (
    calculate_mfu,
    get_flops_formula_for_hf_config,
)

logger = logging.getLogger(__name__)

# Device theoretical FLOPS (FLOPs/s) adapted from https://github.com/verl-project/verl/blob/main/verl/utils/flops_counter.py#L22-L85
_DEVICE_FLOPS: Dict[str, float] = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1336e12,
    "H100": 989e12,
    "H800": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 21.75e12,
}

_UNIT_TO_SCALE = {
    "B": 1e9,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
}


def get_device_flops(unit: str = "T", device_name: Optional[str] = None) -> float:
    """Get theoretical device FLOPS in a requested unit.

    Args:
        unit: One of ``B/K/M/G/T/P``. Default ``T`` (TFLOPs/s).
        device_name: Optional explicit device name for lookup. If ``None``,
            the current torch device name is inferred.

    Returns:
        Theoretical FLOPS in requested unit. Returns ``float("inf")`` for
        unknown devices.
    """
    unit = unit.upper()
    if unit not in _UNIT_TO_SCALE:
        supported = ", ".join(_UNIT_TO_SCALE.keys())
        raise ValueError(f"Unsupported unit '{unit}'. Supported units: {supported}")

    if device_name is None:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            device_name = "CPU"

    flops = float("inf")
    normalized_device = str(device_name).lower()
    for key, value in sorted(_DEVICE_FLOPS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key.lower() in normalized_device:
            flops = value
            break

    return flops / _UNIT_TO_SCALE[unit]


class AutoMFU:
    """Auto MFU calculator - provides MFU calculation for various model architectures.

    This class provides a HuggingFace AutoModel-like interface for calculating
    Model FLOPs Utilization (MFU) during training.
    """
    def __init__(self, config: "PretrainedConfig", device: str = "h100"):
        """Initialize AutoMFU with a model config.

        Args:
            config: HuggingFace PretrainedConfig object
            device: Device name (e.g. ``"h100"``)
        """
        self.config = config
        self.flops_formula = get_flops_formula_for_hf_config(config)
        self.reference_mfu = get_device_flops(unit="T", device_name=device)

    @classmethod
    def register_device(cls, device: str, peak_tflops: float) -> None:
        """Register or override a device peak TFLOPs entry used for MFU calculation."""
        _DEVICE_FLOPS[str(device)] = float(peak_tflops) * 1e12

    @classmethod
    def from_config(
        cls,
        config_or_path_or_model: Union["PretrainedConfig", str, PathLike[str], object],
        device: str = "h100",
        **kwargs,
    ) -> "AutoMFU":
        """Create AutoMFU from a config object, model object, or model path/ID.

        Args:
            config_or_path_or_model: Either a PretrainedConfig object, a model object
                (the .config attribute will be extracted), or a model ID/local path.
            device: Device name (e.g. ``"h100"``)
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained
                when loading from model ID/path.

        Returns:
            AutoMFU instance
        """
        config = config_or_path_or_model
        if hasattr(config, "config"):
            config = config.config
        elif isinstance(config, (str, PathLike)):
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(str(config), **kwargs)
        return cls(config, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_local_path_or_model: Union[str, PathLike[str], object],
        device: str = "h100",
        **kwargs,
    ) -> "AutoMFU":
        """Create AutoMFU from model ID, local path, or a model object.

        Args:
            model_id_or_local_path_or_model: Model ID (e.g., "meta-llama/llama-3-70b"),
                local path, or model object (the .config attribute will be extracted)
            device: Device name (e.g. ``"h100"``)
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained

        Returns:
            AutoMFU instance
        """
        return cls.from_config(model_id_or_local_path_or_model, device=device, **kwargs)

    def __call__(
        self,
        input_ids_or_tensor: Union[torch.Tensor, Tuple[int, int]],
        time_delta: float,
        world_size: int,
    ) -> Optional[float]:
        """Calculate MFU percentage.

        Args:
            input_ids_or_tensor: Either a tensor (batch_size, seq_len) or
                a tuple of (batch_size, seq_len)
            time_delta: Time taken for forward/backward pass in seconds
            world_size: Number of GPUs used for training

        Returns:
            MFU as a percentage, or None if model not supported
        """
        if self.flops_formula is None:
            return None

        if hasattr(input_ids_or_tensor, "shape"):
            batch_size, seq_len = input_ids_or_tensor.shape[:2]
        else:
            batch_size, seq_len = input_ids_or_tensor

        flops = self.flops_formula(self.config, gbs=batch_size, seq_len=seq_len)
        tflops = flops / 1e12
        return calculate_mfu(tflops, world_size, time_delta, reference_mfu=self.reference_mfu)

    def get_flops(
        self,
        input_ids_or_tensor: Union[torch.Tensor, Tuple[int, int]],
    ) -> Optional[float]:
        """Calculate FLOPs for given input shape.

        Args:
            input_ids_or_tensor: Either a tensor (batch_size, seq_len) or
                a tuple of (batch_size, seq_len)

        Returns:
            FLOPs as a float, or None if model not supported
        """
        if self.flops_formula is None:
            return None

        if hasattr(input_ids_or_tensor, "shape"):
            batch_size, seq_len = input_ids_or_tensor.shape[:2]
        else:
            batch_size, seq_len = input_ids_or_tensor

        return self.flops_formula(self.config, gbs=batch_size, seq_len=seq_len)
