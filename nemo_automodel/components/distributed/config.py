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

"""
Strategy-specific distributed training configuration classes.

Design principle:
- Size params (dp_size, dp_replicate_size, tp_size, pp_size, cp_size, ep_size) go directly
  on the from_pretrained/from_config method signature
- dp_replicate_size is FSDP2-only: raises assertion if passed with non-FSDP2 config
- Strategy-specific configs contain only *additional* flags unique to each strategy
- Managers become normal classes that accept (config, device_mesh)

Usage:
    from nemo_automodel.components.distributed.config import FSDP2Config, MegatronFSDPConfig, DDPConfig

    # FSDP2 with custom options
    config = FSDP2Config(sequence_parallel=True, activation_checkpointing=True)

    # MegatronFSDP with custom options
    config = MegatronFSDPConfig(zero_dp_strategy=3, overlap_grad_reduce=True)

    # DDP with activation checkpointing
    config = DDPConfig(activation_checkpointing=True)
"""

from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

# Type alias for API signature
DistributedConfig = Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]


@dataclass
class FSDP2Config:
    """
    Additional configuration for FSDP2 distributed training.

    Note: Size parameters (dp_size, dp_replicate_size, tp_size, pp_size, cp_size, ep_size)
    are passed separately on the from_pretrained/from_config method signature.

    Attributes:
        sequence_parallel (bool): Enable sequence parallelism in TP plan.
        tp_plan (Optional[dict]): Custom TP plan. If None, auto-selected based on model type.
        mp_policy (Optional[MixedPrecisionPolicy]): MixedPrecisionPolicy for FSDP2.
        offload_policy (Optional[CPUOffloadPolicy]): CPUOffloadPolicy for CPU offloading.
        activation_checkpointing (bool): Enable activation checkpointing.
        defer_fsdp_grad_sync (bool): Defer FSDP gradient sync to final micro-batch.
        backend (str): Distributed backend.
    """

    sequence_parallel: bool = False
    tp_plan: Optional[dict] = None
    mp_policy: Optional[MixedPrecisionPolicy] = None
    offload_policy: Optional[CPUOffloadPolicy] = None
    activation_checkpointing: bool = False
    defer_fsdp_grad_sync: bool = True
    backend: str = "nccl"

    def __post_init__(self):
        if self.mp_policy is None:
            self.mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (shallow, preserves policy objects)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class MegatronFSDPConfig:
    """
    Additional configuration for MegatronFSDP distributed training.

    Note: Size parameters (dp_size, tp_size, cp_size) are passed separately on
    the from_pretrained/from_config method signature. MegatronFSDP does not
    support pp_size, dp_replicate_size, or ep_size.

    Attributes:
        sequence_parallel (bool): Enable sequence parallelism in TP plan.
            Note: Not supported with MegatronFSDP right now.
        megatron_fsdp_unit_modules (Optional[List[str]]): List of unit modules to be
            wrapped with MegatronFSDP.
        zero_dp_strategy (int): Data parallel sharding strategy.
        init_fsdp_with_meta_device (bool): Initialize MegatronFSDP with meta device if True.
        grad_reduce_in_fp32 (bool): Reduce gradients in fp32 if True.
        preserve_fp32_weights (bool): Preserve fp32 weights if True.
        overlap_grad_reduce (bool): Overlap gradient reduction if True.
        overlap_param_gather (bool): Overlap parameter gathering if True.
        check_for_nan_in_grad (bool): Check for NaN in gradients if True.
        average_in_collective (bool): Average in collective if True.
        disable_bucketing (bool): Disable bucketing if True.
        calculate_per_token_loss (bool): Calculate per token loss if True.
        keep_fp8_transpose_cache (bool): Keep fp8 transpose cache when using custom FSDP if True.
        nccl_ub (bool): Use NCCL UBs if True.
        fsdp_double_buffer (bool): Use double buffer if True.
        activation_checkpointing (bool): Enable activation checkpointing for transformer
            MLP layers to save memory.
        backend (str): Distributed backend, e.g. 'nccl' or 'gloo'.
    """

    sequence_parallel: bool = False
    tp_plan: InitVar[Optional[dict]] = None
    megatron_fsdp_unit_modules: Optional[List[str]] = None
    zero_dp_strategy: int = 3
    init_fsdp_with_meta_device: bool = False
    grad_reduce_in_fp32: bool = False
    preserve_fp32_weights: bool = False
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    check_for_nan_in_grad: bool = True
    average_in_collective: bool = False
    disable_bucketing: bool = False
    calculate_per_token_loss: bool = False
    keep_fp8_transpose_cache: bool = False
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    activation_checkpointing: bool = False
    backend: str = "nccl"

    def __post_init__(self, tp_plan: Optional[dict]):
        if tp_plan is not None:
            raise ValueError("MegatronFSDPConfig does not support custom TP plans. Use FSDP2Config instead.")
        if self.megatron_fsdp_unit_modules is None:
            self.megatron_fsdp_unit_modules = ["transformers.models.llama.modeling_llama.LlamaDecoderLayer"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (shallow, preserves objects)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class DDPConfig:
    """
    Additional configuration for DDP distributed training.

    Note: DDP does not support tensor parallelism, pipeline parallelism, or expert parallelism.
    Only dp_size is relevant (inferred from world_size).

    Attributes:
        activation_checkpointing (bool): Enable activation checkpointing if True.
        backend (str): Distributed backend, e.g. 'nccl' or 'gloo'.
    """

    activation_checkpointing: bool = False
    backend: str = "nccl"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
