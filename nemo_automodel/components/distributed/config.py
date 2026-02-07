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

from dataclasses import dataclass, field
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
    mp_policy: Optional[MixedPrecisionPolicy] = field(
        default_factory=lambda: MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    )
    offload_policy: Optional[CPUOffloadPolicy] = None
    activation_checkpointing: bool = False
    defer_fsdp_grad_sync: bool = True
    backend: str = "nccl"

    def __init__(
        self,
        sequence_parallel: bool = False,
        tp_plan: Optional[dict] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[CPUOffloadPolicy] = None,
        activation_checkpointing: bool = False,
        defer_fsdp_grad_sync: bool = True,
        backend: str = "nccl",
    ):
        self.sequence_parallel = sequence_parallel
        self.tp_plan = tp_plan
        self.mp_policy = mp_policy or MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
        self.offload_policy = offload_policy
        self.activation_checkpointing = activation_checkpointing
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync
        self.backend = backend

    @classmethod
    def from_config_node(cls, config_node) -> "FSDP2Config":
        """Create FSDP2Config from a configuration node (e.g., Hydra config)."""
        if config_node is None:
            return cls()

        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(config_node, field_name):
                kwargs[field_name] = getattr(config_node, field_name)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sequence_parallel": self.sequence_parallel,
            "tp_plan": self.tp_plan,
            "mp_policy": self.mp_policy,
            "offload_policy": self.offload_policy,
            "activation_checkpointing": self.activation_checkpointing,
            "defer_fsdp_grad_sync": self.defer_fsdp_grad_sync,
            "backend": self.backend,
        }


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
    megatron_fsdp_unit_modules: Optional[List[str]] = field(
        default_factory=lambda: ["transformers.models.llama.modeling_llama.LlamaDecoderLayer"]
    )
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

    def __init__(
        self,
        sequence_parallel: bool = False,
        tp_plan: Optional[dict] = None,
        megatron_fsdp_unit_modules: Optional[List[str]] = None,
        zero_dp_strategy: int = 3,
        init_fsdp_with_meta_device: bool = False,
        grad_reduce_in_fp32: bool = False,
        preserve_fp32_weights: bool = False,
        overlap_grad_reduce: bool = True,
        overlap_param_gather: bool = True,
        check_for_nan_in_grad: bool = True,
        average_in_collective: bool = False,
        disable_bucketing: bool = False,
        calculate_per_token_loss: bool = False,
        keep_fp8_transpose_cache: bool = False,
        nccl_ub: bool = False,
        fsdp_double_buffer: bool = False,
        activation_checkpointing: bool = False,
        backend: str = "nccl",
    ):
        if tp_plan is not None:
            raise ValueError("MegatronFSDPConfig does not support custom TP plans. Use FSDP2Config instead.")
        self.sequence_parallel = sequence_parallel
        self.megatron_fsdp_unit_modules = megatron_fsdp_unit_modules or [
            "transformers.models.llama.modeling_llama.LlamaDecoderLayer"
        ]
        self.zero_dp_strategy = zero_dp_strategy
        self.init_fsdp_with_meta_device = init_fsdp_with_meta_device
        self.grad_reduce_in_fp32 = grad_reduce_in_fp32
        self.preserve_fp32_weights = preserve_fp32_weights
        self.overlap_grad_reduce = overlap_grad_reduce
        self.overlap_param_gather = overlap_param_gather
        self.check_for_nan_in_grad = check_for_nan_in_grad
        self.average_in_collective = average_in_collective
        self.disable_bucketing = disable_bucketing
        self.calculate_per_token_loss = calculate_per_token_loss
        self.keep_fp8_transpose_cache = keep_fp8_transpose_cache
        self.nccl_ub = nccl_ub
        self.fsdp_double_buffer = fsdp_double_buffer
        self.activation_checkpointing = activation_checkpointing
        self.backend = backend

    @classmethod
    def from_config_node(cls, config_node) -> "MegatronFSDPConfig":
        """Create MegatronFSDPConfig from a configuration node (e.g., Hydra config)."""
        if config_node is None:
            return cls()

        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(config_node, field_name):
                kwargs[field_name] = getattr(config_node, field_name)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sequence_parallel": self.sequence_parallel,
            "megatron_fsdp_unit_modules": self.megatron_fsdp_unit_modules,
            "zero_dp_strategy": self.zero_dp_strategy,
            "init_fsdp_with_meta_device": self.init_fsdp_with_meta_device,
            "grad_reduce_in_fp32": self.grad_reduce_in_fp32,
            "preserve_fp32_weights": self.preserve_fp32_weights,
            "overlap_grad_reduce": self.overlap_grad_reduce,
            "overlap_param_gather": self.overlap_param_gather,
            "check_for_nan_in_grad": self.check_for_nan_in_grad,
            "average_in_collective": self.average_in_collective,
            "disable_bucketing": self.disable_bucketing,
            "calculate_per_token_loss": self.calculate_per_token_loss,
            "keep_fp8_transpose_cache": self.keep_fp8_transpose_cache,
            "nccl_ub": self.nccl_ub,
            "fsdp_double_buffer": self.fsdp_double_buffer,
            "activation_checkpointing": self.activation_checkpointing,
            "backend": self.backend,
        }


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

    def __init__(
        self,
        activation_checkpointing: bool = False,
        backend: str = "nccl",
    ):
        self.activation_checkpointing = activation_checkpointing
        self.backend = backend

    @classmethod
    def from_config_node(cls, config_node) -> "DDPConfig":
        """Create DDPConfig from a configuration node (e.g., Hydra config)."""
        if config_node is None:
            return cls()

        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(config_node, field_name):
                kwargs[field_name] = getattr(config_node, field_name)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "activation_checkpointing": self.activation_checkpointing,
            "backend": self.backend,
        }
