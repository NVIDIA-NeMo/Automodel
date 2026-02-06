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

"""Pipeline parallel configuration class.

Design principle:
- Device mesh (world_mesh, moe_mesh) is passed separately to from_pretrained/from_config
- PipelineConfig contains scheduling, splitting, and runtime options
- loss_fn is included here since it's only used for pipelining
- Axis names are inferred automatically from device_mesh in _instantiate_pipeline

Usage:
    from nemo_automodel.components.distributed.pipelining.config import PipelineConfig

    config = PipelineConfig(
        pp_schedule="1f1b",
        pp_microbatch_size=2,
        pp_batch_size=8,
        loss_fn=my_loss_fn,
    )
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import torch


@dataclass
class PipelineConfig:
    """
    Configuration for pipeline parallel training.

    Note: Device mesh (world_mesh, moe_mesh) is passed separately on the
    from_pretrained/from_config method signature. Pipeline parallelism is
    enabled when pp_size > 1. Axis names are inferred automatically from
    the device mesh structure.

    Attributes:
        pp_schedule (Optional[str]): Pipeline schedule type. Supported values:
            "1f1b" (one-forward-one-backward), "gpipe", "interleaved_1f1b",
            "looped_bfs", "dfs", "v_schedule", "zero_bubble". Defaults to "1f1b".
        pp_schedule_csv (Optional[str]): Path to a CSV file defining a custom
            pipeline schedule. If provided, overrides pp_schedule.
        pp_microbatch_size (int): Size of each microbatch for pipeline execution.
            pp_batch_size must be divisible by pp_microbatch_size.
        pp_batch_size (int): Total batch size per pipeline stage. Must be
            divisible by pp_microbatch_size.
        layers_per_stage (Optional[int]): Number of transformer layers per
            pipeline stage. If None, layers are split evenly across stages.
        round_virtual_stages_to_pp_multiple (Optional[Literal["up", "down"]]):
            When using virtual stages (interleaved schedules), round the number
            of virtual stages to a multiple of pp_size. "up" rounds up, "down"
            rounds down. If None, no rounding is applied.
        module_fqns_per_model_part (Optional[List[List[str]]]): Explicit
            specification of which module FQNs belong to each model part/stage.
            If provided, overrides automatic layer splitting.
        patch_inner_model (bool): Apply pipeline patches to the inner model
            (e.g., the base transformer in a CausalLM wrapper). Defaults to True.
        patch_causal_lm_model (bool): Apply pipeline patches to the CausalLM
            wrapper model. Defaults to True.
        patch_stage_backward_maybe_with_nosync (bool): Patch stage backward to
            use no_sync context for gradient accumulation efficiency. Useful
            when combining PP with FSDP.
        dtype (Optional[torch.dtype]): Data type for pipeline computation.
            If None, uses the model's default dtype.
        scale_grads_in_schedule (bool): Scale gradients within the pipeline
            schedule (by 1/n_microbatches). If False, gradients must be scaled
            externally. Defaults to False.
        loss_fn (Optional[Callable]): Loss function used for pipeline training.
            Required when pipeline is enabled. The function signature should be
            compatible with the model's output format.
    """

    pp_schedule: Optional[str] = "1f1b"
    pp_schedule_csv: Optional[str] = None
    pp_microbatch_size: int = 1
    pp_batch_size: int = 1
    layers_per_stage: Optional[int] = None
    round_virtual_stages_to_pp_multiple: Optional[Literal["up", "down"]] = None
    module_fqns_per_model_part: Optional[List[List[str]]] = None
    patch_inner_model: bool = True
    patch_causal_lm_model: bool = True
    patch_stage_backward_maybe_with_nosync: bool = False
    dtype: Optional[torch.dtype] = None
    scale_grads_in_schedule: bool = False
    loss_fn: Optional[Callable] = None

    def __init__(
        self,
        pp_schedule: Optional[str] = "1f1b",
        pp_schedule_csv: Optional[str] = None,
        pp_microbatch_size: int = 1,
        pp_batch_size: int = 1,
        layers_per_stage: Optional[int] = None,
        round_virtual_stages_to_pp_multiple: Optional[Literal["up", "down"]] = None,
        module_fqns_per_model_part: Optional[List[List[str]]] = None,
        patch_inner_model: bool = True,
        patch_causal_lm_model: bool = True,
        patch_stage_backward_maybe_with_nosync: bool = False,
        dtype: Optional[torch.dtype] = None,
        scale_grads_in_schedule: bool = False,
        loss_fn: Optional[Callable] = None,
    ):
        self.pp_schedule = pp_schedule
        self.pp_schedule_csv = pp_schedule_csv
        self.pp_microbatch_size = pp_microbatch_size
        self.pp_batch_size = pp_batch_size
        self.layers_per_stage = layers_per_stage
        self.round_virtual_stages_to_pp_multiple = round_virtual_stages_to_pp_multiple
        self.module_fqns_per_model_part = module_fqns_per_model_part
        self.patch_inner_model = patch_inner_model
        self.patch_causal_lm_model = patch_causal_lm_model
        self.patch_stage_backward_maybe_with_nosync = patch_stage_backward_maybe_with_nosync
        self.dtype = dtype
        self.scale_grads_in_schedule = scale_grads_in_schedule
        self.loss_fn = loss_fn

    @classmethod
    def from_config_node(cls, config_node) -> "PipelineConfig":
        """Create PipelineConfig from a configuration node (e.g., Hydra config)."""
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
            "pp_schedule": self.pp_schedule,
            "pp_schedule_csv": self.pp_schedule_csv,
            "pp_microbatch_size": self.pp_microbatch_size,
            "pp_batch_size": self.pp_batch_size,
            "layers_per_stage": self.layers_per_stage,
            "round_virtual_stages_to_pp_multiple": self.round_virtual_stages_to_pp_multiple,
            "module_fqns_per_model_part": self.module_fqns_per_model_part,
            "patch_inner_model": self.patch_inner_model,
            "patch_causal_lm_model": self.patch_causal_lm_model,
            "patch_stage_backward_maybe_with_nosync": self.patch_stage_backward_maybe_with_nosync,
            "dtype": self.dtype,
            "scale_grads_in_schedule": self.scale_grads_in_schedule,
            "loss_fn": self.loss_fn,
        }
