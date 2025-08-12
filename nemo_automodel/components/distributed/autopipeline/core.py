# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.stage import PipelineStage

from nemo_automodel.components.distributed.autopipeline.functional import pipeline_model
from nemo_automodel.components.distributed.autopipeline.hf_utils import (
    initialize_hf_model_on_meta,
    validate_hf_model_for_pipeline_support,
)
from nemo_automodel.components.distributed.autopipeline.training_utils import (
    pp_clip_grad_norm,
    pp_forward_backward_step,
    pp_scale_grads_by_divisor,
)

if TYPE_CHECKING:
    from transformers import AutoConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineInfo:
    enabled: bool
    schedule: Optional[object]
    has_first_stage: bool
    has_last_stage: bool
    model_parts: Optional[list[nn.Module]]
    stages: Optional[list[PipelineStage]]


class ParallelizeFnProtocol(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        world_mesh: DeviceMesh,
        moe_mesh: DeviceMesh,
        *,
        pp_enabled: bool,
        dp_axis_names: tuple[str, ...],
        cp_axis_name: str | None = None,
        tp_axis_name: str | None = None,
        ep_axis_name: str | None = None,
        ep_shard_axis_names: tuple[str, ...] | None = None,
    ) -> None: ...


@dataclass
class AutoPipelineConfig:
    # Device Mesh
    world_mesh: Optional[DeviceMesh] = None
    moe_mesh: Optional[DeviceMesh] = None
    pp_axis_name: str = "pp"
    dp_axis_names: tuple[str, ...] = ("dp",)
    cp_axis_name: Optional[str] = None
    tp_axis_name: Optional[str] = None
    ep_axis_name: Optional[str] = None
    ep_shard_axis_names: Optional[tuple[str, ...]] = None

    # Pipeline Parallel
    pp_schedule: Optional[str] = "1f1b"
    pp_schedule_csv: Optional[str] = None
    pp_microbatch_size: int = 1
    pp_batch_size: int = 1
    layers_per_stage: Optional[int] = None
    round_virtual_stages_to_pp_multiple: Optional[Literal["up", "down"]] = None
    module_fqns_per_model_part: Optional[list[list[str]]] = None

    # Patching
    patch_inner_model: bool = True
    patch_causal_lm_model: bool = True

    # Runtime
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: Optional[torch.dtype] = None
    scale_grads_in_schedule: bool = False

    # Materialization / weights
    init_weights: bool = True
    empty_weights: bool = False

    # Visualization
    visualization_font_size_offset: int = 0

    def validate(self) -> None:
        if self.pp_schedule_csv is None and self.pp_schedule is None:
            raise ValueError("Either pipeline_parallel_schedule or pipeline_parallel_schedule_csv must be provided")
        if self.pp_batch_size % self.pp_microbatch_size != 0:
            raise ValueError("local_batch_size must be divisible by microbatch_size")


def materialize_meta_model(
    model_parts: list[nn.Module],
    device: torch.device,
    *,
    init_buffers_fn: Callable[[nn.Module, torch.device], None] | None = None,
    init_weights_fn: Callable[[nn.Module, torch.device], None] | None = None,
) -> None:
    for mp in model_parts:
        mp.to_empty(device=device)

        if init_buffers_fn is not None:
            init_buffers_fn(mp, device)

        if init_weights_fn is not None:
            init_weights_fn(mp, device)

        mp.train()
        mp.to(device)


class AutoPipeline:
    """Orchestrates pipeline-parallel training on top of torch.distributed.pipelining.

    Responsibilities:
      0. Validation
      1. Initialize model on meta device
      2. Split model into pipeline stages
      3. Materialize and load weights
      4. Forward/backward step API
    """

    def __init__(self, cfg: AutoPipelineConfig):
        cfg.validate()
        self.cfg = cfg

        if cfg.world_mesh is None:
            raise ValueError("AutopipelineConfig.world_mesh must be provided (DeviceMesh with 'pp' axis)")

        self.world_mesh: DeviceMesh = cfg.world_mesh
        self.pp_mesh: DeviceMesh = self.world_mesh[cfg.pp_axis_name]
        self._device: torch.device = cfg.device

        self._info = PipelineInfo(
            enabled=False,
            schedule=None,
            has_first_stage=False,
            has_last_stage=False,
            model_parts=None,
            stages=None,
        )

    def build(
        self,
        model_or_config: Union[str, "AutoConfig", nn.Module],
        *,
        loss_fn: Optional[Callable] = None,
        parallelize_fn: Optional[ParallelizeFnProtocol] = None,
        initialize_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Build the pipeline: validate -> init meta -> split -> materialize -> schedule."""
        # 0. Validation
        assert loss_fn is not None, "loss_fn must be provided"

        # If model instance provided, use as-is; else expect a config object with instantiate()
        if isinstance(model_or_config, nn.Module):
            model = model_or_config
        else:
            # HF hub id
            model = initialize_hf_model_on_meta(model_or_config, **(initialize_kwargs or {}))
            validate_hf_model_for_pipeline_support(model)

        pp_schedule_obj, model_parts, pp_has_first_stage, pp_has_last_stage, stages = pipeline_model(
            model,
            world_mesh=self.world_mesh,
            moe_mesh=self.cfg.moe_mesh,
            pp_axis_name=self.cfg.pp_axis_name,
            dp_axis_names=self.cfg.dp_axis_names,
            cp_axis_name=self.cfg.cp_axis_name,
            tp_axis_name=self.cfg.tp_axis_name,
            ep_axis_name=self.cfg.ep_axis_name,
            layers_per_stage=self.cfg.layers_per_stage,
            pipeline_parallel_schedule_csv=self.cfg.pp_schedule_csv,
            pipeline_parallel_schedule=self.cfg.pp_schedule,
            microbatch_size=self.cfg.pp_microbatch_size,
            local_batch_size=self.cfg.pp_batch_size,
            device=self.device,
            loss_fn=loss_fn,
            parallelize_fn=parallelize_fn,
            module_fqns_per_model_part=self.cfg.module_fqns_per_model_part,
        )

        # Update PipelineInfo state
        self._info.enabled = True
        self._info.schedule = pp_schedule_obj
        self._info.has_first_stage = pp_has_first_stage
        self._info.has_last_stage = pp_has_last_stage
        self._info.model_parts = model_parts
        self._info.stages = stages

        return self

    def materialize(
        self,
        *,
        init_buffers_fn: Callable[[nn.Module, torch.device], None] | None = None,
        init_weights_fn: Callable[[nn.Module, torch.device], None] | None = None,
    ) -> None:
        if self._info.model_parts is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")

        materialize_meta_model(
            self._info.model_parts,
            device=self.device,
            init_buffers_fn=init_buffers_fn,
            init_weights_fn=init_weights_fn,
        )

    def step(
        self,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
        *,
        train_ctx: Callable,
    ) -> torch.Tensor:
        schedule = self._info.schedule
        if schedule is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")
        return pp_forward_backward_step(
            schedule,
            self._info.has_first_stage,
            self._info.has_last_stage,
            batch,
            labels,
            loss_mask,
            train_ctx,
            self.device,
        )

    def scale_grads_by_divisor(self, divisor: int) -> None:
        if self._info.stages is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")

        pp_scale_grads_by_divisor(self._info.stages, divisor)

    def clip_grad_norm(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
        pp_mesh: DeviceMesh | None = None,
        ep_dense_params_mesh_ndim: int | None = None,
    ) -> torch.Tensor:
        if self._info.stages is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")

        return pp_clip_grad_norm(
            [p for m in self._info.model_parts for p in m.parameters()],
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            ep_dense_params_mesh_ndim,
        )

    @property
    def info(self) -> PipelineInfo:
        return self._info

    @property
    def parts(self) -> list[nn.Module]:
        if self._info.model_parts is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")

        return self._info.model_parts

    @property
    def device(self) -> torch.device:
        return self._device

    def train(self) -> None:
        if self._info.model_parts is None:
            return
        for mp in self._info.model_parts:
            mp.train()

    def eval(self) -> None:
        if self._info.model_parts is None:
            return
        for mp in self._info.model_parts:
            mp.eval()

    def parameters(self) -> list[torch.Tensor]:
        if self._info.model_parts is None:
            return []
        return [p for part in self._info.model_parts for p in part.parameters()]

    # -------------------------- Debug utilities --------------------------
    def list_stage_modules(self) -> list[list[str]]:
        names_per_stage: list[list[str]] = []
        if self._info.model_parts is None:
            return names_per_stage
        for part in self._info.model_parts:
            names = []
            for name, module in part.named_modules():
                if name == "":
                    continue
                names.append(name)
            names_per_stage.append(names)
        return names_per_stage

    def visualize_current_schedule(self, filename: Optional[str] = None) -> None:
        from torch.distributed.pipelining._schedule_visualizer import get_schedule_ops, visualize_schedule

        schedule = self._info.schedule
        ops = get_schedule_ops(schedule, self.pp_mesh.size(), self.cfg.pp_microbatch_size, len(self._info.stages))
        visualize_schedule(ops, filename)

    @staticmethod
    def _count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())

    def get_stage_param_counts(self, trainable_only: bool = False) -> list[int]:
        if not self._info.model_parts:
            return []
        return [self._count_parameters(mp, trainable_only=trainable_only) for mp in self._info.model_parts]

    def get_total_param_count(self, trainable_only: bool = False) -> int:
        return sum(self.get_stage_param_counts(trainable_only=trainable_only))

    def pretty_print_stages(self, max_modules_per_stage: int = 16, trainable_only: bool = False) -> str:
        if not self._info.model_parts:
            return "<no stages>"
        lines: list[str] = []
        param_counts = self.get_stage_param_counts(trainable_only=trainable_only)
        for idx, (mp, nparams) in enumerate(zip(self._info.model_parts, param_counts)):
            tag = []
            if self._info.stages and idx < len(self._info.stages):
                st = self._info.stages[idx]
                if getattr(st, "is_first", False):
                    tag.append("first")
                if getattr(st, "is_last", False):
                    tag.append("last")
            tag_s = f" ({','.join(tag)})" if tag else ""
            lines.append(f"Stage {idx}{tag_s}: params={nparams:,}")
            # list first N module names (excluding top-level empty name)
            mod_names = [n for n, _ in mp.named_modules() if n][:max_modules_per_stage]
            for name in mod_names:
                lines.append(f"  - {name}")
            if len(mod_names) == max_modules_per_stage:
                lines.append("  - ...")
        return "\n".join(lines)

    def debug_summary(self) -> str:
        schedule = self._info.schedule
        n_micro = getattr(schedule, "n_microbatches", None)
        pp_degree = self.pp_mesh.size()
        num_local_stages = len(self._info.stages) if self._info.stages else 0
        total_params = self.get_total_param_count(False)
        trainable_params = self.get_total_param_count(True)
        lines = [
            f"PP degree: {pp_degree}",
            f"Local stages: {num_local_stages}",
            f"Schedule: {type(schedule).__name__ if schedule is not None else None}",
            f"n_microbatches: {n_micro}",
            f"Total params: {total_params:,} (trainable: {trainable_params:,})",
        ]
        return "\n".join(lines)

    def log_debug_summary(self) -> None:
        logger.info("\n%s\n%s", self.debug_summary(), self.pretty_print_stages())
