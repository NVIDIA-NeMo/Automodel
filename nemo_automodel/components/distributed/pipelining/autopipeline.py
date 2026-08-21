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
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.pipelining.stage import PipelineStage

from nemo_automodel.components.distributed.pipelining import model_parts, stage_runtime
from nemo_automodel.components.distributed.pipelining.hf_utils import (
    validate_hf_model_for_pipeline_support,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineInfo:
    """Runtime state produced by pipeline-parallel setup.

    Stage ownership is recorded from the model parts at build time rather than
    read off ``stages``: the PyTorch stages are created lazily on the first
    :meth:`AutoPipeline.step`, but callers such as VLM media staging must know
    which rank owns the first stage before that step runs.
    """

    schedule: _PipelineSchedule | None = None
    model_parts: list[nn.Module] | None = None
    stages: list[PipelineStage] | None = None
    stage_indices: tuple[int, ...] = ()
    num_stages: int = 0

    @property
    def enabled(self) -> bool:
        """Whether pipeline construction has completed."""
        return bool(self.stage_indices)

    @property
    def has_first_stage(self) -> bool:
        """Whether this rank owns the first global stage."""
        return 0 in self.stage_indices

    @property
    def has_last_stage(self) -> bool:
        """Whether this rank owns the last global stage."""
        return self.num_stages > 0 and (self.num_stages - 1) in self.stage_indices


@dataclass(frozen=True)
class _RuntimeKey:
    """Identity of the stage metadata frozen inside a built pipeline runtime.

    ``PipelineStage`` freezes its inferred boundary metadata on first use
    (``_configure_outputs_meta`` refuses to be called twice), so every input
    property that changes those boundary tensors must force a fresh runtime.
    """

    input_shape: tuple[int, ...]
    input_dtype: torch.dtype
    forward_only: bool
    emits_hidden_states: bool


class AutoPipeline:
    """Orchestrates pipeline-parallel training on top of torch.distributed.pipelining."""

    def __init__(
        self,
        # Device Mesh
        world_mesh: DeviceMesh | None = None,
        moe_mesh: DeviceMesh | None = None,
        pp_axis_name: str = "pp",
        dp_axis_names: tuple[str, ...] = ("dp",),
        cp_axis_name: str | None = None,
        tp_axis_name: str | None = None,
        ep_axis_name: str | None = None,
        ep_shard_axis_names: tuple[str, ...] | None = None,
        # Pipeline Parallel
        pp_schedule: str | None = "1f1b",
        pp_schedule_csv: str | None = None,
        pp_microbatch_size: int = 1,
        pp_batch_size: int = 1,
        layers_per_stage: int | None = None,
        round_virtual_stages_to_pp_multiple: Literal["up", "down"] | None = None,
        module_fqns_per_model_part: list[list[str]] | None = None,
        # Patching
        patch_inner_model: bool = True,
        patch_causal_lm_model: bool = True,
        patch_stage_backward_maybe_with_nosync: bool = False,
        defer_fsdp_grad_sync: bool = True,
        # Runtime
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        scale_grads_in_schedule: bool = False,
    ):
        if pp_schedule_csv is None and pp_schedule is None:
            raise ValueError("Either pp_schedule or pp_schedule_csv must be provided")
        if pp_microbatch_size <= 0:
            raise ValueError(f"pp_microbatch_size must be positive, got {pp_microbatch_size}")
        if pp_batch_size % pp_microbatch_size != 0:
            raise ValueError(
                f"pp_batch_size ({pp_batch_size}) must be divisible by pp_microbatch_size ({pp_microbatch_size})"
            )
        if world_mesh is None:
            raise ValueError("world_mesh must be provided (DeviceMesh with a pipeline axis)")

        self.world_mesh: DeviceMesh = world_mesh
        self.moe_mesh = moe_mesh
        self.pp_axis_name = pp_axis_name
        self.dp_axis_names = dp_axis_names
        self.cp_axis_name = cp_axis_name
        self.tp_axis_name = tp_axis_name
        self.ep_axis_name = ep_axis_name
        self.ep_shard_axis_names = ep_shard_axis_names
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
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device: torch.device = device
        self.dtype = dtype
        self.scale_grads_in_schedule = scale_grads_in_schedule

        self.pp_mesh: DeviceMesh = self.world_mesh[pp_axis_name]

        self._info = PipelineInfo()
        self._parts: list[model_parts.PipelineModelPart] = []
        self._loss_fn: Callable | None = None
        self._emits_hidden_states = False
        self._runtime_key: _RuntimeKey | None = None

    def build(
        self,
        model: nn.Module,
        *,
        loss_fn: Callable | None = None,
        parallelize_fn: model_parts.ParallelizeFnProtocol | None = None,
    ) -> "AutoPipeline":
        """Split, parallelize, and prepare a model for pipeline-parallel execution.

        Stages are created without boundary metadata; ``PipelineStage`` measures
        it on the first :meth:`step` from the real inputs.

        Args:
            model: Unsharded model to partition across pipeline stages.
            loss_fn: Loss callable invoked by the schedule on the last stage.
            parallelize_fn: Optional callable applying FSDP/TP/EP to each part.

        Returns:
            This pipeline, ready for :meth:`step`.
        """
        if loss_fn is None:
            raise ValueError("loss_fn must be provided")
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}")
        if self.pp_mesh.size() <= 1:
            raise ValueError("Pipeline parallelism requires a pipeline mesh with at least two ranks")

        validate_hf_model_for_pipeline_support(model)
        self._parts = model_parts.split_model_into_parts(
            model,
            self.pp_mesh,
            self.pp_schedule,
            self.module_fqns_per_model_part,
            layers_per_stage=self.layers_per_stage,
            patch_inner_model=self.patch_inner_model,
            patch_causal_lm_model=self.patch_causal_lm_model,
            round_to_pp_multiple=self.round_virtual_stages_to_pp_multiple,
        )
        self._parts = model_parts.parallelize_model_parts(
            self._parts,
            parallelize_fn,
            world_mesh=self.world_mesh,
            moe_mesh=self.moe_mesh,
            dp_axis_names=self.dp_axis_names,
            cp_axis_name=self.cp_axis_name,
            tp_axis_name=self.tp_axis_name,
            ep_axis_name=self.ep_axis_name,
            ep_shard_axis_names=self.ep_shard_axis_names,
        )
        self._loss_fn = loss_fn
        self._info = PipelineInfo(
            schedule=None,
            model_parts=[part.module for part in self._parts],
            stages=None,
            stage_indices=tuple(part.stage_index for part in self._parts),
            num_stages=self._parts[0].num_stages if self._parts else 0,
        )
        self._runtime_key = None
        return self

    @property
    def info(self) -> PipelineInfo:
        """Runtime state produced by pipeline-parallel setup."""
        return self._info

    @property
    def loss_fn(self) -> Callable | None:
        """Loss callable this pipeline hands to its schedule."""
        return self._loss_fn

    @property
    def emits_hidden_states(self) -> bool:
        """Whether the final stage bypasses its LM head for a fused loss."""
        return self._emits_hidden_states

    @property
    def last_stage_part(self) -> nn.Module | None:
        """Local model part owning the last global stage, if this rank owns it."""
        if not self._parts:
            return None
        return next((part.module for part in self._parts if part.is_last), None)

    def configure_loss_fn(self, loss_fn: Callable, *, emits_hidden_states: bool = False) -> None:
        """Set the schedule loss and the last-stage output contract.

        Both arguments are declarative and identical on every pipeline rank, so
        any resulting runtime rebuild happens uniformly on the next :meth:`step`
        rather than on the last-stage rank alone.

        Args:
            loss_fn: Loss callable invoked by the pipeline schedule.
            emits_hidden_states: Whether the final model part must bypass its LM
                head and emit hidden activations for a fused loss.
        """
        if not self._parts:
            raise RuntimeError("AutoPipeline.build() must be called before configure_loss_fn()")
        self._loss_fn = loss_fn
        self._emits_hidden_states = emits_hidden_states
        self._apply_hidden_state_contract()
        if self._info.schedule is not None and self._runtime_key is not None:
            # A live schedule holds the loss directly; only the output contract
            # can invalidate frozen stage metadata.
            self._info.schedule._loss_fn = loss_fn

    def _apply_hidden_state_contract(self) -> None:
        """Propagate the fused-loss output contract to the last-stage module."""
        last_stage_part = self.last_stage_part
        if last_stage_part is not None:
            last_stage_part._pp_return_hidden_states = self._emits_hidden_states

    def _assert_runtime_key_agrees_across_pp(self, key: _RuntimeKey) -> None:
        """Verify every pipeline rank rebuilds with the same boundary identity.

        Building a runtime starts a group-wide metadata exchange, so ranks that
        disagree about the key would desynchronize. The check runs only on a
        rebuild, and is logged first so an asymmetric hang is diagnosable.

        Args:
            key: The runtime identity this rank is about to build.
        """
        group = self.pp_mesh.get_group(self.pp_axis_name)
        if not dist.is_available() or not dist.is_initialized() or group is None:
            return
        gathered: list[_RuntimeKey | None] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, key, group=group)
        mismatched = [(rank, other) for rank, other in enumerate(gathered) if other != key]
        if mismatched:
            raise RuntimeError(
                "Pipeline ranks disagree about the stage boundary metadata to build. "
                "Every pipeline rank must call step() with the same input shape, dtype, "
                f"and mode. This rank wants {key}; mismatching ranks: {mismatched}."
            )

    def _ensure_runtime_for(self, model_input: torch.Tensor, *, forward_only: bool) -> None:
        """Build or reuse the stage runtime matching the upcoming input.

        Args:
            model_input: Whole-batch first-stage input of shape
                [batch, sequence, ...]; only its shape and dtype are read.
            forward_only: Whether the upcoming step skips the backward pass.
        """
        if not self._parts:
            raise RuntimeError("AutoPipeline.build() must be called before step()")
        if model_input.ndim < 2:
            raise ValueError(f"Pipeline model input must have at least two dimensions, got {tuple(model_input.shape)}")
        if self._loss_fn is None:
            raise RuntimeError("AutoPipeline has no loss function; call build() or configure_loss_fn() first")

        key = _RuntimeKey(
            input_shape=tuple(model_input.shape[1:]),
            input_dtype=model_input.dtype,
            forward_only=forward_only,
            emits_hidden_states=self._emits_hidden_states,
        )
        if key == self._runtime_key:
            return

        logger.info("Building pipeline runtime for %s (previous: %s)", key, self._runtime_key)
        self._assert_runtime_key_agrees_across_pp(key)
        self._apply_hidden_state_contract()

        # Drop the previous stages before allocating new ones so their frozen
        # receive buffers are not held alongside the replacements.
        self._info = PipelineInfo(
            schedule=None,
            model_parts=self._info.model_parts,
            stages=None,
            stage_indices=self._info.stage_indices,
            num_stages=self._info.num_stages,
        )
        self._runtime_key = None

        runtime = stage_runtime.build_pipeline_runtime(
            self._parts,
            self.pp_mesh,
            self.pp_axis_name,
            self.device,
            microbatch_size=self.pp_microbatch_size,
            local_batch_size=self.pp_batch_size,
            schedule_name=self.pp_schedule,
            schedule_csv=self.pp_schedule_csv,
            loss_fn=self._loss_fn,
            scale_grads=self.scale_grads_in_schedule,
            patch_stage_backward_maybe_with_nosync=self.patch_stage_backward_maybe_with_nosync,
            reduce_grad_per_microbatch=not self.defer_fsdp_grad_sync,
        )
        self._info = PipelineInfo(
            schedule=runtime.schedule,
            model_parts=self._info.model_parts,
            stages=runtime.stages,
            stage_indices=self._info.stage_indices,
            num_stages=self._info.num_stages,
        )
        self._runtime_key = key

    def step(
        self,
        model_input: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        losses: list[torch.Tensor] | None = None,
        forward_only: bool = False,
        **kwargs: object,
    ) -> object | None:
        """Run one pipeline schedule step over a whole batch.

        Args:
            model_input: Whole-batch first-stage input of shape
                [batch, sequence, ...]. Every pipeline rank must pass an
                identically shaped tensor; only the first stage consumes values.
            target: Whole-batch loss target consumed by the last stage.
            losses: List that receives one loss tensor per microbatch.
            forward_only: Whether to run without a backward pass.
            **kwargs: Extra model keyword arguments; tensors are split along the
                batch axis in lockstep with ``model_input``.

        Returns:
            The merged last-stage output on the last-stage rank, otherwise None.
        """
        self._ensure_runtime_for(model_input, forward_only=forward_only)
        schedule = self._info.schedule
        if schedule is None:
            raise RuntimeError("Pipeline runtime construction did not produce a schedule")
        schedule_fn = schedule.eval if forward_only else schedule.step
        args = (model_input,) if self._info.has_first_stage else ()
        return schedule_fn(*args, target=target, losses=losses, **kwargs)

    @property
    def parts(self) -> list[nn.Module]:
        """Local model parts owned by this rank."""
        if self._info.model_parts is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")
        return self._info.model_parts

    @property
    def device(self) -> torch.device:
        """Device on which the local pipeline stages execute."""
        return self._device

    # -------------------------- Debug utilities --------------------------
    def get_stage_param_counts(self, trainable_only: bool = False) -> list[int]:
        """Return the parameter count of each local model part."""
        if not self._info.model_parts:
            return []
        return [
            sum(p.numel() for p in mp.parameters() if p.requires_grad or not trainable_only)
            for mp in self._info.model_parts
        ]

    def debug_summary(self) -> str:
        """Return a human-readable summary of the pipeline topology."""
        schedule = self._info.schedule
        param_counts = self.get_stage_param_counts()
        lines = [
            f"PP degree: {self.pp_mesh.size()}",
            f"Local stages: {len(self._info.stages) if self._info.stages else 0}",
            f"Schedule: {type(schedule).__name__ if schedule is not None else 'not built'}",
            f"n_microbatches: {getattr(schedule, '_n_microbatches', None)}",
            f"Runtime key: {self._runtime_key}",
            f"Total params: {sum(param_counts):,}",
        ]
        for idx, nparams in enumerate(param_counts):
            lines.append(f"  Stage part {idx}: params={nparams:,}")
        return "\n".join(lines)

    def log_debug_summary(self) -> None:
        """Log the pipeline topology summary at INFO level."""
        logger.info("\n%s", self.debug_summary())
