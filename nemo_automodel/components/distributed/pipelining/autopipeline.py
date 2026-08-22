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

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.microbatch import BlockMask, TensorChunkSpec
from torch.distributed.pipelining.microbatch import _Replicate as ReplicateChunkSpec
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.pipelining.stage import PipelineStage
from torch.utils._pytree import tree_map

from nemo_automodel.components.distributed.pipelining.functional import (
    ParallelizeFnProtocol,
    pipeline_model,
    reset_pp_stage_shapes,
)
from nemo_automodel.components.distributed.pipelining.hf_utils import (
    validate_hf_model_for_pipeline_support,
)

logger = logging.getLogger(__name__)
_MISSING_STAGE_STATE = object()


@dataclass
class PipelineInfo:
    """Runtime state produced by pipeline-parallel setup."""

    enabled: bool
    schedule: _PipelineSchedule | None
    has_first_stage: bool
    has_last_stage: bool
    model_parts: list[nn.Module] | None
    stages: list[PipelineStage] | None


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
        # Shape inference optimization
        pp_seq_len: int | None = None,
    ):
        # Validation
        if pp_schedule_csv is None and pp_schedule is None:
            raise ValueError("Either pipeline_parallel_schedule or pipeline_parallel_schedule_csv must be provided")
        if pp_batch_size % pp_microbatch_size != 0:
            raise ValueError("local_batch_size must be divisible by microbatch_size")
        if world_mesh is None:
            raise ValueError("world_mesh must be provided (DeviceMesh with 'pp' axis)")

        # Store config attributes
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
        self._device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.scale_grads_in_schedule = scale_grads_in_schedule
        self.pp_seq_len = pp_seq_len

        self.pp_mesh: DeviceMesh = self.world_mesh[pp_axis_name]

        self._info = PipelineInfo(
            enabled=False,
            schedule=None,
            has_first_stage=False,
            has_last_stage=False,
            model_parts=None,
            stages=None,
        )
        self._model_config = None
        self._pp_current_seq_len: int | None = None
        self._pp_current_microbatch_size: int | None = None
        self._pp_current_input_signature: tuple[tuple[int, ...], torch.dtype] | None = None

    def build(
        self,
        model: nn.Module,
        *,
        loss_fn: Callable | None = None,
        parallelize_fn: ParallelizeFnProtocol | None = None,
    ):
        """Build the pipeline: validate -> init meta -> split -> schedule."""
        # 0. Validation
        assert loss_fn is not None, "loss_fn must be provided"
        assert isinstance(model, nn.Module), "model must be a PyTorch module"

        validate_hf_model_for_pipeline_support(model)

        pp_schedule_obj, model_parts, pp_has_first_stage, pp_has_last_stage, stages = pipeline_model(
            model,
            world_mesh=self.world_mesh,
            moe_mesh=self.moe_mesh,
            pp_axis_name=self.pp_axis_name,
            dp_axis_names=self.dp_axis_names,
            cp_axis_name=self.cp_axis_name,
            tp_axis_name=self.tp_axis_name,
            ep_axis_name=self.ep_axis_name,
            ep_shard_axis_names=self.ep_shard_axis_names,
            layers_per_stage=self.layers_per_stage,
            pipeline_parallel_schedule_csv=self.pp_schedule_csv,
            pipeline_parallel_schedule=self.pp_schedule,
            microbatch_size=self.pp_microbatch_size,
            local_batch_size=self.pp_batch_size,
            device=self.device,
            loss_fn=loss_fn,
            parallelize_fn=parallelize_fn,
            module_fqns_per_model_part=self.module_fqns_per_model_part,
            patch_inner_model=self.patch_inner_model,
            patch_causal_lm_model=self.patch_causal_lm_model,
            scale_grads=self.scale_grads_in_schedule,
            round_to_pp_multiple=self.round_virtual_stages_to_pp_multiple,
            patch_stage_backward_maybe_with_nosync=self.patch_stage_backward_maybe_with_nosync,
            reduce_grad_per_microbatch=not self.defer_fsdp_grad_sync,
            seq_len=self.pp_seq_len,
            tensor_dtype=self.dtype,
        )

        # Update PipelineInfo state
        self._info.enabled = True
        self._info.schedule = pp_schedule_obj
        self._info.has_first_stage = pp_has_first_stage
        self._info.has_last_stage = pp_has_last_stage
        self._info.model_parts = model_parts
        self._info.stages = stages

        # Store model config for runtime shape updates
        self._model_config = model.config

        return self

    @property
    def info(self) -> PipelineInfo:
        return self._info

    def update_seq_len(
        self,
        seq_len: int,
        *,
        microbatch_size: int | None = None,
        input_tensor: torch.Tensor | None = None,
    ) -> None:
        """Reset pipeline stage infrastructure for a new input shape.

        VLM training batches can have wildly different sequence lengths across steps
        (image batches vs. text-only batches).  PyTorch's PipelineStage locks in recv
        buffer sizes on the first step, causing a shape-mismatch error on later steps
        with different seq_lens.

        Call this before every ``schedule.step()`` to update the stage shapes without
        running an expensive forward pass.  A no-op when seq_len has not changed.

        Args:
            seq_len: Sequence length of the upcoming batch (``input_ids.shape[1]``).
            microbatch_size: Materialized leading batch extent. Defaults to the
                configured padded microbatch size. Packed THD streams use one
                synthetic batch row even when they were formed from multiple
                source examples.
            input_tensor: Actual positional input for the first stage. When
                provided, its shape and dtype define first-stage metadata;
                this supports floating-point embeddings and flat THD tensors.
        """
        effective_microbatch_size = self.pp_microbatch_size if microbatch_size is None else microbatch_size
        input_signature = (
            (tuple(input_tensor.shape), input_tensor.dtype) if isinstance(input_tensor, torch.Tensor) else None
        )
        if effective_microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be positive, got {effective_microbatch_size}")
        if (
            seq_len == self._pp_current_seq_len
            and effective_microbatch_size == self._pp_current_microbatch_size
            and input_signature == self._pp_current_input_signature
        ):
            return
        if self._model_config is None:
            raise RuntimeError("AutoPipeline.build() must be called before update_seq_len()")
        reset_pp_stage_shapes(
            self._info.schedule,
            self._info.stages,
            self._model_config,
            effective_microbatch_size,
            seq_len,
            tensor_dtype=self.dtype,
            first_stage_input_meta=(
                torch.empty(tuple(input_tensor.shape), device="meta", dtype=input_tensor.dtype)
                if isinstance(input_tensor, torch.Tensor)
                else None
            ),
        )
        self._pp_current_seq_len = seq_len
        self._pp_current_microbatch_size = effective_microbatch_size
        self._pp_current_input_signature = input_signature
        logger.debug(
            "PP stage shapes updated for seq_len=%s, microbatch_size=%s",
            seq_len,
            effective_microbatch_size,
        )

    def _get_schedule_kwargs_chunk_spec(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Build pipeline microbatch chunking metadata for keyword inputs.

        PyTorch's default schedule chunking splits every tensor kwarg on dim 0.
        Most AutoModel batch tensors are batch-major and should keep that
        default, but some model-owned input layouts place batch on another axis.
        The canonical local model part can declare those exceptions by implementing
        ``get_pipeline_kwargs_chunk_dims(kwargs) -> dict[str, int]``.

        Args:
            kwargs: Mapping passed to the pipeline schedule. Tensor values may
                have arbitrary model-defined layouts; the model hook identifies
                any nonstandard batch axis.

        Returns:
            A chunk-spec mapping with the same nested structure as ``kwargs``,
            or ``None`` when PyTorch's default chunking applies.
        """
        model_parts = self._info.model_parts
        if not model_parts:
            raise RuntimeError("AutoPipeline.build() must be called before running a PP schedule step")

        hook = getattr(model_parts[0], "get_pipeline_kwargs_chunk_dims", None)
        if hook is None:
            return None

        custom_chunk_dims = hook(kwargs) or {}
        for key in custom_chunk_dims:
            if key not in kwargs:
                raise ValueError(f"Model PP chunk hook returned unknown kwarg: {key}")

        if not custom_chunk_dims:
            return None

        def default_spec(value):
            if isinstance(value, (torch.Tensor, BlockMask)):
                return TensorChunkSpec(0)
            return ReplicateChunkSpec()

        kwargs_chunk_spec = tree_map(default_spec, kwargs, is_leaf=lambda value: isinstance(value, BlockMask))
        for key, split_dim in custom_chunk_dims.items():
            kwargs_chunk_spec[key] = TensorChunkSpec(split_dim)
        return kwargs_chunk_spec

    def step(
        self,
        model_input: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        losses: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one pipeline schedule step with model-owned input chunking.

        Args:
            model_input: Tensor of shape [batch, ...] containing the first
                pipeline stage's input. Ignored on ranks without the first stage.
            target: Tensor with a model-defined target layout, or ``None`` on
                ranks without the last pipeline stage.
            losses: Mutable list populated with scalar loss tensors, or ``None``
                on ranks without the last pipeline stage.
            **kwargs: Keyword schedule inputs. Tensor values may have arbitrary
                model-defined layouts; model-owned metadata identifies any
                nonstandard batch axis.

        Returns:
            The value returned by the underlying PyTorch pipeline schedule.
        """
        schedule = self._info.schedule
        if schedule is None:
            raise RuntimeError("AutoPipeline.build() must be called before running a PP schedule step")

        schedule_args = (model_input,) if self._info.has_first_stage else ()
        kwargs_chunk_spec = self._get_schedule_kwargs_chunk_spec(kwargs)

        if kwargs_chunk_spec is None:
            return schedule.step(
                *schedule_args,
                target=target,
                losses=losses,
                **kwargs,
            )

        previous_kwargs_chunk_spec = schedule._kwargs_chunk_spec
        schedule._kwargs_chunk_spec = kwargs_chunk_spec
        try:
            return schedule.step(
                *schedule_args,
                target=target,
                losses=losses,
                **kwargs,
            )
        finally:
            schedule._kwargs_chunk_spec = previous_kwargs_chunk_spec

    def step_microbatches(
        self,
        model_inputs: list[dict[str, Any]],
        *,
        loss_fn: Callable[[Any, int], Any],
        losses: list[torch.Tensor] | None = None,
        return_outputs: bool = False,
        finalize_backward: bool = True,
    ) -> Any:
        """Run a schedule step over already prepared model microbatches.

        The caller owns microbatch preparation. This method passes the prepared
        inputs to the PyTorch schedule unchanged, while the schedule continues
        to own execution order and backward calls.

        Args:
            model_inputs: Exactly :attr:`num_microbatches` complete model-input
                mappings. Each mapping contains exactly one of ``input_ids`` or
                ``inputs_embeds``; all remaining items are model keyword inputs.
            loss_fn: Callback invoked as ``loss_fn(output, microbatch_index)``.
                The index identifies the corresponding item in ``model_inputs``
                regardless of the schedule's execution order.
            losses: Mutable list populated by the schedule on the last stage.
            return_outputs: Whether the last stage returns merged model outputs
                when supported by the installed PyTorch version.
            finalize_backward: Whether this schedule invocation ends the
                optimizer's backward window. ``False`` keeps DDP in no-sync
                mode and completes local FSDP post-backward state while
                deferring its gradient collective; per-microbatch gradient
                reduction remains authoritative when configured.

        Returns:
            The value returned by the underlying PyTorch pipeline schedule.

        Raises:
            ValueError: If a non-final call would use schedule-local gradient
                scaling and rescale gradients accumulated by earlier calls.
            RuntimeError: If a non-final call is attempted before
                accumulation-aware pipeline stages have been built.
        """
        if not finalize_backward and self.scale_grads_in_schedule:
            raise ValueError(
                "planned pipeline accumulation requires scale_grads_in_schedule=False; "
                "schedule-local scaling would rescale gradients from earlier calls"
            )
        if not finalize_backward and (
            not self._info.stages
            or any(not vars(stage).get("_nemo_accumulation_aware", False) for stage in self._info.stages)
        ):
            raise RuntimeError(
                "finalize_backward=False requires accumulation-aware pipeline stages; "
                "build the AutoPipeline before running planned accumulation"
            )
        return self._run_prepared_microbatches(
            model_inputs,
            loss_fn=loss_fn,
            losses=losses,
            return_outputs=return_outputs,
            schedule_method="step",
            finalize_backward=finalize_backward,
        )

    def eval_microbatches(
        self,
        model_inputs: list[dict[str, Any]],
        *,
        loss_fn: Callable[[Any, int], Any],
        losses: list[torch.Tensor] | None = None,
        return_outputs: bool = True,
    ) -> Any:
        """Run forward-only evaluation over already prepared model microbatches.

        This is the forward-only counterpart to :meth:`step_microbatches`.
        The caller owns microbatch preparation, while the pipeline schedule owns
        pipeline communication and execution order. The underlying
        ``schedule.eval`` path does not run backward.

        Args:
            model_inputs: Exactly :attr:`num_microbatches` complete model-input
                mappings. Each mapping contains exactly one of ``input_ids`` or
                ``inputs_embeds``; all remaining items are model keyword inputs.
            loss_fn: Callback invoked as ``loss_fn(output, microbatch_index)``.
                The index identifies the corresponding item in ``model_inputs``
                regardless of the schedule's execution order.
            losses: Mutable list populated by the schedule on the last stage.
            return_outputs: Whether the last stage returns merged model outputs
                when supported by the installed PyTorch version.

        Returns:
            The value returned by the underlying PyTorch pipeline schedule.

        Raises:
            NotImplementedError: If the installed PyTorch pipeline schedule
                does not provide forward-only ``eval`` execution.
        """
        return self._run_prepared_microbatches(
            model_inputs,
            loss_fn=loss_fn,
            losses=losses,
            return_outputs=return_outputs,
            schedule_method="eval",
            finalize_backward=True,
        )

    def _run_prepared_microbatches(
        self,
        model_inputs: list[dict[str, Any]],
        *,
        loss_fn: Callable[[Any, int], Any],
        losses: list[torch.Tensor] | None,
        return_outputs: bool,
        schedule_method: Literal["step", "eval"],
        finalize_backward: bool,
    ) -> Any:
        """Run one schedule method with an exact prepared-microbatch split."""
        schedule = self._info.schedule
        if schedule is None:
            raise RuntimeError("AutoPipeline.build() must be called before running a prepared PP schedule")
        if len(model_inputs) != self.num_microbatches:
            raise ValueError(f"Expected {self.num_microbatches} model input microbatches, got {len(model_inputs)}")

        model_args_chunks: list[tuple[Any, ...]] = []
        model_kwargs_chunks: list[dict[str, Any]] = []
        for index, inputs in enumerate(model_inputs):
            if not isinstance(inputs, dict):
                raise TypeError(f"model input microbatch {index} must be a dict")
            primary_names = [name for name in ("input_ids", "inputs_embeds") if name in inputs]
            if len(primary_names) != 1:
                raise ValueError(
                    f"model input microbatch {index} must contain exactly one of input_ids or inputs_embeds"
                )
            kwargs = dict(inputs)
            primary = kwargs.pop(primary_names[0])
            model_args_chunks.append((primary,) if self._info.has_first_stage else ())
            model_kwargs_chunks.append(kwargs)

        def indexed_loss(output: Any, microbatch_id: torch.Tensor) -> Any:
            return loss_fn(output, int(microbatch_id.item()))

        run_schedule = getattr(schedule, schedule_method, None)
        if not callable(run_schedule):
            if schedule_method == "eval":
                raise NotImplementedError(
                    "forward-only pipeline execution requires a PyTorch pipeline schedule with eval(); "
                    "upgrade PyTorch or use non-pipeline Engine.forward"
                )
            raise RuntimeError("PyTorch pipeline schedule has no callable step method")
        # ``schedule.eval`` forwards arbitrary kwargs to ``schedule.step`` and
        # does not list ``return_outputs`` explicitly in supported PyTorch
        # releases. Inspect step for both modes so eval can still suppress the
        # otherwise-unused merged output without breaking older step APIs.
        schedule_options = (
            {"return_outputs": return_outputs}
            if "return_outputs" in inspect.signature(schedule.step).parameters
            else {}
        )
        previous_split_inputs = schedule._split_inputs
        previous_loss_fn = schedule._loss_fn
        previous_stage_state: list[tuple[PipelineStage, object]] = []
        if schedule_method == "step":
            for stage in self._info.stages or ():
                previous_stage_state.append((stage, vars(stage).get("_nemo_finalize_backward", _MISSING_STAGE_STATE)))
                stage._nemo_finalize_backward = finalize_backward or getattr(
                    stage, "_reduce_grad_per_microbatch", False
                )
        schedule._split_inputs = lambda _args, _kwargs=None: (model_args_chunks, model_kwargs_chunks)
        schedule._loss_fn = indexed_loss
        try:
            return run_schedule(
                target=torch.arange(self.num_microbatches, device=self.device),
                losses=losses,
                **schedule_options,
            )
        finally:
            schedule._loss_fn = previous_loss_fn
            schedule._split_inputs = previous_split_inputs
            for stage, previous in previous_stage_state:
                if previous is _MISSING_STAGE_STATE:
                    del stage._nemo_finalize_backward
                else:
                    stage._nemo_finalize_backward = previous

    @property
    def parts(self) -> list[nn.Module]:
        if self._info.model_parts is None:
            raise RuntimeError("Autopipeline not built. Call build() first.")

        return self._info.model_parts

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_microbatches(self) -> int:
        """Number of pipeline microbatches in one local batch."""
        return self.pp_batch_size // self.pp_microbatch_size

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

    def visualize_current_schedule(self, filename: str | None = None) -> None:
        from torch.distributed.pipelining._schedule_visualizer import get_schedule_ops, visualize_schedule

        schedule = self._info.schedule
        ops = get_schedule_ops(schedule, self.pp_mesh.size(), self.pp_microbatch_size, len(self._info.stages))
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
