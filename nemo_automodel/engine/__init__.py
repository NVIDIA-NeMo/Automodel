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

"""A small forward/backward engine for Datum accumulation windows."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from math import prod
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.datasets.datum import (
    CollatedLossInputs,
    Datum,
    LossInputLayout,
    collate_datums,
)
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.models.common.mtp import MTPContextParallelInputs
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.utils import (
    get_expert_tp_replication_factor,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.engine.outputs import LossFnOutputBatch, PerTokenOutput
from nemo_automodel.shared.import_utils import MISSING_TORCHAO_MSG, safe_import_from

CollateFn = Callable[
    [list[Datum]],
    tuple[dict[str, Any], dict[str, torch.Tensor] | CollatedLossInputs],
]
LossInputValue = torch.Tensor | tuple[torch.Tensor, ...]
LossInputs = dict[str, LossInputValue]
BatchContextFn = Callable[[Mapping[str, Any], Mapping[str, LossInputValue]], AbstractContextManager[Any]]
LossFn = Callable[
    [Any, LossInputs],
    torch.Tensor | tuple[torch.Tensor, Sequence[Mapping[str, Any]] | LossFnOutputBatch],
]
ParsedLossOutputs = list[dict[str, Any]] | LossFnOutputBatch | None

_LOSS_FIELD_PREFIX = "__engine_loss__"
_LOSS_METADATA = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "padding_mask")
_T = TypeVar("_T")

__all__ = [
    "Engine",
    "ForwardBackwardResult",
    "ForwardResult",
    "LossFnOutputBatch",
    "OptimStepResult",
    "PerTokenOutput",
    "collate_prebatched",
]


def _nullcontext_for_batch(_model_inputs: dict[str, Any]) -> AbstractContextManager[Any]:
    return nullcontext()


def _nullcontext_for_prepared_batch(
    _model_inputs: Mapping[str, Any],
    _loss_fn_inputs: Mapping[str, LossInputValue],
) -> AbstractContextManager[Any]:
    """Return a no-op context for one prepared eager batch.

    Args:
        _model_inputs: CP-local model mapping with primary token shape
            ``[batch, sequence]`` or ``[tokens]``.
        _loss_fn_inputs: CP-local loss tensors with the same leading token
            axes for ``PER_TOKEN`` fields.

    Returns:
        A no-op context manager.
    """
    return nullcontext()


def _as_tuple(value: _T | Sequence[_T] | None) -> tuple[_T, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _uses_uniform_packed_token_dispatch(model_parts: Sequence[nn.Module]) -> bool:
    """Return whether a model part owns a live dispatcher that requires uniform token extents."""
    for part in model_parts:
        modules = getattr(part, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            dispatcher = getattr(module, "token_dispatcher", None)
            if getattr(dispatcher, "requires_uniform_token_count", False) is True:
                return True
    return False


def _owns_hybridep_packed_cp_equalization(model_parts: Sequence[nn.Module]) -> bool:
    """Return whether a model explicitly owns packed HybridEP equalization after CP preparation."""
    return any(
        bool(getattr(module, "owns_hybridep_packed_cp_equalization", False))
        for part in model_parts
        for module in (part.modules() if callable(getattr(part, "modules", None)) else ())
    )


def _resolve_hybridep_ep_group(
    uses_hybridep: bool,
    mesh_context: MeshContext | None,
) -> tuple[dist.ProcessGroup | None, int]:
    """Resolve the canonical EP group for a live HybridEP token dispatcher."""

    if not uses_hybridep:
        return None, 1
    moe_mesh = mesh_context.moe_mesh if mesh_context is not None else None
    mesh_names = getattr(moe_mesh, "mesh_dim_names", ()) or ()
    if moe_mesh is None or "ep" not in mesh_names:
        raise ValueError("a live HybridEP dispatcher requires mesh_context.moe_mesh with an 'ep' axis")
    ep_mesh = moe_mesh["ep"]
    ep_size = int(ep_mesh.size())
    if ep_size <= 1:
        raise ValueError("a live HybridEP dispatcher requires ep_size > 1")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("a live HybridEP dispatcher requires initialized torch.distributed process groups")
    return ep_mesh.get_group(), ep_size


def _resolve_summed_gradient_reduction(model_parts: Sequence[nn.Module]) -> bool:
    """Resolve whether every local model part uses summed gradient collectives.

    MegatronFSDP exposes ``calculate_per_token_loss`` on its wrapper. Walking
    each part's module tree also finds it below ordinary wrapper layers such as
    DDP or compilation wrappers. A part without a declaration uses the normal
    averaged-gradient contract.

    Args:
        model_parts: Local eager model or pipeline parts after distributed
            wrapping.

    Returns:
        ``True`` when every part declares summed gradients, otherwise ``False``.

    Raises:
        TypeError: If a declaration is not boolean.
        ValueError: If modules or model parts disagree on the reduction mode.
    """
    part_modes: list[bool] = []
    sentinel = object()
    for part_index, part in enumerate(model_parts):
        declared_modes: set[bool] = set()
        for module in part.modules():
            declared_mode = getattr(module, "calculate_per_token_loss", sentinel)
            if declared_mode is sentinel:
                continue
            if not isinstance(declared_mode, bool):
                raise TypeError(
                    "model calculate_per_token_loss declarations must be boolean; "
                    f"part {part_index} has {declared_mode!r}"
                )
            declared_modes.add(declared_mode)
        if len(declared_modes) > 1:
            raise ValueError(
                "model part mixes calculate_per_token_loss=True and False; "
                "Engine requires one gradient-reduction mode per optimizer window"
            )
        part_modes.append(next(iter(declared_modes), False))

    if len(set(part_modes)) > 1:
        raise ValueError(
            "model parts disagree on calculate_per_token_loss; "
            "Engine requires one gradient-reduction mode per optimizer window"
        )
    return bool(part_modes and part_modes[0])


def _resolve_fp8_scale_precompute(
    model_parts: Sequence[nn.Module],
) -> tuple[tuple[nn.Module, ...], Callable[[nn.Module], None] | None]:
    """Resolve the torchao post-step function only for opted-in model parts.

    Args:
        model_parts: Local eager model or pipeline parts after FP8 conversion
            and distributed model partitioning.

    Returns:
        Opted-in local model parts and the torchao precompute function. Both
        are empty when no part requests FP8 FSDP scale precomputation.

    Raises:
        ImportError: If an opted-in part requires a torchao API that is not
            available. Resolution happens during Engine construction, before
            any optimizer mutation.
    """
    capable_parts = tuple(
        part for part in model_parts if getattr(part, "precompute_float8_dynamic_scale_for_fsdp", False)
    )
    if not capable_parts:
        return (), None

    available, precompute = safe_import_from(
        "torchao.float8",
        "precompute_float8_dynamic_scale_for_fsdp",
        msg=MISSING_TORCHAO_MSG,
    )
    if not available:
        raise ImportError(MISSING_TORCHAO_MSG)
    return capable_parts, precompute


def collate_prebatched(datums: list[Datum]) -> tuple[dict[str, Any], CollatedLossInputs | dict[str, torch.Tensor]]:
    """Return one already-collated Datum without changing its layout.

    The Datum represents the whole prebatched item. Consequently, one
    optional ``loss_fn_output`` mapping also describes that whole batch, not
    each sample inside it. A typed token output likewise remains one record
    containing the full inner ``[B, S, ...]`` (or flat THD) tensor; the Engine
    cannot split hidden inner samples. Output records are rejected when PP
    divides that prebatched Datum into multiple inner microbatches.

    Args:
        datums: A one-item list whose model and loss tensor fields already have
            the batch layout expected by the model and loss callback.

    Returns:
        Separate shallow copies of the Datum's model-input and loss-input
        mappings. Tensor shapes, dtypes, devices, and storage are unchanged.
    """
    if len(datums) != 1:
        raise ValueError("collate_prebatched expects exactly one Datum per microbatch")
    datum = datums[0]
    if set(datum.loss_fn_input_layouts) == set(datum.loss_fn_inputs):
        loss_inputs: CollatedLossInputs | dict[str, torch.Tensor] = CollatedLossInputs(
            datum.loss_fn_inputs,
            layouts=datum.loss_fn_input_layouts,
            item_to_datum=None,
            pad_values=datum.loss_fn_input_pad_values,
        )
    else:
        if datum.loss_fn_input_pad_values:
            raise ValueError("prebatched loss input pad values require an explicit layout for every loss field")
        # Source-compatible prebatched callers without complete metadata keep
        # the legacy inference path. It remains fail-closed for ambiguous
        # packed per-Datum fields.
        loss_inputs = dict(datum.loss_fn_inputs)
    return dict(datum.model_inputs), loss_inputs


@dataclass(frozen=True)
class _LossBatchLayout:
    """Field semantics and logical-item routing captured at collate time."""

    fields: Mapping[str, LossInputLayout]
    item_to_datum: tuple[int, ...] | None
    pad_values: Mapping[str, float | int | bool]
    unresolved_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _OutputRestorePlan:
    """How to turn one CP-local callback token stream back into Datums."""

    sharder: ContextParallelSharder
    is_thd: bool
    item_to_datum: tuple[int, ...] | None
    real_lengths: tuple[int, ...] | None
    padded_lengths: tuple[int, ...] | None
    token_mask: torch.Tensor | None
    synthetic_suffix_tokens: int = 0


@dataclass(frozen=True)
class ForwardResult:
    """Forward-only loss statistics and per-Datum outputs.

    ``loss_sum`` and ``weight_sum`` are complete across model-parallel CP and
    PP ranks, but remain local to one data-parallel replica. The Engine adds no
    per-call DP loss-statistic collective, so callers reduce the two sums once
    at the end of an evaluation epoch. Distributed model wrappers may still
    communicate during forward and therefore retain their own call-alignment
    requirements. ``loss_fn_outputs`` remains local to the replica's input
    Datums. PP stages receive identical detached copies. Legacy output mappings
    remain opaque and therefore CP-local; fields declared through
    :class:`LossFnOutputBatch` are restored to full token order across CP before
    being split back into per-Datum records.

    Attributes:
        loss_sum: Detached weighted numerator for this Datum window.
        weight_sum: Detached full-sequence weight denominator for this window.
        loss_fn_outputs: Detached per-Datum mappings in input order.
    """

    loss_sum: torch.Tensor
    weight_sum: torch.Tensor
    loss_fn_outputs: list[dict[str, Any]]


@dataclass(frozen=True)
class ForwardBackwardResult:
    """One complete optimizer window's loss statistics and callback outputs.

    The numerator is summed across the DP-CP gradient group. The full-sequence
    denominator is summed across DP only because CP ranks begin with replicated
    weights; both are synchronized across PP stages. ``loss`` is
    ``loss_sum / weight_sum`` when the denominator is nonzero and zero
    otherwise. ``loss_fn_outputs`` remains local to one data-parallel replica,
    is restored to full token order across CP when explicitly typed, and is
    identical on every PP stage in that replica.

    Attributes:
        loss: Detached weighted mean for the complete Datum window.
        loss_sum: Detached numerator summed across DP and CP, then synchronized
            across PP stages.
        weight_sum: Detached window denominator, summed across DP but
            not CP, then synchronized across PP stages.
        loss_fn_outputs: Detached per-Datum mappings in input order.
    """

    loss: torch.Tensor
    loss_sum: torch.Tensor
    weight_sum: torch.Tensor
    loss_fn_outputs: list[dict[str, Any]]


@dataclass(frozen=True)
class OptimStepResult:
    """Statistics from one completed optimizer step.

    Attributes:
        grad_norm: Gradient norm reported before clipping. This is a scalar
            tensor on the gradients' device, or ``0.0`` when clipping is
            disabled by ``max_grad_norm=None``.
        learning_rates: Learning rates of every optimizer parameter group after
            the configured schedulers advance.
    """

    grad_norm: torch.Tensor | float
    learning_rates: tuple[float, ...]


class Engine:
    """Run model forward or forward/backward over Datum windows.

    The model and distributed topology are already constructed when they are
    passed here. The Engine owns batching and model-parallel execution.
    :meth:`forward` performs evaluation without gradients;
    :meth:`forward_backward` additionally owns global weight normalization,
    gradient-accumulation synchronization, and backward. When optimizers are
    provided, :meth:`optim_step` owns distributed gradient finalization,
    clipping, parameter updates, gradient clearing, model post-step hooks, and
    LR-scheduler advancement. One :meth:`forward_backward` call is the complete
    optimizer accumulation window consumed by :meth:`optim_step`. Dynamic loss
    scaling and overflow-skipped updates are not part of this contract.

    Args:
        model: An already configured and distributed model, or a built
            :class:`AutoPipeline`.
        device: Device on which model inputs and losses are evaluated.
        mesh_context: Runtime topology. Required for an ``AutoPipeline``. For
            eager models, an initialized default process group is treated as
            pure data parallelism when omitted.
        microbatch_size: Number of Datum items collated into each outer batch.
            Use one with :func:`collate_prebatched`.
        collate_fn: Batches one microbatch of Datums into separate model and
            loss inputs. The default supports padded and packed text. VLMs pass
            a model-specific collater. Existing recipes whose dataloaders
            already collate can use :func:`collate_prebatched`; the Engine then
            applies any remaining CP/THD preparation. The callable must keep
            model inputs and loss inputs aligned and preserve the sum of
            ``weights``.
        padding_token_id: Token used when the CP sharder pads ``input_ids``.
        mtp_ignore_index: Label value used for invalid globally shifted MTP
            targets before context-parallel sharding.
        context_fn: Creates an optional context from the CP-prepared model-input
            mapping. It covers model forward, loss, and backward, or the full
            pipeline schedule. Recipes use it for FP8 and model input staging.
        batch_context_fn: Creates an optional context from the CP/packing-
            prepared model-input and loss/side-channel mappings. It covers
            eager model forward, loss, backward, and activation-checkpoint
            recomputation. With AutoPipeline, the factory is selected by the
            logical inner-microbatch id and entered separately for every stage
            forward, loss, and backward phase; it must therefore be repeatable.
            Rank-local callback failures are process-fatal in distributed
            execution, like failures from ``context_fn`` or model forward.
        defer_fsdp_grad_sync: Defer FSDP/DDP gradient synchronization until the
            final microbatch.
        optimizers: Already-built optimizer or optimizers for these model parts.
            The Engine retains the same objects; it does not build or copy them.
        lr_schedulers: Already-built optimizer parameter scheduler or schedulers.
            They advance once after a completed optimizer update.
        max_grad_norm: Maximum gradient norm. ``None`` preserves gradients
            without clipping while still running distributed expert-gradient
            finalization.

    Note:
        Context-parallel input layout and transport are delegated to
        :class:`ContextParallelSharder`. With an :class:`AutoPipeline`, the
        pipeline schedule owns execution order and backward calls; the Engine
        owns exact microbatch materialization, the complete outer accumulation
        window, and loss normalization. Pipeline output mappings are
        synchronized to every physical PP rank. Magi's current packed contract
        uses one inner pipeline microbatch; recipes enforce that configuration.
        Packed pipeline batches with multiple inner microbatches must split at
        sequence boundaries into equal-width token chunks. Token-aligned loss
        fields follow those chunks. Layout-aware collaters may additionally
        route per-Datum scalar fields from THD sequence boundaries and preserve
        replicated fields unchanged. A ``PER_DATUM`` field is copied to every
        CP rank; a loss callback combines it with that rank's CP-local token
        contribution so scalar numerators remain additive across CP. Custom or
        prebatched collaters that hide the sequence-to-Datum relationship
        remain fail-closed for per-Datum fields instead of guessing inner
        sample boundaries.
    """

    def __init__(
        self,
        model: nn.Module | AutoPipeline,
        *,
        device: torch.device | str,
        mesh_context: MeshContext | None = None,
        microbatch_size: int = 1,
        collate_fn: CollateFn = collate_datums,
        padding_token_id: int = 0,
        mtp_ignore_index: int = -100,
        context_fn: Callable[[dict[str, Any]], AbstractContextManager[Any]] = _nullcontext_for_batch,
        batch_context_fn: BatchContextFn | None = None,
        defer_fsdp_grad_sync: bool = True,
        optimizers: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None = None,
        lr_schedulers: OptimizerParamScheduler | Sequence[OptimizerParamScheduler] | None = None,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be a positive integer, got {microbatch_size!r}")
        if isinstance(mtp_ignore_index, bool) or not isinstance(mtp_ignore_index, int):
            raise ValueError(f"mtp_ignore_index must be an integer, got {mtp_ignore_index!r}")
        self.pipeline = model if isinstance(model, AutoPipeline) else None
        self.model_parts = model.parts if self.pipeline is not None else [model]
        self.model = self.model_parts[0]
        self._summed_gradient_reduction = _resolve_summed_gradient_reduction(self.model_parts)
        self._fp8_scale_precompute_parts, self._fp8_scale_precompute_fn = _resolve_fp8_scale_precompute(
            self.model_parts
        )
        self.device = torch.device(device)
        self.mesh_context = mesh_context
        uses_hybridep = _uses_uniform_packed_token_dispatch(self.model_parts)
        pipeline_uses_hybridep = uses_hybridep
        if (
            self.pipeline is not None
            and mesh_context is not None
            and mesh_context.pp_size > 1
            and getattr(mesh_context, "ep_size", 1) > 1
        ):
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("pipeline HybridEP capability validation requires initialized process groups")
            any_hybridep = torch.tensor(int(uses_hybridep), dtype=torch.int64, device=self.device)
            dist.all_reduce(any_hybridep, op=dist.ReduceOp.MAX, group=self.pipeline.pp_mesh.get_group())
            pipeline_uses_hybridep = int(any_hybridep.item()) != 0
        self._pipeline_uses_hybridep = pipeline_uses_hybridep
        self._hybridep_ep_group, self._hybridep_ep_size = _resolve_hybridep_ep_group(
            uses_hybridep,
            mesh_context,
        )
        self._model_owns_hybridep_packed_cp_equalization = _owns_hybridep_packed_cp_equalization(self.model_parts)
        self.microbatch_size = microbatch_size
        self.collate_fn = collate_fn
        self.padding_token_id = padding_token_id
        self.mtp_ignore_index = mtp_ignore_index
        self.context_fn = context_fn
        self.batch_context_fn = _nullcontext_for_prepared_batch if batch_context_fn is None else batch_context_fn
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync
        self.optimizers = _as_tuple(optimizers)
        self.lr_schedulers = _as_tuple(lr_schedulers)
        self.max_grad_norm = max_grad_norm
        self._optim_step_consumed = False
        self._grads_finalized = False
        self._finalized_grad_norm: torch.Tensor | float | None = None
        self._optim_step_in_progress = False
        self._backward_status = "idle"

    @torch.no_grad()
    def forward(
        self,
        datums: Sequence[Datum],
        loss_fn: LossFn,
    ) -> ForwardResult:
        """Run a forward-only Datum window.

        The Engine groups and collates the flat Datum sequence exactly as in
        :meth:`forward_backward`, then applies the same device movement,
        context-parallel layout, packed metadata, batch contexts, and pipeline
        microbatch materialization. Model parts run in evaluation mode and no
        autograd graph, gradient synchronization, or backward lifecycle is
        created.

        Loss statistics are reduced only across model-parallel CP and PP
        ranks. They deliberately remain local to one DP replica, avoiding an
        Engine-introduced per-call DP loss collective; callers perform one
        dataset-level DP reduction afterwards. Distributed wrappers such as
        DDP/FSDP may still require aligned forward calls for their own model
        communication.

        Args:
            datums: Flat sequence of Datum items for this forward-only window.
            loss_fn: Computes a per-element loss tensor or scalar local
                weighted numerator from the raw model output and CP-local loss
                inputs. It may additionally return opaque mappings per Datum,
                or a :class:`LossFnOutputBatch` whose explicitly typed token
                streams the Engine restores across CP.

        Returns:
            Detached model-parallel-complete loss statistics and per-Datum
            outputs. The sums and outputs are local to one data-parallel
            replica. A prebatched Datum produces one outer record, not one
            record per hidden inner sample, and cannot produce records when PP
            splits it into multiple inner microbatches.
        """
        microbatches = self._group_datums(datums)
        self._validate_execution_parallelism()
        cp_group, cp_size = self._cp_group_and_size()
        self._validate_window_size_across_group(len(microbatches), cp_group, cp_size)
        weight_sum = self._local_weight_sum(microbatches)
        zero_weight_sum = bool(weight_sum == 0)
        self._validate_pipeline_window(len(microbatches), weight_sum)

        for part in self.model_parts:
            part.eval()
        inner_microbatches = self.pipeline.num_microbatches if self.pipeline is not None else 1
        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_fn_outputs: list[dict[str, Any]] = []
        returns_outputs: bool | None = None

        for batch_datums in microbatches:
            cp_context, model_inputs, loss_inputs, loss_batch_layout, output_restore_plan = self._prepare_batch(
                batch_datums, inner_microbatches
            )
            if self.pipeline is not None:
                batch_returns_outputs, batch_outputs, batch_error = self._pipeline_execute(
                    model_inputs,
                    loss_inputs,
                    batch_datums,
                    loss_fn,
                    local_loss_sum,
                    cp_context,
                    loss_batch_layout,
                    output_restore_plan,
                    backward_scale=None,
                    zero_weight_sum=zero_weight_sum,
                )
                if batch_error is not None:
                    raise batch_error
                if batch_returns_outputs is not None:
                    if returns_outputs is None:
                        returns_outputs = batch_returns_outputs
                    elif returns_outputs != batch_returns_outputs:
                        raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
                    loss_fn_outputs.extend(batch_outputs)
                continue

            loss_inputs = _with_loss_metadata(model_inputs, loss_inputs)
            with self.context_fn(model_inputs), cp_context(), self.batch_context_fn(model_inputs, loss_inputs):
                forward_inputs = filter_forward_kwargs(self.model, model_inputs)
                output = self.model(**forward_inputs)
                numerator, parsed_outputs, output_parse_error = _parse_loss_result(
                    loss_fn(output, loss_inputs), loss_inputs["weights"]
                )
                self._validate_loss_fn_outputs_across_cp(
                    parsed_outputs,
                    loss_inputs.get("weights"),
                    expected_records=len(batch_datums),
                    local_error=output_parse_error,
                    restore_plan=output_restore_plan,
                    datum_indices=tuple(range(len(batch_datums))),
                )
                returns_outputs = _update_output_mode(returns_outputs, parsed_outputs)
                if zero_weight_sum:
                    numerator = numerator * 0
            if parsed_outputs is not None:
                outputs = self._restore_loss_fn_outputs(
                    parsed_outputs,
                    loss_inputs,
                    output_restore_plan,
                    datum_indices=tuple(range(len(batch_datums))),
                    chunk_index=None,
                )
                if len(outputs) != len(batch_datums):
                    raise ValueError("loss_fn outputs must contain one mapping per Datum")
                loss_fn_outputs.extend(outputs)
            local_loss_sum.add_(numerator.detach().to(torch.float64))

        if cp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=cp_group)
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=pp_group)

        return ForwardResult(
            loss_sum=local_loss_sum.detach(),
            weight_sum=weight_sum.detach(),
            loss_fn_outputs=loss_fn_outputs,
        )

    def forward_backward(
        self,
        datums: Sequence[Datum],
        loss_fn: LossFn,
    ) -> ForwardBackwardResult:
        """Run one complete optimizer accumulation window.

        A second call with configured optimizers is rejected until
        :meth:`optim_step` consumes the first call's gradients. A failed
        partial backward poisons the window so its gradients cannot be reused.
        """
        if self.optimizers:
            if self._backward_status == "ready":
                raise RuntimeError(
                    "forward_backward already produced the current optimizer window; call optim_step first"
                )
            if self._backward_status == "broken":
                raise RuntimeError("the previous backward window failed and this Engine cannot be reused")
            if self._backward_status == "running":
                raise RuntimeError("a forward_backward call is already running")
        if self._grads_finalized:
            raise RuntimeError("gradients were already finalized; retry optim_step before forward_backward")
        if self.optimizers:
            self._backward_status = "running"
        try:
            result = self._forward_backward_window(datums, loss_fn)
        except BaseException:
            if self.optimizers:
                self._backward_status = "broken"
            raise
        self._optim_step_consumed = False
        self._grads_finalized = False
        self._finalized_grad_norm = None
        if self.optimizers:
            self._backward_status = "ready"
        return result

    def _forward_backward_window(
        self,
        datums: Sequence[Datum],
        loss_fn: LossFn,
    ) -> ForwardBackwardResult:
        """Accumulate gradients for a complete optimizer window.

        ``datums`` is a flat optimizer accumulation window. The Engine groups
        it into outer batches of ``microbatch_size`` and invokes ``collate_fn``
        once per group. A Datum normally represents one sample. Recipes that
        retain worker-side collation instead wrap each prepared batch in one
        Datum and configure ``microbatch_size=1`` with
        :func:`collate_prebatched`.

        ``loss_fn`` receives the raw model output and CP-local
        ``loss_fn_inputs``. It returns either per-element losses with exactly
        the same shape as
        ``loss_fn_inputs["weights"]``, or a scalar local weighted-sum
        numerator. For a scalar, the callback must apply weights and masks;
        the Engine will only apply global normalization. The callback may
        also return one opaque output mapping per Datum, or a
        :class:`LossFnOutputBatch`. Explicit token streams in that envelope are
        detached, restored from CP-local to full token order, and then split
        back into per-Datum records; ordinary mappings remain opaque. Under
        pipeline parallelism the last stage performs CP restoration before it
        broadcasts the records to every stage in the pipeline group.
        Values in pipeline output records must therefore be pickle-compatible;
        large records also incur CPU serialization and PP broadcast cost.

        A prebatched Datum intentionally hides its inner sample boundaries.
        It can therefore return a single output mapping only when a pipeline
        schedule has one inner microbatch. Callers that need one output per
        sample use ordinary flat Datums.

        Args:
            datums: Flat sequence of Datum items in the complete optimizer
                accumulation window. A Datum's token weights may have shape
                [tokens] or the custom collater's batched token layout; the
                loss tensor must use the identical shape.
            loss_fn: Computes either that per-token loss tensor or a scalar
                local weighted-sum numerator from the raw model output and
                collated loss inputs.

        Returns:
            Structured loss statistics and callback outputs. ``loss_sum`` is
            reduced across the DP-CP gradient group, while ``weight_sum`` is
            reduced across DP only so replicated CP weights are not counted
            twice. Both are synchronized across PP stages, and ``loss`` is
            their safe quotient.
            ``loss_fn_outputs`` contains mappings for this DP replica's outer
            Datums in window order; pipeline execution returns the same
            mappings on every physical stage rank in that replica. Model
            parameters are unchanged, but their gradients contain the complete
            window's globally normalized backward result.
        """
        microbatches = self._group_datums(datums)
        self._validate_parallelism()
        dp_group, dp_size = self._dp_group_and_size()
        grad_group, grad_group_size = self._gradient_group_and_size(dp_group, dp_size)
        gradient_reduction_multiplier = 1 if self._summed_gradient_reduction else grad_group_size
        self._validate_window_size_across_group(len(microbatches), grad_group, grad_group_size)
        denominator = self._global_weight_sum(microbatches, dp_group, dp_size)
        zero_denominator = bool(denominator == 0)
        safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        self._validate_pipeline_window(len(microbatches), denominator)

        pp_enabled = self.pipeline is not None
        for part in self.model_parts:
            part.train()
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=pp_enabled)
        inner_microbatches = self.pipeline.num_microbatches if self.pipeline is not None else 1
        effective_total_microbatches = len(microbatches) * inner_microbatches
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            self._cp_size() * gradient_reduction_multiplier / (grad_group_size * effective_total_microbatches)
        )

        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_fn_outputs: list[dict[str, Any]] = []
        returns_outputs: bool | None = None
        output_error: Exception | None = None
        backward_scale = (
            safe_denominator.new_zeros(())
            if zero_denominator
            else safe_denominator.new_tensor(gradient_reduction_multiplier) / safe_denominator
        )

        for index, datums in enumerate(microbatches):
            is_last = index == len(microbatches) - 1
            if is_last:
                prepare_for_final_backward(self.model_parts, pp_enabled=pp_enabled)

            cp_context, model_inputs, loss_inputs, loss_batch_layout, output_restore_plan = self._prepare_batch(
                datums, inner_microbatches
            )

            if self.pipeline is not None:
                batch_returns_outputs, batch_outputs, batch_error = self._pipeline_execute(
                    model_inputs,
                    loss_inputs,
                    datums,
                    loss_fn,
                    local_loss_sum,
                    cp_context,
                    loss_batch_layout,
                    output_restore_plan,
                    backward_scale=backward_scale,
                    zero_weight_sum=zero_denominator,
                )
                if output_error is None:
                    if batch_error is not None:
                        output_error = batch_error
                    else:
                        try:
                            if batch_returns_outputs is not None:
                                if returns_outputs is None:
                                    returns_outputs = batch_returns_outputs
                                elif returns_outputs != batch_returns_outputs:
                                    raise ValueError(
                                        "loss_fn must return per-Datum outputs for every microbatch or none of them"
                                    )
                                loss_fn_outputs.extend(batch_outputs)
                        except Exception as error:
                            output_error = error
            else:
                loss_inputs = _with_loss_metadata(model_inputs, loss_inputs)
                with (
                    get_sync_ctx(self.model, is_last, self.defer_fsdp_grad_sync),
                    self.context_fn(model_inputs),
                    cp_context(),
                    self.batch_context_fn(model_inputs, loss_inputs),
                ):
                    forward_inputs = filter_forward_kwargs(self.model, model_inputs)
                    output = self.model(**forward_inputs)
                    numerator, parsed_outputs, output_parse_error = _parse_loss_result(
                        loss_fn(output, loss_inputs), loss_inputs["weights"]
                    )
                    if output_error is None and dp_size <= 1:
                        self._validate_loss_fn_outputs_across_cp(
                            parsed_outputs,
                            loss_inputs.get("weights"),
                            expected_records=len(datums),
                            local_error=output_parse_error,
                            restore_plan=output_restore_plan,
                            datum_indices=tuple(range(len(datums))),
                        )
                        returns_outputs = _update_output_mode(returns_outputs, parsed_outputs)
                    if zero_denominator:
                        numerator = numerator * 0
                    (numerator * backward_scale).backward()

                if output_error is None:
                    try:
                        if dp_size > 1:
                            self._validate_loss_fn_outputs_across_cp(
                                parsed_outputs,
                                loss_inputs.get("weights"),
                                expected_records=len(datums),
                                local_error=output_parse_error,
                                restore_plan=output_restore_plan,
                                datum_indices=tuple(range(len(datums))),
                            )
                            returns_outputs = _update_output_mode(returns_outputs, parsed_outputs)
                        if parsed_outputs is not None:
                            outputs = self._restore_loss_fn_outputs(
                                parsed_outputs,
                                loss_inputs,
                                output_restore_plan,
                                datum_indices=tuple(range(len(datums))),
                                chunk_index=None,
                            )
                            if len(outputs) != len(datums):
                                raise ValueError("loss_fn outputs must contain one mapping per Datum")
                            loss_fn_outputs.extend(outputs)
                    except Exception as error:
                        output_error = error
                local_loss_sum.add_(numerator.detach().to(torch.float64))
            if index == 0:
                prepare_after_first_microbatch()

        # Piggyback the output-error bit on the existing end-of-window loss
        # reductions. Every gradient rank therefore finishes backward before
        # any replica raises a data-local output-routing error.
        step_state = torch.stack((local_loss_sum, local_loss_sum.new_tensor(int(output_error is not None))))
        if grad_group_size > 1:
            dist.all_reduce(step_state, op=dist.ReduceOp.SUM, group=grad_group)
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size > 1:
            dist.all_reduce(step_state, op=dist.ReduceOp.SUM, group=pp_group)
        if bool(step_state[1] > 0):
            if output_error is not None:
                raise output_error
            raise RuntimeError("another model-parallel rank failed while restoring loss_fn outputs")

        loss_sum = step_state[0].detach()
        loss = (loss_sum / safe_denominator).detach()
        return ForwardBackwardResult(
            loss=loss,
            loss_sum=loss_sum,
            weight_sum=denominator.detach(),
            loss_fn_outputs=loss_fn_outputs,
        )

    @torch.no_grad()
    def optim_step(
        self,
        *,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> OptimStepResult:
        """Finalize accumulated gradients and perform one optimizer update.

        Gradient normalization performed by :meth:`forward_backward` is not
        repeated here. This method applies the repository's model-parallel
        expert-gradient correction and global clipping once, then invokes an
        optional mutation fence before any optimizer changes. Async
        checkpointers use that fence to preserve ``finalize/clip -> wait ->
        step`` overlap. After updating weights, it runs model maintenance for
        MoE gate bias and opted-in FP8 FSDP scale precomputation before
        advancing LR schedulers.

        Once the first optimizer mutation begins, failures from an optimizer,
        model post-step hook, or scheduler cannot be rolled back and are
        process-fatal in distributed execution.

        Args:
            before_optimizer_step: Optional callback invoked exactly once after
                gradient finalization and clipping, but before the first
                optimizer step. If it raises, parameters, optimizer state,
                model post-step state, and schedulers remain untouched.

        Returns:
            Gradient norm and post-scheduler learning rates for the completed
            optimizer update.

        Raises:
            RuntimeError: If this Engine was constructed without optimizers.
        """
        if not self.optimizers:
            raise RuntimeError("Engine.optim_step requires at least one optimizer")
        if before_optimizer_step is not None and not callable(before_optimizer_step):
            raise TypeError("before_optimizer_step must be callable or None")
        if self._optim_step_in_progress:
            raise RuntimeError("Engine.optim_step is already running")
        if self._backward_status == "broken":
            raise RuntimeError("the previous backward window failed and this Engine cannot be optimized")
        if self._backward_status == "running":
            raise RuntimeError("a forward_backward call is still running")
        if self._optim_step_consumed:
            raise RuntimeError("optim_step already consumed the current gradients; run forward_backward first")

        device_mesh = self.mesh_context.device_mesh if self.mesh_context is not None else None
        moe_mesh = self.mesh_context.moe_mesh if self.mesh_context is not None else None
        dp_group, dp_size = self._dp_group_and_size()
        _, grad_group_size = self._gradient_group_and_size(dp_group, dp_size)
        pp_enabled = self.pipeline is not None
        self._optim_step_in_progress = True
        mutation_started = False
        try:
            if not self._grads_finalized:
                self._finalized_grad_norm = scale_grads_and_clip_grad_norm(
                    max_grad_norm=self.max_grad_norm,
                    model_parts=self.model_parts,
                    norm_type=2.0,
                    pp_enabled=pp_enabled,
                    device_mesh=device_mesh,
                    moe_mesh=moe_mesh,
                    ep_axis_name="ep" if moe_mesh is not None and "ep" in (moe_mesh.mesh_dim_names or ()) else None,
                    pp_axis_name="pp" if pp_enabled else None,
                    foreach=True,
                    num_label_tokens=None,
                    dp_group_size=grad_group_size,
                    expert_tp_replication_factor=get_expert_tp_replication_factor(self.model_parts, device_mesh),
                )
                if self._finalized_grad_norm is None:
                    raise RuntimeError("gradient finalization did not return a gradient norm")
                self._grads_finalized = True
            grad_norm = self._finalized_grad_norm
            assert grad_norm is not None

            if before_optimizer_step is not None:
                before_optimizer_step()

            mutation_started = True
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for part in self.model_parts:
                update_moe_gate_bias = getattr(part, "update_moe_gate_bias", None)
                if callable(update_moe_gate_bias):
                    update_moe_gate_bias()

            if self._fp8_scale_precompute_fn is not None:
                for part in self._fp8_scale_precompute_parts:
                    self._fp8_scale_precompute_fn(part)

            for scheduler in self.lr_schedulers:
                scheduler.step(1)

            learning_rates = tuple(
                float(group["lr"]) for optimizer in self.optimizers for group in optimizer.param_groups
            )
        except Exception:
            if mutation_started or not self._grads_finalized:
                self._optim_step_consumed = True
                self._backward_status = "broken"
            raise
        finally:
            self._optim_step_in_progress = False

        self._optim_step_consumed = True
        self._grads_finalized = False
        self._finalized_grad_norm = None
        self._backward_status = "idle"
        return OptimStepResult(grad_norm=grad_norm, learning_rates=learning_rates)

    def _group_datums(self, datums: Sequence[Datum]) -> list[list[Datum]]:
        if not isinstance(datums, Sequence) or isinstance(datums, (str, bytes)) or not datums:
            raise ValueError("Engine requires a non-empty flat sequence of Datum")
        if not all(isinstance(datum, Datum) for datum in datums):
            raise TypeError("Engine received a value that is not a Datum")
        return [
            list(datums[start : start + self.microbatch_size]) for start in range(0, len(datums), self.microbatch_size)
        ]

    def _hybridep_packed_token_target(self, model_inputs: Mapping[str, Any]) -> int | None:
        """Return the EP-wide raw-THD token width required by HybridEP.

        Every rank enters one fixed-size metadata collective before local THD
        validation. This makes a packed/non-packed or raw/final-THD mismatch a
        group-wide error instead of letting one rank enter HybridEP with a
        different routing-map shape. This path deliberately owns only the
        CP=1 layout used by MOLT and the canonical Datum collaters; DSV4 keeps
        its existing model-owned CP>1 equalization path.
        """
        if self.pipeline is not None and self._pipeline_uses_hybridep:
            if model_inputs.get("qkv_format") == "thd":
                raise NotImplementedError("HybridEP packed token equalization currently supports eager PP size 1 only")
            return None
        if self._hybridep_ep_group is None:
            return None

        packed = model_inputs.get("qkv_format") == "thd"
        primary_names = [name for name in ("input_ids", "inputs_embeds") if name in model_inputs]
        primary = model_inputs.get(primary_names[0]) if len(primary_names) == 1 else None
        primary_valid = isinstance(primary, torch.Tensor) and (
            (primary_names == ["input_ids"] and primary.ndim == 2)
            or (primary_names == ["inputs_embeds"] and primary.ndim == 3)
        )
        batch_size = int(primary.shape[0]) if primary_valid else 0
        token_width = int(primary.shape[1]) if primary_valid else 0
        raw_thd = packed and all(
            isinstance(model_inputs.get(name), torch.Tensor) for name in ("seq_lens", "seq_lens_padded")
        )
        local = torch.tensor(
            (int(packed), int(raw_thd), int(primary_valid), batch_size, token_width),
            dtype=torch.int64,
            device=self.device,
        )
        gathered = [torch.empty_like(local) for _ in range(self._hybridep_ep_size)]
        dist.all_gather(gathered, local, group=self._hybridep_ep_group)
        metadata = torch.stack(gathered).cpu().tolist()

        packed_flags = {row[0] for row in metadata}
        if len(packed_flags) != 1:
            raise ValueError("HybridEP ranks must all use packed THD inputs or all use non-packed inputs")
        if packed_flags == {0}:
            return None
        if self._cp_size() > 1:
            if self._model_owns_hybridep_packed_cp_equalization:
                return None
            raise NotImplementedError(
                "generic HybridEP packed token equalization currently supports CP size 1 only; "
                "DSV4 retains its model-owned CP equalization path"
            )
        if any(row[1] != 1 for row in metadata):
            raise NotImplementedError(
                "HybridEP packed token equalization currently requires raw THD seq_lens/seq_lens_padded inputs"
            )
        if any(row[2] != 1 or row[3] != 1 or row[4] <= 0 for row in metadata):
            raise ValueError(
                "HybridEP packed token equalization requires one non-empty raw THD token row per EP rank; "
                f"got metadata {metadata}"
            )
        return max(row[4] for row in metadata)

    @staticmethod
    def _resolve_loss_batch_layout(
        datums: list[Datum],
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, LossInputValue],
    ) -> _LossBatchLayout:
        """Resolve collater metadata without exposing it to the loss callback."""
        datum_layouts: dict[str, LossInputLayout] = {}
        for name in sorted({name for datum in datums for name in datum.loss_fn_input_layouts}):
            declared = [datum.loss_fn_input_layouts[name] for datum in datums if name in datum.loss_fn_input_layouts]
            if len(declared) != len(datums):
                raise ValueError(f"every Datum must declare the loss layout for field {name!r}")
            if any(layout != declared[0] for layout in declared[1:]):
                raise ValueError(f"Datum items disagree on the loss layout for field {name!r}")
            datum_layouts[name] = declared[0]

        datum_pad_values: dict[str, float | int | bool] = {}
        for name in sorted({name for datum in datums for name in datum.loss_fn_input_pad_values}):
            declared = [
                datum.loss_fn_input_pad_values[name] for datum in datums if name in datum.loss_fn_input_pad_values
            ]
            if len(declared) != len(datums):
                raise ValueError(f"every Datum must declare the loss pad value for field {name!r}")
            if any(value != declared[0] for value in declared[1:]):
                raise ValueError(f"Datum items disagree on the loss pad value for field {name!r}")
            datum_pad_values[name] = declared[0]

        missing_fields = (set(datum_layouts) | set(datum_pad_values)) - set(loss_inputs)
        if missing_fields:
            raise ValueError(f"collate_fn dropped explicitly declared loss fields: {sorted(missing_fields)}")

        if isinstance(loss_inputs, CollatedLossInputs):
            if set(loss_inputs.layouts) != set(loss_inputs):
                raise ValueError("CollatedLossInputs.layouts must describe every loss field exactly once")
            for name, layout in datum_layouts.items():
                if loss_inputs.layouts[name] is not layout:
                    raise ValueError(
                        f"CollatedLossInputs layout for field {name!r} disagrees with its Datum declaration"
                    )
            for name, pad_value in datum_pad_values.items():
                if name not in loss_inputs.pad_values or loss_inputs.pad_values[name] != pad_value:
                    raise ValueError(
                        f"CollatedLossInputs pad value for field {name!r} disagrees with its Datum declaration"
                    )
            item_to_datum = loss_inputs.item_to_datum
            if item_to_datum is not None and item_to_datum != tuple(range(len(datums))):
                raise ValueError(
                    "Engine currently requires collater item_to_datum to preserve outer Datum order; "
                    f"got {list(item_to_datum)} for {len(datums)} Datums"
                )
            return _LossBatchLayout(
                fields=dict(loss_inputs.layouts),
                item_to_datum=item_to_datum,
                pad_values=dict(loss_inputs.pad_values),
            )

        if datum_pad_values:
            raise ValueError(
                "custom collaters must return CollatedLossInputs to preserve "
                f"loss_fn_input_pad_values for fields {sorted(datum_pad_values)}"
            )

        weights = loss_inputs.get("weights")
        if not isinstance(weights, torch.Tensor):
            raise ValueError("collate_fn must return a Tensor loss input named 'weights'")

        fields: dict[str, LossInputLayout] = {}
        unresolved: set[str] = set()
        for name, value in loss_inputs.items():
            if name in datum_layouts:
                fields[name] = datum_layouts[name]
                continue

            if isinstance(value, torch.Tensor) and _loss_sequence_dim(dict(model_inputs), value) is not None:
                fields[name] = LossInputLayout.PER_TOKEN
                continue

            # Preserve source compatibility for older prepared/custom batches
            # whose output weights are not token-shaped. This is deliberately
            # unresolved rather than a new public PER_DATUM-weight contract:
            # padded PP keeps its historical shape slicing, while packed PP
            # fails closed without explicit token weights.
            if name == "weights":
                fields[name] = LossInputLayout.REPLICATED
                unresolved.add(name)
                continue

            datum_values = [datum.loss_fn_inputs.get(name) for datum in datums]
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == len(datums)
                and all(isinstance(item, torch.Tensor) and item.numel() == 1 for item in datum_values)
            ):
                fields[name] = LossInputLayout.PER_DATUM
            else:
                fields[name] = LossInputLayout.REPLICATED
            unresolved.add(name)

        # Legacy padded collaters historically preserve one input row per
        # Datum. Keep that path source compatible; packed collaters must opt in
        # explicitly because a THD sequence is not necessarily an outer Datum.
        item_to_datum: tuple[int, ...] | None = None
        primary = model_inputs.get("inputs_embeds", model_inputs.get("input_ids"))
        if (
            model_inputs.get("qkv_format") != "thd"
            and isinstance(primary, torch.Tensor)
            and primary.ndim > 0
            and primary.shape[0] == len(datums)
        ):
            item_to_datum = tuple(range(len(datums)))
        return _LossBatchLayout(
            fields=fields,
            item_to_datum=item_to_datum,
            pad_values={},
            unresolved_fields=frozenset(unresolved),
        )

    def _prepare_batch(
        self,
        datums: list[Datum],
        num_pipeline_microbatches: int,
    ) -> tuple[
        Callable[[], AbstractContextManager[Any]],
        dict[str, Any],
        LossInputs,
        _LossBatchLayout,
        _OutputRestorePlan,
    ]:
        """Collate, move, and CP-shard one outer batch.

        Args:
            datums: Datum items in one outer batch. Tensor layouts are defined
                by the configured collater.
            num_pipeline_microbatches: Number of pipeline microbatches that
                the prepared outer batch must materialize.

        Returns:
            The CP context factory, CP-local model inputs, CP-local loss
            inputs, collated field-layout metadata, and an internal plan for
            restoring explicitly declared token outputs. Token-aligned model
            and loss tensors use the same padded, packed THD, Magi, or
            model-owned local sequence layout.
        """
        model_inputs, collated_loss_inputs = self.collate_fn(datums)
        hybridep_target_tokens = self._hybridep_packed_token_target(model_inputs)
        hybridep_suffix_tokens = 0
        if hybridep_target_tokens is not None:
            token_template = _model_token_template(model_inputs)
            if token_template.ndim != 2:
                raise ValueError("HybridEP raw THD equalization requires a two-dimensional token template")
            hybridep_suffix_tokens = hybridep_target_tokens - int(token_template.shape[1])
        loss_batch_layout = self._resolve_loss_batch_layout(datums, model_inputs, collated_loss_inputs)
        loss_inputs = dict(collated_loss_inputs)
        weight_layout = loss_batch_layout.fields.get("weights")
        if weight_layout is not LossInputLayout.PER_TOKEN and not (
            weight_layout is LossInputLayout.REPLICATED and "weights" in loss_batch_layout.unresolved_fields
        ):
            raise ValueError("loss input 'weights' must use the PER_TOKEN layout")
        if "labels" in loss_inputs and loss_batch_layout.fields["labels"] is not LossInputLayout.PER_TOKEN:
            raise ValueError("loss input 'labels' must use the PER_TOKEN layout")
        self._validate_collated_weights(datums, loss_inputs)
        token_reference_name, loss_seq_dim = self._validate_loss_batch_layout(
            datums,
            model_inputs,
            loss_inputs,
            loss_batch_layout.fields,
        )
        if hybridep_target_tokens is not None:
            if loss_seq_dim is None:
                raise ValueError("HybridEP packed token equalization requires token-aligned loss inputs")
            _pad_hybridep_packed_thd(
                model_inputs,
                loss_inputs,
                loss_batch_layout.fields,
                loss_batch_layout.pad_values,
                loss_seq_dim=loss_seq_dim,
                target_tokens=hybridep_target_tokens,
                padding_token_id=self.padding_token_id,
            )
        output_routing = (
            _output_sequence_lengths(
                datums,
                model_inputs,
                loss_inputs,
                loss_batch_layout.item_to_datum,
                is_thd=model_inputs.get("qkv_format") == "thd",
            )
            if loss_seq_dim is not None and weight_layout is LossInputLayout.PER_TOKEN
            else (None, None, None)
        )
        model_inputs = _to_device(model_inputs, self.device)
        loss_inputs = _to_device(loss_inputs, self.device)
        token_reference = (
            loss_inputs[token_reference_name]
            if token_reference_name is not None
            else _model_token_template(model_inputs)
        )

        cp_batch = dict(model_inputs)
        labels = loss_inputs.get("labels")
        cp_batch["labels"] = (
            labels.clone() if isinstance(labels, torch.Tensor) else torch.zeros_like(token_reference, dtype=torch.long)
        )
        is_thd = cp_batch.get("qkv_format") == "thd"
        position_ids = cp_batch.get("position_ids")
        if (
            is_thd
            and isinstance(position_ids, torch.Tensor)
            and position_ids.ndim == 3
            and (self._cp_size() > 1 or num_pipeline_microbatches > 1)
        ):
            raise NotImplementedError(
                "THD pipeline/context parallelism does not yet support three-dimensional mRoPE position_ids"
            )

        thd_loss_fields: list[str] = []
        if is_thd:
            nonzero_pad_fields = [
                name
                for name, pad_value in loss_batch_layout.pad_values.items()
                if pad_value != 0 and loss_batch_layout.fields[name] is LossInputLayout.PER_TOKEN
            ]
            if num_pipeline_microbatches > 1 and nonzero_pad_fields:
                raise NotImplementedError(
                    "packed pipeline microbatching does not yet preserve nonzero PER_TOKEN pad sentinels for "
                    f"{nonzero_pad_fields}"
                )
            if "weights" in loss_batch_layout.unresolved_fields:
                raise ValueError("packed THD execution requires token-aligned loss weights")
            unresolved_non_token_fields = [
                name
                for name in loss_batch_layout.unresolved_fields
                if loss_batch_layout.fields[name] is not LossInputLayout.PER_TOKEN
            ]
            if num_pipeline_microbatches > 1 and unresolved_non_token_fields:
                raise NotImplementedError(
                    "packed pipeline microbatching requires an explicit PER_DATUM or REPLICATED layout for "
                    f"non-token loss fields {unresolved_non_token_fields}"
                )
            per_datum_fields = [
                name for name, layout in loss_batch_layout.fields.items() if layout is LossInputLayout.PER_DATUM
            ]
            if num_pipeline_microbatches > 1 and per_datum_fields and loss_batch_layout.item_to_datum is None:
                raise NotImplementedError(
                    "packed pipeline microbatching requires collater item_to_datum metadata for per-Datum "
                    f"loss fields {per_datum_fields}; use collate_datums or an explicitly layout-aware collater"
                )
            for name, value in loss_inputs.items():
                if name == "labels" or loss_batch_layout.fields[name] is not LossInputLayout.PER_TOKEN:
                    continue
                if not _is_token_aligned(value, token_reference):
                    raise ValueError(f"per-token loss field {name!r} does not match the collated token layout")
                key = f"{_LOSS_FIELD_PREFIX}{name}"
                if key in cp_batch:
                    raise ValueError(f"model inputs contain reserved Engine key {key!r}")
                if loss_batch_layout.pad_values.get(name, 0) == 0:
                    cp_batch[key] = value
                thd_loss_fields.append(name)

        device_mesh = self.mesh_context.device_mesh if self.mesh_context is not None else None
        sharder = ContextParallelSharder(
            self.model,
            device_mesh,
            cp_batch,
            padding_token_id=self.padding_token_id,
            num_chunks=num_pipeline_microbatches,
        )
        mtp_cp_inputs = self._prepare_mtp_cp_inputs(cp_batch)
        cp_context, model_inputs = sharder.shard(cp_batch)
        if model_inputs.get("qkv_format") == "thd" and (
            "seq_lens" in model_inputs or "seq_lens_padded" in model_inputs
        ):
            raise ValueError(
                "ContextParallelSharder could not prepare raw THD inputs for this model; "
                "use a THD-capable attention backend"
            )

        local_labels = model_inputs.pop("labels")
        if is_thd:
            local_loss_inputs = dict(loss_inputs)
            for name in thd_loss_fields:
                key = f"{_LOSS_FIELD_PREFIX}{name}"
                candidate = model_inputs.pop(key, None)
                if isinstance(candidate, torch.Tensor) and _matches_primary_token_layout(candidate, model_inputs):
                    local_loss_inputs[name] = candidate
                else:
                    local_loss_inputs[name] = sharder.shard_token_tensor(
                        loss_inputs[name],
                        seq_dim=loss_seq_dim or 0,
                        fill=loss_batch_layout.pad_values.get(name, 0),
                    )
            loss_inputs = local_loss_inputs
        else:
            loss_inputs = self._shard_loss_inputs(
                sharder,
                loss_inputs,
                loss_seq_dim,
                loss_batch_layout.fields,
                loss_batch_layout.pad_values,
                token_reference,
                loss_batch_layout.unresolved_fields,
            )
        if labels is not None:
            loss_inputs["labels"] = local_labels
        if mtp_cp_inputs is not None:
            loss_inputs = self._attach_mtp_cp_inputs(sharder, mtp_cp_inputs, model_inputs, loss_inputs)
            loss_batch_layout = _LossBatchLayout(
                fields={
                    **loss_batch_layout.fields,
                    "mtp_per_depth_targets": LossInputLayout.PER_TOKEN,
                },
                item_to_datum=loss_batch_layout.item_to_datum,
                pad_values=loss_batch_layout.pad_values,
                unresolved_fields=loss_batch_layout.unresolved_fields,
            )
        real_lengths, padded_lengths, token_mask = output_routing
        return (
            cp_context,
            model_inputs,
            loss_inputs,
            loss_batch_layout,
            _OutputRestorePlan(
                sharder=sharder,
                is_thd=is_thd,
                item_to_datum=loss_batch_layout.item_to_datum,
                real_lengths=real_lengths,
                padded_lengths=padded_lengths,
                token_mask=token_mask,
                synthetic_suffix_tokens=hybridep_suffix_tokens,
            ),
        )

    def _prepare_mtp_cp_inputs(self, batch: dict[str, Any]) -> MTPContextParallelInputs | None:
        """Prepare global MTP future-token tensors before CP shards the batch.

        Args:
            batch: Unsharded model batch. ``input_ids`` and ``labels`` have
                shape [batch, sequence]; ``position_ids`` has shape [batch,
                sequence] or [axes, batch, sequence]. Model-owned CP setup has
                already populated any required position metadata.

        Returns:
            Globally ordered per-depth MTP tensors, or ``None`` when CP or MTP
            is inactive. Token tensors have shape [batch, sequence]; multi-axis
            positions have shape [axes, batch, sequence].
        """
        if self._cp_size() <= 1:
            return None

        supports = getattr(self.model, "supports", None)
        if supports is None or not bool(getattr(supports, "mtp_enabled", False)):
            return None
        if not bool(getattr(supports, "supports_mtp_cp", False)):
            raise NotImplementedError(f"{type(self.model).__name__} does not support MTP with context parallelism")
        if self.pipeline is not None and not bool(getattr(supports, "supports_mtp_cp_pp", False)):
            raise NotImplementedError(
                "MTP with context and pipeline parallelism is not supported; use PP size 1 or CP size 1"
            )

        prepare = getattr(self.model, "prepare_mtp_inputs_for_cp", None)
        if not callable(prepare):
            raise RuntimeError(
                f"{type(self.model).__name__} declares MTP+CP support but has no prepare_mtp_inputs_for_cp hook"
            )
        prepared = prepare(batch, ignore_index=self.mtp_ignore_index)
        if not isinstance(prepared, MTPContextParallelInputs):
            raise TypeError(
                "prepare_mtp_inputs_for_cp must return MTPContextParallelInputs when MTP and CP are enabled"
            )
        return prepared

    def _attach_mtp_cp_inputs(
        self,
        sharder: ContextParallelSharder,
        prepared: MTPContextParallelInputs,
        model_inputs: dict[str, Any],
        loss_inputs: LossInputs,
    ) -> LossInputs:
        """Shard prepared MTP tensors with the main batch's captured CP layout.

        Args:
            sharder: Sharder that has already partitioned the main batch and
                captured its local token layout.
            prepared: Global per-depth token IDs, positions, targets, and masks.
                Token tensors have shape [batch, sequence]; multi-axis positions
                have shape [axes, batch, sequence].
            model_inputs: CP-local model mapping updated in place. Added token
                tensors have shape [batch, local_sequence] (or [local_tokens]
                for THD), and multi-axis positions have shape [axes, batch,
                local_sequence].
            loss_inputs: CP-local loss mapping. ``mtp_per_depth_targets`` is
                added with the same local token layout as labels.

        Returns:
            A shallow copy of ``loss_inputs`` containing CP-local per-depth
            targets. Existing tensor storage is preserved.
        """
        model_inputs["mtp_per_depth_input_ids"] = tuple(
            sharder.shard_token_tensor(value, seq_dim=1, fill=0) for value in prepared.input_ids
        )
        model_inputs["mtp_per_depth_position_ids"] = tuple(
            sharder.shard_token_tensor(value, seq_dim=prepared.position_ids_seq_dim, fill=0)
            for value in prepared.position_ids
        )
        model_inputs["mtp_per_depth_valid_masks"] = tuple(
            sharder.shard_token_tensor(value, seq_dim=1, fill=False) for value in prepared.valid_masks
        )
        result = dict(loss_inputs)
        result["mtp_per_depth_targets"] = tuple(
            sharder.shard_token_tensor(value, seq_dim=1, fill=self.mtp_ignore_index) for value in prepared.targets
        )
        return result

    def _pipeline_execute(
        self,
        model_inputs: dict[str, Any],
        loss_inputs: LossInputs,
        datums: Sequence[Datum],
        loss_fn: LossFn,
        local_loss_sum: torch.Tensor,
        cp_context: Callable[[], AbstractContextManager[Any]],
        loss_batch_layout: _LossBatchLayout,
        output_restore_plan: _OutputRestorePlan,
        *,
        backward_scale: torch.Tensor | None,
        zero_weight_sum: bool,
    ) -> tuple[bool | None, list[dict[str, Any]], Exception | None]:
        """Run prepared pipeline microbatches in training or forward-only mode.

        Args:
            model_inputs: CP-prepared outer-batch model inputs.
            loss_inputs: CP-prepared outer-batch loss inputs.
            datums: Datum items represented by the outer batch.
            loss_fn: Model-output loss callback.
            local_loss_sum: Accumulator updated with detached numerators.
            cp_context: Context covering the complete pipeline schedule.
            loss_batch_layout: Semantic layout of every loss field plus the
                collater's logical item-to-Datum routing, when available.
            output_restore_plan: Captured full-token routing and CP sharder for
                explicit per-token callback outputs.
            backward_scale: Multiplier returned to the training schedule for
                backward, or ``None`` to run the forward-only schedule.
            zero_weight_sum: Whether reporting numerators must be forced to
                graph-connected zero.

        Returns:
            Whether the callback returned outputs, its detached outputs in
            logical Datum order, and any post-schedule output-restoration
            error synchronized across PP stages.
        """
        outputs_by_microbatch: list[list[dict[str, Any]] | None] = [None] * self.pipeline.num_microbatches
        outputs_by_datum: list[dict[str, Any] | None] = [None] * len(datums)
        parsed_outputs_by_microbatch: list[ParsedLossOutputs] = [None] * self.pipeline.num_microbatches
        output_parse_errors_by_microbatch: list[Exception | None] = [None] * self.pipeline.num_microbatches
        loss_called_by_microbatch = [False] * self.pipeline.num_microbatches
        returns_outputs: bool | None = None

        with cp_context():
            primary_microbatch, is_thd, batch_size = self._plan_pipeline_batch(model_inputs)
            num_microbatches = self.pipeline.num_microbatches
            seq_len = (
                primary_microbatch.shape[0]
                if is_thd and num_microbatches == 1
                else primary_microbatch.shape[min(primary_microbatch.ndim - 1, 1)]
            )
            self.pipeline.update_seq_len(
                seq_len,
                microbatch_size=1 if is_thd else primary_microbatch.shape[0],
                input_tensor=primary_microbatch,
            )

            # Batch contexts may inspect the stage metadata to decide whether a
            # real forward will be consumed for dynamic shape inference. Update
            # that metadata first so VLM media cursors are not reset after the
            # first actual pipeline microbatch.
            with self.context_fn(model_inputs):
                model_microbatches, loss_microbatches, datum_indices_by_microbatch = (
                    self._materialize_pipeline_microbatches(
                        model_inputs,
                        loss_inputs,
                        loss_batch_layout,
                        num_datums=len(datums),
                        is_thd=is_thd,
                        batch_size=batch_size,
                    )
                )
                batch_context_fns: list[Callable[[], AbstractContextManager[Any]]] = [
                    partial(self.batch_context_fn, model_inputs_mb, loss_inputs_mb)
                    for model_inputs_mb, loss_inputs_mb in zip(model_microbatches, loss_microbatches)
                ]

                def pipeline_loss(output: Any, microbatch_index: int) -> torch.Tensor:
                    loss_inputs_mb = loss_microbatches[microbatch_index]
                    with batch_context_fns[microbatch_index]():
                        numerator, batch_outputs, output_parse_error = _parse_loss_result(
                            loss_fn(output, loss_inputs_mb), loss_inputs_mb["weights"]
                        )
                    if loss_called_by_microbatch[microbatch_index]:
                        raise RuntimeError(f"pipeline evaluated loss_fn twice for microbatch {microbatch_index}")
                    loss_called_by_microbatch[microbatch_index] = True
                    parsed_outputs_by_microbatch[microbatch_index] = batch_outputs
                    output_parse_errors_by_microbatch[microbatch_index] = output_parse_error
                    if zero_weight_sum:
                        numerator = numerator * 0
                    local_loss_sum.add_(numerator.detach().to(torch.float64))
                    return numerator if backward_scale is None else numerator * backward_scale

                losses = [] if self.pipeline.info.has_last_stage else None
                if backward_scale is None:
                    self.pipeline.eval_microbatches(
                        model_microbatches,
                        loss_fn=pipeline_loss,
                        losses=losses,
                        return_outputs=False,
                        batch_context_fns=batch_context_fns,
                    )
                else:
                    self.pipeline.step_microbatches(
                        model_microbatches,
                        loss_fn=pipeline_loss,
                        losses=losses,
                        return_outputs=False,
                        batch_context_fns=batch_context_fns,
                    )

        outputs: list[dict[str, Any]] = []
        serialized_outputs: bytes | None = None
        output_error: Exception | None = None
        if self.pipeline.info.has_last_stage:
            try:
                for microbatch_index, batch_outputs in enumerate(parsed_outputs_by_microbatch):
                    local_error = output_parse_errors_by_microbatch[microbatch_index]
                    if not loss_called_by_microbatch[microbatch_index]:
                        local_error = RuntimeError(
                            f"pipeline did not evaluate loss_fn for microbatch {microbatch_index}"
                        )
                    weights = loss_microbatches[microbatch_index].get("weights")
                    datum_indices = (
                        None if datum_indices_by_microbatch is None else datum_indices_by_microbatch[microbatch_index]
                    )
                    expected_records = (
                        len(datum_indices)
                        if datum_indices is not None
                        else (1 if len(datums) == 1 and self.pipeline.num_microbatches == 1 else None)
                    )
                    self._validate_loss_fn_outputs_across_cp(
                        batch_outputs,
                        weights,
                        expected_records=expected_records,
                        microbatch_index=microbatch_index,
                        local_error=local_error,
                        restore_plan=output_restore_plan,
                        datum_indices=datum_indices,
                        chunk_index=(microbatch_index if is_thd and self.pipeline.num_microbatches > 1 else None),
                    )
                    returns_outputs = _update_output_mode(returns_outputs, batch_outputs)
                    if batch_outputs is None:
                        continue
                    restored = (
                        batch_outputs
                        if isinstance(batch_outputs, list)
                        else self._restore_loss_fn_outputs(
                            batch_outputs,
                            loss_microbatches[microbatch_index],
                            output_restore_plan,
                            datum_indices=datum_indices,
                            chunk_index=(microbatch_index if is_thd and self.pipeline.num_microbatches > 1 else None),
                        )
                    )
                    if datum_indices is None:
                        outputs_by_microbatch[microbatch_index] = restored
                        continue
                    if len(restored) != len(datum_indices):
                        raise RuntimeError("restored token outputs do not match their pipeline Datum routing")
                    for datum_index, item in zip(datum_indices, restored):
                        if outputs_by_datum[datum_index] is not None:
                            raise RuntimeError(f"pipeline returned more than one output for Datum {datum_index}")
                        outputs_by_datum[datum_index] = item
                if returns_outputs:
                    if datum_indices_by_microbatch is not None:
                        if any(item is None for item in outputs_by_datum):
                            raise RuntimeError("pipeline schedule did not return exactly one output for every Datum")
                        outputs = [item for item in outputs_by_datum if item is not None]
                    else:
                        if any(items is None for items in outputs_by_microbatch):
                            raise RuntimeError(
                                "pipeline schedule did not evaluate loss_fn for every logical microbatch"
                            )
                        outputs = [item for items in outputs_by_microbatch if items is not None for item in items]
                        if len(outputs) != len(datums):
                            raise ValueError(
                                f"pipeline loss_fn returned {len(outputs)} outputs across the outer batch, "
                                f"expected one for each of its {len(datums)} Datums"
                            )
                if outputs:
                    # broadcast_object_list serializes only on the source
                    # stage. Preflight while errors can still be propagated to
                    # every PP stage instead of leaving peers in the broadcast.
                    # Broadcast these already validated bytes so arbitrary
                    # record objects are never pickled again inside a PP
                    # collective.
                    serialized_outputs = pickle.dumps(_to_device(outputs, torch.device("cpu")))
            except Exception as error:
                output_error = error
        output_error = self._synchronize_pipeline_output_error(output_error)
        if output_error is None:
            outputs = self._broadcast_pipeline_outputs(outputs, serialized_outputs=serialized_outputs)
        return bool(outputs), outputs, output_error

    def _restore_loss_fn_outputs(
        self,
        outputs: list[dict[str, Any]] | LossFnOutputBatch,
        loss_inputs: LossInputs,
        plan: _OutputRestorePlan,
        *,
        datum_indices: tuple[int, ...] | None,
        chunk_index: int | None,
    ) -> list[dict[str, Any]]:
        """Restore explicitly typed token streams and merge per-Datum records."""
        weights = loss_inputs.get("weights")
        if not isinstance(weights, torch.Tensor):
            raise ValueError("loss_fn token outputs require Tensor loss weights")
        if isinstance(outputs, list):
            return outputs

        if datum_indices is None:
            if plan.item_to_datum is not None:
                raise RuntimeError("loss_fn output routing is missing the collater's Datum mapping")
            if chunk_index is not None:
                raise NotImplementedError(
                    "a prebatched Datum cannot restore token outputs across multiple pipeline microbatches"
                )
            datum_indices = (0,)

        source_records = ({},) * len(datum_indices) if outputs.per_datum is None else outputs.per_datum
        records = [dict(record) for record in source_records]
        if len(records) != len(datum_indices):
            raise ValueError(
                f"LossFnOutputBatch.per_datum contains {len(records)} records, expected {len(datum_indices)}"
            )
        record_keys = {key for record in records for key in record}
        collisions = record_keys & set(outputs.per_token)
        if collisions:
            raise ValueError(f"per-token output keys collide with per-Datum records: {sorted(collisions)}")

        restored_fields: list[tuple[str, torch.Tensor, int]] = []
        for name in sorted(outputs.per_token):
            spec = outputs.per_token[name]
            tensor = spec.tensor.detach()
            if tensor.shape[: weights.ndim] != weights.shape:
                raise ValueError(
                    f"per-token output {name!r} must start with the loss weight shape {tuple(weights.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.device != weights.device:
                raise ValueError(
                    f"per-token output {name!r} must be on the loss weight device {weights.device}, got {tensor.device}"
                )
            seq_dim = _loss_sequence_dim_from_weights(weights)
            shard_layout = plan.sharder.shard_layout
            selected_layout = shard_layout
            if chunk_index is not None:
                if shard_layout is None or shard_layout.chunk_layouts is None:
                    if self._cp_size() > 1:
                        raise NotImplementedError(
                            "the active context-parallel backend does not report reversible per-pipeline-chunk "
                            "token layouts; typed per-token outputs are unavailable for this packed PP+CP batch"
                        )
                    selected_layout = None
                else:
                    if chunk_index < 0 or chunk_index >= len(shard_layout.chunk_layouts):
                        raise IndexError(
                            f"pipeline chunk {chunk_index} is out of range for "
                            f"{len(shard_layout.chunk_layouts)} reported CP layouts"
                        )
                    selected_layout = shard_layout.chunk_layouts[chunk_index]
            if self._cp_size() > 1 and selected_layout is None:
                raise NotImplementedError(
                    "the active context-parallel backend does not report a reversible token layout; "
                    "typed per-token outputs are unavailable for this batch"
                )
            if selected_layout is not None:
                tensor = plan.sharder.gather_token_tensor(
                    tensor,
                    seq_dim=seq_dim,
                    trim=True,
                    fill=spec.fill_value,
                    chunk_index=chunk_index,
                )
                if selected_layout.input_row_shape is not None:
                    seq_dim = len(selected_layout.input_row_shape) - 1
            restored_fields.append((name, tensor, seq_dim))

        # Complete every field's CP collective before Datum-local splitting.
        # A routing error can then no longer leave a peer entering the next
        # field gather while this rank exits early.
        for name, tensor, seq_dim in restored_fields:
            pieces = _split_restored_token_output(
                tensor,
                plan,
                datum_indices=datum_indices,
                seq_dim=seq_dim,
            )
            if len(pieces) != len(records):
                raise RuntimeError(f"restored per-token output {name!r} does not match its Datum records")
            for record, piece in zip(records, pieces):
                record[name] = piece.detach()
        return records

    def _validate_loss_fn_outputs_across_cp(
        self,
        outputs: ParsedLossOutputs,
        weights: Any,
        *,
        expected_records: int | None,
        microbatch_index: int | None = None,
        local_error: Exception | None = None,
        restore_plan: _OutputRestorePlan | None = None,
        datum_indices: tuple[int, ...] | None = None,
        chunk_index: int | None = None,
    ) -> None:
        """Validate output routing and reach CP consensus before token gathers."""
        if local_error is None:
            error, schema = _loss_fn_output_contract(
                outputs,
                weights,
                expected_records=expected_records,
                microbatch_index=microbatch_index,
            )
        else:
            error, schema = str(local_error), ("invalid-output", type(local_error).__name__)
        if error is None and isinstance(outputs, LossFnOutputBatch) and restore_plan is not None:
            restore_error, restore_schema = _loss_fn_output_restore_contract(
                outputs,
                weights,
                restore_plan,
                datum_indices=datum_indices,
                chunk_index=chunk_index,
                cp_size=self._cp_size(),
            )
            error = restore_error
            schema = (*schema, restore_schema)
        cp_group, cp_size = self._cp_group_and_size()
        if cp_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            if error is not None:
                raise ValueError(error)
            return
        digest = int.from_bytes(hashlib.sha256(repr(schema).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        local = torch.tensor([int(error is not None), digest], dtype=torch.int64, device=self.device)
        gathered = torch.empty(cp_size * 2, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(gathered, local, group=cp_group)
        gathered = gathered.view(cp_size, 2)
        if bool((gathered[:, 0] != 0).any()):
            detail = f": {error}" if error is not None else ""
            raise ValueError(f"invalid loss_fn outputs on one or more context-parallel ranks{detail}")
        if not bool((gathered[:, 1] == gathered[0, 1]).all()):
            raise ValueError("context-parallel ranks returned different loss output schemas")

    def _synchronize_pipeline_output_error(self, error: Exception | None) -> Exception | None:
        """Propagate output failures across CP lanes and then PP stages."""
        failed = torch.tensor(int(error is not None), dtype=torch.int64, device=self.device)
        cp_group, cp_size = self._cp_group_and_size()
        if cp_size > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=cp_group)
            if bool(failed.item()) and error is None:
                error = RuntimeError("another context-parallel rank failed while restoring loss_fn outputs")

        pp_group, pp_size = self._pp_group_and_size()
        if pp_size <= 1:
            return error
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=pp_group)
        if not bool(failed.item()):
            return None
        return error or RuntimeError("pipeline last stage failed while restoring loss_fn outputs")

    def _broadcast_pipeline_outputs(
        self,
        outputs: list[dict[str, Any]],
        *,
        serialized_outputs: bytes | None,
    ) -> list[dict[str, Any]]:
        """Broadcast last-stage per-Datum outputs to every pipeline stage."""
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size <= 1:
            return outputs

        has_last_stage = self.pipeline.info.has_last_stage
        local_state = torch.tensor(
            [int(has_last_stage), int(has_last_stage and bool(outputs))], dtype=torch.int64, device=self.device
        )
        stage_states = torch.empty(pp_size * 2, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(stage_states, local_state, group=pp_group)
        stage_states = stage_states.view(pp_size, 2)
        source_ranks = (stage_states[:, 0] == 1).nonzero(as_tuple=False).flatten()
        if source_ranks.numel() != 1:
            raise RuntimeError("pipeline output synchronization requires exactly one physical last-stage rank")

        source_group_rank = int(source_ranks.item())
        if not bool(stage_states[source_group_rank, 1]):
            return []
        source_global_rank = dist.get_global_rank(pp_group, source_group_rank)
        object_list: list[Any] = [serialized_outputs if dist.get_rank(group=pp_group) == source_group_rank else None]
        dist.broadcast_object_list(object_list, src=source_global_rank, group=pp_group, device=self.device)
        payload = object_list[0]
        if not isinstance(payload, bytes):
            raise RuntimeError("pipeline output synchronization received an invalid serialized payload")
        # The payload was serialized by this Engine's trusted PP source rank
        # immediately above; this is not an external deserialization boundary.
        received = pickle.loads(payload)  # noqa: S301
        if not isinstance(received, list) or not all(isinstance(item, dict) for item in received):
            raise RuntimeError("pipeline output synchronization received invalid per-Datum outputs")
        return _to_device(received, self.device)

    def _plan_pipeline_batch(self, model_inputs: Mapping[str, Any]) -> tuple[torch.Tensor, bool, int | None]:
        """Validate the PP outer-batch shape once for metadata and slicing."""
        primary_name = _primary_name(model_inputs)
        primary = model_inputs[primary_name]
        if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
            raise ValueError("pipeline Engine requires a tensor input_ids or inputs_embeds")

        num_microbatches = self.pipeline.num_microbatches
        is_thd = model_inputs.get("qkv_format") == "thd"
        if num_microbatches == 1:
            return primary, is_thd, None
        if is_thd:
            if primary.shape[0] != num_microbatches:
                raise ValueError(
                    f"THD sharder produced {primary.shape[0]} chunks, expected {num_microbatches} pipeline microbatches"
                )
            return primary.narrow(0, 0, 1), True, None

        batch_size = primary.shape[0]
        if batch_size % num_microbatches != 0:
            raise ValueError(
                f"pipeline outer batch size {batch_size} must be divisible by {num_microbatches} microbatches"
            )
        materialized_batch_size = batch_size // num_microbatches
        if materialized_batch_size != self.pipeline.pp_microbatch_size:
            raise ValueError(
                f"materialized pipeline microbatch has batch size {materialized_batch_size}, "
                f"but AutoPipeline is configured for pp_microbatch_size={self.pipeline.pp_microbatch_size}"
            )
        return primary.narrow(0, 0, materialized_batch_size), False, batch_size

    def _materialize_pipeline_microbatches(
        self,
        model_inputs: dict[str, Any],
        loss_inputs: LossInputs,
        loss_batch_layout: _LossBatchLayout,
        *,
        num_datums: int,
        is_thd: bool,
        batch_size: int | None,
    ) -> tuple[list[dict[str, Any]], list[LossInputs], list[tuple[int, ...]] | None]:
        """Split one CP-prepared outer batch into exact pipeline inputs.

        Args:
            model_inputs: CP-local model tensors. Padded tensors use shape
                [batch, sequence, ...]. Chunked THD tensors use shape
                [microbatches, tokens, ...].
            loss_inputs: CP-local loss tensors. Token-aligned fields have the
                same leading token axes as the primary model tensor.
            loss_batch_layout: Semantic field layouts and optional collater
                item-to-Datum mapping.
            num_datums: Number of outer Datum items represented by this batch.

        Returns:
            Parallel lists of complete model and loss mappings, plus optional
            logical Datum indices for each microbatch. Every list has exactly
            ``pipeline.num_microbatches`` items. Token slicing retains a
            size-one pipeline microbatch axis.
        """
        num_microbatches = self.pipeline.num_microbatches
        if num_microbatches == 1:
            datum_indices = [tuple(range(num_datums))]
            if loss_batch_layout.item_to_datum is not None:
                datum_indices = _pipeline_datum_indices(
                    [model_inputs],
                    loss_batch_layout.item_to_datum,
                    num_datums=num_datums,
                    is_thd=is_thd,
                )
                assert datum_indices is not None
            return (
                [dict(model_inputs)],
                [_with_loss_metadata(model_inputs, loss_inputs)],
                datum_indices,
            )

        if is_thd:
            model_microbatches = [
                _select_chunk(model_inputs, index, num_microbatches) for index in range(num_microbatches)
            ]
        else:
            assert batch_size is not None
            custom_dims: dict[str, int] = {}
            chunk_dims = getattr(self.model, "get_pipeline_kwargs_chunk_dims", None)
            if chunk_dims is not None:
                custom_dims = chunk_dims(model_inputs) or {}
            model_microbatches = [
                _slice_batch_mapping(model_inputs, index, num_microbatches, batch_size, custom_dims)
                for index in range(num_microbatches)
            ]

        datum_indices_by_microbatch = _pipeline_datum_indices(
            model_microbatches,
            loss_batch_layout.item_to_datum,
            num_datums=num_datums,
            is_thd=is_thd,
        )
        loss_microbatches = [
            _materialize_loss_mapping(
                loss_inputs,
                loss_batch_layout.fields,
                loss_batch_layout.unresolved_fields,
                index=index,
                num_chunks=num_microbatches,
                batch_size=None if is_thd else batch_size,
                datum_indices=(None if datum_indices_by_microbatch is None else datum_indices_by_microbatch[index]),
                num_datums=num_datums,
                is_thd=is_thd,
            )
            for index in range(num_microbatches)
        ]

        loss_microbatches = [
            _with_loss_metadata(model_microbatch, loss_microbatch)
            for model_microbatch, loss_microbatch in zip(model_microbatches, loss_microbatches)
        ]
        return model_microbatches, loss_microbatches, datum_indices_by_microbatch

    def _validate_parallelism(self) -> None:
        """Validate topology plus backward-specific distributed contracts."""
        self._validate_execution_parallelism()
        if self.pipeline is not None and self.pipeline.scale_grads_in_schedule:
            raise ValueError("Engine requires AutoPipeline scale_grads_in_schedule=False")

    def _validate_execution_parallelism(self) -> None:
        """Validate model-parallel topology shared by forward and backward."""
        if (
            self.pipeline is not None
            and self._cp_size() > 1
            and any(
                bool(getattr(getattr(part, "supports", None), "mtp_enabled", False))
                and not bool(getattr(getattr(part, "supports", None), "supports_mtp_cp_pp", False))
                for part in self.model_parts
            )
        ):
            raise NotImplementedError(
                "MTP with context and pipeline parallelism is not supported; use PP size 1 or CP size 1"
            )
        if self.pipeline is not None and self.mesh_context is None:
            raise ValueError("pipeline Engine requires mesh_context")
        if self.mesh_context is None:
            return
        if self.mesh_context.pp_size > 1 and self.pipeline is None:
            raise NotImplementedError("pipeline parallelism requires an AutoPipeline")
        if self.mesh_context.cp_size > 1 and self.mesh_context.device_mesh is None:
            raise ValueError("context parallelism requires a device mesh")

    def _cp_size(self) -> int:
        return self.mesh_context.cp_size if self.mesh_context is not None else 1

    def _cp_group_and_size(self) -> tuple[dist.ProcessGroup | None, int]:
        if (
            self.mesh_context is None
            or self.mesh_context.device_mesh is None
            or self.mesh_context.cp_size <= 1
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return None, 1
        cp_mesh = self.mesh_context.device_mesh["cp"]
        size = int(cp_mesh.size())
        return (cp_mesh.get_group() if size > 1 else None), size

    def _dp_group_and_size(self) -> tuple[dist.ProcessGroup | None, int]:
        if self.mesh_context is not None and self.mesh_context.device_mesh is not None:
            dp_mesh = get_flat_mesh(self.mesh_context.device_mesh, "dp")
            size = int(dp_mesh.size())
            return (dp_mesh.get_group() if size > 1 else None), size

        group = self.mesh_context.process_group if self.mesh_context is not None else None
        if dist.is_available() and dist.is_initialized():
            return group, dist.get_world_size(group=group)
        return None, 1

    def _gradient_group_and_size(
        self,
        dp_group: dist.ProcessGroup | None,
        dp_size: int,
    ) -> tuple[dist.ProcessGroup | None, int]:
        if self.mesh_context is None or self.mesh_context.device_mesh is None or self._cp_size() == 1:
            return dp_group, dp_size
        dp_cp_mesh = get_flat_mesh(self.mesh_context.device_mesh, "dp_cp")
        size = int(dp_cp_mesh.size())
        return (dp_cp_mesh.get_group() if size > 1 else None), size

    def _pp_group_and_size(self) -> tuple[dist.ProcessGroup | None, int]:
        if self.pipeline is None or not dist.is_available() or not dist.is_initialized():
            return None, 1
        size = int(self.pipeline.pp_mesh.size())
        return (self.pipeline.pp_mesh.get_group() if size > 1 else None), size

    def _validate_pipeline_window(self, size: int, denominator: torch.Tensor) -> None:
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size <= 1:
            return
        local = torch.stack((denominator.new_tensor(size), denominator))
        gathered = torch.empty(pp_size * 2, dtype=denominator.dtype, device=denominator.device)
        dist.all_gather_into_tensor(gathered, local, group=pp_group)
        gathered = gathered.view(pp_size, 2)
        if not bool((gathered[:, 0] == gathered[0, 0]).all()):
            raise ValueError(f"pipeline stages must use the same outer window size; got {gathered[:, 0].tolist()}")
        if not torch.allclose(gathered[:, 1], gathered[0, 1].expand(pp_size), rtol=1e-8, atol=1e-12):
            raise ValueError(f"pipeline stages must use the same weight denominator; got {gathered[:, 1].tolist()}")

    def _global_weight_sum(
        self,
        microbatches: list[list[Datum]],
        dp_group: dist.ProcessGroup | None,
        dp_size: int,
    ) -> torch.Tensor:
        denominator = self._local_weight_sum(microbatches)
        if dp_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=dp_group)
        return denominator

    def _local_weight_sum(self, microbatches: list[list[Datum]]) -> torch.Tensor:
        """Return the full-sequence denominator for one DP replica."""
        local_sum = 0.0
        for datum in (datum for microbatch in microbatches for datum in microbatch):
            weights = datum.loss_fn_inputs.get("weights")
            if not isinstance(weights, torch.Tensor):
                raise ValueError("every Datum must contain a Tensor loss_fn_inputs['weights']")
            if weights.numel() == 0 or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("Datum weights must be non-empty, finite, and non-negative")
            local_sum += float(weights.to(torch.float64).sum())

        denominator = torch.tensor(local_sum, dtype=torch.float64, device=self.device)
        self._validate_weight_sum_across_cp(denominator)
        return denominator

    def _validate_window_size_across_group(
        self,
        size: int,
        group: dist.ProcessGroup | None,
        group_size: int,
    ) -> None:
        if group_size <= 1:
            return
        local_size = torch.tensor([size], dtype=torch.int64, device=self.device)
        sizes = torch.empty(group_size, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(sizes, local_size, group=group)
        if not bool((sizes == sizes[0]).all()):
            raise ValueError(f"every participating rank must use the same number of microbatches; got {sizes.tolist()}")

    def _validate_weight_sum_across_cp(self, local_weight_sum: torch.Tensor) -> None:
        """Verify that CP replicas started from the same full-sequence weights."""
        if self._cp_size() <= 1 or not dist.is_available() or not dist.is_initialized():
            return
        cp_group = self.mesh_context.device_mesh["cp"].get_group()
        values = torch.empty(self._cp_size(), dtype=local_weight_sum.dtype, device=local_weight_sum.device)
        dist.all_gather_into_tensor(values, local_weight_sum.reshape(1), group=cp_group)
        if not torch.allclose(values, values[0].expand_as(values), rtol=1e-8, atol=1e-12):
            raise ValueError(f"context-parallel ranks must use identical full-sequence weights; got {values.tolist()}")

    def _shard_loss_inputs(
        self,
        sharder: ContextParallelSharder,
        loss_inputs: LossInputs,
        seq_dim: int | None,
        layouts: Mapping[str, LossInputLayout],
        pad_values: Mapping[str, float | int | bool],
        token_reference: torch.Tensor,
        unresolved_fields: frozenset[str],
    ) -> LossInputs:
        """Apply the model batch's CP token layout to loss-only tensors.

        Args:
            sharder: Sharder after it has prepared the current model batch.
            loss_inputs: Tensors in the pre-CP layout. ``weights`` has shape
                ``[batch, sequence]`` or ``[tokens]``; any tensor with those
                leading token axes is sharded identically.
            seq_dim: Sequence axis in the pre-CP loss tensors, or ``None`` when
                the weights do not follow the model's token axes.
            layouts: Explicit semantic layout for every loss field. Only
                ``PER_TOKEN`` fields follow the context-parallel token shard.
            pad_values: Explicit fills for ``PER_TOKEN`` fields whose padding
                sentinel is not zero.
            token_reference: Tensor carrying the full collated token axes.
            unresolved_fields: Legacy fields whose semantics were not declared
                by their collater. Non-token legacy weights cannot cross a CP
                layout change safely.

        Returns:
            Loss tensors in the model output's CP-local token layout. Non-token
            tensors are returned unchanged; the input mapping is not mutated.
        """
        shard_layout = sharder.shard_layout
        layout_changed = self._cp_size() > 1 or (
            shard_layout is not None
            and (
                shard_layout.input_row_shape is not None
                or shard_layout.input_token_stream_positions is not None
                or shard_layout.original_seq_len != shard_layout.padded_seq_len
            )
        )
        if "weights" in unresolved_fields and layout_changed:
            raise ValueError("context-parallel loss weights must match the model's token axes")
        if self._cp_size() == 1 and shard_layout is None:
            return {name: value for name, value in loss_inputs.items() if name != "labels"}
        if seq_dim is None:
            if any(field_layout is LossInputLayout.PER_TOKEN for field_layout in layouts.values()):
                raise ValueError("context-parallel per-token loss inputs must match the model's token axes")
            return {name: value for name, value in loss_inputs.items() if name != "labels"}

        local: LossInputs = {}
        for name, value in loss_inputs.items():
            if name == "labels":
                continue
            if layouts[name] is not LossInputLayout.PER_TOKEN:
                local[name] = value
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"per-token loss field {name!r} must be a Tensor before CP sharding")
            token_aligned = _is_token_aligned(value, token_reference)
            if not token_aligned:
                raise ValueError(f"per-token loss field {name!r} does not match the collated token layout")
            local[name] = sharder.shard_token_tensor(value, seq_dim=seq_dim, fill=pad_values.get(name, 0))
        return local

    @staticmethod
    def _validate_loss_batch_layout(
        datums: Sequence[Datum],
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, LossInputValue],
        layouts: Mapping[str, LossInputLayout],
    ) -> tuple[str | None, int | None]:
        """Validate semantic loss layouts before any CP/PP transformation."""
        if set(layouts) != set(loss_inputs):
            raise ValueError("loss input layouts must describe every collated loss field exactly once")
        weights = loss_inputs.get("weights")
        if not isinstance(weights, torch.Tensor):
            raise ValueError("collate_fn must return a Tensor loss input named 'weights'")
        token_fields = [name for name, layout in layouts.items() if layout is LossInputLayout.PER_TOKEN]
        token_reference_name = "weights" if layouts.get("weights") is LossInputLayout.PER_TOKEN else None
        if token_reference_name is None and "labels" in token_fields:
            token_reference_name = "labels"
        if token_reference_name is None and token_fields:
            token_reference_name = token_fields[0]
        token_reference = loss_inputs.get(token_reference_name) if token_reference_name is not None else None
        if token_reference_name is not None and not isinstance(token_reference, torch.Tensor):
            raise TypeError(f"per-token loss field {token_reference_name!r} must be a Tensor")
        loss_seq_dim = (
            _loss_sequence_dim(model_inputs, token_reference) if isinstance(token_reference, torch.Tensor) else None
        )
        if token_reference_name is not None and loss_seq_dim is None:
            raise ValueError(f"per-token loss field {token_reference_name!r} must match the primary model token axes")

        for name, value in loss_inputs.items():
            layout = layouts[name]
            if layout is LossInputLayout.PER_TOKEN:
                if not isinstance(value, torch.Tensor) or not _is_token_aligned(value, token_reference):
                    raise ValueError(f"per-token loss field {name!r} does not match the collated token layout")
            elif layout is LossInputLayout.PER_DATUM:
                if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != len(datums):
                    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                    raise ValueError(f"per-Datum loss field {name!r} must have leading size {len(datums)}, got {shape}")
        return token_reference_name, loss_seq_dim

    @staticmethod
    def _validate_collated_weights(
        datums: list[Datum],
        loss_inputs: Mapping[str, LossInputValue],
    ) -> None:
        weights = loss_inputs.get("weights")
        if not isinstance(weights, torch.Tensor):
            raise ValueError("collate_fn must return a Tensor loss input named 'weights'")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("collated weights must be finite and non-negative")

        expected_sum = sum(float(datum.loss_fn_inputs["weights"].to(torch.float64).sum()) for datum in datums)
        actual_sum = weights.to(torch.float64).sum()
        if not torch.isclose(actual_sum, actual_sum.new_tensor(expected_sum)):
            raise ValueError("collate_fn changed the sum of Datum weights")


def _to_device(value: Any, device: torch.device) -> Any:
    """Move tensors in common model-input containers without changing layout."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _pad_tensor_axis(tensor: torch.Tensor, *, dim: int, amount: int, value: int | float | bool) -> torch.Tensor:
    """Right-pad one tensor axis without disturbing trailing feature axes."""
    if amount <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = amount
    suffix = torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, suffix), dim=dim)


def _pad_hybridep_packed_thd(
    model_inputs: dict[str, Any],
    loss_inputs: LossInputs,
    layouts: Mapping[str, LossInputLayout],
    pad_values: Mapping[str, int | float | bool],
    *,
    loss_seq_dim: int,
    target_tokens: int,
    padding_token_id: int,
) -> None:
    """Pad one raw THD row to the HybridEP group token width in place.

    Args:
        model_inputs: Raw packed mapping whose primary token stream is
            ``input_ids [1, T]`` or ``inputs_embeds [1, T, H]``. Standard and
            multi-axis position ids use ``[1, T]`` and ``[A, 1, T]``.
        loss_inputs: Collated loss mapping. Every ``PER_TOKEN`` tensor has
            token axes ``[1, T, ...]``.
        layouts: Semantic layout for every loss field.
        pad_values: Explicit per-token fill values such as ``-1`` for routing
            replay. Undeclared fields use zero, except labels use ``-100``.
        loss_seq_dim: Sequence axis shared by all per-token loss tensors.
        target_tokens: EP-wide physical token width after equalization.
        padding_token_id: Fill value for synthetic ``input_ids`` tokens.

    Returns:
        None. The mappings are updated in place; real sequence lengths stay
        unchanged and only the final padded document extent grows.
    """
    token_template = _model_token_template(model_inputs)
    if token_template.ndim != 2 or token_template.shape[0] != 1:
        raise ValueError("HybridEP packed token equalization requires a raw THD primary tensor with token shape [1, T]")
    local_tokens = int(token_template.shape[1])
    if target_tokens < local_tokens:
        raise ValueError(f"HybridEP target width {target_tokens} is smaller than local width {local_tokens}")
    amount = target_tokens - local_tokens

    seq_lens = model_inputs.get("seq_lens")
    seq_lens_padded = model_inputs.get("seq_lens_padded")
    if not isinstance(seq_lens, torch.Tensor) or not isinstance(seq_lens_padded, torch.Tensor):
        raise ValueError("raw THD equalization requires Tensor seq_lens and seq_lens_padded")
    if seq_lens.shape != seq_lens_padded.shape or seq_lens.ndim != 2 or seq_lens.shape[0] != 1:
        raise ValueError(
            "raw THD seq_lens and seq_lens_padded must have matching shape [1, documents]; "
            f"got {tuple(seq_lens.shape)} and {tuple(seq_lens_padded.shape)}"
        )
    valid = seq_lens_padded[0] != -1000
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("raw THD metadata must describe at least one packed document")
    real_values = seq_lens[0, valid].to(torch.long)
    original_padded_values = seq_lens_padded[0, valid].to(torch.long)
    if bool((real_values < 0).any()) or bool((real_values > original_padded_values).any()):
        raise ValueError("raw THD metadata requires 0 <= seq_lens <= seq_lens_padded")
    if int(original_padded_values.sum()) != local_tokens:
        raise ValueError(
            "raw THD seq_lens_padded must cover the original physical token row; "
            f"got {int(original_padded_values.sum())} and {local_tokens}"
        )

    model_token_fields = {
        "input_ids": padding_token_id,
        "inputs_embeds": 0,
        "attention_mask": 0,
        "mm_token_type_ids": 0,
        "token_type_ids": 0,
        "_packed_seq_ids": 0,
        "packed_seq_ids": 0,
    }
    model_replacements: dict[str, torch.Tensor] = {}
    for name, fill in model_token_fields.items():
        tensor = model_inputs.get(name)
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[1] != local_tokens:
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise ValueError(f"packed model field {name!r} must use token axis [1, {local_tokens}, ...], got {shape}")
        model_replacements[name] = _pad_tensor_axis(tensor, dim=1, amount=amount, value=fill)

    position_ids = model_inputs.get("position_ids")
    if position_ids is not None:
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError("packed position_ids must be a Tensor")
        position_seq_dim = 2 if position_ids.ndim == 3 else 1
        expected_batch = position_ids.shape[1] if position_ids.ndim == 3 else position_ids.shape[0]
        if (
            position_ids.ndim not in {2, 3}
            or expected_batch != 1
            or position_ids.shape[position_seq_dim] != local_tokens
        ):
            raise ValueError(
                "packed position_ids must have shape [1, T] or [axes, 1, T]; "
                f"got {tuple(position_ids.shape)} for T={local_tokens}"
            )
        if amount:
            increment_shape = [1] * position_ids.ndim
            increment_shape[position_seq_dim] = amount
            increments = torch.arange(
                1,
                amount + 1,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).reshape(increment_shape)
            suffix = position_ids.narrow(position_seq_dim, local_tokens - 1, 1) + increments
            model_replacements["position_ids"] = torch.cat((position_ids, suffix), dim=position_seq_dim)
        else:
            model_replacements["position_ids"] = position_ids

    loss_replacements: dict[str, torch.Tensor] = {}
    for name, layout in layouts.items():
        if layout is not LossInputLayout.PER_TOKEN:
            continue
        tensor = loss_inputs[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[loss_seq_dim] != local_tokens:
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise ValueError(f"packed per-token loss field {name!r} has incompatible shape {shape}")
        fill = pad_values.get(name, -100 if name == "labels" else 0)
        loss_replacements[name] = _pad_tensor_axis(tensor, dim=loss_seq_dim, amount=amount, value=fill)

    padded = seq_lens_padded.clone()
    padded[0, int(valid_indices[-1])] += amount
    padded_values = padded[0, valid].to(torch.long)
    if int(padded_values.sum()) != target_tokens:
        raise ValueError(
            "raw THD seq_lens_padded must cover the physical token row after HybridEP equalization; "
            f"got {int(padded_values.sum())} and {target_tokens}"
        )
    padding_mask = torch.ones((1, target_tokens), dtype=torch.bool, device=token_template.device)
    offset = 0
    for real, padded_length in zip(real_values.tolist(), padded_values.tolist()):
        padding_mask[0, offset : offset + real] = False
        offset += padded_length

    model_inputs.update(model_replacements)
    loss_inputs.update(loss_replacements)
    model_inputs["seq_lens_padded"] = padded
    model_inputs["padding_mask"] = padding_mask


def _model_token_template(model_inputs: Mapping[str, Any]) -> torch.Tensor:
    """Return a tensor with exactly the primary model input's token axes."""
    primary_name = _primary_name(model_inputs)
    primary = model_inputs[primary_name]
    if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
        raise ValueError("model primary input must be a non-scalar Tensor")
    if primary_name == "inputs_embeds":
        if primary.ndim < 2 or primary.shape[-1] == 0:
            raise ValueError("inputs_embeds must contain token and hidden dimensions")
        return primary.select(-1, 0)
    return primary


def _loss_sequence_dim(model_inputs: dict[str, Any], value: torch.Tensor) -> int | None:
    """Find the sequence axis shared by primary model tokens and loss weights.

    Args:
        model_inputs: Pre-CP model mapping whose ``input_ids`` has shape
            ``[batch, sequence]`` or ``[tokens]``, or whose ``inputs_embeds``
            has shape ``[batch, sequence, hidden]``.
        value: Candidate token-aligned tensor. It may have trailing feature
            dimensions after the primary input's token axes.

    Returns:
        The sequence axis in ``value``, or ``None`` when the layouts do not
        describe the same token stream.
    """
    try:
        token_template = _model_token_template(model_inputs)
    except ValueError:
        return None
    token_dims = token_template.ndim
    if (
        token_dims in {1, 2}
        and value.ndim >= token_dims
        and tuple(value.shape[:token_dims]) == tuple(token_template.shape)
    ):
        return token_dims - 1
    return None


def _is_token_aligned(value: Any, weights: torch.Tensor) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and weights.ndim > 0
        and value.ndim >= weights.ndim
        and tuple(value.shape[: weights.ndim]) == tuple(weights.shape)
    )


def _primary_name(model_inputs: Mapping[str, Any]) -> str:
    names = [name for name in ("input_ids", "inputs_embeds") if name in model_inputs]
    if len(names) != 1:
        raise ValueError("model inputs must contain exactly one of input_ids or inputs_embeds")
    return names[0]


def _matches_primary_token_layout(tensor: torch.Tensor, model_inputs: Mapping[str, Any]) -> bool:
    primary_name = _primary_name(model_inputs)
    primary = model_inputs[primary_name]
    if not isinstance(primary, torch.Tensor) or tensor.ndim == 0:
        return False
    if model_inputs.get("qkv_format") == "thd":
        token_dims = primary.ndim - int(primary_name == "inputs_embeds")
    else:
        token_dims = 1 if primary.ndim == 1 else 2
    return tensor.ndim >= token_dims and tuple(tensor.shape[:token_dims]) == tuple(primary.shape[:token_dims])


def _with_loss_metadata(model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, LossInputValue]) -> LossInputs:
    result = dict(loss_inputs)
    for name in _LOSS_METADATA:
        value = model_inputs.get(name)
        if isinstance(value, torch.Tensor):
            result[name] = value
    return result


def _select_chunk(value: Any, index: int, num_chunks: int) -> Any:
    """Select one already materialized THD chunk without dropping its batch axis.

    Args:
        value: Tensor leaves use shape [microbatches, ...] when chunked;
            arbitrary nested containers and replicated metadata are accepted.
        index: Pipeline microbatch index.
        num_chunks: Expected leading microbatch extent.

    Returns:
        A matching container whose chunked tensor leaves retain shape [1, ...].
    """
    if isinstance(value, torch.Tensor):
        return value.narrow(0, index, 1) if value.ndim > 0 and value.shape[0] == num_chunks else value
    if isinstance(value, dict):
        return {name: _select_chunk(item, index, num_chunks) for name, item in value.items()}
    if isinstance(value, list):
        return [_select_chunk(item, index, num_chunks) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_chunk(item, index, num_chunks) for item in value)
    return value


def _slice_batch_mapping(
    values: Mapping[str, Any],
    index: int,
    num_chunks: int,
    batch_size: int,
    custom_dims: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Slice batch-aligned tensor leaves into one padded microbatch.

    Args:
        values: Mapping whose batch-aligned tensor leaves have shape
            [batch, ...], except keys listed in ``custom_dims``.
        index: Pipeline microbatch index.
        num_chunks: Number of equal pipeline microbatches.
        batch_size: Full outer batch extent.
        custom_dims: Optional top-level key to batch-axis mapping. A tensor for
            key ``name`` then has shape [..., batch, ...] with the batch axis at
            ``custom_dims[name]``.

    Returns:
        Shallow container copy whose batch-aligned tensors are narrow views;
        scalar and non-batch metadata is replicated.
    """
    custom_dims = custom_dims or {}

    def slice_value(value: Any, dim: int | None = None) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            resolved_dim = dim if dim is not None else (0 if value.shape[0] == batch_size else None)
            if resolved_dim is None:
                return value
            if value.shape[resolved_dim] != batch_size:
                raise ValueError(
                    f"pipeline batch axis {resolved_dim} has length {value.shape[resolved_dim]}, expected {batch_size}"
                )
            chunk_size = batch_size // num_chunks
            return value.narrow(resolved_dim, index * chunk_size, chunk_size)
        if isinstance(value, dict):
            return {name: slice_value(item) for name, item in value.items()}
        if isinstance(value, list):
            return [slice_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(slice_value(item) for item in value)
        return value

    return {
        name: slice_value(value, custom_dims.get(name))
        for name, value in values.items()
        if value is not None and not (isinstance(value, dict) and not value)
    }


def _pipeline_datum_indices(
    model_microbatches: Sequence[Mapping[str, Any]],
    item_to_datum: tuple[int, ...] | None,
    *,
    num_datums: int,
    is_thd: bool,
) -> list[tuple[int, ...]] | None:
    """Map each materialized model microbatch back to its outer Datums."""
    if item_to_datum is None:
        if is_thd:
            return None
        row_count = sum(_padded_microbatch_size(microbatch) for microbatch in model_microbatches)
        if row_count != num_datums:
            return None
        item_to_datum = tuple(range(num_datums))

    if len(item_to_datum) != num_datums or sorted(item_to_datum) != list(range(num_datums)):
        raise ValueError(
            "collater item_to_datum must contain every outer Datum index exactly once; "
            f"got {list(item_to_datum)} for {num_datums} Datums"
        )

    counts = [
        _thd_microbatch_sequence_count(microbatch) if is_thd else _padded_microbatch_size(microbatch)
        for microbatch in model_microbatches
    ]
    if sum(counts) != len(item_to_datum):
        raise ValueError(
            "collater item_to_datum does not match the materialized model items; "
            f"microbatch counts are {counts}, mapping has {len(item_to_datum)} entries"
        )

    result: list[tuple[int, ...]] = []
    start = 0
    for count in counts:
        result.append(item_to_datum[start : start + count])
        start += count
    return result


def _padded_microbatch_size(model_inputs: Mapping[str, Any]) -> int:
    primary = model_inputs[_primary_name(model_inputs)]
    if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
        raise ValueError("padded pipeline microbatch requires a batched primary tensor")
    return int(primary.shape[0])


def _thd_microbatch_sequence_count(model_inputs: Mapping[str, Any]) -> int:
    cu_seqlens = model_inputs.get("cu_seqlens")
    if not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("packed per-Datum routing requires tensor cu_seqlens in every pipeline microbatch")
    valid = cu_seqlens.reshape(-1)
    valid = valid[valid >= 0]
    if valid.numel() < 2:
        raise ValueError("packed pipeline microbatch must contain at least one sequence")
    return int(valid.numel() - 1)


def _materialize_loss_mapping(
    loss_inputs: Mapping[str, LossInputValue],
    layouts: Mapping[str, LossInputLayout],
    unresolved_fields: frozenset[str],
    *,
    index: int,
    num_chunks: int,
    batch_size: int | None,
    datum_indices: tuple[int, ...] | None,
    num_datums: int,
    is_thd: bool,
) -> LossInputs:
    """Materialize token, Datum, and replicated loss fields by semantics."""
    result: LossInputs = {}
    for name, value in loss_inputs.items():
        if name in unresolved_fields:
            if is_thd:
                raise NotImplementedError(
                    f"packed pipeline microbatching requires an explicit layout for loss field {name!r}"
                )
            assert batch_size is not None
            result[name] = _slice_batch_mapping({name: value}, index, num_chunks, batch_size)[name]
            continue
        layout = layouts[name]
        if layout is LossInputLayout.PER_TOKEN:
            if is_thd:
                result[name] = _select_chunk(value, index, num_chunks)
            else:
                assert batch_size is not None
                result[name] = _slice_batch_mapping({name: value}, index, num_chunks, batch_size)[name]
        elif layout is LossInputLayout.PER_DATUM:
            if datum_indices is None:
                raise NotImplementedError(
                    f"pipeline microbatching cannot route per-Datum loss field {name!r} without "
                    "collater item_to_datum metadata"
                )
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != num_datums:
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(f"per-Datum loss field {name!r} must have leading size {num_datums}, got {shape}")
            indices = torch.tensor(datum_indices, dtype=torch.long, device=value.device)
            result[name] = value.index_select(0, indices)
        elif layout is LossInputLayout.REPLICATED:
            result[name] = value
        else:  # pragma: no cover - normalized by Datum/CollatedLossInputs
            raise ValueError(f"unsupported loss layout {layout!r} for field {name!r}")
    return result


def _weighted_numerator(losses: Any, weights: torch.Tensor) -> torch.Tensor:
    if not isinstance(losses, torch.Tensor):
        raise TypeError("loss_fn must return a Tensor, optionally followed by per-Datum outputs")
    if losses.ndim == 0:
        numerator = losses
    elif losses.shape == weights.shape:
        numerator = (losses * weights.to(losses)).sum()
    else:
        raise ValueError(
            "loss_fn must return a scalar local weighted sum or losses with exactly the same shape as weights; "
            f"got losses={tuple(losses.shape)}, weights={tuple(weights.shape)}"
        )
    if losses.device != weights.device:
        raise ValueError("loss_fn losses and weights must be on the same device")
    return numerator


def _output_sequence_lengths(
    datums: Sequence[Datum],
    model_inputs: Mapping[str, Any],
    loss_inputs: Mapping[str, LossInputValue],
    item_to_datum: tuple[int, ...] | None,
    *,
    is_thd: bool,
) -> tuple[
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    torch.Tensor | None,
]:
    """Capture real and padded sequence lengths before CP mutates the batch."""
    if item_to_datum is None:
        return None, None, None
    if item_to_datum != tuple(range(len(datums))):
        raise ValueError("token output restoration requires collater items to preserve Datum order")

    weights = loss_inputs.get("weights")
    if not isinstance(weights, torch.Tensor):
        raise ValueError("token output restoration requires Tensor loss weights")
    if is_thd:
        if isinstance(model_inputs.get("seq_lens"), torch.Tensor):
            real_lengths = _valid_row_values(model_inputs["seq_lens"])
            padded_source = model_inputs.get("seq_lens_padded", model_inputs["seq_lens"])
            if not isinstance(padded_source, torch.Tensor):
                raise ValueError("packed THD seq_lens_padded must be a Tensor")
            padded_lengths = _valid_row_values(padded_source)
        else:
            cu_seqlens = model_inputs.get("cu_seqlens")
            if not isinstance(cu_seqlens, torch.Tensor):
                raise ValueError("packed token outputs require seq_lens or cu_seqlens metadata")
            real_lengths = _lengths_from_cu_seqlens(cu_seqlens)
            padded_cu = model_inputs.get("cu_seqlens_padded", cu_seqlens)
            if not isinstance(padded_cu, torch.Tensor):
                raise ValueError("packed THD cu_seqlens_padded must be a Tensor")
            padded_lengths = _lengths_from_cu_seqlens(padded_cu)
        if len(real_lengths) != len(item_to_datum) or len(padded_lengths) != len(item_to_datum):
            raise ValueError(
                "collater item_to_datum does not match packed token metadata; expected one real and padded "
                "length per Datum, "
                f"got real={real_lengths}, padded={padded_lengths}, Datums={len(item_to_datum)}"
            )
        if any(real < 0 or padded < 0 or real > padded for real, padded in zip(real_lengths, padded_lengths)):
            raise ValueError(
                "packed token metadata requires 0 <= real_length <= padded_length for every Datum; "
                f"got real={real_lengths}, padded={padded_lengths}"
            )
        if sum(padded_lengths) != weights.numel():
            raise ValueError(
                f"packed padded lengths sum to {sum(padded_lengths)}, but loss weights have token width "
                f"{weights.numel()}"
            )
        return real_lengths, padded_lengths, None

    if weights.ndim < 2 or weights.shape[0] != len(item_to_datum):
        raise ValueError("padded token outputs require weights with one row per Datum")
    width = int(weights.shape[1])
    attention_mask = model_inputs.get("attention_mask")
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 2
        and tuple(attention_mask.shape) == tuple(weights.shape[:2])
    ):
        token_mask = attention_mask.to(torch.bool)
        # Keep token routing compact. Turning every active position into a
        # Python int is prohibitively expensive for long-context batches and
        # would penalize ordinary training that never returns token outputs.
        real_lengths = tuple(int(length) for length in token_mask.sum(dim=1).tolist())
    else:
        inferred: list[int] = []
        for datum in datums:
            try:
                inferred.append(datum.seq_len)
            except ValueError:
                inferred.append(width)
        real_lengths = tuple(inferred)
        token_mask = None
    if any(length < 0 or length > width for length in real_lengths):
        raise ValueError(f"Datum token lengths must be within padded width {width}; got {real_lengths}")
    return real_lengths, (width,) * len(item_to_datum), token_mask


def _valid_row_values(values: torch.Tensor, sentinel: int = -1000) -> tuple[int, ...]:
    rows = values.reshape(1, -1) if values.ndim == 1 else values.reshape(values.shape[0], -1)
    result: list[int] = []
    for row in rows:
        result.extend(int(value) for value in row.tolist() if int(value) != sentinel)
    return tuple(result)


def _lengths_from_cu_seqlens(cu_seqlens: torch.Tensor, sentinel: int = -1000) -> tuple[int, ...]:
    rows = cu_seqlens.reshape(1, -1) if cu_seqlens.ndim == 1 else cu_seqlens.reshape(cu_seqlens.shape[0], -1)
    result: list[int] = []
    for row in rows:
        valid = row[row != sentinel]
        if valid.numel() < 2:
            continue
        lengths = valid[1:] - valid[:-1]
        if bool((lengths < 0).any()):
            raise ValueError("cu_seqlens must be monotonically non-decreasing within each row")
        result.extend(int(value) for value in lengths.tolist())
    return tuple(result)


def _loss_sequence_dim_from_weights(weights: torch.Tensor) -> int:
    if weights.ndim == 1:
        return 0
    if weights.ndim == 2:
        return 1
    raise ValueError(f"per-token output weights must be one- or two-dimensional, got {tuple(weights.shape)}")


def _split_restored_token_output(
    tensor: torch.Tensor,
    plan: _OutputRestorePlan,
    *,
    datum_indices: tuple[int, ...],
    seq_dim: int,
) -> list[torch.Tensor]:
    """Split one restored collated token tensor into input-Datum coordinates."""
    if plan.item_to_datum is None:
        if datum_indices != (0,):
            raise NotImplementedError("prebatched token outputs can only produce their single outer Datum record")
        if plan.synthetic_suffix_tokens:
            restored_width = int(tensor.shape[seq_dim])
            if plan.synthetic_suffix_tokens >= restored_width:
                raise ValueError(
                    "HybridEP synthetic output suffix must be shorter than the restored token stream; "
                    f"got suffix={plan.synthetic_suffix_tokens}, width={restored_width}"
                )
            tensor = tensor.narrow(seq_dim, 0, restored_width - plan.synthetic_suffix_tokens).contiguous()
        return [tensor]
    if plan.real_lengths is None or plan.padded_lengths is None:
        raise RuntimeError("token output routing metadata is incomplete")

    if plan.is_thd:
        expected_width = sum(plan.padded_lengths[index] for index in datum_indices)
        if tensor.shape[seq_dim] != expected_width and seq_dim == 1 and tensor.ndim >= 2:
            # Some THD backends restore the caller's pre-flatten [B, S]
            # coordinates. Sequence metadata is a row-major flat stream, so
            # collapse those two token axes before routing individual Datums.
            if tensor.shape[0] * tensor.shape[1] == expected_width:
                tensor = tensor.flatten(0, 1)
                seq_dim = 0
        if tensor.shape[seq_dim] != expected_width:
            raise ValueError(
                f"restored packed token output has width {tensor.shape[seq_dim]}, expected {expected_width}"
            )
        pieces: list[torch.Tensor] = []
        start = 0
        for datum_index in datum_indices:
            real_length = plan.real_lengths[datum_index]
            padded_length = plan.padded_lengths[datum_index]
            piece = tensor.narrow(seq_dim, start, real_length)
            if seq_dim == 1 and piece.shape[0] == 1:
                piece = piece.squeeze(0)
            pieces.append(piece.contiguous())
            start += padded_length
        return pieces

    if tensor.ndim < 2 or tensor.shape[0] != len(datum_indices):
        raise ValueError(f"restored padded token output must have {len(datum_indices)} rows, got {tuple(tensor.shape)}")
    pieces = []
    for row_index, datum_index in enumerate(datum_indices):
        row = tensor.select(0, row_index)
        if plan.token_mask is not None:
            mask = plan.token_mask[datum_index].to(device=row.device)
            pieces.append(row[mask].contiguous())
        else:
            pieces.append(row.narrow(0, 0, plan.real_lengths[datum_index]).contiguous())
    return pieces


def _loss_fn_output_contract(
    outputs: ParsedLossOutputs,
    weights: Any,
    *,
    expected_records: int | None,
    microbatch_index: int | None,
) -> tuple[str | None, tuple[Any, ...]]:
    """Return a local validation error and a rank-comparable output schema."""
    if outputs is None:
        return None, ("none",)
    if expected_records is None:
        return (
            "a prebatched Datum may return outputs only when num_microbatches=1 because its inner sample "
            "boundaries are not part of the Datum contract",
            ("unsupported", type(outputs).__name__),
        )
    if not isinstance(weights, torch.Tensor):
        return "loss_fn outputs require Tensor loss weights", ("invalid-weights", type(weights).__name__)

    if isinstance(outputs, list):
        error = None
        if len(outputs) != expected_records:
            error = (
                f"pipeline loss_fn returned {len(outputs)} outputs for microbatch {microbatch_index}, "
                f"expected {expected_records} from its Datum mapping"
                if microbatch_index is not None
                else f"loss_fn outputs must contain one mapping per Datum; got {len(outputs)} for {expected_records}"
            )
        # Legacy records are deliberately opaque and may contain rank-local
        # fields. Only their presence and routing count participate in CP
        # consensus; typed outputs opt into stronger schema agreement.
        return error, ("legacy", len(outputs))

    error = None
    if weights.ndim not in (1, 2):
        error = f"per-token output weights must be one- or two-dimensional, got {tuple(weights.shape)}"
    records = outputs.per_datum
    if records is not None and len(records) != expected_records:
        error = (
            f"LossFnOutputBatch.per_datum contains {len(records)} records, expected {expected_records}"
            if error is None
            else error
        )
    record_keys = () if records is None else tuple(tuple(sorted(record)) for record in records)
    field_schema: list[tuple[Any, ...]] = []
    for name, spec in sorted(outputs.per_token.items()):
        tensor = spec.tensor
        if tensor.shape[: weights.ndim] != weights.shape and error is None:
            error = (
                f"per-token output {name!r} must start with the loss weight shape {tuple(weights.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != weights.device and error is None:
            error = f"per-token output {name!r} must be on the loss weight device {weights.device}, got {tensor.device}"
        field_schema.append(
            (
                name,
                str(tensor.dtype),
                tensor.device.type,
                tuple(tensor.shape),
                type(spec.fill_value).__name__,
                repr(spec.fill_value),
            )
        )
    return error, ("typed", len(records) if records is not None else None, record_keys, tuple(field_schema))


def _token_mask_schema(token_mask: torch.Tensor | None) -> tuple[tuple[int, ...], str] | None:
    """Return compact, rank-comparable routing metadata for a padded token mask."""
    if token_mask is None:
        return None
    compact = token_mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256(compact.numpy().tobytes()).hexdigest()
    return tuple(compact.shape), digest


def _loss_fn_output_restore_contract(
    outputs: LossFnOutputBatch,
    weights: Any,
    plan: _OutputRestorePlan,
    *,
    datum_indices: tuple[int, ...] | None,
    chunk_index: int | None,
    cp_size: int,
) -> tuple[str | None, tuple[Any, ...]]:
    """Preflight one typed restore before any field enters a CP collective."""
    try:
        if not isinstance(weights, torch.Tensor):
            raise ValueError("loss_fn token outputs require Tensor loss weights")
        seq_dim = _loss_sequence_dim_from_weights(weights)
        if datum_indices is None:
            if plan.item_to_datum is not None:
                raise RuntimeError("loss_fn output routing is missing the collater's Datum mapping")
            if chunk_index is not None:
                raise NotImplementedError(
                    "a prebatched Datum cannot restore token outputs across multiple pipeline microbatches"
                )
            datum_indices = (0,)

        if any(index < 0 for index in datum_indices):
            raise ValueError(f"loss_fn output Datum indices must be non-negative, got {datum_indices}")
        if plan.item_to_datum is not None:
            if plan.real_lengths is None or plan.padded_lengths is None:
                raise RuntimeError("token output routing metadata is incomplete")
            if any(index >= len(plan.real_lengths) or index >= len(plan.padded_lengths) for index in datum_indices):
                raise ValueError(f"loss_fn output Datum indices are out of range: {datum_indices}")

        layout = plan.sharder.shard_layout
        selected_layout = layout
        if chunk_index is not None:
            if layout is None or layout.chunk_layouts is None:
                if cp_size > 1:
                    raise NotImplementedError(
                        "the active context-parallel backend does not report reversible per-pipeline-chunk "
                        "token layouts; typed per-token outputs are unavailable for this packed PP+CP batch"
                    )
                selected_layout = None
            else:
                if chunk_index < 0 or chunk_index >= len(layout.chunk_layouts):
                    raise IndexError(
                        f"pipeline chunk {chunk_index} is out of range for {len(layout.chunk_layouts)} reported CP layouts"
                    )
                selected_layout = layout.chunk_layouts[chunk_index]
        if cp_size > 1 and selected_layout is None:
            raise NotImplementedError(
                "the active context-parallel backend does not report a reversible token layout; "
                "typed per-token outputs are unavailable for this batch"
            )

        local_width = int(weights.shape[seq_dim])
        global_width = local_width * cp_size
        if selected_layout is not None:
            captured = selected_layout.local_token_global_indices
            if captured is not None and captured.numel() != local_width:
                raise ValueError(
                    f"the reported CP layout has {captured.numel()} local token indices, "
                    f"but loss outputs have local width {local_width}"
                )
            if selected_layout.padded_seq_len is not None and selected_layout.padded_seq_len != global_width:
                raise ValueError(
                    f"the reported CP layout has padded width {selected_layout.padded_seq_len}, "
                    f"but loss outputs imply global width {global_width}"
                )

        restored_width = global_width
        if selected_layout is not None:
            if selected_layout.input_token_stream_positions is not None:
                positions = selected_layout.input_token_stream_positions
                if bool((positions < -1).any()) or bool((positions >= global_width).any()):
                    raise ValueError(f"the reported CP position map must contain -1 or indices below {global_width}")
                restored_width = positions.numel() if plan.is_thd else int(positions.shape[1])
            elif selected_layout.input_row_shape is not None:
                restored_width = prod(selected_layout.input_row_shape)
            elif selected_layout.original_seq_len is not None:
                restored_width = selected_layout.original_seq_len
        if (
            plan.is_thd
            and weights.ndim == 2
            and (
                selected_layout is None
                or (selected_layout.input_token_stream_positions is None and selected_layout.input_row_shape is None)
            )
        ):
            # Some model-owned THD paths retain caller [B, S] rows instead of
            # reporting an explicit input_row_shape. Routing flattens those
            # token axes row-major after restoration.
            restored_width *= int(weights.shape[0])

        if plan.item_to_datum is not None:
            assert plan.real_lengths is not None and plan.padded_lengths is not None
            if plan.is_thd:
                expected_width = sum(plan.padded_lengths[index] for index in datum_indices)
                if restored_width != expected_width:
                    raise ValueError(
                        f"restored packed token output has width {restored_width}, expected {expected_width}"
                    )
            else:
                if weights.ndim != 2 or weights.shape[0] != len(datum_indices):
                    raise ValueError(
                        f"restored padded token output must have {len(datum_indices)} rows, "
                        f"got local loss weights {tuple(weights.shape)}"
                    )
                expected_widths = {plan.padded_lengths[index] for index in datum_indices}
                if expected_widths != {restored_width}:
                    raise ValueError(
                        f"restored padded token output has width {restored_width}, expected {sorted(expected_widths)}"
                    )
                positions = selected_layout.input_token_stream_positions if selected_layout is not None else None
                if positions is not None and positions.shape[0] != len(datum_indices):
                    raise ValueError(
                        f"the reported CP position map has {positions.shape[0]} rows, expected {len(datum_indices)}"
                    )

        layout_schema = None
        if selected_layout is not None:
            layout_schema = (
                selected_layout.original_seq_len,
                selected_layout.padded_seq_len,
                selected_layout.input_row_shape,
                (
                    None
                    if selected_layout.input_token_stream_positions is None
                    else tuple(selected_layout.input_token_stream_positions.shape)
                ),
                (
                    None
                    if selected_layout.local_token_global_indices is None
                    else selected_layout.local_token_global_indices.numel()
                ),
            )
        schema = (
            "restore",
            plan.is_thd,
            datum_indices,
            plan.real_lengths,
            plan.padded_lengths,
            _token_mask_schema(plan.token_mask),
            plan.synthetic_suffix_tokens,
            seq_dim,
            tuple(weights.shape),
            layout_schema,
            tuple(sorted(outputs.per_token)),
        )
        return None, schema
    except Exception as error:
        return str(error), ("invalid-restore", type(error).__name__)


def _parse_loss_result(
    result: torch.Tensor | tuple[torch.Tensor, Sequence[Mapping[str, Any]] | LossFnOutputBatch],
    weights: torch.Tensor,
) -> tuple[torch.Tensor, ParsedLossOutputs, Exception | None]:
    """Normalize one loss callback result without applying Datum routing.

    Output-only contract errors are returned separately so distributed peers
    can finish the same backward schedule and reach a common error decision.
    Loss-tensor errors still raise immediately because no valid backward value
    exists in that case.
    """
    if not isinstance(result, tuple):
        return _weighted_numerator(result, weights), None, None
    if not result:
        raise ValueError("loss_fn returned an empty tuple instead of a loss Tensor")
    numerator = _weighted_numerator(result[0], weights)
    if len(result) != 2:
        return numerator, None, ValueError("loss_fn must return a loss Tensor or a two-item (loss, outputs) tuple")
    outputs = result[1]
    try:
        if isinstance(outputs, LossFnOutputBatch):
            detached = LossFnOutputBatch(
                per_token={
                    name: PerTokenOutput(spec.tensor.detach(), fill_value=spec.fill_value)
                    for name, spec in outputs.per_token.items()
                },
                per_datum=(None if outputs.per_datum is None else [_detach(record) for record in outputs.per_datum]),
            )
            return numerator, detached, None
        if (
            not isinstance(outputs, Sequence)
            or isinstance(outputs, (str, bytes))
            or not all(isinstance(item, Mapping) for item in outputs)
        ):
            raise ValueError("loss_fn outputs must be a sequence of mappings")
        return numerator, [_detach(dict(item)) for item in outputs], None
    except Exception as error:
        return numerator, None, error


def _update_output_mode(previous: bool | None, outputs: ParsedLossOutputs) -> bool:
    current = outputs is not None
    if previous is not None and previous != current:
        raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
    return current


def _detach(value: Any) -> Any:
    """Detach tensor leaves without changing an output record's structure."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    return value
