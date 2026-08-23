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

"""Forward/backward execution for Datum accumulation windows."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.datasets.datum import CollatedLossInputs, Datum, LossInputLayout, collate_datums
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
from nemo_automodel.engine._batch import (
    CollateFn,
    LossInputs,
    LossInputValue,
    _is_token_aligned,
    _loss_sequence_dim,
    _LossBatchLayout,
    _materialize_loss_mapping,
    _model_token_template,
    _pad_hybridep_packed_thd,
    _pad_hybridep_padded_sequence,
    _pipeline_datum_indices,
    _primary_name,
    _resolve_loss_batch_layout,
    _select_chunk,
    _slice_batch_mapping,
    _to_device,
    _validate_collated_weights,
    _validate_loss_batch_layout,
    _with_loss_metadata,
)
from nemo_automodel.engine.outputs import (
    EvaluationResult,
    ForwardBackwardResult,
    LossOutput,
    ParsedLossOutputs,
    StepResult,
    _loss_output_contract,
    _loss_output_restore_contract,
    _output_sequence_lengths,
    _OutputRestorePlan,
    _parse_loss_result,
    _split_restored_token_output,
    _token_sequence_dim,
)
from nemo_automodel.shared.import_utils import MISSING_TORCHAO_MSG, safe_import_from

BatchContextFn = Callable[[Mapping[str, Any], Mapping[str, LossInputValue]], AbstractContextManager[Any]]
LossFn = Callable[
    [Any, LossInputs],
    torch.Tensor | LossOutput,
]
OutputFn = Callable[[Any, LossInputs], torch.Tensor]
_LOSS_FIELD_PREFIX = "__engine_loss__"
_OUTPUT_TOKEN_REFERENCE = "__engine_token_reference__"
_T = TypeVar("_T")


def _nullcontext_for_batch(_model_inputs: dict[str, Any]) -> AbstractContextManager[Any]:
    return nullcontext()


def _nullcontext_for_prepared_batch(
    _model_inputs: Mapping[str, Any],
    _loss_fn_inputs: Mapping[str, LossInputValue],
) -> AbstractContextManager[Any]:
    """Return a no-op context for one prepared eager batch.

    Args:
        _model_inputs: CP-local model mapping with primary token shape
            ``[batch, sequence]``, packed ``[1, tokens]``, or a model-owned
            flat THD ``[tokens]`` layout.
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


def _layout_schema(value: Any) -> tuple[Any, ...]:
    """Describe nested input layout without inspecting tensor values."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, Mapping):
        return ("mapping", tuple((str(name), _layout_schema(item)) for name, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_layout_schema(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    return (type(value).__module__, type(value).__qualname__)


@contextmanager
def _model_mode(model_parts: Sequence[nn.Module], *, training: bool):
    """Temporarily set every local model part's train/eval mode."""
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for part in model_parts:
        for module in part.modules():
            if id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
    previous = tuple(module.training for module in modules)
    try:
        for part in model_parts:
            part.train(training)
        yield
    finally:
        # Calling parent.train(mode) would recursively erase intentionally
        # mixed child modes, such as a frozen vision tower kept in eval.
        for module, mode in zip(modules, previous):
            module.training = mode


def _hybridep_capabilities(model_parts: Sequence[nn.Module]) -> tuple[bool, bool]:
    """Resolve live uniform-token dispatch and model-owned packed-CP handling."""
    uses_uniform_tokens = False
    owns_packed_cp_equalization = False
    for part in model_parts:
        modules = getattr(part, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            dispatcher = getattr(module, "token_dispatcher", None)
            uses_uniform_tokens |= getattr(dispatcher, "requires_uniform_token_count", False) is True
            owns_packed_cp_equalization |= bool(getattr(module, "owns_hybridep_packed_cp_equalization", False))
    return uses_uniform_tokens, owns_packed_cp_equalization


def _resolve_hybridep_equalization_groups(
    uses_hybridep: bool,
    mesh_context: MeshContext | None,
) -> tuple[dist.ProcessGroup | None, ...]:
    """Resolve ordered mesh-axis reductions used for HybridEP token equalization.

    HybridEP dispatch itself communicates along ``ep``. With context
    parallelism, however, one CP replica can be split across different EP rows
    when ``ep_size`` and ``cp_size`` do not nest. Reducing first along ``ep``
    and then along ``ep_shard`` computes one maximum over the complete
    non-pipeline MoE mesh without creating a new process group. Consequently,
    every CP replica starts from the same padded global width and every
    HybridEP group receives the same local token extent after CP sharding.
    """

    if not uses_hybridep:
        return ()
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

    groups = [ep_mesh.get_group()]
    cp_size = int(getattr(mesh_context, "cp_size", 1))
    if cp_size > 1:
        if "ep_shard" not in mesh_names:
            raise ValueError("HybridEP with context parallelism requires mesh_context.moe_mesh with an 'ep_shard' axis")
        ep_shard_mesh = moe_mesh["ep_shard"]
        ep_shard_size = int(ep_shard_mesh.size())
        if ep_shard_size > 1:
            groups.append(ep_shard_mesh.get_group())
    return tuple(groups)


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


class Engine:
    """Run model forward or forward/backward over Datum windows.

    The model and distributed topology are already constructed when they are
    passed here. The Engine owns batching and model-parallel execution.
    :meth:`forward` returns restored token outputs without a loss;
    :meth:`evaluate` computes loss statistics without gradients; and
    :meth:`forward_backward` owns global weight normalization,
    gradient-accumulation synchronization, and backward. When optimizers are
    provided, :meth:`step` owns distributed gradient finalization,
    clipping, parameter updates, gradient clearing, model post-step hooks, and
    LR-scheduler advancement. One :meth:`forward_backward` call is the complete
    optimizer accumulation window consumed by :meth:`step`. Dynamic loss
    scaling and overflow-skipped updates are not part of this contract.

    Args:
        model: An already configured and distributed model, or a built
            :class:`AutoPipeline`.
        device: Device on which model inputs and losses are evaluated.
        mesh_context: Runtime topology. Required for an ``AutoPipeline``. For
            eager models, an initialized default process group is treated as
            pure data parallelism when omitted.
        collate_fn: Batches one microbatch of Datums into separate model and
            loss inputs. The default supports padded and packed text. VLMs pass
            a model-specific collater. Existing recipes whose dataloaders
            already collate can use :func:`nemo_automodel.engine.collate_prebatched`; the Engine then
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
            Rank-local function failures are process-fatal in distributed
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
        window, and loss normalization. Callers provide explicit outer Datum
        groups; each group must still satisfy the
        AutoPipeline schedule's fixed inner microbatch constraints. Pipeline
        output mappings are synchronized to every physical PP rank. Magi's
        current packed contract uses one inner pipeline microbatch; recipes
        enforce that configuration.
        Packed pipeline batches with multiple inner microbatches must split at
        sequence boundaries into equal-width token chunks. Token-aligned loss
        fields follow those chunks. Layout-aware collaters may additionally
        route per-Datum scalar fields from THD sequence boundaries and preserve
        replicated fields unchanged. A ``PER_DATUM`` field is copied to every
        CP rank; a loss function combines it with that rank's CP-local token
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
        uses_hybridep, model_owns_hybridep_packed_cp_equalization = _hybridep_capabilities(self.model_parts)
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
        self._hybridep_equalization_groups = _resolve_hybridep_equalization_groups(
            uses_hybridep,
            mesh_context,
        )
        self._model_owns_hybridep_packed_cp_equalization = model_owns_hybridep_packed_cp_equalization
        self.collate_fn = collate_fn
        self.padding_token_id = padding_token_id
        self.mtp_ignore_index = mtp_ignore_index
        self.context_fn = context_fn
        self.batch_context_fn = _nullcontext_for_prepared_batch if batch_context_fn is None else batch_context_fn
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync
        self.optimizers = _as_tuple(optimizers)
        self.lr_schedulers = _as_tuple(lr_schedulers)
        self.max_grad_norm = max_grad_norm
        self._step_consumed = False
        self._finalized_grad_norm: torch.Tensor | float | None = None
        self._step_in_progress = False
        self._backward_status = "idle"

    @torch.no_grad()
    def forward(
        self,
        datums: Sequence[Datum],
        compute_output: OutputFn,
    ) -> list[torch.Tensor]:
        """Compute and restore one token-aligned tensor per Datum.

        ``compute_output`` runs on the physical model output and side inputs
        after padding, packing, pipeline, and context-parallel preparation. It
        returns one tensor whose leading axes match that physical token stream;
        trailing feature axes are preserved. The Engine returns tensors in the
        input Datum order and logical token coordinates.

        Forward Datums need only model inputs and task side inputs. They do not
        need labels, weights, or a placeholder loss.

        Args:
            datums: One non-empty logical batch. Ordinary Datums contain a
                token vector ``[sequence]``; the configured collater may form
                padded ``[batch, sequence]``, packed ``[1, tokens]``, or
                model-owned flat THD ``[tokens]`` physical model inputs.
            compute_output: Computes one Tensor from the raw model output and
                prepared side inputs. Its leading axes must equal the physical
                model token axes: ``[batch, sequence, ...]`` for padded input
                and ``[1, tokens, ...]`` or model-owned ``[tokens, ...]`` for
                THD input. Any trailing feature axes are preserved.

        Returns:
            One detached Tensor per input Datum in input order. Each has shape
            ``[logical_tokens, ...]`` after CP restoration and removal of
            packing or padding positions.
        """
        batch = self._validate_datums(datums)
        if not callable(compute_output):
            raise TypeError("compute_output must be callable")
        with _model_mode(self.model_parts, training=False):
            return self._forward_outputs(batch, compute_output)

    @torch.no_grad()
    def evaluate(
        self,
        batches: Sequence[Sequence[Datum]],
        loss_fn: LossFn,
    ) -> EvaluationResult:
        """Evaluate explicit non-empty Datum batches without backward.

        Args:
            batches: Non-empty outer sequence of non-empty Datum batches. Each
                inner batch is collated and executed once; AutoPipeline may
                split that physical batch internally.
            loss_fn: Computes a scalar local weighted numerator from the raw
                model output and prepared loss inputs. It may return
                :class:`LossOutput` to request token-aligned or opaque batch
                values alongside that numerator.

        Returns:
            Detached model-parallel-complete loss sums plus ``token_outputs``
            and ``batch_outputs`` whose outer length exactly matches
            ``batches``.
        """
        resolved_batches = self._validate_batches(batches)
        with _model_mode(self.model_parts, training=False):
            return self._evaluate(resolved_batches, loss_fn)

    def _evaluate(
        self,
        microbatches: list[list[Datum]],
        loss_fn: LossFn,
    ) -> EvaluationResult:
        """Run a forward-only Datum window.

        The Engine collates each explicit inner Datum batch exactly as in
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
            microbatches: Explicit Datum batches for this evaluation window.
            loss_fn: Computes a scalar local weighted numerator from the raw
                model output and CP-local loss inputs. A :class:`LossOutput`
                may additionally name token streams that the Engine restores
                across CP and one opaque batch mapping that it leaves
                uninterpreted.

        Returns:
            Detached model-parallel-complete loss statistics and explicit
            output channels. The sums and outputs are local to one
            data-parallel replica.
        """
        self._validate_execution_parallelism()
        cp_group, cp_size = self._cp_group_and_size()
        self._validate_window_size_across_group(len(microbatches), cp_group, cp_size)
        weight_sum = self._local_weight_sum(microbatches)
        zero_weight_sum = bool(weight_sum == 0)
        self._validate_pipeline_window(len(microbatches), weight_sum)

        inner_microbatches = self.pipeline.num_microbatches if self.pipeline is not None else 1
        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        result_token_outputs: list[dict[str, list[torch.Tensor]]] = []
        result_batch_outputs: list[Mapping[str, Any] | None] = []

        for batch_datums in microbatches:
            cp_context, model_inputs, loss_inputs, loss_batch_layout, output_restore_plan = self._prepare_batch(
                batch_datums, inner_microbatches
            )
            if self.pipeline is not None:
                token_outputs, batch_output, batch_error = self._pipeline_execute(
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
                result_token_outputs.append(token_outputs)
                result_batch_outputs.append(batch_output)
                continue

            loss_inputs = _with_loss_metadata(model_inputs, loss_inputs)
            with self.context_fn(model_inputs), cp_context(), self.batch_context_fn(model_inputs, loss_inputs):
                forward_inputs = filter_forward_kwargs(self.model, model_inputs)
                output = self.model(**forward_inputs)
                numerator, parsed_outputs, output_parse_error = _parse_loss_result(
                    loss_fn(output, loss_inputs), loss_inputs["weights"]
                )
                self._validate_outputs_across_cp(
                    parsed_outputs,
                    loss_inputs.get("weights"),
                    expected_datums=len(batch_datums),
                    local_error=output_parse_error,
                    restore_plan=output_restore_plan,
                    datum_indices=tuple(range(len(batch_datums))),
                )
                if zero_weight_sum:
                    numerator = numerator * 0
            token_outputs: dict[str, list[torch.Tensor]] = {}
            batch_output = None if parsed_outputs is None else parsed_outputs.batch_output
            if parsed_outputs is not None and parsed_outputs.token_outputs:
                token_outputs = self._restore_outputs(
                    parsed_outputs,
                    loss_inputs["weights"],
                    output_restore_plan,
                    datum_indices=tuple(range(len(batch_datums))),
                    chunk_index=None,
                )
                if any(len(values) != len(batch_datums) for values in token_outputs.values()):
                    raise ValueError("every loss output field must contain one tensor per Datum")
            result_token_outputs.append(token_outputs)
            result_batch_outputs.append(batch_output)
            local_loss_sum.add_(numerator.detach().to(torch.float64))

        if cp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=cp_group)
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=pp_group)

        return EvaluationResult(
            loss_sum=local_loss_sum.detach(),
            weight_sum=weight_sum.detach(),
            token_outputs=result_token_outputs,
            batch_outputs=result_batch_outputs,
        )

    def _forward_outputs(self, datums: list[Datum], compute_output: OutputFn) -> list[torch.Tensor]:
        """Execute one scoring batch and restore its physical token output."""
        self._validate_execution_parallelism()
        cp_group, cp_size = self._cp_group_and_size()
        self._validate_forward_batch_across_model_parallel(datums, cp_group, cp_size)
        inner_microbatches = self.pipeline.num_microbatches if self.pipeline is not None else 1
        cp_context, model_inputs, side_inputs, batch_layout, restore_plan = self._prepare_batch(
            datums,
            inner_microbatches,
            require_loss=False,
        )

        def compute_as_loss(output: Any, prepared_inputs: LossInputs) -> LossOutput:
            task_inputs = {name: value for name, value in prepared_inputs.items() if name != _OUTPUT_TOKEN_REFERENCE}
            tensor = compute_output(output, task_inputs)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("compute_output must return a torch.Tensor")
            token_reference = prepared_inputs[_OUTPUT_TOKEN_REFERENCE]
            zero = token_reference.new_zeros((), dtype=torch.float32)
            return LossOutput(loss_sum=zero, token_outputs={"output": tensor})

        if self.pipeline is not None:
            local_zero = torch.zeros((), dtype=torch.float64, device=self.device)
            fields, batch_output, error = self._pipeline_execute(
                model_inputs,
                side_inputs,
                datums,
                compute_as_loss,
                local_zero,
                cp_context,
                batch_layout,
                restore_plan,
                backward_scale=None,
                zero_weight_sum=False,
                token_reference_key=_OUTPUT_TOKEN_REFERENCE,
            )
            if error is not None:
                raise error
            if batch_output is not None:
                raise RuntimeError("internal forward unexpectedly produced a batch output")
            if set(fields) != {"output"} or len(fields["output"]) != len(datums):
                raise RuntimeError("pipeline forward did not return exactly one token output per Datum")
            return fields["output"]

        token_reference = side_inputs.pop(_OUTPUT_TOKEN_REFERENCE)
        task_inputs = _with_loss_metadata(model_inputs, side_inputs)
        with self.context_fn(model_inputs), cp_context(), self.batch_context_fn(model_inputs, task_inputs):
            forward_inputs = filter_forward_kwargs(self.model, model_inputs)
            output = self.model(**forward_inputs)
            parsed_output: LossOutput | None = None
            output_error: Exception | None = None
            try:
                parsed_output = compute_as_loss(output, {**task_inputs, _OUTPUT_TOKEN_REFERENCE: token_reference})
            except Exception as error:
                output_error = error
        self._validate_outputs_across_cp(
            parsed_output,
            token_reference,
            expected_datums=len(datums),
            local_error=output_error,
            restore_plan=restore_plan,
            datum_indices=tuple(range(len(datums))),
        )
        assert parsed_output is not None
        fields = self._restore_outputs(
            parsed_output,
            token_reference,
            restore_plan,
            datum_indices=tuple(range(len(datums))),
            chunk_index=None,
        )
        if set(fields) != {"output"} or len(fields["output"]) != len(datums):
            raise RuntimeError("forward did not return exactly one token output per Datum")
        return fields["output"]

    def forward_backward(
        self,
        batches: Sequence[Sequence[Datum]],
        loss_fn: LossFn,
        *,
        accumulate_gradients: bool = False,
    ) -> ForwardBackwardResult:
        """Run one complete optimizer accumulation window.

        A second call with configured optimizers is rejected until
        :meth:`step` consumes the first call's gradients unless
        ``accumulate_gradients`` is true. A failed partial backward poisons the
        window so its gradients cannot be reused.

        Args:
            batches: Non-empty outer sequence of explicit non-empty Datum
                batches in the complete optimizer window. Each inner batch is
                collated once into padded ``[batch, sequence, ...]``, packed
                ``[1, tokens, ...]``, or model-owned flat THD
                ``[tokens, ...]`` physical tensors.
            loss_fn: Computes a scalar local weighted numerator from the raw
                model output and CP-local loss inputs. A :class:`LossOutput`
                may additionally expose token-aligned physical tensors or one
                opaque mapping for that outer batch.
            accumulate_gradients: Whether to add this complete window's
                gradients to a preceding successful call before one shared
                :meth:`step`. Each call retains its own loss
                normalization. The default preserves the one-call-per-step
                safety check.

        Returns:
            Globally normalized loss statistics. ``token_outputs`` and
            ``batch_outputs`` have the same outer length as ``batches``;
            token fields contain one logical ``[tokens, ...]`` Tensor per
            input Datum.
        """
        resolved_batches = self._validate_batches(batches)
        if not isinstance(accumulate_gradients, bool):
            raise TypeError("accumulate_gradients must be a bool")
        if self.optimizers:
            if self._backward_status == "ready" and not accumulate_gradients:
                raise RuntimeError("forward_backward already produced the current optimizer window; call step first")
            if self._backward_status == "broken":
                raise RuntimeError("the previous backward window failed and this Engine cannot be reused")
            if self._backward_status == "running":
                raise RuntimeError("a forward_backward call is already running")
        if self._finalized_grad_norm is not None:
            raise RuntimeError("gradients were already finalized; retry step before forward_backward")
        if self.optimizers:
            self._backward_status = "running"
        try:
            with _model_mode(self.model_parts, training=True):
                result = self._forward_backward_window(resolved_batches, loss_fn)
        except BaseException:
            if self.optimizers:
                self._backward_status = "broken"
            raise
        self._step_consumed = False
        self._finalized_grad_norm = None
        if self.optimizers:
            self._backward_status = "ready"
        return result

    def _forward_backward_window(
        self,
        microbatches: list[list[Datum]],
        loss_fn: LossFn,
    ) -> ForwardBackwardResult:
        """Accumulate gradients for a complete optimizer window.

        ``microbatches`` is the explicit optimizer accumulation window. The
        Engine invokes ``collate_fn`` once per group. A Datum normally
        represents one sample. Recipes that retain worker-side collation wrap
        each prepared batch in a one-Datum group with
        :func:`nemo_automodel.engine.collate_prebatched`.

        ``loss_fn`` receives the raw model output and CP-local
        ``loss_fn_inputs``. It returns a scalar local weighted-sum numerator.
        The loss function must apply weights and masks; the Engine
        applies only global normalization. It may
        also return a :class:`LossOutput`. Named token streams in that envelope
        are detached, restored from CP-local to full token order, and split
        into one tensor per Datum. Under pipeline parallelism the last stage
        performs CP restoration before broadcasting the field mapping to every
        stage in the pipeline group.

        Args:
            microbatches: Non-empty outer sequence of explicit Datum batches
                in the complete optimizer window. Each inner batch is collated
                once. Ordinary padded loss fields use ``[batch, sequence, ...]``;
                THD fields use ``[1, tokens, ...]`` or a model-owned flat
                ``[tokens, ...]`` layout.
            loss_fn: Computes a scalar local weighted-sum numerator from the
                raw model output and collated loss inputs. A
                :class:`LossOutput` may also expose physical token tensors or
                one caller-owned mapping for the current outer batch.

        Returns:
            Structured loss statistics and optional outputs. ``loss_sum`` is
            reduced across the DP-CP gradient group, while ``weight_sum`` is
            reduced across DP only so replicated CP weights are not counted
            twice. Both are synchronized across PP stages, and ``loss`` is
            their safe quotient.
            ``token_outputs`` and ``batch_outputs`` align one-to-one with the
            explicit outer batches; pipeline execution returns the same values
            on every physical stage rank in that replica. Model
            parameters are unchanged, but their gradients contain the complete
            window's globally normalized backward result.
        """
        self._validate_parallelism()
        dp_group, dp_size = self._dp_group_and_size()
        grad_group, grad_group_size = self._gradient_group_and_size(dp_group, dp_size)
        gradient_reduction_multiplier = 1 if self._summed_gradient_reduction else grad_group_size
        self._validate_window_size_across_group(len(microbatches), grad_group, grad_group_size)
        denominator = self._local_weight_sum(microbatches)
        if dp_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=dp_group)
        zero_denominator = bool(denominator == 0)
        safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        self._validate_pipeline_window(len(microbatches), denominator)

        pp_enabled = self.pipeline is not None
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=pp_enabled)
        inner_microbatches = self.pipeline.num_microbatches if self.pipeline is not None else 1
        effective_total_microbatches = len(microbatches) * inner_microbatches
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            self._cp_size() * gradient_reduction_multiplier / (grad_group_size * effective_total_microbatches)
        )

        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        result_token_outputs: list[dict[str, list[torch.Tensor]]] = []
        result_batch_outputs: list[Mapping[str, Any] | None] = []
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
                token_outputs, batch_output, batch_error = self._pipeline_execute(
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
                            result_token_outputs.append(token_outputs)
                            result_batch_outputs.append(batch_output)
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
                        self._validate_outputs_across_cp(
                            parsed_outputs,
                            loss_inputs.get("weights"),
                            expected_datums=len(datums),
                            local_error=output_parse_error,
                            restore_plan=output_restore_plan,
                            datum_indices=tuple(range(len(datums))),
                        )
                    if zero_denominator:
                        numerator = numerator * 0
                    (numerator * backward_scale).backward()

                if output_error is None:
                    try:
                        if dp_size > 1:
                            self._validate_outputs_across_cp(
                                parsed_outputs,
                                loss_inputs.get("weights"),
                                expected_datums=len(datums),
                                local_error=output_parse_error,
                                restore_plan=output_restore_plan,
                                datum_indices=tuple(range(len(datums))),
                            )
                        token_outputs: dict[str, list[torch.Tensor]] = {}
                        batch_output = None if parsed_outputs is None else parsed_outputs.batch_output
                        if parsed_outputs is not None and parsed_outputs.token_outputs:
                            token_outputs = self._restore_outputs(
                                parsed_outputs,
                                loss_inputs["weights"],
                                output_restore_plan,
                                datum_indices=tuple(range(len(datums))),
                                chunk_index=None,
                            )
                            if any(len(values) != len(datums) for values in token_outputs.values()):
                                raise ValueError("every loss output field must contain one tensor per Datum")
                        result_token_outputs.append(token_outputs)
                        result_batch_outputs.append(batch_output)
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
            token_outputs=result_token_outputs,
            batch_outputs=result_batch_outputs,
        )

    @torch.no_grad()
    def step(
        self,
        *,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> StepResult:
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
            raise RuntimeError("Engine.step requires at least one optimizer")
        if before_optimizer_step is not None and not callable(before_optimizer_step):
            raise TypeError("before_optimizer_step must be callable or None")
        if self._step_in_progress:
            raise RuntimeError("Engine.step is already running")
        if self._backward_status == "broken":
            raise RuntimeError("the previous backward window failed and this Engine cannot be optimized")
        if self._backward_status == "running":
            raise RuntimeError("a forward_backward call is still running")
        if self._step_consumed:
            raise RuntimeError("step already consumed the current gradients; run forward_backward first")

        device_mesh = self.mesh_context.device_mesh if self.mesh_context is not None else None
        moe_mesh = self.mesh_context.moe_mesh if self.mesh_context is not None else None
        dp_group, dp_size = self._dp_group_and_size()
        _, grad_group_size = self._gradient_group_and_size(dp_group, dp_size)
        pp_enabled = self.pipeline is not None
        self._step_in_progress = True
        mutation_started = False
        try:
            if self._finalized_grad_norm is None:
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
            if mutation_started or self._finalized_grad_norm is None:
                self._step_consumed = True
                self._backward_status = "broken"
            raise
        finally:
            self._step_in_progress = False

        self._step_consumed = True
        self._finalized_grad_norm = None
        self._backward_status = "idle"
        return StepResult(grad_norm=grad_norm, learning_rates=learning_rates)

    @staticmethod
    def _validate_datums(datums: Sequence[Datum]) -> list[Datum]:
        """Return one explicit non-empty Datum batch."""
        if not isinstance(datums, Sequence) or isinstance(datums, (str, bytes)) or not datums:
            raise ValueError("Engine requires a non-empty sequence of Datum")
        if not all(isinstance(datum, Datum) for datum in datums):
            raise TypeError("Engine received a value that is not a Datum")
        return list(datums)

    @classmethod
    def _validate_batches(cls, batches: Sequence[Sequence[Datum]]) -> list[list[Datum]]:
        """Return explicit non-empty batches without inferring boundaries."""
        if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)) or not batches:
            raise ValueError("Engine requires a non-empty sequence of Datum batches")
        return [cls._validate_datums(batch) for batch in batches]

    def _hybridep_equalization_target(
        self,
        model_inputs: Mapping[str, Any],
    ) -> tuple[bool, int] | None:
        """Return the MoE-stage-wide physical token shape required by HybridEP.

        Every rank enters fixed-size extrema collectives before local THD
        validation. At CP size one the reduction is scoped to the exact EP
        group. At CP size greater than one it runs in the fixed ``ep`` then
        ``ep_shard`` order, producing one target over the complete non-PP MoE
        mesh. Raw THD batches use the maximum physical token width. Padded
        batches may vary only in sequence width: all ranks keep their existing
        batch size and pad to the maximum width. This preserves dynamic
        batching's token budget instead of materializing ``max(B) * max(S)``.
        DSV4 keeps its existing model-owned packed-CP equalization path.

        Args:
            model_inputs: Unsharded collater output. ``input_ids`` is a Tensor
                of shape ``[batch, tokens]``, or ``inputs_embeds`` is a Tensor
                of shape ``[batch, tokens, hidden]``. The generic raw-THD path
                requires ``batch == 1`` and ``seq_lens`` plus
                ``seq_lens_padded`` as Tensors of shape ``[1, documents]``.
                Model-owned layouts need only rank-consistent packed flags;
                their remaining fields are opaque after that consensus. All
                tensors are local to the rank before CP sharding.

        Returns:
            An ``(is_thd, sequence_width)`` target to materialize before CP
            sharding, or ``None`` when no Engine padding is needed.
        """
        if self.pipeline is not None and self._pipeline_uses_hybridep:
            if model_inputs.get("qkv_format") == "thd":
                raise NotImplementedError("HybridEP packed token equalization currently supports eager PP size 1 only")
            return None
        if not self._hybridep_equalization_groups:
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
        # All metadata are nonnegative. Packing each value with its negation
        # obtains both extrema through one MAX per mesh axis, keeping this
        # per-outer-microbatch guard to one collective at CP=1 and two at CP>1.
        extrema = torch.cat((local, -local))
        for group in self._hybridep_equalization_groups:
            dist.all_reduce(extrema, op=dist.ReduceOp.MAX, group=group)
        upper = extrema[: local.numel()].cpu().tolist()
        lower = (-extrema[local.numel() :]).cpu().tolist()

        if lower[0] != upper[0]:
            raise ValueError("HybridEP ranks must all use packed THD inputs or all use non-packed inputs")
        if upper[0] == 0:
            if lower[2] != 1 or lower[3] <= 0 or lower[4] <= 0:
                raise ValueError(
                    "HybridEP padded equalization requires a non-empty batched primary tensor on every rank; "
                    f"got metadata extrema min={lower}, max={upper}"
                )
            if lower[3] != upper[3]:
                raise NotImplementedError(
                    "HybridEP padded equalization requires the same batch size on every participating rank; "
                    "use token-budget grouping with a common number of samples"
                )
            if lower[4] == upper[4]:
                return None
            return False, int(upper[4])
        if self._cp_size() > 1 and self._model_owns_hybridep_packed_cp_equalization:
            # All ranks completed the packed/non-packed consensus above,
            # preventing a split before the model's own collective. Do not
            # impose the generic raw-THD/B=1 contract: the model validates and
            # repads its own source layout before equalizing it.
            return None
        if lower[1] != 1:
            raise NotImplementedError(
                "HybridEP packed token equalization currently requires raw THD seq_lens/seq_lens_padded inputs"
            )
        if lower[2] != 1 or lower[3] != 1 or upper[3] != 1 or lower[4] <= 0:
            raise ValueError(
                "HybridEP packed token equalization requires one non-empty raw THD token row per EP rank; "
                f"got metadata extrema min={lower}, max={upper}"
            )
        return True, int(upper[4])

    def _prepare_batch(
        self,
        datums: list[Datum],
        num_pipeline_microbatches: int,
        *,
        require_loss: bool = True,
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
            and loss tensors use the same padded, packed THD, indexed-mask,
            Magi, or model-owned local sequence layout. Indexed-mask packing
            remains physical ``[1, tokens]`` and requires eager CP1/PP1.
        """
        model_inputs, collated_loss_inputs = self.collate_fn(datums)
        packing_layout = (
            getattr(collated_loss_inputs, "_engine_packing_layout", None)
            if isinstance(collated_loss_inputs, CollatedLossInputs)
            else None
        )
        if packing_layout not in (None, "indexed_mask"):
            raise ValueError(f"unsupported Engine packing layout {packing_layout!r}")
        indexed_mask_packing = packing_layout == "indexed_mask"
        if indexed_mask_packing:
            if self._cp_size() > 1:
                raise NotImplementedError("indexed_mask packing currently requires context-parallel size 1")
            if self.pipeline is not None:
                raise NotImplementedError("indexed_mask packing currently requires eager execution with PP size 1")
            if self._hybridep_equalization_groups:
                raise NotImplementedError("indexed_mask packing is not supported with HybridEP")
            if model_inputs.get("qkv_format") == "thd":
                raise ValueError("indexed_mask packing cannot be combined with qkv_format='thd'")
            packed_seq_ids = model_inputs.get("_packed_seq_ids")
            if not isinstance(packed_seq_ids, torch.Tensor):
                raise TypeError("indexed_mask packing requires Tensor _packed_seq_ids")
            primary = model_inputs.get("inputs_embeds", model_inputs.get("input_ids"))
            if not isinstance(primary, torch.Tensor) or primary.ndim < 2 or primary.shape[0] != 1:
                shape = tuple(primary.shape) if isinstance(primary, torch.Tensor) else type(primary).__name__
                raise ValueError(f"indexed_mask packing requires a [1, tokens, ...] primary tensor, got {shape}")
            item_to_datum = (
                collated_loss_inputs.item_to_datum if isinstance(collated_loss_inputs, CollatedLossInputs) else None
            )
            if item_to_datum != tuple(range(len(datums))):
                raise ValueError("indexed_mask packing requires identity item_to_datum metadata")
            attention_mask = model_inputs.get("attention_mask")
            if not isinstance(attention_mask, torch.Tensor) or not torch.equal(attention_mask, packed_seq_ids):
                raise ValueError("indexed_mask packing requires attention_mask and _packed_seq_ids to match")
        else:
            packed_seq_ids = None
        hybridep_target = self._hybridep_equalization_target(model_inputs)
        hybridep_suffix_tokens = 0
        if hybridep_target is not None:
            hybridep_is_thd, hybridep_target_width = hybridep_target
            token_template = _model_token_template(model_inputs)
            if token_template.ndim != 2:
                raise ValueError("HybridEP equalization requires a two-dimensional token template")
            hybridep_suffix_tokens = hybridep_target_width - int(token_template.shape[1])
        loss_batch_layout = _resolve_loss_batch_layout(
            datums,
            model_inputs,
            collated_loss_inputs,
            require_weights=require_loss,
        )
        loss_inputs = dict(collated_loss_inputs)
        weight_layout = loss_batch_layout.fields.get("weights")
        if (
            require_loss
            and weight_layout is not LossInputLayout.PER_TOKEN
            and not (weight_layout is LossInputLayout.REPLICATED and "weights" in loss_batch_layout.unresolved_fields)
        ):
            raise ValueError("loss input 'weights' must use the PER_TOKEN layout")
        if "labels" in loss_inputs and loss_batch_layout.fields["labels"] is not LossInputLayout.PER_TOKEN:
            raise ValueError("loss input 'labels' must use the PER_TOKEN layout")
        if require_loss:
            _validate_collated_weights(datums, loss_inputs)
        token_reference_name, loss_seq_dim = _validate_loss_batch_layout(
            datums,
            model_inputs,
            loss_inputs,
            loss_batch_layout.fields,
        )
        if hybridep_target is not None:
            if loss_seq_dim is None:
                loss_seq_dim = _model_token_template(model_inputs).ndim - 1
            if hybridep_is_thd:
                _pad_hybridep_packed_thd(
                    model_inputs,
                    loss_inputs,
                    loss_batch_layout.fields,
                    loss_batch_layout.pad_values,
                    loss_seq_dim=loss_seq_dim,
                    target_tokens=hybridep_target_width,
                    padding_token_id=self.padding_token_id,
                )
            else:
                candidate_model_inputs, candidate_loss_inputs = dict(model_inputs), dict(loss_inputs)
                padding_error: Exception | None = None
                try:
                    _pad_hybridep_padded_sequence(
                        candidate_model_inputs,
                        candidate_loss_inputs,
                        loss_batch_layout.fields,
                        loss_batch_layout.pad_values,
                        target_sequence_length=hybridep_target_width,
                        padding_token_id=self.padding_token_id,
                    )
                except Exception as error:
                    padding_error = error
                failure = 2 if isinstance(padding_error, NotImplementedError) else int(padding_error is not None)
                padding_status = torch.tensor(failure, dtype=torch.int64, device=self.device)
                for group in self._hybridep_equalization_groups:
                    dist.all_reduce(padding_status, op=dist.ReduceOp.MAX, group=group)
                if padding_status.item():
                    detail = (
                        str(padding_error)
                        if padding_error is not None
                        else "another participating rank rejected its local batch"
                    )
                    error_type = NotImplementedError if padding_status.item() == 2 else ValueError
                    raise error_type(f"HybridEP sequence-padding preflight failed: {detail}")
                model_inputs, loss_inputs = candidate_model_inputs, candidate_loss_inputs
        output_routing = _output_sequence_lengths(
            datums,
            model_inputs,
            _model_token_template(model_inputs),
            loss_batch_layout.item_to_datum,
            is_thd=model_inputs.get("qkv_format") == "thd",
            packed_seq_ids=packed_seq_ids,
        )
        model_inputs = _to_device(model_inputs, self.device)
        loss_inputs = _to_device(loss_inputs, self.device)
        token_reference = (
            loss_inputs[token_reference_name]
            if require_loss and token_reference_name is not None
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
                if isinstance(candidate, torch.Tensor) and _loss_sequence_dim(model_inputs, candidate) is not None:
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
        if not require_loss:
            loss_inputs[_OUTPUT_TOKEN_REFERENCE] = local_labels
            loss_batch_layout = _LossBatchLayout(
                fields={**loss_batch_layout.fields, _OUTPUT_TOKEN_REFERENCE: LossInputLayout.PER_TOKEN},
                item_to_datum=loss_batch_layout.item_to_datum,
                pad_values=loss_batch_layout.pad_values,
                unresolved_fields=loss_batch_layout.unresolved_fields,
            )
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
        prepared_weights = loss_inputs.get("weights")
        if (
            self.pipeline is None
            and hybridep_target is not None
            and not hybridep_is_thd
            and isinstance(prepared_weights, torch.Tensor)
            and self._cp_size() > 1
        ):
            local_tokens = prepared_weights.numel()
            extent = torch.tensor((local_tokens, -local_tokens), dtype=torch.int64, device=prepared_weights.device)
            for group in self._hybridep_equalization_groups:
                dist.all_reduce(extent, op=dist.ReduceOp.MAX, group=group)
            if extent[0].item() != -extent[1].item():
                raise RuntimeError("HybridEP context-parallel preparation produced unequal local token extents")
        real_lengths, padded_lengths, token_mask = output_routing
        return (
            cp_context,
            model_inputs,
            loss_inputs,
            loss_batch_layout,
            _OutputRestorePlan(
                sharder=sharder,
                is_thd=is_thd,
                packed_seq_ids=packed_seq_ids,
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
        token_reference_key: str = "weights",
    ) -> tuple[dict[str, list[torch.Tensor]], Mapping[str, Any] | None, Exception | None]:
        """Run prepared pipeline microbatches in training or forward-only mode.

        Args:
            model_inputs: CP-prepared outer-batch model inputs.
            loss_inputs: CP-prepared outer-batch loss inputs.
            datums: Datum items represented by the outer batch.
            loss_fn: Model-output loss function.
            local_loss_sum: Accumulator updated with detached numerators.
            cp_context: Context covering the complete pipeline schedule.
            loss_batch_layout: Semantic layout of every loss field plus the
                collater's logical item-to-Datum routing, when available.
            output_restore_plan: Captured full-token routing and CP sharder for
                explicit per-token outputs.
            backward_scale: Multiplier returned to the training schedule for
                backward, or ``None`` to run the forward-only schedule.
            zero_weight_sum: Whether reporting numerators must be forced to
                graph-connected zero.

        Returns:
            Restored token fields, an optional caller-owned batch mapping, and
            any post-schedule output-restoration error synchronized across PP
            stages.
        """
        parsed_outputs_by_microbatch: list[ParsedLossOutputs] = [None] * self.pipeline.num_microbatches
        output_parse_errors_by_microbatch: list[Exception | None] = [None] * self.pipeline.num_microbatches
        loss_called_by_microbatch = [False] * self.pipeline.num_microbatches

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
                context_inputs = [
                    (
                        {name: value for name, value in loss_inputs_mb.items() if name != token_reference_key}
                        if token_reference_key == _OUTPUT_TOKEN_REFERENCE
                        else loss_inputs_mb
                    )
                    for loss_inputs_mb in loss_microbatches
                ]
                batch_context_fns: list[Callable[[], AbstractContextManager[Any]]] = [
                    partial(self.batch_context_fn, model_inputs_mb, side_inputs_mb)
                    for model_inputs_mb, side_inputs_mb in zip(model_microbatches, context_inputs)
                ]

                def pipeline_loss(output: Any, microbatch_index: int) -> torch.Tensor:
                    loss_inputs_mb = loss_microbatches[microbatch_index]
                    with batch_context_fns[microbatch_index]():
                        numerator, parsed_outputs, output_parse_error = _parse_loss_result(
                            loss_fn(output, loss_inputs_mb), loss_inputs_mb[token_reference_key]
                        )
                    if loss_called_by_microbatch[microbatch_index]:
                        raise RuntimeError(f"pipeline evaluated loss_fn twice for microbatch {microbatch_index}")
                    loss_called_by_microbatch[microbatch_index] = True
                    parsed_outputs_by_microbatch[microbatch_index] = parsed_outputs
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

        token_outputs: dict[str, list[torch.Tensor]] = {}
        batch_output: Mapping[str, Any] | None = None
        outputs_by_field: dict[str, list[torch.Tensor | None]] = {}
        expected_fields: frozenset[str] | None = None
        returns_token_outputs: bool | None = None
        serialized_outputs: bytes | None = None
        output_error: Exception | None = None
        if self.pipeline.info.has_last_stage:
            try:
                for microbatch_index, parsed_outputs in enumerate(parsed_outputs_by_microbatch):
                    local_error = output_parse_errors_by_microbatch[microbatch_index]
                    if not loss_called_by_microbatch[microbatch_index]:
                        local_error = RuntimeError(
                            f"pipeline did not evaluate loss_fn for microbatch {microbatch_index}"
                        )
                    token_reference = loss_microbatches[microbatch_index].get(token_reference_key)
                    datum_indices = (
                        None if datum_indices_by_microbatch is None else datum_indices_by_microbatch[microbatch_index]
                    )
                    expected_datums = (
                        len(datum_indices)
                        if datum_indices is not None
                        else (1 if len(datums) == 1 and self.pipeline.num_microbatches == 1 else None)
                    )
                    self._validate_outputs_across_cp(
                        parsed_outputs,
                        token_reference,
                        expected_datums=expected_datums,
                        local_error=local_error,
                        restore_plan=output_restore_plan,
                        datum_indices=datum_indices,
                        chunk_index=(microbatch_index if is_thd and self.pipeline.num_microbatches > 1 else None),
                    )
                    current_returns_tokens = bool(parsed_outputs is not None and parsed_outputs.token_outputs)
                    if returns_token_outputs is None:
                        returns_token_outputs = current_returns_tokens
                    elif returns_token_outputs != current_returns_tokens:
                        raise ValueError("pipeline loss must return token outputs for every inner microbatch or none")
                    if parsed_outputs is None:
                        continue
                    if parsed_outputs.batch_output is not None:
                        if self.pipeline.num_microbatches > 1:
                            raise NotImplementedError(
                                "pipeline batch_output requires num_microbatches=1 because caller-owned values "
                                "cannot be merged without exposing physical pipeline microbatches"
                            )
                        # LossOutput freezes its public mappings. Cross the PP
                        # serialization boundary with an ordinary mapping so
                        # pickle and recursive tensor movement stay generic.
                        batch_output = dict(parsed_outputs.batch_output)
                    restored = (
                        self._restore_outputs(
                            parsed_outputs,
                            loss_microbatches[microbatch_index][token_reference_key],
                            output_restore_plan,
                            datum_indices=datum_indices,
                            chunk_index=(microbatch_index if is_thd and self.pipeline.num_microbatches > 1 else None),
                        )
                        if parsed_outputs.token_outputs
                        else {}
                    )
                    if datum_indices is None:
                        datum_indices = (0,)
                    fields = frozenset(restored)
                    if expected_fields is None:
                        expected_fields = fields
                        outputs_by_field = {name: [None] * len(datums) for name in fields}
                    elif fields != expected_fields:
                        raise ValueError("pipeline loss outputs must use the same fields for every microbatch")
                    for name, pieces in restored.items():
                        if len(pieces) != len(datum_indices):
                            raise RuntimeError(f"restored token output {name!r} does not match Datum routing")
                        for datum_index, piece in zip(datum_indices, pieces):
                            if outputs_by_field[name][datum_index] is not None:
                                raise RuntimeError(
                                    f"pipeline returned more than one {name!r} output for Datum {datum_index}"
                                )
                            outputs_by_field[name][datum_index] = piece
                if returns_token_outputs:
                    for name, values in outputs_by_field.items():
                        if any(value is None for value in values):
                            raise RuntimeError(
                                f"pipeline schedule did not return exactly one {name!r} output for every Datum"
                            )
                        token_outputs[name] = [value for value in values if value is not None]
                if token_outputs or batch_output is not None:
                    # broadcast_object_list serializes only on the source
                    # stage. Preflight while errors can still be propagated to
                    # every PP stage instead of leaving peers in the broadcast.
                    # Broadcast these already validated bytes so arbitrary
                    # field tensors are never pickled again inside a PP
                    # collective.
                    serialized_outputs = pickle.dumps(
                        _to_device(
                            {"token_outputs": token_outputs, "batch_output": batch_output},
                            torch.device("cpu"),
                        )
                    )
            except Exception as error:
                output_error = error
        output_error = self._synchronize_pipeline_output_error(output_error)
        if output_error is None:
            token_outputs, batch_output = self._broadcast_pipeline_outputs(
                token_outputs,
                batch_output,
                serialized_outputs=serialized_outputs,
            )
        return token_outputs, batch_output, output_error

    def _restore_outputs(
        self,
        outputs: LossOutput,
        token_reference: torch.Tensor,
        plan: _OutputRestorePlan,
        *,
        datum_indices: tuple[int, ...] | None,
        chunk_index: int | None,
    ) -> dict[str, list[torch.Tensor]]:
        """Restore each token field into one tensor per Datum."""
        # _validate_outputs_across_cp has already checked this frozen
        # output envelope on every CP rank before any field enters a gather.
        if datum_indices is None:
            datum_indices = (0,)

        local_seq_dim = _token_sequence_dim(token_reference)
        selected_layout = plan.sharder.shard_layout
        if chunk_index is not None:
            selected_layout = (
                None
                if selected_layout is None or selected_layout.chunk_layouts is None
                else selected_layout.chunk_layouts[chunk_index]
            )

        restored_fields: list[tuple[str, torch.Tensor, int]] = []
        for name in sorted(outputs.token_outputs):
            tensor = outputs.token_outputs[name]
            restored_seq_dim = local_seq_dim
            if selected_layout is not None:
                tensor = plan.sharder.gather_token_tensor(
                    tensor,
                    seq_dim=local_seq_dim,
                    trim=True,
                    fill=0,
                    chunk_index=chunk_index,
                )
                if selected_layout.input_row_shape is not None:
                    restored_seq_dim = len(selected_layout.input_row_shape) - 1
            restored_fields.append((name, tensor, restored_seq_dim))

        # Complete every field's CP collective before Datum-local splitting.
        # A routing error can then no longer leave a peer entering the next
        # field gather while this rank exits early.
        restored: dict[str, list[torch.Tensor]] = {}
        for name, tensor, seq_dim in restored_fields:
            pieces = _split_restored_token_output(
                tensor,
                plan,
                datum_indices=datum_indices,
                seq_dim=seq_dim,
            )
            if len(pieces) != len(datum_indices):
                raise RuntimeError(f"restored token output {name!r} does not match its Datum batch")
            restored[name] = [piece.detach() for piece in pieces]
        return restored

    def _validate_outputs_across_cp(
        self,
        outputs: ParsedLossOutputs,
        token_reference: Any,
        *,
        expected_datums: int | None,
        local_error: Exception | None = None,
        restore_plan: _OutputRestorePlan | None = None,
        datum_indices: tuple[int, ...] | None = None,
        chunk_index: int | None = None,
    ) -> None:
        """Validate output routing and reach CP consensus before token gathers."""
        if local_error is None:
            error, schema = _loss_output_contract(
                outputs,
                token_reference,
                expected_datums=expected_datums,
            )
        else:
            error, schema = str(local_error), ("invalid-output", type(local_error).__name__)
        if error is None and isinstance(outputs, LossOutput) and outputs.token_outputs and restore_plan is not None:
            restore_error, restore_schema = _loss_output_restore_contract(
                outputs,
                token_reference,
                restore_plan,
                datum_indices=datum_indices,
                chunk_index=chunk_index,
                cp_size=self._cp_size(),
            )
            error = restore_error
            schema = (*schema, restore_schema)
        cp_group, cp_size = self._cp_group_and_size()
        if cp_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            if local_error is not None:
                raise local_error
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
        token_outputs: dict[str, list[torch.Tensor]],
        batch_output: Mapping[str, Any] | None,
        *,
        serialized_outputs: bytes | None,
    ) -> tuple[dict[str, list[torch.Tensor]], Mapping[str, Any] | None]:
        """Broadcast last-stage token and batch channels to every PP stage."""
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size <= 1:
            return token_outputs, batch_output

        has_last_stage = self.pipeline.info.has_last_stage
        local_state = torch.tensor(
            [int(has_last_stage), int(has_last_stage and (bool(token_outputs) or batch_output is not None))],
            dtype=torch.int64,
            device=self.device,
        )
        stage_states = torch.empty(pp_size * 2, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(stage_states, local_state, group=pp_group)
        stage_states = stage_states.view(pp_size, 2)
        source_ranks = (stage_states[:, 0] == 1).nonzero(as_tuple=False).flatten()
        if source_ranks.numel() != 1:
            raise RuntimeError("pipeline output synchronization requires exactly one physical last-stage rank")

        source_group_rank = int(source_ranks.item())
        if not bool(stage_states[source_group_rank, 1]):
            return {}, None
        source_global_rank = dist.get_global_rank(pp_group, source_group_rank)
        object_list: list[Any] = [serialized_outputs if dist.get_rank(group=pp_group) == source_group_rank else None]
        dist.broadcast_object_list(object_list, src=source_global_rank, group=pp_group, device=self.device)
        payload = object_list[0]
        if not isinstance(payload, bytes):
            raise RuntimeError("pipeline output synchronization received an invalid serialized payload")
        # The payload was serialized by this Engine's trusted PP source rank
        # immediately above; this is not an external deserialization boundary.
        received = pickle.loads(payload)  # noqa: S301
        if not isinstance(received, dict) or set(received) != {"token_outputs", "batch_output"}:
            raise RuntimeError("pipeline output synchronization received an invalid payload")
        received_token_outputs = received["token_outputs"]
        received_batch_output = received["batch_output"]
        if not isinstance(received_token_outputs, dict) or not all(
            isinstance(name, str)
            and isinstance(values, list)
            and all(isinstance(value, torch.Tensor) for value in values)
            for name, values in received_token_outputs.items()
        ):
            raise RuntimeError("pipeline output synchronization received invalid token outputs")
        if received_batch_output is not None and not isinstance(received_batch_output, Mapping):
            raise RuntimeError("pipeline output synchronization received an invalid batch output")
        moved = _to_device(received, self.device)
        return moved["token_outputs"], moved["batch_output"]

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

    def _validate_forward_batch_across_model_parallel(
        self,
        datums: list[Datum],
        cp_group: dist.ProcessGroup | None,
        cp_size: int,
    ) -> None:
        """Fail before collation when CP or PP ranks disagree on scoring layout."""
        schema = tuple(
            (
                datum.seq_len,
                _layout_schema(datum.model_inputs),
                _layout_schema(datum.loss_fn_inputs),
                tuple(sorted((name, layout.value) for name, layout in datum.loss_fn_input_layouts.items())),
                tuple(sorted(datum.loss_fn_input_pad_values.items())),
            )
            for datum in datums
        )
        digest = int.from_bytes(hashlib.sha256(repr(schema).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        local = torch.tensor([len(datums), digest], dtype=torch.int64, device=self.device)
        groups = (("context-parallel", cp_group, cp_size), ("pipeline", *self._pp_group_and_size()))
        for name, group, size in groups:
            if size <= 1:
                continue
            gathered = torch.empty(size * 2, dtype=torch.int64, device=self.device)
            dist.all_gather_into_tensor(gathered, local, group=group)
            gathered = gathered.view(size, 2)
            if not bool((gathered == gathered[0]).all()):
                raise ValueError(
                    f"{name} ranks must use the same scoring Datum count and tensor layouts; got {gathered.tolist()}"
                )

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
                ``[batch, sequence]``, packed ``[1, tokens]``, or model-owned
                flat THD ``[tokens]``; any tensor with those leading token
                axes is sharded identically.
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
