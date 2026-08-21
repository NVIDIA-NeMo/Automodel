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

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

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
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

CollateFn = Callable[
    [list[Datum]],
    tuple[dict[str, Any], dict[str, torch.Tensor] | CollatedLossInputs],
]
LossInputValue = torch.Tensor | tuple[torch.Tensor, ...]
LossInputs = dict[str, LossInputValue]
LossFn = Callable[
    [Any, LossInputs],
    torch.Tensor | tuple[torch.Tensor, Sequence[Mapping[str, Any]]],
]

_LOSS_FIELD_PREFIX = "__engine_loss__"
_LOSS_METADATA = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "padding_mask")

__all__ = ["Engine", "ForwardResult", "collate_prebatched"]


def _nullcontext_for_batch(_model_inputs: dict[str, Any]) -> AbstractContextManager[Any]:
    return nullcontext()


def collate_prebatched(datums: list[Datum]) -> tuple[dict[str, Any], CollatedLossInputs | dict[str, torch.Tensor]]:
    """Return one already-collated Datum without changing its layout.

    The Datum represents the whole prebatched item. Consequently, one
    optional ``loss_fn_output`` mapping also describes that whole batch, not
    each sample inside it.

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
        )
    else:
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
    unresolved_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ForwardResult:
    """Forward-only loss statistics and per-Datum outputs.

    ``loss_sum`` and ``weight_sum`` are complete across model-parallel CP and
    PP ranks, but remain local to one data-parallel replica. The Engine adds no
    per-call DP loss-statistic collective, so callers reduce the two sums once
    at the end of an evaluation epoch. Distributed model wrappers may still
    communicate during forward and therefore retain their own call-alignment
    requirements. ``loss_fn_outputs`` remains local to the replica's input
    Datums; PP stages receive identical detached copies, while any CP-local
    tensor layout inside those mappings remains caller defined.

    Attributes:
        loss_sum: Detached weighted numerator for this Datum window.
        weight_sum: Detached full-sequence weight denominator for this window.
        loss_fn_outputs: Detached per-Datum mappings in input order.
    """

    loss_sum: torch.Tensor
    weight_sum: torch.Tensor
    loss_fn_outputs: list[dict[str, Any]]


class Engine:
    """Run model forward or forward/backward over Datum windows.

    The model and distributed topology are already constructed when they are
    passed here. The Engine owns batching and model-parallel execution.
    :meth:`forward` performs evaluation without gradients;
    :meth:`forward_backward` additionally owns global weight normalization,
    gradient-accumulation synchronization, and backward. The Engine
    deliberately does not zero, clip, finalize expert gradients, or step them;
    callers choose the optimizer boundary and retain the repository's existing
    distributed gradient-finalization path.

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
        defer_fsdp_grad_sync: Defer FSDP/DDP gradient synchronization until the
            final microbatch.

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
        defer_fsdp_grad_sync: bool = True,
    ) -> None:
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be a positive integer, got {microbatch_size!r}")
        if isinstance(mtp_ignore_index, bool) or not isinstance(mtp_ignore_index, int):
            raise ValueError(f"mtp_ignore_index must be an integer, got {mtp_ignore_index!r}")
        self.pipeline = model if isinstance(model, AutoPipeline) else None
        self.model_parts = model.parts if self.pipeline is not None else [model]
        self.model = self.model_parts[0]
        self.device = torch.device(device)
        self.mesh_context = mesh_context
        self.microbatch_size = microbatch_size
        self.collate_fn = collate_fn
        self.padding_token_id = padding_token_id
        self.mtp_ignore_index = mtp_ignore_index
        self.context_fn = context_fn
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync

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
                inputs. It may additionally return one output mapping per
                Datum.

        Returns:
            Detached model-parallel-complete loss statistics and per-Datum
            outputs. The sums are local to one data-parallel replica.
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
            cp_context, model_inputs, loss_inputs, loss_batch_layout = self._prepare_batch(
                batch_datums, inner_microbatches
            )
            if self.pipeline is not None:
                batch_returns_outputs, batch_outputs = self._pipeline_execute(
                    model_inputs,
                    loss_inputs,
                    batch_datums,
                    loss_fn,
                    local_loss_sum,
                    cp_context,
                    loss_batch_layout,
                    backward_scale=None,
                    zero_weight_sum=zero_weight_sum,
                )
                if batch_returns_outputs is not None:
                    if returns_outputs is None:
                        returns_outputs = batch_returns_outputs
                    elif returns_outputs != batch_returns_outputs:
                        raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
                    loss_fn_outputs.extend(batch_outputs)
                continue

            loss_inputs = _with_loss_metadata(model_inputs, loss_inputs)
            with self.context_fn(model_inputs), cp_context():
                forward_inputs = filter_forward_kwargs(self.model, model_inputs)
                output = self.model(**forward_inputs)
                result = loss_fn(output, loss_inputs)
                has_outputs = isinstance(result, tuple)
                if returns_outputs is None:
                    returns_outputs = has_outputs
                elif returns_outputs != has_outputs:
                    raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
                if isinstance(result, tuple):
                    losses, outputs = result
                    if (
                        not isinstance(outputs, Sequence)
                        or isinstance(outputs, (str, bytes))
                        or len(outputs) != len(batch_datums)
                        or not all(isinstance(item, Mapping) for item in outputs)
                    ):
                        raise ValueError("loss_fn outputs must contain one mapping per Datum")
                    loss_fn_outputs.extend(_detach(dict(item)) for item in outputs)
                else:
                    losses = result
                numerator = _weighted_numerator(losses, loss_inputs["weights"])
                if zero_weight_sum:
                    numerator = numerator * 0
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
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
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
        also return one output mapping per Datum.
        Those mappings are detached and preserved in input order; the Engine
        deliberately does not interpret or reduce them. Under pipeline
        parallelism the last stage computes the mappings and broadcasts them
        to every stage in the pipeline group.

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
            ``(loss, loss_fn_outputs)``. ``loss`` is a detached scalar reduced
            over the DP-CP gradient group and, for pipeline execution,
            synchronized across PP stages. ``loss_fn_outputs`` contains
            per-Datum mappings in window order; pipeline execution returns the
            same mappings on every physical stage rank. Model parameters are
            unchanged, but their gradients contain the complete window's
            globally normalized backward result.
        """
        microbatches = self._group_datums(datums)
        self._validate_parallelism()
        dp_group, dp_size = self._dp_group_and_size()
        grad_group, grad_group_size = self._gradient_group_and_size(dp_group, dp_size)
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
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            self._cp_size() / (len(microbatches) * inner_microbatches)
        )

        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_fn_outputs: list[dict[str, Any]] = []
        returns_outputs: bool | None = None

        for index, datums in enumerate(microbatches):
            is_last = index == len(microbatches) - 1
            if is_last:
                prepare_for_final_backward(self.model_parts, pp_enabled=pp_enabled)

            cp_context, model_inputs, loss_inputs, loss_batch_layout = self._prepare_batch(datums, inner_microbatches)

            if self.pipeline is not None:
                backward_scale = (
                    safe_denominator.new_zeros(())
                    if zero_denominator
                    else safe_denominator.new_tensor(grad_group_size) / safe_denominator
                )
                batch_returns_outputs, batch_outputs = self._pipeline_execute(
                    model_inputs,
                    loss_inputs,
                    datums,
                    loss_fn,
                    local_loss_sum,
                    cp_context,
                    loss_batch_layout,
                    backward_scale=backward_scale,
                    zero_weight_sum=zero_denominator,
                )
                if batch_returns_outputs is not None:
                    if returns_outputs is None:
                        returns_outputs = batch_returns_outputs
                    elif returns_outputs != batch_returns_outputs:
                        raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
                    loss_fn_outputs.extend(batch_outputs)
            else:
                loss_inputs = _with_loss_metadata(model_inputs, loss_inputs)
                with (
                    get_sync_ctx(self.model, is_last, self.defer_fsdp_grad_sync),
                    self.context_fn(model_inputs),
                    cp_context(),
                ):
                    forward_inputs = filter_forward_kwargs(self.model, model_inputs)
                    output = self.model(**forward_inputs)
                    result = loss_fn(output, loss_inputs)
                    has_outputs = isinstance(result, tuple)
                    if returns_outputs is None:
                        returns_outputs = has_outputs
                    elif returns_outputs != has_outputs:
                        raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
                    if isinstance(result, tuple):
                        losses, outputs = result
                        if (
                            not isinstance(outputs, Sequence)
                            or isinstance(outputs, (str, bytes))
                            or len(outputs) != len(datums)
                            or not all(isinstance(item, Mapping) for item in outputs)
                        ):
                            raise ValueError("loss_fn outputs must contain one mapping per Datum")
                        loss_fn_outputs.extend(_detach(dict(item)) for item in outputs)
                    else:
                        losses = result
                    numerator = _weighted_numerator(losses, loss_inputs["weights"])
                    if zero_denominator:
                        numerator = numerator * 0
                    (numerator * (grad_group_size / safe_denominator)).backward()

                local_loss_sum.add_(numerator.detach().to(torch.float64))
            if index == 0:
                prepare_after_first_microbatch()

        if grad_group_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=grad_group)
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=pp_group)

        loss = (local_loss_sum / safe_denominator).detach()
        return loss, loss_fn_outputs

    def _group_datums(self, datums: Sequence[Datum]) -> list[list[Datum]]:
        if not isinstance(datums, Sequence) or isinstance(datums, (str, bytes)) or not datums:
            raise ValueError("Engine requires a non-empty flat sequence of Datum")
        if not all(isinstance(datum, Datum) for datum in datums):
            raise TypeError("Engine received a value that is not a Datum")
        return [
            list(datums[start : start + self.microbatch_size]) for start in range(0, len(datums), self.microbatch_size)
        ]

    @staticmethod
    def _resolve_loss_batch_layout(
        datums: list[Datum],
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, LossInputValue],
    ) -> _LossBatchLayout:
        """Resolve collater metadata without exposing it to the loss callback."""
        if isinstance(loss_inputs, CollatedLossInputs):
            if set(loss_inputs.layouts) != set(loss_inputs):
                raise ValueError("CollatedLossInputs.layouts must describe every loss field exactly once")
            item_to_datum = loss_inputs.item_to_datum
            if item_to_datum is not None and item_to_datum != tuple(range(len(datums))):
                raise ValueError(
                    "Engine currently requires collater item_to_datum to preserve outer Datum order; "
                    f"got {list(item_to_datum)} for {len(datums)} Datums"
                )
            return _LossBatchLayout(
                fields=dict(loss_inputs.layouts),
                item_to_datum=item_to_datum,
            )

        weights = loss_inputs.get("weights")
        if not isinstance(weights, torch.Tensor):
            raise ValueError("collate_fn must return a Tensor loss input named 'weights'")

        fields: dict[str, LossInputLayout] = {}
        unresolved: set[str] = set()
        for name, value in loss_inputs.items():
            declared = {datum.loss_fn_input_layouts[name] for datum in datums if name in datum.loss_fn_input_layouts}
            if len(declared) > 1:
                raise ValueError(f"Datum items disagree on the loss layout for field {name!r}")
            if declared:
                if not all(name in datum.loss_fn_input_layouts for datum in datums):
                    raise ValueError(f"every Datum must declare the loss layout for field {name!r}")
                fields[name] = next(iter(declared))
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
            unresolved_fields=frozenset(unresolved),
        )

    def _prepare_batch(
        self,
        datums: list[Datum],
        num_pipeline_microbatches: int,
    ) -> tuple[Callable[[], AbstractContextManager[Any]], dict[str, Any], LossInputs, _LossBatchLayout]:
        """Collate, move, and CP-shard one outer batch.

        Args:
            datums: Datum items in one outer batch. Tensor layouts are defined
                by the configured collater.
            num_pipeline_microbatches: Number of pipeline microbatches that
                the prepared outer batch must materialize.

        Returns:
            The CP context factory, CP-local model inputs, CP-local loss
            inputs, and collated field-layout metadata. Token-aligned model
            and loss tensors use the same padded, packed THD, Magi, or
            model-owned local sequence layout.
        """
        model_inputs, collated_loss_inputs = self.collate_fn(datums)
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
                        loss_inputs[name], seq_dim=loss_seq_dim or 0, fill=0
                    )
            loss_inputs = local_loss_inputs
        else:
            loss_inputs = self._shard_loss_inputs(
                sharder,
                loss_inputs,
                loss_seq_dim,
                loss_batch_layout.fields,
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
                unresolved_fields=loss_batch_layout.unresolved_fields,
            )
        return cp_context, model_inputs, loss_inputs, loss_batch_layout

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
        *,
        backward_scale: torch.Tensor | None,
        zero_weight_sum: bool,
    ) -> tuple[bool | None, list[dict[str, Any]]]:
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
            backward_scale: Multiplier returned to the training schedule for
                backward, or ``None`` to run the forward-only schedule.
            zero_weight_sum: Whether reporting numerators must be forced to
                graph-connected zero.

        Returns:
            Whether the callback returned outputs, and its detached outputs in
            logical Datum order.
        """
        outputs_by_microbatch: list[list[dict[str, Any]] | None] = [None] * self.pipeline.num_microbatches
        outputs_by_datum: list[dict[str, Any] | None] = [None] * len(datums)
        returns_outputs: bool | None = None

        with cp_context():
            primary_name = _primary_name(model_inputs)
            primary = model_inputs[primary_name]
            if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
                raise ValueError("pipeline Engine requires a tensor input_ids or inputs_embeds")

            num_microbatches = self.pipeline.num_microbatches
            is_thd = model_inputs.get("qkv_format") == "thd"
            if num_microbatches == 1:
                primary_microbatch = primary
            elif is_thd:
                if primary.shape[0] != num_microbatches:
                    raise ValueError(
                        f"THD sharder produced {primary.shape[0]} chunks, "
                        f"expected {num_microbatches} pipeline microbatches"
                    )
                primary_microbatch = primary.narrow(0, 0, 1)
            else:
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
                primary_microbatch = primary.narrow(0, 0, materialized_batch_size)

            if is_thd and num_microbatches == 1:
                seq_len = primary_microbatch.shape[0]
            else:
                seq_len = primary_microbatch.shape[1] if primary_microbatch.ndim >= 2 else primary_microbatch.shape[0]
            effective_microbatch_size = 1 if is_thd else primary_microbatch.shape[0]
            self.pipeline.update_seq_len(
                seq_len,
                microbatch_size=effective_microbatch_size,
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
                    )
                )

                def pipeline_loss(output: Any, microbatch_index: int) -> torch.Tensor:
                    nonlocal returns_outputs
                    loss_inputs_mb = loss_microbatches[microbatch_index]
                    result = loss_fn(output, loss_inputs_mb)
                    has_outputs = isinstance(result, tuple)
                    if returns_outputs is None:
                        returns_outputs = has_outputs
                    elif returns_outputs != has_outputs:
                        raise ValueError("loss_fn must return per-Datum outputs for every microbatch or none of them")
                    if isinstance(result, tuple):
                        losses, batch_outputs = result
                        if (
                            not isinstance(batch_outputs, Sequence)
                            or isinstance(batch_outputs, (str, bytes))
                            or not all(isinstance(item, Mapping) for item in batch_outputs)
                        ):
                            raise ValueError("loss_fn outputs must be a sequence of mappings")
                        datum_indices = (
                            None
                            if datum_indices_by_microbatch is None
                            else datum_indices_by_microbatch[microbatch_index]
                        )
                        if datum_indices is None and len(datums) == 1 and self.pipeline.num_microbatches > 1:
                            raise ValueError(
                                "a prebatched Datum may return outputs only when num_microbatches=1 because "
                                "its inner sample boundaries are not part of the Datum contract"
                            )
                        detached_outputs = [_detach(dict(item)) for item in batch_outputs]
                        if datum_indices is not None:
                            if len(detached_outputs) != len(datum_indices):
                                raise ValueError(
                                    f"pipeline loss_fn returned {len(detached_outputs)} outputs for microbatch "
                                    f"{microbatch_index}, expected {len(datum_indices)} from its Datum mapping"
                                )
                            for datum_index, item in zip(datum_indices, detached_outputs):
                                if outputs_by_datum[datum_index] is not None:
                                    raise RuntimeError(
                                        f"pipeline returned more than one output for Datum {datum_index}"
                                    )
                                outputs_by_datum[datum_index] = item
                        else:
                            outputs_by_microbatch[microbatch_index] = detached_outputs
                    else:
                        losses = result
                    numerator = _weighted_numerator(losses, loss_inputs_mb["weights"])
                    if zero_weight_sum:
                        numerator = numerator * 0
                    local_loss_sum.add_(numerator.detach().to(torch.float64))
                    return numerator if backward_scale is None else numerator * backward_scale

                losses = [] if self.pipeline.info.has_last_stage else None
                run_microbatches = (
                    self.pipeline.eval_microbatches if backward_scale is None else self.pipeline.step_microbatches
                )
                run_microbatches(
                    model_microbatches,
                    loss_fn=pipeline_loss,
                    losses=losses,
                    return_outputs=False,
                )

        outputs: list[dict[str, Any]] = []
        if self.pipeline.info.has_last_stage and returns_outputs:
            if datum_indices_by_microbatch is not None:
                if any(item is None for item in outputs_by_datum):
                    raise RuntimeError("pipeline schedule did not return exactly one output for every Datum")
                outputs = [item for item in outputs_by_datum if item is not None]
            else:
                if any(items is None for items in outputs_by_microbatch):
                    raise RuntimeError("pipeline schedule did not evaluate loss_fn for every logical microbatch")
                outputs = [item for items in outputs_by_microbatch if items is not None for item in items]
                if len(outputs) != len(datums):
                    raise ValueError(
                        f"pipeline loss_fn returned {len(outputs)} outputs across the outer batch, "
                        f"expected one for each of its {len(datums)} Datums"
                    )
        outputs = self._broadcast_pipeline_outputs(outputs)
        return bool(outputs), outputs

    def _broadcast_pipeline_outputs(self, outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        object_list: list[Any] = [
            _to_device(outputs, torch.device("cpu")) if dist.get_rank(group=pp_group) == source_group_rank else None
        ]
        dist.broadcast_object_list(object_list, src=source_global_rank, group=pp_group, device=self.device)
        received = object_list[0]
        if not isinstance(received, list) or not all(isinstance(item, dict) for item in received):
            raise RuntimeError("pipeline output synchronization received invalid per-Datum outputs")
        return _to_device(received, self.device)

    def _materialize_pipeline_microbatches(
        self,
        model_inputs: dict[str, Any],
        loss_inputs: LossInputs,
        loss_batch_layout: _LossBatchLayout,
        *,
        num_datums: int,
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
                    is_thd=model_inputs.get("qkv_format") == "thd",
                )
                assert datum_indices is not None
            return (
                [dict(model_inputs)],
                [_with_loss_metadata(model_inputs, loss_inputs)],
                datum_indices,
            )

        primary_name = _primary_name(model_inputs)
        primary = model_inputs[primary_name]
        if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
            raise ValueError(f"pipeline Engine requires tensor {primary_name}")

        is_thd = model_inputs.get("qkv_format") == "thd"
        if is_thd:
            if primary.shape[0] != num_microbatches:
                raise ValueError(
                    f"THD sharder produced {primary.shape[0]} chunks, expected {num_microbatches} pipeline microbatches"
                )
            model_microbatches = [
                _select_chunk(model_inputs, index, num_microbatches) for index in range(num_microbatches)
            ]
        else:
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
        if any(
            bool(getattr(module, "calculate_per_token_loss", False))
            for part in self.model_parts
            for module in part.modules()
        ):
            raise NotImplementedError(
                "Engine.forward_backward requires averaged distributed gradients; "
                "MegatronFSDP calculate_per_token_loss=True uses summed gradients"
            )
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
            local[name] = sharder.shard_token_tensor(value, seq_dim=seq_dim, fill=0)
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
