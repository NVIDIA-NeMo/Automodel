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
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.datasets.datum import Datum, collate_datums
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

CollateFn = Callable[[list[Datum]], tuple[dict[str, Any], dict[str, torch.Tensor]]]
LossFn = Callable[
    [Any, dict[str, torch.Tensor]],
    torch.Tensor | tuple[torch.Tensor, Sequence[Mapping[str, Any]]],
]

_LOSS_FIELD_PREFIX = "__engine_loss__"
_LOSS_METADATA = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "padding_mask")

__all__ = ["Engine", "collate_prebatched"]


def _nullcontext_for_batch(_model_inputs: dict[str, Any]) -> AbstractContextManager[Any]:
    return nullcontext()


def collate_prebatched(datums: list[Datum]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
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
    return dict(datum.model_inputs), dict(datum.loss_fn_inputs)


class Engine:
    """Run model forward/backward over one optimizer accumulation window.

    The model and distributed topology are already constructed when they are
    passed here. The Engine owns batching, global weight normalization,
    gradient-accumulation synchronization, and backward. It deliberately does
    not zero, clip, finalize expert gradients, or step them; callers choose the
    optimizer boundary and retain the repository's existing distributed
    gradient-finalization path.

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
        fields follow those chunks. Per-Datum scalar loss fields do not yet
        carry enough boundary metadata to be split in that layout and are
        rejected instead of being replicated silently.
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
        context_fn: Callable[[dict[str, Any]], AbstractContextManager[Any]] = _nullcontext_for_batch,
        defer_fsdp_grad_sync: bool = True,
    ) -> None:
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be a positive integer, got {microbatch_size!r}")
        self.pipeline = model if isinstance(model, AutoPipeline) else None
        self.model_parts = model.parts if self.pipeline is not None else [model]
        self.model = self.model_parts[0]
        self.device = torch.device(device)
        self.mesh_context = mesh_context
        self.microbatch_size = microbatch_size
        self.collate_fn = collate_fn
        self.padding_token_id = padding_token_id
        self.context_fn = context_fn
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync

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

            cp_context, model_inputs, loss_inputs = self._prepare_batch(datums, inner_microbatches)

            if self.pipeline is not None:
                batch_returns_outputs, batch_outputs = self._pipeline_step(
                    model_inputs,
                    loss_inputs,
                    datums,
                    loss_fn,
                    safe_denominator,
                    zero_denominator,
                    grad_group_size,
                    local_loss_sum,
                    cp_context,
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
            raise ValueError("forward_backward requires a non-empty flat sequence of Datum")
        if not all(isinstance(datum, Datum) for datum in datums):
            raise TypeError("forward_backward received a value that is not a Datum")
        return [
            list(datums[start : start + self.microbatch_size]) for start in range(0, len(datums), self.microbatch_size)
        ]

    def _prepare_batch(
        self,
        datums: list[Datum],
        num_pipeline_microbatches: int,
    ) -> tuple[Callable[[], AbstractContextManager[Any]], dict[str, Any], dict[str, torch.Tensor]]:
        """Collate, move, and CP-shard one outer batch.

        Args:
            datums: Datum items in one outer batch. Tensor layouts are defined
                by the configured collater.
            num_pipeline_microbatches: Number of pipeline microbatches that
                the prepared outer batch must materialize.

        Returns:
            The CP context factory, CP-local model inputs, and CP-local loss
            inputs. Token-aligned model and loss tensors use the same padded,
            packed THD, Magi, or model-owned local sequence layout.
        """
        model_inputs, loss_inputs = self.collate_fn(datums)
        self._validate_collated_weights(datums, loss_inputs)
        model_inputs = _to_device(model_inputs, self.device)
        loss_inputs = _to_device(loss_inputs, self.device)
        full_weights = loss_inputs["weights"]
        loss_seq_dim = _loss_sequence_dim(model_inputs, full_weights)

        cp_batch = dict(model_inputs)
        labels = loss_inputs.get("labels")
        cp_batch["labels"] = (
            labels.clone() if isinstance(labels, torch.Tensor) else torch.zeros_like(full_weights, dtype=torch.long)
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
            ambiguous_per_datum_fields = [
                name
                for name, value in loss_inputs.items()
                if name != "labels"
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == len(datums)
                and len(datums) > 1
                and not _is_token_aligned(value, full_weights)
            ]
            if num_pipeline_microbatches > 1 and ambiguous_per_datum_fields:
                raise NotImplementedError(
                    "packed pipeline microbatching cannot yet split per-Datum loss fields "
                    f"{ambiguous_per_datum_fields}; use token-aligned fields or a prepared collater"
                )
            for name, value in loss_inputs.items():
                if name == "labels" or not _is_token_aligned(value, full_weights):
                    continue
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
            loss_inputs = self._shard_loss_inputs(sharder, loss_inputs, loss_seq_dim)
        if labels is not None:
            loss_inputs["labels"] = local_labels
        return cp_context, model_inputs, loss_inputs

    def _pipeline_step(
        self,
        model_inputs: dict[str, Any],
        loss_inputs: dict[str, torch.Tensor],
        datums: Sequence[Datum],
        loss_fn: LossFn,
        denominator: torch.Tensor,
        zero_denominator: bool,
        grad_group_size: int,
        local_loss_sum: torch.Tensor,
        cp_context: Callable[[], AbstractContextManager[Any]],
    ) -> tuple[bool | None, list[dict[str, Any]]]:
        outputs_by_microbatch: list[list[dict[str, Any]] | None] = [None] * self.pipeline.num_microbatches
        returns_outputs: bool | None = None

        with self.context_fn(model_inputs), cp_context():
            model_microbatches, loss_microbatches = self._materialize_pipeline_microbatches(model_inputs, loss_inputs)
            primary = model_microbatches[0].get("inputs_embeds", model_microbatches[0].get("input_ids"))
            if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
                raise ValueError("pipeline Engine requires a tensor input_ids or inputs_embeds")
            if model_microbatches[0].get("qkv_format") == "thd" and self.pipeline.num_microbatches == 1:
                seq_len = primary.shape[0]
            else:
                seq_len = primary.shape[1] if primary.ndim >= 2 else primary.shape[0]
            effective_microbatch_size = 1 if model_microbatches[0].get("qkv_format") == "thd" else primary.shape[0]
            self.pipeline.update_seq_len(
                seq_len,
                microbatch_size=effective_microbatch_size,
                input_tensor=primary,
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
                    if len(datums) == 1 and self.pipeline.num_microbatches > 1:
                        raise ValueError(
                            "a prebatched Datum may return outputs only when num_microbatches=1 because "
                            "its inner sample boundaries are not part of the Datum contract"
                        )
                    outputs_by_microbatch[microbatch_index] = [_detach(dict(item)) for item in batch_outputs]
                else:
                    losses = result
                numerator = _weighted_numerator(losses, loss_inputs_mb["weights"])
                if zero_denominator:
                    numerator = numerator * 0
                local_loss_sum.add_(numerator.detach().to(torch.float64))
                return numerator * (grad_group_size / denominator)

            losses = [] if self.pipeline.info.has_last_stage else None
            self.pipeline.step_microbatches(
                model_microbatches,
                loss_fn=pipeline_loss,
                losses=losses,
                return_outputs=False,
            )

        outputs: list[dict[str, Any]] = []
        if self.pipeline.info.has_last_stage and returns_outputs:
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
        loss_inputs: dict[str, torch.Tensor],
    ) -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]]]:
        """Split one CP-prepared outer batch into exact pipeline inputs.

        Args:
            model_inputs: CP-local model tensors. Padded tensors use shape
                [batch, sequence, ...]. Chunked THD tensors use shape
                [microbatches, tokens, ...].
            loss_inputs: CP-local loss tensors. Token-aligned fields have the
                same leading token axes as the primary model tensor.

        Returns:
            Parallel lists of complete model and loss mappings, each with
            exactly ``pipeline.num_microbatches`` items. Tensor slicing returns
            views that retain a size-one pipeline microbatch axis.
        """
        num_microbatches = self.pipeline.num_microbatches
        if num_microbatches == 1:
            return [dict(model_inputs)], [_with_loss_metadata(model_inputs, loss_inputs)]

        primary_name = _primary_name(model_inputs)
        primary = model_inputs[primary_name]
        if not isinstance(primary, torch.Tensor) or primary.ndim == 0:
            raise ValueError(f"pipeline Engine requires tensor {primary_name}")

        if model_inputs.get("qkv_format") == "thd":
            if primary.shape[0] != num_microbatches:
                raise ValueError(
                    f"THD sharder produced {primary.shape[0]} chunks, expected {num_microbatches} pipeline microbatches"
                )
            model_microbatches = [
                _select_chunk(model_inputs, index, num_microbatches) for index in range(num_microbatches)
            ]
            loss_microbatches = [
                _select_chunk(loss_inputs, index, num_microbatches) for index in range(num_microbatches)
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
            loss_microbatches = [
                _slice_batch_mapping(loss_inputs, index, num_microbatches, batch_size)
                for index in range(num_microbatches)
            ]

        loss_microbatches = [
            _with_loss_metadata(model_microbatch, loss_microbatch)
            for model_microbatch, loss_microbatch in zip(model_microbatches, loss_microbatches)
        ]
        return model_microbatches, loss_microbatches

    def _validate_parallelism(self) -> None:
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
            raise ValueError(
                f"pipeline stages must use the same DP-reduced weight denominator; got {gathered[:, 1].tolist()}"
            )

    def _global_weight_sum(
        self,
        microbatches: list[list[Datum]],
        dp_group: dist.ProcessGroup | None,
        dp_size: int,
    ) -> torch.Tensor:
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
        if dp_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=dp_group)
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
            raise ValueError(f"every gradient rank must use the same number of microbatches; got {sizes.tolist()}")

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
        loss_inputs: dict[str, torch.Tensor],
        seq_dim: int | None,
    ) -> dict[str, torch.Tensor]:
        """Apply the model batch's CP token layout to loss-only tensors.

        Args:
            sharder: Sharder after it has prepared the current model batch.
            loss_inputs: Tensors in the pre-CP layout. ``weights`` has shape
                ``[batch, sequence]`` or ``[tokens]``; any tensor with those
                leading token axes is sharded identically.
            seq_dim: Sequence axis in the pre-CP loss tensors, or ``None`` when
                the weights do not follow the model's token axes.

        Returns:
            Loss tensors in the model output's CP-local token layout. Non-token
            tensors are returned unchanged; the input mapping is not mutated.
        """
        weights = loss_inputs["weights"]
        layout = sharder.shard_layout
        if self._cp_size() == 1 and layout is None:
            return {name: value for name, value in loss_inputs.items() if name != "labels"}
        layout_changed = self._cp_size() > 1 or (
            layout is not None
            and (
                layout.input_row_shape is not None
                or layout.input_token_stream_positions is not None
                or layout.original_seq_len != layout.padded_seq_len
            )
        )
        if seq_dim is None:
            if layout_changed:
                raise ValueError("context-parallel loss weights must match the model's token axes")
            return dict(loss_inputs)

        local: dict[str, torch.Tensor] = {}
        for name, value in loss_inputs.items():
            if name == "labels":
                continue
            token_aligned = value.ndim >= weights.ndim and tuple(value.shape[: weights.ndim]) == tuple(weights.shape)
            local[name] = sharder.shard_token_tensor(value, seq_dim=seq_dim, fill=0) if token_aligned else value
        return local

    @staticmethod
    def _validate_collated_weights(
        datums: list[Datum],
        loss_inputs: dict[str, torch.Tensor],
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


def _loss_sequence_dim(model_inputs: dict[str, Any], weights: torch.Tensor) -> int | None:
    """Find the sequence axis shared by primary model tokens and loss weights.

    Args:
        model_inputs: Pre-CP model mapping whose ``input_ids`` has shape
            ``[batch, sequence]`` or ``[tokens]``, or whose ``inputs_embeds``
            has shape ``[batch, sequence, hidden]``.
        weights: Loss weights of shape ``[batch, sequence]`` or ``[tokens]``.

    Returns:
        The sequence axis in ``weights``, or ``None`` when the layouts do not
        describe the same token stream.
    """
    primary = model_inputs.get("inputs_embeds", model_inputs.get("input_ids"))
    if not isinstance(primary, torch.Tensor):
        return None
    if primary.ndim >= 2 and weights.ndim >= 2 and tuple(weights.shape[:2]) == tuple(primary.shape[:2]):
        return 1
    if primary.ndim == 1 and weights.ndim == 1 and weights.shape == primary.shape:
        return 0
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


def _with_loss_metadata(
    model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
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
