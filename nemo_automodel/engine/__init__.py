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
from nemo_automodel.components.distributed.context_parallel.sharder import (
    identity_local_indices,
    shard_batch_identity,
)
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
    [Any, dict[str, torch.Tensor], Sequence[Datum], dict[str, Any]],
    torch.Tensor | tuple[torch.Tensor, Sequence[Mapping[str, Any]]],
]

__all__ = ["Engine", "collate_prebatched"]


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
        collate_fn: Batches one microbatch of Datums into separate model and
            loss inputs. The default supports padded and packed text. VLMs pass
            a model-specific collater. Existing recipes whose dataloaders
            already collate can use :func:`collate_prebatched`; the Engine then
            applies any remaining CP/THD preparation. The callable must keep
            model inputs and loss inputs aligned and preserve the sum of
            ``weights``.
        padding_token_id: Token used when the CP sharder pads ``input_ids``.
        context_fn: Creates an optional context around model forward, loss,
            and backward. Recipes use this for runtime contexts such as FP8.
        defer_fsdp_grad_sync: Defer FSDP/DDP gradient synchronization until the
            final microbatch.

    Note:
        Context-parallel input layout and transport are delegated to
        :class:`ContextParallelSharder`. With an :class:`AutoPipeline`, the
        pipeline schedule owns its internal microbatching and backward calls;
        the Engine still owns the complete outer accumulation window and loss
        normalization. Pipeline execution currently requires context-parallel
        size one.
    """

    def __init__(
        self,
        model: nn.Module | AutoPipeline,
        *,
        device: torch.device | str,
        mesh_context: MeshContext | None = None,
        collate_fn: CollateFn = collate_datums,
        padding_token_id: int = 0,
        context_fn: Callable[[], AbstractContextManager[Any]] = nullcontext,
        defer_fsdp_grad_sync: bool = True,
    ) -> None:
        self.pipeline = model if isinstance(model, AutoPipeline) else None
        self.model_parts = model.parts if self.pipeline is not None else [model]
        self.model = self.model_parts[0]
        self.device = torch.device(device)
        self.mesh_context = mesh_context
        self.collate_fn = collate_fn
        self.padding_token_id = padding_token_id
        self.context_fn = context_fn
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync

    def forward_backward(
        self,
        window: Sequence[Sequence[Datum]],
        loss_fn: LossFn,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Accumulate gradients for a complete optimizer window.

        ``window`` is explicit: each inner sequence is one eager microbatch, or
        one outer pipeline batch that the schedule splits internally. Pipeline
        batches currently contain exactly one already-batched Datum.
        ``loss_fn`` receives the raw model output, CP-local
        ``loss_fn_inputs``, the original Datums, and the final CP-local model
        inputs produced by the sharder. It returns either per-element losses
        with exactly the same shape as
        ``loss_fn_inputs["weights"]``, or a scalar local weighted-sum
        numerator. For a scalar, the callback must apply weights and masks;
        the Engine will only apply global normalization. The callback may
        also return one output mapping per Datum during eager execution.
        Those mappings are detached and preserved in input order; the Engine
        deliberately does not interpret or reduce them. Pipeline execution
        does not yet support per-Datum outputs.

        Args:
            window: The complete optimizer accumulation window. Each inner
                sequence is one eager microbatch or outer pipeline batch of
                Datums. A pipeline batch must contain exactly one prebatched
                Datum. A Datum's token weights
                may have shape ``[tokens]`` or the custom collater's batched
                token layout; the loss tensor must use the identical shape.
            loss_fn: Computes either that per-token loss tensor or a scalar
                local weighted-sum numerator from the raw model output and
                collated loss inputs.

        Returns:
            ``(loss, loss_fn_outputs)``. ``loss`` is a detached scalar reduced
            over the DP-CP gradient group and, for pipeline execution,
            synchronized across PP stages. ``loss_fn_outputs`` contains
            local-rank, per-Datum mappings in window order. Model parameters
            are unchanged, but their gradients contain the complete window's
            globally normalized backward result.
        """
        microbatches = self._validate_window(window)
        self._validate_parallelism()
        if self.pipeline is not None and any(len(microbatch) != 1 for microbatch in microbatches):
            raise ValueError("pipeline Engine requires exactly one prebatched Datum in each outer batch")
        dp_group, dp_size = self._dp_group_and_size()
        grad_group, grad_group_size = self._gradient_group_and_size(dp_group, dp_size)
        self._validate_window_size_across_group(len(microbatches), grad_group, grad_group_size)
        denominator = self._global_weight_sum(microbatches, dp_group, dp_size)
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

            model_inputs, loss_inputs = self.collate_fn(datums)
            self._validate_collated_weights(datums, loss_inputs)
            model_inputs = _to_device(model_inputs, self.device)
            loss_inputs = _to_device(loss_inputs, self.device)
            full_weights = loss_inputs["weights"]
            loss_seq_dim = _loss_sequence_dim(model_inputs, full_weights)

            # ContextParallelSharder is the single owner of padded, THD, Magi,
            # and model-specific CP layouts. Labels are temporarily present in
            # its batch because each backend historically shards them with the
            # model inputs; all other loss tensors use the sharder token verb.
            cp_batch = dict(model_inputs)
            labels = loss_inputs.get("labels")
            cp_batch["labels"] = (
                labels.clone() if isinstance(labels, torch.Tensor) else torch.zeros_like(full_weights, dtype=torch.long)
            )
            device_mesh = self.mesh_context.device_mesh if self.mesh_context is not None else None
            final_thd = _is_final_thd(cp_batch)
            if final_thd and self._cp_size() > 1:
                raise ValueError(
                    "context parallelism requires raw THD inputs so ContextParallelSharder can partition them"
                )
            if final_thd:
                if self.pipeline is not None and self.pipeline.num_microbatches > 1:
                    raise ValueError(
                        "pipeline Engine requires raw THD inputs so ContextParallelSharder can split them "
                        "for the schedule's internal microbatches"
                    )
                sharder = ContextParallelSharder(
                    device_mesh=device_mesh,
                    shard_batch=shard_batch_identity,
                    local_token_global_indices=identity_local_indices,
                    padding_token_id=self.padding_token_id,
                )
            else:
                sharder = ContextParallelSharder(
                    self.model,
                    device_mesh,
                    cp_batch,
                    padding_token_id=self.padding_token_id,
                    num_chunks=inner_microbatches,
                )
            cp_context, model_inputs = sharder.shard(cp_batch)
            if model_inputs.get("qkv_format") == "thd" and (
                "seq_lens" in model_inputs or "seq_lens_padded" in model_inputs
            ):
                raise ValueError(
                    "ContextParallelSharder could not prepare raw THD inputs for this model; "
                    "use a THD-capable attention backend or provide final THD inputs at cp_size=1"
                )
            local_labels = model_inputs.pop("labels")
            loss_inputs = self._shard_loss_inputs(sharder, loss_inputs, loss_seq_dim)
            if labels is not None:
                loss_inputs["labels"] = local_labels
            weights = loss_inputs["weights"]

            if self.pipeline is not None:
                self._pipeline_step(
                    model_inputs,
                    loss_inputs,
                    datums,
                    loss_fn,
                    denominator,
                    grad_group_size,
                    local_loss_sum,
                    cp_context,
                )
            else:
                forward_inputs = filter_forward_kwargs(self.model, model_inputs)
                with get_sync_ctx(self.model, is_last, self.defer_fsdp_grad_sync), self.context_fn(), cp_context():
                    output = self.model(**forward_inputs)
                    result = loss_fn(output, loss_inputs, datums, model_inputs)
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
                    numerator = _weighted_numerator(losses, weights)
                    (numerator * (grad_group_size / denominator)).backward()

                local_loss_sum.add_(numerator.detach().to(torch.float64))
            if index == 0:
                prepare_after_first_microbatch()

        if grad_group_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=grad_group)
        pp_group, pp_size = self._pp_group_and_size()
        if pp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=pp_group)

        loss = (local_loss_sum / denominator).detach()
        return loss, loss_fn_outputs

    @staticmethod
    def _validate_window(window: Sequence[Sequence[Datum]]) -> list[list[Datum]]:
        if not isinstance(window, Sequence) or isinstance(window, (str, bytes)) or not window:
            raise ValueError("forward_backward requires a non-empty accumulation window")
        microbatches: list[list[Datum]] = []
        for index, microbatch in enumerate(window):
            if not isinstance(microbatch, Sequence) or isinstance(microbatch, (str, bytes)) or not microbatch:
                raise ValueError(f"microbatch {index} must be a non-empty sequence of Datum")
            if not all(isinstance(datum, Datum) for datum in microbatch):
                raise TypeError(f"microbatch {index} contains a value that is not a Datum")
            microbatches.append(list(microbatch))
        return microbatches

    def _pipeline_step(
        self,
        model_inputs: dict[str, Any],
        loss_inputs: dict[str, torch.Tensor],
        datums: Sequence[Datum],
        loss_fn: LossFn,
        denominator: torch.Tensor,
        grad_group_size: int,
        local_loss_sum: torch.Tensor,
        cp_context: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        primary_names = [name for name in ("input_ids", "inputs_embeds") if name in model_inputs]
        if len(primary_names) != 1:
            raise ValueError("pipeline Engine requires exactly one of input_ids or inputs_embeds")
        primary_name = primary_names[0]
        primary = model_inputs.pop(primary_name)
        if not isinstance(primary, torch.Tensor) or primary.ndim < 2:
            raise ValueError(f"pipeline Engine requires batched {primary_name} with a sequence dimension")

        self.pipeline.update_seq_len(primary.shape[1])
        pipeline_kwargs = {
            name: value
            for name, value in model_inputs.items()
            if value is not None and not (isinstance(value, dict) and not value)
        }

        def pipeline_loss(output, loss_inputs_mb, model_args_mb, model_kwargs_mb):
            if not model_args_mb:
                raise RuntimeError("AutoPipeline loss callback did not receive the primary model input")
            final_model_inputs = {primary_name: model_args_mb[0], **model_kwargs_mb}
            result = loss_fn(output, loss_inputs_mb, datums, final_model_inputs)
            if isinstance(result, tuple):
                raise ValueError("pipeline Engine does not yet support per-Datum loss_fn outputs")
            numerator = _weighted_numerator(result, loss_inputs_mb["weights"])
            local_loss_sum.add_(numerator.detach().to(torch.float64))
            return numerator * (grad_group_size / denominator)

        with self.context_fn(), cp_context():
            self.pipeline.step(
                primary,
                loss_inputs=loss_inputs,
                loss_fn=pipeline_loss,
                return_outputs=False,
                **pipeline_kwargs,
            )

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
        if self.pipeline is not None and self._cp_size() > 1:
            raise NotImplementedError("pipeline Engine does not yet support context parallelism")
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
        if float(denominator) <= 0:
            raise ValueError("forward_backward requires a positive global weight sum")
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


def _is_final_thd(model_inputs: dict[str, Any]) -> bool:
    """Return whether a CP1 caller already supplied the final flat THD layout."""
    if model_inputs.get("qkv_format") != "thd" or "cu_seqlens" not in model_inputs:
        return False
    if "seq_lens" in model_inputs or "seq_lens_padded" in model_inputs:
        raise ValueError("THD inputs cannot contain both raw seq_lens and final cu_seqlens metadata")
    return True


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
