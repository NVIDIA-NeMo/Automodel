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
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)

CollateFn = Callable[[list[Datum]], tuple[dict[str, Any], dict[str, torch.Tensor]]]
LossFn = Callable[
    [Any, dict[str, torch.Tensor], Sequence[Datum]],
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
        model: An already configured and distributed model.
        device: Device on which model inputs and losses are evaluated.
        mesh_context: Runtime topology. When omitted, an initialized default
            process group is treated as pure data parallelism.
        collate_fn: Batches one microbatch of Datums into separate model and
            loss inputs. The default supports padded text. Packed text and
            VLMs pass a model-specific collater that returns final model-ready
            inputs. Existing recipes whose dataloaders already collate can use
            :func:`collate_prebatched`. The callable must keep model inputs and
            loss inputs aligned and preserve the sum of ``weights``.
        context_fn: Creates an optional context around model forward, loss,
            and backward. Recipes use this for runtime contexts such as FP8.
        defer_fsdp_grad_sync: Defer FSDP/DDP gradient synchronization until the
            final microbatch.

    Note:
        This first execution backend is eager and weight-normalized. Pipeline
        and context parallel schedules are intentionally deferred.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
        mesh_context: MeshContext | None = None,
        collate_fn: CollateFn = collate_datums,
        context_fn: Callable[[], AbstractContextManager[Any]] = nullcontext,
        defer_fsdp_grad_sync: bool = True,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.mesh_context = mesh_context
        self.collate_fn = collate_fn
        self.context_fn = context_fn
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync

    def forward_backward(
        self,
        window: Sequence[Sequence[Datum]],
        loss_fn: LossFn,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Accumulate gradients for a complete optimizer window.

        ``window`` is explicit: each inner sequence is one eager microbatch.
        ``loss_fn`` receives the raw model output, collated
        ``loss_fn_inputs``, and the original Datums for that microbatch. It
        returns either per-element losses with exactly the same shape as
        ``loss_fn_inputs["weights"]``, or a scalar local weighted-sum
        numerator. For a scalar, the callback must apply weights and masks;
        the Engine will only apply global normalization. The callback may
        also return one output mapping per Datum.
        Those mappings are detached and preserved in input order; the Engine
        deliberately does not interpret or reduce them.

        Args:
            window: The complete optimizer accumulation window. Each inner
                sequence is one microbatch of Datums. A Datum's token weights
                may have shape ``[tokens]`` or the custom collater's batched
                token layout; the loss tensor must use the identical shape.
            loss_fn: Computes either that per-token loss tensor or a scalar
                local weighted-sum numerator from the raw model output and
                collated loss inputs.

        Returns:
            ``(loss, loss_fn_outputs)``. ``loss`` is a detached, DP-reduced
            scalar. ``loss_fn_outputs`` contains local-rank, per-Datum mappings
            in window order. Model parameters are unchanged, but their
            gradients contain the complete window's globally normalized
            backward result.
        """
        microbatches = self._validate_window(window)
        self._validate_parallelism()
        dp_group, dp_size = self._dp_group_and_size()
        self._validate_window_size_across_dp(len(microbatches), dp_group, dp_size)
        denominator = self._global_weight_sum(microbatches, dp_group, dp_size)

        self.model.train()
        prepare_for_grad_accumulation([self.model], pp_enabled=False)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0 / len(microbatches))

        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_fn_outputs: list[dict[str, Any]] = []
        returns_outputs: bool | None = None

        for index, datums in enumerate(microbatches):
            is_last = index == len(microbatches) - 1
            if is_last:
                prepare_for_final_backward([self.model], pp_enabled=False)

            model_inputs, loss_inputs = self.collate_fn(datums)
            if model_inputs.get("qkv_format") == "thd" and (
                "seq_lens" in model_inputs or "seq_lens_padded" in model_inputs
            ):
                raise ValueError(
                    "packed collate_fn must return final model-ready THD inputs, not seq_lens packing metadata"
                )
            self._validate_collated_weights(datums, loss_inputs)
            model_inputs = _to_device(model_inputs, self.device)
            loss_inputs = _to_device(loss_inputs, self.device)
            weights = loss_inputs["weights"]

            with get_sync_ctx(self.model, is_last, self.defer_fsdp_grad_sync), self.context_fn():
                output = self.model(**model_inputs)
                result = loss_fn(output, loss_inputs, datums)
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

                (numerator * (dp_size / denominator)).backward()

            local_loss_sum.add_(numerator.detach().to(torch.float64))
            if index == 0:
                prepare_after_first_microbatch()

        if dp_size > 1:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=dp_group)

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

    def _validate_parallelism(self) -> None:
        if self.mesh_context is None:
            return
        if self.mesh_context.pp_size > 1:
            raise NotImplementedError("Engine.forward_backward does not yet support pipeline parallelism")
        if self.mesh_context.cp_size > 1:
            raise NotImplementedError("Engine.forward_backward does not yet support context parallelism")

    def _dp_group_and_size(self) -> tuple[dist.ProcessGroup | None, int]:
        if self.mesh_context is not None and self.mesh_context.device_mesh is not None:
            dp_mesh = get_flat_mesh(self.mesh_context.device_mesh, "dp")
            size = int(dp_mesh.size())
            return (dp_mesh.get_group() if size > 1 else None), size

        group = self.mesh_context.process_group if self.mesh_context is not None else None
        if dist.is_available() and dist.is_initialized():
            return group, dist.get_world_size(group=group)
        return None, 1

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
        if dp_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=dp_group)
        if float(denominator) <= 0:
            raise ValueError("forward_backward requires a positive global weight sum")
        return denominator

    def _validate_window_size_across_dp(
        self,
        size: int,
        dp_group: dist.ProcessGroup | None,
        dp_size: int,
    ) -> None:
        if dp_size <= 1:
            return
        local_size = torch.tensor([size], dtype=torch.int64, device=self.device)
        sizes = torch.empty(dp_size, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(sizes, local_size, group=dp_group)
        if not bool((sizes == sizes[0]).all()):
            raise ValueError(f"every data-parallel rank must use the same number of microbatches; got {sizes.tolist()}")

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
