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

"""Structured Engine outputs and their token-layout restoration helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any

import torch

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder

LossInputValue = torch.Tensor | tuple[torch.Tensor, ...]

__all__ = [
    "ForwardBackwardResult",
    "ForwardResult",
    "LossFnOutputBatch",
    "OptimStepResult",
    "PerTokenOutput",
]


def _validate_field_name(name: Any, *, container: str) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError(f"{container} field names must be non-empty strings, got {name!r}")


@dataclass(frozen=True, eq=False)
class PerTokenOutput:
    """One callback tensor aligned with the current local token stream.

    The tensor's leading dimensions must exactly match the current CP-local
    loss ``weights`` shape (``[B, S]`` or ``[T]``); only trailing feature
    dimensions may be added. The tensor must also be on the same device as the
    weights. The Engine therefore knows the token axis without guessing from
    arbitrary shapes.
    ``fill_value`` is used only when restoring a caller position for which a
    backend did not retain a token (for example, a dropped slot in a repacked
    sequence).

    The tensor is deliberately retained by reference rather than cloned. This
    preserves its device, autograd relationship, and avoids copying potentially
    large token outputs; the Engine detaches returned records at its API
    boundary.

    Args:
        tensor: Tensor covering the complete CP-local token stream for the
            current loss-callback invocation.
        fill_value: Scalar value used for positions introduced by layout
            restoration. It must be representable by ``tensor.dtype`` when the
            Engine performs that restoration.
    """

    tensor: torch.Tensor
    fill_value: float | int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"PerTokenOutput.tensor must be a torch.Tensor, got {type(self.tensor).__name__}")
        if self.tensor.ndim == 0:
            raise ValueError("PerTokenOutput.tensor must have at least one dimension")
        if not isinstance(self.fill_value, (int, float)):
            raise TypeError(f"PerTokenOutput.fill_value must be an int or float, got {type(self.fill_value).__name__}")


@dataclass(frozen=True, eq=False)
class LossFnOutputBatch:
    """Layout-aware outputs from one loss-callback invocation.

    ``per_token`` holds batch-level local token streams. Callbacks should not
    split those streams into per-Datum tensors themselves: under packed context
    parallelism, different ranks can own different numbers of tokens from one
    Datum. The Engine can restore each complete stream first and then route it
    to the final per-Datum records.

    Restoration requires the active context-parallel backend to report a
    reversible token layout. Backends that do not report one fail explicitly
    when this typed output is requested; legacy output records remain usable.

    ``per_datum`` optionally supplies the ordinary output records that the
    restored token fields will augment. It follows the existing callback
    convention of one mapping per logical Datum in the current microbatch.
    These records are opaque and must already be identical on CP ranks (for
    example, sample IDs copied from a PER_DATUM loss input). A token-derived
    CP-local value belongs in ``per_token`` so the Engine can restore it before
    producing records.

    Input mappings, the record sequence, and each record mapping are copied and
    exposed as read-only views. Tensor and other nested values are intentionally
    retained by reference so the envelope is cheap and preserves autograd until
    the Engine consumes it.

    Args:
        per_token: Non-empty mapping from output field name to its local token
            tensor and restoration fill value. Tensor leading dimensions and
            device must match the callback's loss weights.
        per_datum: Optional sequence of existing per-Datum output mappings.
            These mappings may not already contain a field named by
            ``per_token`` because the Engine will add those fields after token
            restoration.
    """

    per_token: Mapping[str, PerTokenOutput]
    per_datum: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.per_token, Mapping):
            raise TypeError(f"LossFnOutputBatch.per_token must be a mapping, got {type(self.per_token).__name__}")
        copied_per_token = dict(self.per_token)
        if not copied_per_token:
            raise ValueError("LossFnOutputBatch.per_token cannot be empty")
        for name, output in copied_per_token.items():
            _validate_field_name(name, container="per_token")
            if not isinstance(output, PerTokenOutput):
                raise TypeError(f"per_token field {name!r} must be a PerTokenOutput, got {type(output).__name__}")

        object.__setattr__(self, "per_token", MappingProxyType(copied_per_token))
        if self.per_datum is None:
            return
        if not isinstance(self.per_datum, Sequence) or isinstance(self.per_datum, (str, bytes)):
            raise TypeError(
                "LossFnOutputBatch.per_datum must be a sequence of mappings or None, "
                f"got {type(self.per_datum).__name__}"
            )

        copied_records: list[Mapping[str, Any]] = []
        token_names = set(copied_per_token)
        for index, record in enumerate(self.per_datum):
            if not isinstance(record, Mapping):
                raise TypeError(f"per_datum record {index} must be a mapping, got {type(record).__name__}")
            copied_record = dict(record)
            for name in copied_record:
                _validate_field_name(name, container=f"per_datum record {index}")
            conflicts = token_names.intersection(copied_record)
            if conflicts:
                raise ValueError(f"per_datum record {index} conflicts with per_token fields {sorted(conflicts)}")
            copied_records.append(MappingProxyType(copied_record))
        object.__setattr__(self, "per_datum", tuple(copied_records))


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


ParsedLossOutputs = list[dict[str, Any]] | LossFnOutputBatch | None


@dataclass(frozen=True)
class _OutputRestorePlan:
    """How to turn one CP-local callback token stream back into Datums."""

    sharder: ContextParallelSharder
    is_thd: bool
    packed_seq_ids: torch.Tensor | None
    item_to_datum: tuple[int, ...] | None
    real_lengths: tuple[int, ...] | None
    padded_lengths: tuple[int, ...] | None
    token_mask: torch.Tensor | None
    synthetic_suffix_tokens: int = 0


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
    packed_seq_ids: torch.Tensor | None = None,
) -> tuple[
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    torch.Tensor | None,
]:
    """Capture real and padded sequence lengths before CP mutates the batch.

    Args:
        datums: Logical items represented by the collated token tensors.
        model_inputs: Model mapping whose token fields have padded
            ``[batch, sequence]`` or packed ``[1, tokens]`` layout.
        loss_inputs: Loss mapping whose per-token ``weights`` has the same
            leading token axes as ``model_inputs``.
        item_to_datum: Identity mapping from collated logical items to Datums.
        is_thd: Whether the physical model layout is THD.
        packed_seq_ids: Optional 1-based document map of shape ``[1, tokens]``
            for a physically BSH indexed-mask pack; zero denotes padding.

    Returns:
        Real lengths, padded lengths, and an optional padded-batch boolean mask.
        Length tuples contain one value per Datum. The mask has shape
        ``[batch, sequence]`` when ordinary padded rows need sparse routing.
    """
    if item_to_datum is None:
        return None, None, None
    if item_to_datum != tuple(range(len(datums))):
        raise ValueError("token output restoration requires collater items to preserve Datum order")

    weights = loss_inputs.get("weights")
    if not isinstance(weights, torch.Tensor):
        raise ValueError("token output restoration requires Tensor loss weights")
    if packed_seq_ids is not None:
        if is_thd:
            raise ValueError("indexed-mask output routing cannot be combined with a THD model layout")
        if packed_seq_ids.ndim != 2 or packed_seq_ids.shape[0] != 1:
            raise ValueError(f"_packed_seq_ids must have shape [1, tokens], got {tuple(packed_seq_ids.shape)}")
        if packed_seq_ids.dtype == torch.bool or packed_seq_ids.is_floating_point():
            raise TypeError("_packed_seq_ids must use an integer dtype")
        if weights.ndim < 2 or tuple(weights.shape[:2]) != tuple(packed_seq_ids.shape):
            raise ValueError(
                "indexed-mask token outputs require weights on the same [1, tokens] axes as _packed_seq_ids"
            )
        if bool((packed_seq_ids < 0).any()):
            raise ValueError("_packed_seq_ids must contain only zero padding or positive document IDs")
        flat_seq_ids = packed_seq_ids[0]
        valid_tokens = int((flat_seq_ids > 0).sum())
        if not bool((flat_seq_ids[:valid_tokens] > 0).all()) or bool((flat_seq_ids[valid_tokens:] != 0).any()):
            raise ValueError("_packed_seq_ids zero padding must be one trailing suffix")
        valid_seq_ids = flat_seq_ids[:valid_tokens]
        observed = tuple(int(value) for value in torch.unique(valid_seq_ids).tolist())
        expected = tuple(range(1, len(item_to_datum) + 1))
        if observed != expected:
            raise ValueError(f"_packed_seq_ids must contain every document ID 1..{len(expected)}, got {observed}")
        if valid_seq_ids.numel() > 1:
            transitions = valid_seq_ids[1:] - valid_seq_ids[:-1]
            if bool(((transitions < 0) | (transitions > 1)).any()):
                raise ValueError("_packed_seq_ids must contain one contiguous segment for each document in order")
        real_lengths = tuple(int((packed_seq_ids == index).sum()) for index in expected)
        return real_lengths, real_lengths, None
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
    """Split one restored collated token tensor into input-Datum coordinates.

    Args:
        tensor: Restored token output of shape ``[batch, sequence, ...]`` or
            a flat THD-equivalent ``[tokens, ...]`` layout.
        plan: Captured physical layout and logical document routing metadata.
        datum_indices: Outer Datum indices represented by ``tensor``.
        seq_dim: Sequence axis in ``tensor`` before any packed-row squeeze.

    Returns:
        One contiguous tensor per requested Datum, each with shape
        ``[datum_tokens, ...]``.
    """
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

    if plan.packed_seq_ids is not None:
        packed_seq_ids = plan.packed_seq_ids
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != tuple(packed_seq_ids.shape):
            raise ValueError(
                "restored indexed-mask token output must start with the original [1, tokens] axes; "
                f"got {tuple(tensor.shape)} and ids {tuple(packed_seq_ids.shape)}"
            )
        if plan.item_to_datum != datum_indices:
            raise ValueError("indexed-mask output routing requires the complete Datum group in input order")
        row = tensor.select(0, 0)
        valid_tokens = sum(plan.real_lengths)
        return [piece.contiguous() for piece in torch.split(row[:valid_tokens], plan.real_lengths, dim=0)]

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
            elif plan.packed_seq_ids is not None:
                if plan.item_to_datum != datum_indices:
                    raise ValueError("indexed-mask output routing requires the complete Datum group in input order")
                if weights.ndim != 2 or tuple(weights.shape) != tuple(plan.packed_seq_ids.shape):
                    raise ValueError(
                        "indexed-mask token outputs require weights with the original [1, tokens] shape; "
                        f"got {tuple(weights.shape)} and ids {tuple(plan.packed_seq_ids.shape)}"
                    )
                if restored_width != plan.packed_seq_ids.shape[1]:
                    raise ValueError(
                        f"restored indexed-mask token output has width {restored_width}, "
                        f"expected {plan.packed_seq_ids.shape[1]}"
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
