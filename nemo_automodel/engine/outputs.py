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
    "EvaluationResult",
    "ForwardBackwardResult",
    "LossOutput",
    "StepResult",
]


def _validate_field_name(name: Any, *, container: str) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError(f"{container} field names must be non-empty strings, got {name!r}")


@dataclass(frozen=True, eq=False)
class LossOutput:
    """A local loss numerator and explicitly laid-out values returned with it.

    ``loss_sum`` is the already-weighted scalar numerator for the current
    physical batch. ``token_outputs`` contains tensors aligned with that
    batch's complete local token stream; the Engine restores them through
    packing and context parallelism before returning one tensor per Datum.
    ``batch_output`` is one opaque mapping for the complete physical batch.
    The Engine detaches and transports it across pipeline stages, but does not
    interpret or restore its contents across context parallelism.

    Args:
        loss_sum: Scalar local loss numerator.
        token_outputs: Named tensors whose leading axes exactly match prepared
            token weights: padded ``[batch, sequence, ...]``, canonical packed
            ``[1, tokens, ...]``, or active model-owned THD
            ``[tokens, ...]``. Trailing feature dimensions are preserved.
        batch_output: Caller-owned values for the complete batch, such as
            aggregate metrics. This is separate from token outputs because it
            has no Datum or token layout known to the Engine.
    """

    loss_sum: torch.Tensor
    token_outputs: Mapping[str, torch.Tensor] | None = None
    batch_output: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.loss_sum, torch.Tensor) or self.loss_sum.ndim != 0:
            raise TypeError("LossOutput.loss_sum must be a scalar torch.Tensor")
        token_outputs = {} if self.token_outputs is None else dict(self.token_outputs)
        if self.token_outputs is not None and not token_outputs:
            raise ValueError("LossOutput.token_outputs cannot be an empty mapping")
        for name, tensor in token_outputs.items():
            _validate_field_name(name, container="token_outputs")
            if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
                raise TypeError(f"token output {name!r} must be a non-scalar torch.Tensor")
        batch_output = None if self.batch_output is None else dict(self.batch_output)
        if self.batch_output is not None and not batch_output:
            raise ValueError("LossOutput.batch_output cannot be an empty mapping")
        if batch_output is not None:
            for name in batch_output:
                _validate_field_name(name, container="batch_output")
        if not token_outputs and batch_output is None:
            raise ValueError("LossOutput requires token_outputs or batch_output; return the scalar loss Tensor instead")
        object.__setattr__(self, "token_outputs", MappingProxyType(token_outputs))
        object.__setattr__(
            self,
            "batch_output",
            None if batch_output is None else MappingProxyType(batch_output),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation loss statistics and optional typed outputs.

    ``loss_sum`` and ``weight_sum`` are complete across model-parallel CP and
    PP ranks, but remain local to one data-parallel replica. The Engine adds no
    per-call DP loss-statistic collective, so callers reduce the two sums once
    at the end of an evaluation epoch. Distributed model wrappers may still
    communicate during forward and therefore retain their own call-alignment
    requirements. ``token_outputs`` remains local to the replica's input
    Datums. PP stages receive identical detached copies. Token fields declared
    through :class:`LossOutput` are restored to full token order across CP
    before being split into one tensor per Datum. ``batch_outputs`` is opaque,
    remains local to each context-parallel rank, and is copied only across the
    corresponding pipeline lane.

    Attributes:
        loss_sum: Detached weighted numerator for this Datum window.
        weight_sum: Detached full-sequence weight denominator for this window.
        token_outputs: One field-to-Datum-tensors mapping per input batch. A batch
            that requested no token outputs has an empty mapping.
        batch_outputs: One caller-owned mapping or ``None`` per input batch.
    """

    loss_sum: torch.Tensor
    weight_sum: torch.Tensor
    token_outputs: list[dict[str, list[torch.Tensor]]]
    batch_outputs: list[Mapping[str, Any] | None]


@dataclass(frozen=True)
class ForwardBackwardResult:
    """One complete optimizer window's loss statistics and optional outputs.

    The numerator is summed across the DP-CP gradient group. The full-sequence
    denominator is summed across DP only because CP ranks begin with replicated
    weights; both are synchronized across PP stages. ``loss`` is
    ``loss_sum / weight_sum`` when the denominator is nonzero and zero
    otherwise. ``token_outputs`` remains local to one data-parallel replica,
    is restored to full token order across CP when explicitly typed, and is
    identical on every PP stage in that replica.

    Attributes:
        loss: Detached weighted mean for the complete Datum window.
        loss_sum: Detached numerator summed across DP and CP, then synchronized
            across PP stages.
        weight_sum: Detached window denominator, summed across DP but
            not CP, then synchronized across PP stages.
        token_outputs: One field-to-Datum-tensors mapping per input batch. A batch
            that requested no token outputs has an empty mapping.
        batch_outputs: One caller-owned mapping or ``None`` per input batch.
            Values are not restored across context parallelism.
    """

    loss: torch.Tensor
    loss_sum: torch.Tensor
    weight_sum: torch.Tensor
    token_outputs: list[dict[str, list[torch.Tensor]]]
    batch_outputs: list[Mapping[str, Any] | None]


@dataclass(frozen=True)
class StepResult:
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


ParsedLossOutputs = LossOutput | None


@dataclass(frozen=True)
class _OutputRestorePlan:
    """How to turn one CP-local token stream back into Datums."""

    sharder: ContextParallelSharder
    is_thd: bool
    packed_seq_ids: torch.Tensor | None
    item_to_datum: tuple[int, ...] | None
    real_lengths: tuple[int, ...] | None
    padded_lengths: tuple[int, ...] | None
    token_mask: torch.Tensor | None
    synthetic_suffix_tokens: int = 0


def _scalar_loss_sum(loss_sum: Any, weights: torch.Tensor) -> torch.Tensor:
    if not isinstance(loss_sum, torch.Tensor) or loss_sum.ndim != 0:
        raise TypeError("loss_fn must return a scalar Tensor or LossOutput")
    if loss_sum.device != weights.device:
        raise ValueError("loss_fn loss and weights must be on the same device")
    return loss_sum


def _output_sequence_lengths(
    datums: Sequence[Datum],
    model_inputs: Mapping[str, Any],
    token_reference: torch.Tensor,
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
        token_reference: Tensor with the same leading token axes as the model
            output. This is internal routing metadata, not a user loss field.
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

    if packed_seq_ids is not None:
        if is_thd:
            raise ValueError("indexed-mask output routing cannot be combined with a THD model layout")
        if packed_seq_ids.ndim != 2 or packed_seq_ids.shape[0] != 1:
            raise ValueError(f"_packed_seq_ids must have shape [1, tokens], got {tuple(packed_seq_ids.shape)}")
        if packed_seq_ids.dtype == torch.bool or packed_seq_ids.is_floating_point():
            raise TypeError("_packed_seq_ids must use an integer dtype")
        if token_reference.ndim < 2 or tuple(token_reference.shape[:2]) != tuple(packed_seq_ids.shape):
            raise ValueError(
                "indexed-mask token outputs require model tokens on the same [1, tokens] axes as _packed_seq_ids"
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
        if sum(padded_lengths) != token_reference.numel():
            raise ValueError(
                f"packed padded lengths sum to {sum(padded_lengths)}, but model inputs have token width "
                f"{token_reference.numel()}"
            )
        return real_lengths, padded_lengths, None

    if token_reference.ndim < 2 or token_reference.shape[0] != len(item_to_datum):
        raise ValueError("padded token outputs require model tokens with one row per Datum")
    width = int(token_reference.shape[1])
    attention_mask = model_inputs.get("attention_mask")
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 2
        and tuple(attention_mask.shape) == tuple(token_reference.shape[:2])
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


def _token_sequence_dim(token_reference: torch.Tensor) -> int:
    if token_reference.ndim == 1:
        return 0
    if token_reference.ndim == 2:
        return 1
    raise ValueError(f"token routing reference must be one- or two-dimensional, got {tuple(token_reference.shape)}")


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
            raise NotImplementedError("prebatched token outputs can only produce their single outer Datum tensor")
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


def _loss_output_contract(
    outputs: ParsedLossOutputs,
    token_reference: Any,
    *,
    expected_datums: int | None,
) -> tuple[str | None, tuple[Any, ...]]:
    """Return a local validation error and a rank-comparable output schema."""
    if outputs is None:
        return None, ("none",)
    token_outputs = outputs.token_outputs
    if token_outputs and expected_datums is None:
        return (
            "a prebatched Datum may return token outputs only when num_microbatches=1 because its inner sample "
            "boundaries are not part of the Datum contract",
            ("unsupported", type(outputs).__name__),
        )
    if token_outputs and not isinstance(token_reference, torch.Tensor):
        return "token outputs require an internal token reference", ("invalid-token-reference",)

    error = None
    if token_outputs and token_reference.ndim not in (1, 2):
        error = f"token reference must be one- or two-dimensional, got {tuple(token_reference.shape)}"
    field_schema: list[tuple[Any, ...]] = []
    for name, tensor in sorted(token_outputs.items()):
        if tensor.shape[: token_reference.ndim] != token_reference.shape and error is None:
            error = (
                f"token output {name!r} must start with the physical token shape {tuple(token_reference.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != token_reference.device and error is None:
            error = (
                f"token output {name!r} must be on the model token device {token_reference.device}, got {tensor.device}"
            )
        field_schema.append(
            (
                name,
                str(tensor.dtype),
                tensor.device.type,
                tuple(tensor.shape),
            )
        )
    return error, (
        "typed",
        expected_datums if token_outputs else None,
        tuple(field_schema),
        outputs.batch_output is not None,
    )


def _token_mask_schema(token_mask: torch.Tensor | None) -> tuple[tuple[int, ...], str] | None:
    """Return compact, rank-comparable routing metadata for a padded token mask."""
    if token_mask is None:
        return None
    compact = token_mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256(compact.numpy().tobytes()).hexdigest()
    return tuple(compact.shape), digest


def _loss_output_restore_contract(
    outputs: LossOutput,
    token_reference: Any,
    plan: _OutputRestorePlan,
    *,
    datum_indices: tuple[int, ...] | None,
    chunk_index: int | None,
    cp_size: int,
) -> tuple[str | None, tuple[Any, ...]]:
    """Preflight one typed restore before any field enters a CP collective."""
    try:
        if not isinstance(token_reference, torch.Tensor):
            raise ValueError("token outputs require an internal token reference")
        seq_dim = _token_sequence_dim(token_reference)
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

        local_width = int(token_reference.shape[seq_dim])
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
            and token_reference.ndim == 2
            and (
                selected_layout is None
                or (selected_layout.input_token_stream_positions is None and selected_layout.input_row_shape is None)
            )
        ):
            # Some model-owned THD paths retain caller [B, S] rows instead of
            # reporting an explicit input_row_shape. Routing flattens those
            # token axes row-major after restoration.
            restored_width *= int(token_reference.shape[0])

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
                if token_reference.ndim != 2 or tuple(token_reference.shape) != tuple(plan.packed_seq_ids.shape):
                    raise ValueError(
                        "indexed-mask token outputs require the original [1, tokens] model shape; "
                        f"got {tuple(token_reference.shape)} and ids {tuple(plan.packed_seq_ids.shape)}"
                    )
                if restored_width != plan.packed_seq_ids.shape[1]:
                    raise ValueError(
                        f"restored indexed-mask token output has width {restored_width}, "
                        f"expected {plan.packed_seq_ids.shape[1]}"
                    )
            else:
                if token_reference.ndim != 2 or token_reference.shape[0] != len(datum_indices):
                    raise ValueError(
                        f"restored padded token output must have {len(datum_indices)} rows, "
                        f"got local model tokens {tuple(token_reference.shape)}"
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
            tuple(token_reference.shape),
            layout_schema,
            tuple(sorted(outputs.token_outputs)),
        )
        return None, schema
    except Exception as error:
        return str(error), ("invalid-restore", type(error).__name__)


def _parse_loss_result(
    result: torch.Tensor | LossOutput,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, ParsedLossOutputs, Exception | None]:
    """Normalize one loss function result without applying Datum routing.

    Output-only contract errors are returned separately so distributed peers
    can finish the same backward schedule and reach a common error decision.
    Loss-tensor errors still raise immediately because no valid backward value
    exists in that case.
    """
    if isinstance(result, torch.Tensor):
        return _scalar_loss_sum(result, weights), None, None
    if not isinstance(result, LossOutput):
        raise TypeError("loss_fn must return a Tensor or LossOutput")
    numerator = result.loss_sum
    if numerator.device != weights.device:
        raise ValueError("LossOutput.loss_sum and weights must be on the same device")
    try:
        detached = LossOutput(
            loss_sum=numerator.detach(),
            token_outputs=(
                {name: tensor.detach() for name, tensor in result.token_outputs.items()}
                if result.token_outputs
                else None
            ),
            batch_output=None if result.batch_output is None else _detach(result.batch_output),
        )
        return numerator, detached, None
    except Exception as error:
        return numerator, None, error


def _detach(value: Any) -> Any:
    """Detach tensor leaves without changing a caller-owned batch value."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    return value
