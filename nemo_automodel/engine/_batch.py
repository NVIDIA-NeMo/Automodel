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

"""Batch collation, padding, and slicing helpers for the Datum Engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nemo_automodel.components.datasets.datum import CollatedLossInputs, Datum, LossInputLayout
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS

CollateFn = Callable[
    [list[Datum]],
    tuple[dict[str, Any], dict[str, torch.Tensor] | CollatedLossInputs],
]
LossInputValue = torch.Tensor | tuple[torch.Tensor, ...]
LossInputs = dict[str, LossInputValue]

_LOSS_METADATA = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "padding_mask")


def collate_prebatched(datums: list[Datum]) -> tuple[dict[str, Any], CollatedLossInputs | dict[str, torch.Tensor]]:
    """Return one already-collated Datum without changing its layout.

    The Datum represents the whole prebatched item. Consequently, the Engine
    cannot split a :class:`LossOutput` token field into hidden inner samples;
    it returns the complete inner ``[B, S, ...]`` (or flat THD) tensor for that
    Datum. ``batch_output`` describes the same complete outer item and is
    rejected when AutoPipeline divides it into multiple inner microbatches,
    because caller-owned values have no generic merge operation.

    Args:
        datums: A one-item list whose model and loss tensor fields already have
            the batch layout expected by the model and loss function.

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


def _resolve_loss_batch_layout(
    datums: list[Datum],
    model_inputs: Mapping[str, Any],
    loss_inputs: Mapping[str, LossInputValue],
    *,
    require_weights: bool = True,
) -> _LossBatchLayout:
    """Resolve collater metadata without exposing it to the loss function."""
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
        declared = [datum.loss_fn_input_pad_values[name] for datum in datums if name in datum.loss_fn_input_pad_values]
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
                raise ValueError(f"CollatedLossInputs layout for field {name!r} disagrees with its Datum declaration")
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
    if require_weights and not isinstance(weights, torch.Tensor):
        raise ValueError("collate_fn must return a Tensor loss input named 'weights'")
    if weights is not None and not isinstance(weights, torch.Tensor):
        raise TypeError("collated loss input 'weights' must be a Tensor")

    fields: dict[str, LossInputLayout] = {}
    unresolved: set[str] = set()
    for name, value in loss_inputs.items():
        if name in datum_layouts:
            fields[name] = datum_layouts[name]
            continue

        if isinstance(value, torch.Tensor) and _loss_sequence_dim(model_inputs, value) is not None:
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


def _validate_loss_batch_layout(
    datums: Sequence[Datum],
    model_inputs: Mapping[str, Any],
    loss_inputs: Mapping[str, LossInputValue],
    layouts: Mapping[str, LossInputLayout],
) -> tuple[str | None, int | None]:
    """Validate semantic loss layouts before any CP/PP transformation."""
    if set(layouts) != set(loss_inputs):
        raise ValueError("loss input layouts must describe every collated loss field exactly once")
    token_fields = [name for name, layout in layouts.items() if layout is LossInputLayout.PER_TOKEN]
    sequence_dims: dict[str, int] = {}
    for name in token_fields:
        value = loss_inputs[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"per-token loss field {name!r} must be a Tensor")
        sequence_dim = _loss_sequence_dim(model_inputs, value)
        if sequence_dim is None:
            raise ValueError(f"per-token loss field {name!r} must match the primary model token axes")
        sequence_dims[name] = sequence_dim
    if len(set(sequence_dims.values())) > 1:
        raise ValueError(f"per-token loss fields disagree on the model sequence axis: {sequence_dims}")

    # A reference carries only the common token prefix used by CP/PP slicing.
    # Choose the lowest-rank field so a routed-expert tensor [B, S, L, K], for
    # example, cannot make an ordinary target tensor [B, S] look misaligned.
    token_reference_name = None
    if token_fields:
        minimum_rank = min(loss_inputs[name].ndim for name in token_fields)
        minimum_rank_fields = [name for name in token_fields if loss_inputs[name].ndim == minimum_rank]
        token_reference_name = next(
            (name for name in ("weights", "labels") if name in minimum_rank_fields),
            min(minimum_rank_fields),
        )
    loss_seq_dim = None if token_reference_name is None else sequence_dims[token_reference_name]

    for name, value in loss_inputs.items():
        layout = layouts[name]
        if layout is LossInputLayout.PER_DATUM:
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != len(datums):
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(f"per-Datum loss field {name!r} must have leading size {len(datums)}, got {shape}")
    return token_reference_name, loss_seq_dim


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


def _pin_memory(value: Any) -> Any:
    """Page-lock CPU tensors in the container shapes accepted by Engine inputs."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.is_pinned():
            return value
        return value.pin_memory()
    if isinstance(value, dict):
        return {key: _pin_memory(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pin_memory(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pin_memory(item) for item in value)
    return value


def _pad_tensor_axis(tensor: torch.Tensor, *, dim: int, amount: int, value: int | float | bool) -> torch.Tensor:
    """Right-pad one tensor axis without disturbing trailing feature axes."""
    if amount <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = amount
    suffix = torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, suffix), dim=dim)


def _pad_hybridep_padded_sequence(
    model_inputs: dict[str, Any],
    loss_inputs: LossInputs,
    layouts: Mapping[str, LossInputLayout],
    pad_values: Mapping[str, int | float | bool],
    *,
    target_sequence_length: int,
    padding_token_id: int,
) -> None:
    """Pad one BSH batch to the HybridEP-wide sequence width in place.

    Args:
        model_inputs: Padded model mapping with primary shape ``[batch,
            sequence]`` or ``[batch, sequence, hidden]``. Declared token side
            channels use the same leading axes; media stays item-aligned.
        loss_inputs: Mapping whose ``PER_TOKEN`` tensors start with ``[batch,
            sequence]``. Other layouts remain unchanged.
        layouts: Semantic layout for every loss field.
        pad_values: Explicit per-token fills; labels default to ``-100``.
        target_sequence_length: HybridEP-wide physical sequence extent.
        padding_token_id: Fill value for synthetic ``input_ids`` tokens.

    Returns:
        None. Model and loss mappings are updated in place. Every synthetic
        token has ``padding_mask=True`` and every synthetic loss weight is zero.
    """
    token_template = _model_token_template(model_inputs)
    if token_template.ndim != 2:
        raise ValueError("HybridEP padded equalization requires a [batch, sequence] token template")
    batch_size, sequence_length = map(int, token_template.shape)
    if target_sequence_length < sequence_length:
        raise ValueError(f"HybridEP target width {target_sequence_length} is below local width {sequence_length}")
    sequence_padding = target_sequence_length - sequence_length

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None and (not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2):
        shape = (
            tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else type(attention_mask).__name__
        )
        raise NotImplementedError(f"HybridEP padding requires a 2-D attention_mask, got {shape}")
    expected_shape = (batch_size, sequence_length)
    if isinstance(attention_mask, torch.Tensor):
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(f"attention_mask must have shape {expected_shape}, got {tuple(attention_mask.shape)}")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("two-dimensional attention_mask must contain only binary padding values")

    padding_mask = model_inputs.get("padding_mask")
    if padding_mask is None:
        padding_mask = (
            attention_mask == 0
            if isinstance(attention_mask, torch.Tensor)
            else torch.zeros_like(token_template, dtype=torch.bool)
        )
    if not isinstance(padding_mask, torch.Tensor) or tuple(padding_mask.shape) != expected_shape:
        shape = tuple(padding_mask.shape) if isinstance(padding_mask, torch.Tensor) else type(padding_mask).__name__
        raise ValueError(f"padding_mask must have shape {expected_shape}, got {shape}")
    padding_mask = padding_mask.to(torch.bool)

    model_fills: dict[str, int | float | bool] = {
        "input_ids": padding_token_id,
        "inputs_embeds": 0,
        "attention_mask": 0,
        "padding_mask": True,
        "mm_token_type_ids": 0,
        "token_type_ids": 0,
        "_packed_seq_ids": 0,
        "packed_seq_ids": 0,
    }
    model_replacements: dict[str, torch.Tensor] = {}
    for name, fill in model_fills.items():
        tensor = padding_mask if name == "padding_mask" else model_inputs.get(name)
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tuple(tensor.shape[:2]) != expected_shape:
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise ValueError(f"model field {name!r} must start with {expected_shape}, got {shape}")
        model_replacements[name] = _pad_tensor_axis(tensor, dim=1, amount=sequence_padding, value=fill)

    position_ids = model_inputs.get("position_ids")
    if position_ids is not None:
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError("position_ids must be a Tensor")
        if position_ids.ndim == 1 and tuple(position_ids.shape) == (sequence_length,):
            sequence_dim = 0
        elif (
            position_ids.ndim == 2
            and position_ids.shape[1] == sequence_length
            and position_ids.shape[0] in (1, batch_size)
        ):
            sequence_dim = 1
        elif position_ids.ndim == 3 and tuple(position_ids.shape[1:]) == expected_shape:
            sequence_dim = 2
        else:
            raise ValueError(f"unsupported padded position_ids shape {tuple(position_ids.shape)}")
        model_replacements["position_ids"] = _pad_tensor_axis(
            position_ids, dim=sequence_dim, amount=sequence_padding, value=0
        )

    cache_position = model_inputs.get("cache_position")
    if cache_position is not None:
        if not isinstance(cache_position, torch.Tensor) or tuple(cache_position.shape) != (sequence_length,):
            shape = (
                tuple(cache_position.shape)
                if isinstance(cache_position, torch.Tensor)
                else type(cache_position).__name__
            )
            raise ValueError(f"cache_position must have shape {(sequence_length,)}, got {shape}")
        suffix = torch.arange(
            sequence_length, target_sequence_length, dtype=cache_position.dtype, device=cache_position.device
        )
        model_replacements["cache_position"] = torch.cat((cache_position, suffix))

    known_fields = set(model_fills) | {"position_ids", "cache_position"} | set(VLM_INPUT_KEYS)
    for name, value in model_inputs.items():
        if name not in known_fields and isinstance(value, torch.Tensor) and value.ndim >= 2:
            if tuple(value.shape[:2]) == expected_shape:
                raise ValueError(f"token-aligned model field {name!r} has no HybridEP padding sentinel")

    loss_replacements: dict[str, torch.Tensor] = {}
    for name, layout in layouts.items():
        tensor = loss_inputs[name]
        if layout is LossInputLayout.PER_TOKEN:
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tuple(tensor.shape[:2]) != expected_shape:
                shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
                raise ValueError(f"per-token loss field {name!r} has incompatible shape {shape}")
            loss_replacements[name] = _pad_tensor_axis(
                tensor,
                dim=1,
                amount=sequence_padding,
                value=0 if name == "weights" else pad_values.get(name, -100 if name == "labels" else 0),
            )

    model_inputs.update(model_replacements)
    loss_inputs.update(loss_replacements)


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


def _loss_sequence_dim(model_inputs: Mapping[str, Any], value: torch.Tensor) -> int | None:
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
