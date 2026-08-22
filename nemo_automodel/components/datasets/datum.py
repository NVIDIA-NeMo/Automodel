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

"""The model-input and loss-input boundary used by training engines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.utils import (
    default_collater,
    pack_features_for_thd,
    packed_sequence_thd_collater,
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin

CROSS_ENTROPY_IGNORE_IDX = -100

__all__ = ["CollatedLossInputs", "Datum", "LossInputLayout", "collate_datums", "collate_vlm_datums"]


def _pin_memory_tree(value: Any) -> Any:
    """Pin tensors in the common container shapes accepted by Engine inputs."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, dict):
        return {name: _pin_memory_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_pin_memory_tree(item) for item in value]
    if isinstance(value, tuple):
        pinned = tuple(_pin_memory_tree(item) for item in value)
        return type(value)(*pinned) if hasattr(value, "_fields") else pinned
    pin_memory = getattr(value, "pin_memory", None)
    return pin_memory() if callable(pin_memory) else value


class LossInputLayout(str, Enum):
    """How one loss input relates to the Datums being collated.

    ``PER_TOKEN`` values have a leading token axis and follow token padding,
    packing, CP sharding, and PP microbatching; trailing feature axes are
    preserved. ``PER_DATUM`` values contain one scalar for each outer
    :class:`Datum`; CP ranks receive the same scalars, so a loss callback uses
    them with its CP-local token contribution rather than returning a repeated
    full-sequence scalar. ``REPLICATED`` values are batch-level metadata copied
    unchanged to every CP/PP microbatch.
    """

    PER_TOKEN = "per_token"
    PER_DATUM = "per_datum"
    REPLICATED = "replicated"


class CollatedLossInputs(dict[str, torch.Tensor]):
    """Collated loss tensors with layout metadata outside the tensor mapping.

    This remains a normal ``dict`` for source compatibility. ``layouts`` is a
    complete mapping over the dictionary's initial keys. ``item_to_datum``
    maps each padded row or valid THD sequence back to the input Datum index;
    the current Engine accepts the identity mapping (one item per Datum, in
    input order). It is ``None`` when a collater cannot expose its inner
    boundaries, as with an already-prebatched batch. ``copy()`` preserves the
    side channel. ``pad_values`` supplies nonzero sentinels for explicitly
    padded ``PER_TOKEN`` fields. Converting this object to a plain ``dict``
    intentionally drops it and opts back into the Engine's conservative legacy
    inference.
    """

    def __init__(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        layouts: Mapping[str, LossInputLayout],
        item_to_datum: tuple[int, ...] | None,
        pad_values: Mapping[str, float | int | bool] | None = None,
    ) -> None:
        """Attach semantic layout metadata to collated loss tensors.

        Args:
            values: Collated tensors. ``PER_TOKEN`` values have shape
                ``[batch, sequence, ...]`` or ``[tokens, ...]``;
                ``PER_DATUM`` values have leading Datum axis.
            layouts: Complete semantic layout mapping for ``values``.
            item_to_datum: Optional mapping from collated rows/sequences to
                outer Datum indices.
            pad_values: Optional scalar fills for padded ``PER_TOKEN`` fields.
        """
        super().__init__(values)
        if set(layouts) != set(self):
            raise ValueError("layouts must contain exactly the CollatedLossInputs keys")
        if not all(isinstance(layout, LossInputLayout) for layout in layouts.values()):
            raise TypeError("every loss input layout must be a LossInputLayout")
        resolved_item_to_datum = None if item_to_datum is None else tuple(item_to_datum)
        if resolved_item_to_datum is not None and not all(isinstance(index, int) for index in resolved_item_to_datum):
            raise TypeError("item_to_datum must contain integer Datum indices")
        resolved_pad_values = dict(pad_values or {})
        unknown_pad_values = set(resolved_pad_values) - set(self)
        if unknown_pad_values:
            raise ValueError(f"pad_values contains unknown loss inputs: {sorted(unknown_pad_values)}")
        if not all(isinstance(value, (bool, int, float)) for value in resolved_pad_values.values()):
            raise TypeError("every loss input pad value must be a bool, int, or float")
        non_token_pad_values = {name for name in resolved_pad_values if layouts[name] is not LossInputLayout.PER_TOKEN}
        if non_token_pad_values:
            raise ValueError(f"pad_values are only valid for PER_TOKEN loss inputs: {sorted(non_token_pad_values)}")

        # Store a normal dict so the public collate result remains pickleable
        # across DataLoader worker boundaries. Expose only a read-only view.
        self._layouts = dict(layouts)
        self._pad_values = resolved_pad_values
        self.item_to_datum = resolved_item_to_datum

    @property
    def layouts(self) -> Mapping[str, LossInputLayout]:
        """Complete, read-only field-layout mapping."""
        return MappingProxyType(self._layouts)

    @property
    def pad_values(self) -> Mapping[str, float | int | bool]:
        """Read-only explicit fill values for padded ``PER_TOKEN`` fields."""
        return MappingProxyType(self._pad_values)

    def copy(self) -> CollatedLossInputs:
        """Return a shallow copy that retains the layout side channel."""
        return type(self)(
            self,
            layouts=self.layouts,
            item_to_datum=self.item_to_datum,
            pad_values=self.pad_values,
        )

    def __copy__(self) -> CollatedLossInputs:
        return self.copy()


@dataclass(init=False)
class Datum:
    """One processor-ready training item.

    ``model_inputs`` contains exactly the keyword arguments needed by the
    model. The default collater treats each Datum as one text sequence. A
    custom collater may also consume processor-ready multimodal fields or an
    already-collated batch. ``loss_fn_inputs`` is kept separate so algorithm
    data such as targets, weights, old log-probabilities, and advantages is
    never passed to the model.

    Args:
        model_inputs: Model-ready values for one training item. Values are
            model-specific because LLM and VLM processors emit different
            fields.
        loss_fn_inputs: Tensor values consumed by the loss function.
        loss_fn_input_layouts: Optional semantic layouts for loss fields. The
            canonical collater infers omitted fields using its legacy
            token-aligned-versus-scalar rules.
        loss_fn_input_pad_values: Optional nonzero padding sentinels for
            ``PER_TOKEN`` loss/side-channel fields. For example, routing replay
            uses ``-1`` to mean that a padded token keeps its live route.
        input_ids: Deprecated convenience spelling for the old text-only API.
            It cannot be combined with ``model_inputs``.
    """

    model_inputs: dict[str, Any]
    loss_fn_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    loss_fn_input_layouts: dict[str, LossInputLayout] = field(default_factory=dict)
    loss_fn_input_pad_values: dict[str, float | int | bool] = field(default_factory=dict)

    def __init__(
        self,
        model_inputs: dict[str, Any] | torch.Tensor | list[int] | None = None,
        loss_fn_inputs: dict[str, torch.Tensor] | None = None,
        *,
        input_ids: torch.Tensor | list[int] | None = None,
        loss_fn_input_layouts: Mapping[str, LossInputLayout] | None = None,
        loss_fn_input_pad_values: Mapping[str, float | int | bool] | None = None,
    ) -> None:
        """Initialize one processor-ready item and its loss side channels.

        Args:
            model_inputs: Model keyword mapping, or legacy 1-D ``input_ids``.
                Tensor layouts are model-specific.
            loss_fn_inputs: Loss/algorithm tensors before collation. A
                ``PER_TOKEN`` tensor has shape ``[sequence, ...]``.
            input_ids: Legacy 1-D ``[sequence]`` token ids.
            loss_fn_input_layouts: Optional semantic layout per loss field.
            loss_fn_input_pad_values: Optional scalar padding sentinel per
                ``PER_TOKEN`` loss field.
        """
        # Preserve the old positional ``Datum(input_ids, loss_fn_inputs)`` form
        # while downstream users move to the model-ready mapping.
        if model_inputs is not None and not isinstance(model_inputs, dict):
            if input_ids is not None:
                raise ValueError("pass either model_inputs or input_ids, not both")
            input_ids = model_inputs
            model_inputs = None
        if model_inputs is not None and input_ids is not None:
            raise ValueError("pass either model_inputs or input_ids, not both")
        if model_inputs is None:
            if input_ids is None:
                raise ValueError("Datum requires model_inputs")
            model_inputs = {"input_ids": input_ids}

        self.model_inputs = dict(model_inputs)
        self.loss_fn_inputs = dict(loss_fn_inputs or {})
        self.loss_fn_input_layouts = dict(loss_fn_input_layouts or {})
        self.loss_fn_input_pad_values = dict(loss_fn_input_pad_values or {})
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.model_inputs:
            raise ValueError("Datum.model_inputs cannot be empty")

        input_ids = self.model_inputs.get("input_ids")
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids, dtype=torch.long)
                self.model_inputs["input_ids"] = input_ids
            if input_ids.ndim == 0:
                raise ValueError("Datum.model_inputs['input_ids'] must have at least one dimension")

        for key, value in self.loss_fn_inputs.items():
            if not isinstance(value, torch.Tensor):
                self.loss_fn_inputs[key] = torch.as_tensor(value)

        unknown_layouts = set(self.loss_fn_input_layouts) - set(self.loss_fn_inputs)
        if unknown_layouts:
            raise ValueError(f"loss_fn_input_layouts contains unknown loss inputs: {sorted(unknown_layouts)}")
        if not all(isinstance(layout, LossInputLayout) for layout in self.loss_fn_input_layouts.values()):
            raise TypeError("every loss_fn_input_layouts value must be a LossInputLayout")
        unknown_pad_values = set(self.loss_fn_input_pad_values) - set(self.loss_fn_inputs)
        if unknown_pad_values:
            raise ValueError(f"loss_fn_input_pad_values contains unknown loss inputs: {sorted(unknown_pad_values)}")
        if not all(isinstance(value, (bool, int, float)) for value in self.loss_fn_input_pad_values.values()):
            raise TypeError("every loss_fn_input_pad_values value must be a bool, int, or float")

    @property
    def input_ids(self) -> torch.Tensor:
        """The text token sequence, retained for source compatibility."""
        input_ids = self.model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise AttributeError("this Datum has no tensor model input named 'input_ids'")
        return input_ids

    @property
    def seq_len(self) -> int:
        """Token length used by the default text collater."""
        if isinstance(self.model_inputs.get("input_ids"), torch.Tensor):
            input_ids = self.model_inputs["input_ids"]
            if input_ids.ndim == 1:
                return int(input_ids.shape[0])
            raise ValueError("the default collater requires 1-D input_ids; use a custom collate_fn")
        for key in ("target_tokens", "weights"):
            value = self.loss_fn_inputs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                return int(value.shape[0])
        raise ValueError("cannot infer token length; use a model-specific collate_fn for this Datum")

    def pin_memory(self) -> Datum:
        """Pin tensor inputs when a DataLoader pins this custom batch object.

        PyTorch delegates custom batch pinning to ``pin_memory()``. Recursively
        pin common model-input containers, then update this Datum only after
        both input mappings succeed. Non-tensor metadata and loss layout
        metadata remain unchanged.

        Returns:
            This Datum with model and loss tensors replaced by their
            pinned-memory counterparts.
        """
        model_inputs = _pin_memory_tree(self.model_inputs)
        loss_fn_inputs = _pin_memory_tree(self.loss_fn_inputs)
        self.model_inputs.clear()
        self.model_inputs.update(model_inputs)
        self.loss_fn_inputs.clear()
        self.loss_fn_inputs.update(loss_fn_inputs)
        return self

    def to_features(self, *, ignore_index: int = CROSS_ENTROPY_IGNORE_IDX) -> dict[str, Any]:
        """Convert one text Datum for the repository's canonical collaters.

        Token-aligned 1-D model inputs become Python lists so
        :func:`default_collater` can pad ragged examples. Other model inputs are
        passed through unchanged. A model-specific collater should be used for
        layouts such as multimodal mRoPE positions or variable media tensors.
        """
        if "input_ids" not in self.model_inputs:
            raise ValueError("the default collater requires model_inputs['input_ids']")
        if self.input_ids.ndim != 1:
            raise ValueError("the default collater requires 1-D input_ids; use a custom collate_fn")

        features: dict[str, Any] = {}
        for key, value in self.model_inputs.items():
            if isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] == self.seq_len:
                if value.is_floating_point():
                    raise ValueError(
                        f"the default text collater cannot preserve floating-point token field {key!r}; "
                        "use a model-specific collate_fn"
                    )
                features[key] = value.tolist()
            else:
                features[key] = value
        features.setdefault("attention_mask", [1] * self.seq_len)

        if "target_tokens" in self.loss_fn_inputs:
            labels = self.loss_fn_inputs["target_tokens"].clone()
            weights = self.loss_fn_inputs.get("weights")
            if weights is not None:
                labels = labels.masked_fill(weights == 0, ignore_index)
            features["labels"] = labels.tolist()
        return features


def collate_datums(
    datums: list[Datum],
    *,
    packed: bool = False,
    pad_seq_len_divisible: int | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> tuple[dict[str, Any], CollatedLossInputs]:
    """Collate text Datums into separate model and loss inputs.

    This is the default text collater. Callers with model-specific VLM
    batching rules pass a thin callable around their existing collater to
    ``Engine`` instead.

    Args:
        datums: Non-empty training items for one microbatch.
        packed: Produce one flat THD token row instead of a padded batch.
        pad_seq_len_divisible: Round the padded token width to this multiple.
        ignore_index: Label fill value used internally by the THD collater.

    Returns:
        ``(model_inputs, loss_fn_inputs)``. The second item remains a ``dict``
        and also exposes complete ``layouts`` and ``item_to_datum`` metadata.
        Per-token loss inputs have shape ``[B, T, ...]`` in padded mode and
        ``[1, total_tokens, ...]`` in packed mode; per-Datum scalar loss inputs
        have shape ``[B]``. Replicated inputs retain one copy of their original
        shape.
    """
    if not datums:
        raise ValueError("collate_datums requires at least one Datum")

    model_keys = set(datums[0].model_inputs)
    loss_keys = set(datums[0].loss_fn_inputs)
    for datum in datums[1:]:
        if set(datum.model_inputs) != model_keys:
            raise ValueError("every Datum in a microbatch must carry the same model_inputs keys")
        if set(datum.loss_fn_inputs) != loss_keys:
            raise ValueError("every Datum in a microbatch must carry the same loss_fn_inputs keys")

    features = [datum.to_features(ignore_index=ignore_index) for datum in datums]
    if packed:
        unsupported = model_keys - {"input_ids", "attention_mask"}
        if unsupported:
            raise ValueError(
                "the default packed collater only supports text input_ids; "
                f"use a model-specific collate_fn for {sorted(unsupported)}"
            )
        for datum in datums:
            attention_mask = datum.model_inputs.get("attention_mask")
            if attention_mask is not None and not bool(torch.as_tensor(attention_mask).bool().all()):
                raise ValueError(
                    "packed Datums must contain only real tokens; explicit attention_mask padding is unsupported"
                )
        model_inputs = packed_sequence_thd_collater([pack_features_for_thd(features, ignore_index=ignore_index)])
    else:
        model_inputs = default_collater([dict(feature) for feature in features], pad_seq_len_divisible)

    # Labels are loss data, not model input. The canonical collaters only see
    # them so their padding/THD machinery can be reused unchanged.
    model_inputs.pop("labels", None)
    width = int(model_inputs["input_ids"].shape[-1])

    loss_inputs: dict[str, torch.Tensor] = {}
    loss_layouts: dict[str, LossInputLayout] = {}
    loss_pad_values: dict[str, float | int | bool] = {}
    for key in sorted(loss_keys):
        values = [datum.loss_fn_inputs[key] for datum in datums]
        declared_layouts = [datum.loss_fn_input_layouts[key] for datum in datums if key in datum.loss_fn_input_layouts]
        if declared_layouts and len(declared_layouts) != len(datums):
            raise ValueError(f"every Datum must declare the layout for loss input {key!r}, or none may declare it")
        explicit_layouts = set(declared_layouts)
        if len(explicit_layouts) > 1:
            raise ValueError(f"every Datum must use the same explicit layout for loss input {key!r}")
        explicit_layout = next(iter(explicit_layouts), None)
        declared_pad_values = [
            datum.loss_fn_input_pad_values[key] for datum in datums if key in datum.loss_fn_input_pad_values
        ]
        if declared_pad_values and len(declared_pad_values) != len(datums):
            raise ValueError(f"every Datum must declare the pad value for loss input {key!r}, or none may declare it")
        if len(set(declared_pad_values)) > 1:
            raise ValueError(f"every Datum must use the same pad value for loss input {key!r}")

        token_aligned = [value.ndim >= 1 and value.shape[0] == datum.seq_len for value, datum in zip(values, datums)]
        legacy_token_aligned = [value.ndim == 1 and aligned for value, aligned in zip(values, token_aligned)]
        if explicit_layout is LossInputLayout.PER_TOKEN and not all(token_aligned):
            shapes = [tuple(value.shape) for value in values]
            raise ValueError(
                f"PER_TOKEN loss input {key!r} must have a leading axis matching each Datum's token length; "
                f"got {shapes}"
            )

        scalar_per_datum = [value.numel() == 1 for value in values]
        if explicit_layout is LossInputLayout.PER_DATUM and not all(scalar_per_datum):
            shapes = [tuple(value.shape) for value in values]
            raise ValueError(f"PER_DATUM loss input {key!r} must contain one value per Datum; got {shapes}")

        layout = explicit_layout
        if layout is None:
            if all(legacy_token_aligned):
                layout = LossInputLayout.PER_TOKEN
            elif all(scalar_per_datum):
                layout = LossInputLayout.PER_DATUM
            else:
                shapes = [tuple(value.shape) for value in values]
                raise ValueError(
                    "the default collater infers only scalar or 1-D token-aligned loss inputs; "
                    f"declare an explicit layout for {key!r} with shapes {shapes}"
                )

        loss_layouts[key] = layout
        if layout is LossInputLayout.PER_TOKEN:
            pad_value = declared_pad_values[0] if declared_pad_values else 0
            if declared_pad_values:
                loss_pad_values[key] = pad_value
            first = values[0]
            if not all(
                value.shape[1:] == first.shape[1:] and value.dtype == first.dtype and value.device == first.device
                for value in values[1:]
            ):
                shapes = [tuple(value.shape) for value in values]
                raise ValueError(
                    f"PER_TOKEN loss input {key!r} must use one trailing shape, dtype, and device; got {shapes}"
                )
            if packed:
                loss_inputs[key] = torch.cat(values, dim=0).unsqueeze(0)
            else:
                loss_inputs[key] = torch.stack(
                    [
                        F.pad(
                            value,
                            (*([0, 0] * (value.ndim - 1)), 0, width - value.shape[0]),
                            value=pad_value,
                        )
                        for value in values
                    ]
                )
            continue

        if declared_pad_values:
            raise ValueError(f"loss input {key!r} declares a pad value but does not use the PER_TOKEN layout")

        if layout is LossInputLayout.PER_DATUM:
            loss_inputs[key] = torch.stack([value.reshape(()) for value in values])
            continue

        first = values[0]
        if not all(
            value.shape == first.shape
            and value.dtype == first.dtype
            and value.device == first.device
            and torch.equal(value, first)
            for value in values[1:]
        ):
            raise ValueError(f"REPLICATED loss input {key!r} must have the same value in every Datum")
        loss_inputs[key] = first

    return model_inputs, CollatedLossInputs(
        loss_inputs,
        layouts=loss_layouts,
        item_to_datum=tuple(range(len(datums))),
        pad_values=loss_pad_values,
    )


def collate_vlm_datums(
    datums: list[Datum],
    *,
    processor: "ProcessorMixin",
    packed: bool = False,
    get_rope_index: Callable[..., object] | None = None,
    sequence_alignment: int = 1,
) -> tuple[dict[str, Any], CollatedLossInputs]:
    """Collate pre-tokenized VLM SFT Datums with processor-specific media inputs.

    The input Datums retain their unshifted token stream because
    :func:`~nemo_automodel.components.datasets.vlm.collate_fns.pad_collate_fn`
    owns the autoregressive shift. ``labels`` and ``weights`` therefore also
    use the unshifted token axis, with zero weight at target position zero.
    In packed mode, every Datum becomes one THD document and the collater
    preserves its real and aligned sequence lengths. Common VLM fields are
    handled by the canonical VLM collaters; additional processor tensor fields
    retain their leading media axis and are concatenated.

    Args:
        datums: Non-empty processor-ready VLM items. Each item has
            ``input_ids``, ``labels``, and ``weights`` tensors of shape
            ``[sequence]`` on matching unshifted token axes. Optional processor
            fields carry their processor-defined token or leading media axes.
        processor: Hugging Face processor (or compatible object) that supplies
            the tokenizer padding token.
        packed: Pack the Datums as THD documents instead of padding a batch.
        get_rope_index: Optional bound model callable used to materialize
            multi-axis VLM position IDs before packing.
        sequence_alignment: Per-document THD alignment. Context-parallel
            callers use ``2 * cp_size``. Multi-axis mRoPE with alignment above
            one is rejected by the canonical packed-VLM implementation.

    Returns:
        A pair of shifted/padded model inputs and layout-aware loss inputs.
        In padded mode, token model fields, labels, and weights have shape
        ``[batch, padded_sequence - 1]``. In packed mode they have shape
        ``[1, aligned_tokens]`` and model inputs include THD sequence metadata.
        Media tensors retain arbitrary trailing dimensions and are concatenated
        on their leading media axis.
    """
    if not datums:
        raise ValueError("collate_vlm_datums requires at least one Datum")
    if isinstance(sequence_alignment, bool) or not isinstance(sequence_alignment, int) or sequence_alignment < 1:
        raise ValueError(f"sequence_alignment must be a positive integer, got {sequence_alignment!r}")

    from nemo_automodel.components.datasets.vlm.collate_fns import pad_collate_fn

    examples = []
    for datum in datums:
        labels = datum.loss_fn_inputs.get("labels")
        weights = datum.loss_fn_inputs.get("weights")
        input_ids = datum.model_inputs.get("input_ids")
        if not all(isinstance(value, torch.Tensor) and value.ndim == 1 for value in (input_ids, labels, weights)):
            raise ValueError("VLM Datums require 1-D input_ids, labels, and weights")
        if datum.seq_len < 2:
            raise ValueError("VLM Datums require at least two tokens for autoregressive shifting")
        if labels.shape != input_ids.shape or weights.shape != input_ids.shape:
            raise ValueError("VLM labels and weights must match the unshifted input_ids shape")
        if bool(weights[0] != 0):
            raise ValueError("VLM weight at target position zero must be zero before autoregressive shift")
        examples.append({**datum.model_inputs, "labels": labels})

    if packed:
        from nemo_automodel.components.datasets.vlm.collate_fns import packed_sequence_thd_vlm_collater
        from nemo_automodel.components.datasets.vlm.neat_packing_vlm import (
            _aligned_length,
            _build_packed_vlm_sample,
            _compute_mrope_position_ids,
            _shift_sample,
        )

        position_ids = (
            [_compute_mrope_position_ids(example, get_rope_index) for example in examples]
            if get_rope_index is not None
            else [None] * len(examples)
        )
        if any(position is not None for position in position_ids) and not all(
            position is not None for position in position_ids
        ):
            raise ValueError("get_rope_index must return position IDs for every VLM Datum or none of them")
        has_mrope = bool(position_ids and position_ids[0] is not None)
        shifted_examples = []
        shifted_weights = []
        for example, position, datum in zip(examples, position_ids, datums):
            if position is not None:
                example["position_ids"] = position
            shifted_examples.append(_shift_sample(example, has_mrope=has_mrope))
            shifted_weights.append(datum.loss_fn_inputs["weights"][1:])

        tokenizer = getattr(processor, "tokenizer", processor)
        padding_idx = getattr(tokenizer, "pad_token_id", 0) or 0
        pack_size = sum(_aligned_length(item["input_ids"].shape[0], sequence_alignment) for item in shifted_examples)
        packed_sample = _build_packed_vlm_sample(
            shifted_examples,
            pack_size=pack_size,
            padding_idx=padding_idx,
            has_mrope=has_mrope,
            sequence_alignment=sequence_alignment,
        )
        model_inputs = packed_sequence_thd_vlm_collater([packed_sample], padding_idx=padding_idx)
        labels = model_inputs.pop("labels")
        weights = torch.cat(
            [
                F.pad(weight, (0, _aligned_length(weight.shape[0], sequence_alignment) - weight.shape[0]))
                for weight in shifted_weights
            ]
        ).unsqueeze(0)
    else:
        model_inputs = pad_collate_fn(examples, processor)
        labels = model_inputs.pop("labels")

        target_width = labels.shape[-1] + 1
        shifted_weights = [
            F.pad(datum.loss_fn_inputs["weights"], (0, target_width - datum.seq_len))[1:] for datum in datums
        ]
        weights = torch.stack(shifted_weights)

    unhandled_keys = set().union(*(datum.model_inputs.keys() for datum in datums)) - set(model_inputs)
    unhandled_keys -= {"input_ids", "attention_mask"}
    for key in sorted(unhandled_keys):
        values = [datum.model_inputs[key] for datum in datums if key in datum.model_inputs]
        if not values:
            continue
        if not all(isinstance(value, torch.Tensor) and value.ndim > 0 for value in values):
            raise TypeError(f"VLM processor field {key!r} must contain tensors with a leading media axis")
        try:
            model_inputs[key] = torch.cat(values, dim=0)
        except RuntimeError as exc:
            raise ValueError(f"VLM processor field {key!r} cannot be concatenated across samples") from exc

    return model_inputs, CollatedLossInputs(
        {"labels": labels, "weights": weights},
        layouts={
            "labels": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
        },
        item_to_datum=tuple(range(len(datums))),
        pad_values={"labels": CROSS_ENTROPY_IGNORE_IDX},
    )
