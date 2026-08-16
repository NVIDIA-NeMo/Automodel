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

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.utils import (
    default_collater,
    pack_features_for_thd,
    packed_sequence_thd_collater,
)

CROSS_ENTROPY_IGNORE_IDX = -100

__all__ = ["Datum", "collate_datums"]


@dataclass(init=False)
class Datum:
    """One processor-ready training example.

    ``model_inputs`` contains exactly the keyword arguments needed by the
    model for one example. Text examples usually contain ``input_ids``;
    multimodal examples may additionally contain fields such as
    ``pixel_values`` and ``image_grid_thw``. ``loss_fn_inputs`` is kept
    separate so algorithm data such as targets, weights, old log-probabilities,
    and advantages is never passed to the model.

    Args:
        model_inputs: Model-ready values for one example. Values are
            model-specific because LLM and VLM processors emit different
            fields.
        loss_fn_inputs: Tensor values consumed by the loss function.
        input_ids: Deprecated convenience spelling for the old text-only API.
            It cannot be combined with ``model_inputs``.
    """

    model_inputs: dict[str, Any]
    loss_fn_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __init__(
        self,
        model_inputs: dict[str, Any] | torch.Tensor | list[int] | None = None,
        loss_fn_inputs: dict[str, torch.Tensor] | None = None,
        *,
        input_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
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
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.model_inputs:
            raise ValueError("Datum.model_inputs cannot be empty")

        input_ids = self.model_inputs.get("input_ids")
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids, dtype=torch.long)
                self.model_inputs["input_ids"] = input_ids
            if input_ids.ndim != 1:
                raise ValueError(
                    "Datum.model_inputs['input_ids'] must be 1-D [T] for one token sequence; "
                    f"got shape {tuple(input_ids.shape)}"
                )

        for key, value in self.loss_fn_inputs.items():
            if not isinstance(value, torch.Tensor):
                self.loss_fn_inputs[key] = torch.as_tensor(value)

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
            return int(self.model_inputs["input_ids"].shape[0])
        for key in ("target_tokens", "weights"):
            value = self.loss_fn_inputs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                return int(value.shape[0])
        raise ValueError("cannot infer token length; use a model-specific collate_fn for this Datum")

    def to_features(self, *, ignore_index: int = CROSS_ENTROPY_IGNORE_IDX) -> dict[str, Any]:
        """Convert one text Datum for the repository's canonical collaters.

        Token-aligned 1-D model inputs become Python lists so
        :func:`default_collater` can pad ragged examples. Other model inputs are
        passed through unchanged. A model-specific collater should be used for
        layouts such as multimodal mRoPE positions or variable media tensors.
        """
        if "input_ids" not in self.model_inputs:
            raise ValueError("the default collater requires model_inputs['input_ids']")

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
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Collate text Datums into separate model and loss inputs.

    This is the default text collater. Callers with model-specific VLM
    batching rules pass a thin callable around their existing collater to
    ``Engine`` instead.

    Args:
        datums: Non-empty examples for one microbatch.
        packed: Produce one flat THD token row instead of a padded batch.
        pad_seq_len_divisible: Round the padded token width to this multiple.
        ignore_index: Label fill value used internally by the THD collater.

    Returns:
        ``(model_inputs, loss_fn_inputs)``. Per-token loss inputs have shape
        ``[B, T]`` in padded mode and ``[1, total_tokens]`` in packed mode;
        scalar loss inputs have shape ``[B]``.
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
    for key in sorted(loss_keys):
        values = [datum.loss_fn_inputs[key] for datum in datums]
        per_token = all(value.ndim == 1 and value.shape[0] == datum.seq_len for value, datum in zip(values, datums))
        if per_token:
            if packed:
                loss_inputs[key] = torch.cat(values).unsqueeze(0)
            else:
                loss_inputs[key] = torch.stack([F.pad(value, (0, width - value.shape[0])) for value in values])
            continue

        if not all(value.numel() == 1 for value in values):
            shapes = [tuple(value.shape) for value in values]
            raise ValueError(
                f"the default collater only supports scalar or 1-D token-aligned loss inputs; {key!r} has {shapes}"
            )
        loss_inputs[key] = torch.stack([value.reshape(()) for value in values])

    return model_inputs, loss_inputs
