# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Typed input contract for training: :class:`Datum` and :func:`collate_datums`.

A ``Datum`` is the single-example input boundary between user/algorithm code
(SFT, RL post-training) and the training loop. It lives in ``components.datasets``
because feeding and collating examples is a data concern — and, crucially,
because that lets :func:`collate_datums` **reuse the canonical collaters**
(``default_collater`` for padded ``[B, T]`` and ``packed_sequence_thd_collater``
for THD) instead of forking a second padding/packing implementation that could
drift from them.

The companion output contract (``ModelOutput`` and the per-token extraction
helpers) lives in ``components.training`` — that side touches model logits, so
it is a forward concern, not a dataset one.

Conventions
-----------
* A ``Datum`` holds **one** sequence. ``input_ids`` is 1-D, shape ``[T]``.
* ``loss_inputs`` carries everything the loss needs, aligned to ``input_ids``
  token positions (length ``T``) for per-token entries:

  ===============  =======================================================
  key              meaning
  ===============  =======================================================
  ``target_tokens``  next-token targets, shape ``[T]`` (becomes ``labels``)
  ``weights``        per-token loss mask / weight (0 disables a position)
  ``logprobs``       old/behavior-policy logprobs (importance sampling)
  ``advantages``     advantage signal (PPO/GRPO), per-token or per-sample
  ===============  =======================================================

* Masking convention matches the codebase: a target position with
  ``weights == 0`` becomes ``ignore_index`` (default ``-100``) in ``labels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.utils import default_collater, packed_sequence_thd_collater

CROSS_ENTROPY_IGNORE_IDX = -100

__all__ = ["Datum", "collate_datums"]


@dataclass
class Datum:
    """A single training example.

    Args:
        input_ids: 1-D ``LongTensor`` of token ids, shape ``[T]``.
        loss_inputs: per-key tensors the loss consumes. Per-token entries are
            1-D and length ``T``; per-sample entries are scalar or shape ``[1]``.
            See the module docstring for the well-known keys.
    """

    input_ids: torch.Tensor
    loss_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, torch.Tensor):
            self.input_ids = torch.as_tensor(self.input_ids, dtype=torch.long)
        if self.input_ids.dim() != 1:
            raise ValueError(f"Datum.input_ids must be 1-D [T]; got shape {tuple(self.input_ids.shape)}")
        for key, value in self.loss_inputs.items():
            if not isinstance(value, torch.Tensor):
                self.loss_inputs[key] = torch.as_tensor(value)

    @property
    def seq_len(self) -> int:
        """Number of tokens in this example."""
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "Datum":
        """Return a copy with all tensors moved to ``device``."""
        return Datum(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            loss_inputs={k: v.to(device, non_blocking=non_blocking) for k, v in self.loss_inputs.items()},
        )

    def to_features(self, *, ignore_index: int = CROSS_ENTROPY_IGNORE_IDX) -> dict[str, list[int]]:
        """Emit the per-example dict the canonical collaters expect.

        ``labels`` is included only when ``loss_inputs["target_tokens"]`` is
        present, with positions where ``loss_inputs["weights"] == 0`` set to
        ``ignore_index``. Only integer token fields are emitted here — the
        collaters cast to ``LongTensor``; float side-inputs are batched
        separately by :func:`collate_datums`.

        Returns:
            ``{"input_ids": [...], "labels": [...]}`` as plain ``list[int]``.
        """
        features: dict[str, list[int]] = {"input_ids": self.input_ids.tolist()}
        if "target_tokens" in self.loss_inputs:
            labels = self.loss_inputs["target_tokens"].clone()
            weights = self.loss_inputs.get("weights")
            if weights is not None:
                labels = labels.masked_fill(weights == 0, ignore_index)
            features["labels"] = labels.tolist()
        return features

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain tensors (round-trips with :meth:`from_dict`)."""
        return {"input_ids": self.input_ids, "loss_inputs": dict(self.loss_inputs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Datum":
        return cls(input_ids=data["input_ids"], loss_inputs=dict(data.get("loss_inputs", {})))


def collate_datums(
    datums: list[Datum],
    *,
    packed: bool = False,
    pad_seq_len_divisible: int | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> dict[str, torch.Tensor]:
    """Collate a list of :class:`Datum` into a model-ready batch dict.

    Token fields are delegated to the **existing** canonical collaters so the
    padded / THD schema (``attention_mask`` / ``qkv_format`` / ``seq_lens``) is
    produced by the same code paths the dataset pipeline uses — no fork:

    * ``packed=False`` → ``default_collater`` (padded ``[B, T]``).
    * ``packed=True``  → ``packed_sequence_thd_collater`` (THD).

    Float per-token side-inputs (every ``loss_inputs`` key shared by all datums
    except ``target_tokens``, e.g. ``weights`` / ``logprobs`` / ``advantages``)
    are right-padded to the collated width and stacked under their own key —
    this is the part the token collaters cannot carry (they cast to
    ``LongTensor``). Per-sample (scalar / length-1) entries are stacked into a
    ``[B]`` tensor without padding.

    Args:
        datums: examples for this microbatch. Must be non-empty. One ``Datum``
            is treated as one sequence.
        packed: emit THD packed layout instead of padded ``[B, T]``.
        pad_seq_len_divisible: pad sequence length to a multiple of this value
            (padded mode only; TP/CP/FP8 alignment).
        ignore_index: label value for masked positions.

    Returns:
        The collater output dict, augmented with the float side-input tensors.
    """
    if len(datums) == 0:
        raise ValueError("collate_datums requires at least one Datum")

    features = [datums[i].to_features(ignore_index=ignore_index) for i in range(len(datums))]
    if packed:
        batch = packed_sequence_thd_collater([dict(f) for f in features])
    else:
        batch = default_collater([dict(f) for f in features], pad_seq_len_divisible)

    width = int(batch["input_ids"].shape[-1])
    shared_keys = set(datums[0].loss_inputs)
    for d in datums[1:]:
        shared_keys &= set(d.loss_inputs)
    for key in sorted(shared_keys - {"target_tokens"}):
        rows = [d.loss_inputs[key].to(torch.float).flatten() for d in datums]
        if all(r.shape[0] == d.seq_len for r, d in zip(rows, datums)):
            # Per-token field: right-pad to the collated width and stack.
            batch[key] = torch.stack([F.pad(r, (0, width - r.shape[0])) for r in rows])
        else:
            # Per-sample field: one value per datum.
            batch[key] = torch.stack([r.reshape(-1)[0] for r in rows])
    return batch
