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

import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.utils import (
    default_collater,
    pack_features_for_thd,
    packed_sequence_thd_collater,
)

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

    def to_features(self, *, ignore_index: int = CROSS_ENTROPY_IGNORE_IDX) -> dict[str, list[int]]:
        """Emit the per-example dict the canonical collaters expect.

        Every position of a ``Datum`` is a real token, so ``attention_mask`` is
        all ones: it tells the padded collater exactly which positions it added,
        instead of leaving it to infer them from the pad token *value* — which
        misreads a real token that happens to equal the pad id (commonly
        ``pad_token_id == eos_token_id``) as padding.

        ``labels`` is included only when ``loss_inputs["target_tokens"]`` is
        present, with positions where ``loss_inputs["weights"] == 0`` set to
        ``ignore_index``. Only integer token fields are emitted here — the
        collaters cast to ``LongTensor``; float side-inputs are batched
        separately by :func:`collate_datums`.

        Returns:
            ``{"input_ids": [...], "attention_mask": [...], "labels": [...]}``
            as plain ``list[int]``.
        """
        features: dict[str, list[int]] = {
            "input_ids": self.input_ids.tolist(),
            "attention_mask": [1] * self.seq_len,
        }
        if "target_tokens" in self.loss_inputs:
            labels = self.loss_inputs["target_tokens"].clone()
            weights = self.loss_inputs.get("weights")
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
) -> dict[str, torch.Tensor]:
    """Collate a list of :class:`Datum` into a model-ready batch dict.

    Token fields are delegated to the **existing** canonical collaters so the
    padded / THD schema (``attention_mask`` / ``qkv_format`` / ``seq_lens``) is
    produced by the same code paths the dataset pipeline uses — no fork:

    * ``packed=False`` → ``default_collater`` (padded ``[B, T]``).
    * ``packed=True``  → :func:`pack_features_for_thd` concatenates all datums
      into one pre-packed record, then ``packed_sequence_thd_collater`` emits
      the flat ``[1, total_tokens]`` THD schema (``qkv_format="thd"``,
      per-sequence ``seq_lens`` for splitting outputs back per datum).

    Float per-token side-inputs (every ``loss_inputs`` key shared by all datums
    except ``target_tokens``, e.g. ``weights`` / ``logprobs`` / ``advantages``)
    are batched under their own key — this is the part the token collaters
    cannot carry (they cast to ``LongTensor``). Padded mode right-pads them to
    the collated width and stacks to ``[B, T]``; packed mode concatenates them
    in datum order to ``[1, total_tokens]``, aligned with ``input_ids``.
    Per-sample (scalar / length-1) entries are stacked into a ``[num_datums]``
    tensor without padding in both modes. A length-1 entry on a single-token
    sequence matches both shapes; it is read as per-token.

    Args:
        datums: examples for this microbatch. Must be non-empty. One ``Datum``
            is treated as one sequence.
        packed: pack all datums into one flat ``[1, total_tokens]`` THD row
            instead of the padded ``[B, T]`` layout.
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
        # Concatenate all datums into one pre-packed record and let the
        # canonical THD collater produce the [1, total] schema.
        batch = packed_sequence_thd_collater([pack_features_for_thd(features, ignore_index=ignore_index)])
    else:
        batch = default_collater([dict(f) for f in features], pad_seq_len_divisible)

    width = int(batch["input_ids"].shape[-1])
    keys = set(datums[0].loss_inputs)
    for d in datums[1:]:
        if set(d.loss_inputs) != keys:
            raise ValueError(
                "every Datum in a batch must carry the same loss_inputs keys "
                f"(got {sorted(keys)} and {sorted(d.loss_inputs)}); a missing key would "
                "silently drop it -- for `weights` that means losing the loss mask"
            )
    for key in sorted(keys - {"target_tokens"}):
        rows = [d.loss_inputs[key].to(torch.float).flatten() for d in datums]
        if all(r.shape[0] == d.seq_len for r, d in zip(rows, datums)):
            if packed:
                # Per-token field, packed: concatenate in datum order so the
                # values ride the same flat [1, total] axis as input_ids.
                batch[key] = torch.cat(rows).unsqueeze(0)
            else:
                # Per-token field, padded: right-pad to the collated width and stack.
                batch[key] = torch.stack([F.pad(r, (0, width - r.shape[0])) for r in rows])
        else:
            # Per-sample field: one value per datum (shape [num_datums], which in
            # packed mode is deliberately NOT the batch dim of the [1, total] rows).
            batch[key] = torch.stack([r.reshape(-1)[0] for r in rows])
    return batch
