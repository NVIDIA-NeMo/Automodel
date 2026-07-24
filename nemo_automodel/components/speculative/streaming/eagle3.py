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

"""EAGLE-3 schema for the streaming data plane.

Mirrors the EAGLE-3 row of the speculative-decoding train/inference
disaggregation RFC's "Feature schema per algorithm" table:

| Required feature keys                              | Supervision                                  |
|----------------------------------------------------|----------------------------------------------|
| aux_hidden_states (3 aux concat, H*3),             | logits [B,S,V] OR target_probs + position_mask
| input_ids, attention_mask, loss_mask               | (draft-vocab; mutually exclusive)             |

"Exactly one supervision encoding" is the rule. The colocated target
backend ships full ``logits``; the remote target backend ships ``
target_probs`` + ``position_mask`` so the wire never carries a
full-vocab tensor.

DFlash and DSpark schemas land alongside their producer / loader.

This is a greenfield API: bump ``EAGLE3_SCHEMA_VERSION`` when the feature
layout changes; consumers reject mismatched refs rather than migrating older
samples.
"""

from __future__ import annotations

import logging

import torch

from nemo_automodel.components.speculative.streaming.refs import FeatureAlgorithm, FeatureSpec, SampleRef

logger = logging.getLogger(__name__)

EAGLE3_SCHEMA_VERSION = 1

EAGLE3_CORE_FEATURES: tuple[str, ...] = (
    "aux_hidden_states",
    "input_ids",
    "attention_mask",
    "loss_mask",
)

EAGLE3_LOGITS_SUPERVISION: tuple[str, ...] = ("logits",)

EAGLE3_DRAFT_VOCAB_SUPERVISION: tuple[str, ...] = ("target_probs", "position_mask")

EAGLE3_SUPERVISION_ENCODINGS: tuple[str, ...] = EAGLE3_CORE_FEATURES + EAGLE3_LOGITS_SUPERVISION

EAGLE3_PACKING_FEATURES: tuple[str, ...] = ("position_ids", "seq_lens", "doc_remaining")


def validate_eagle3_packing_inputs(
    *,
    position_ids: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
    doc_remaining: torch.Tensor | None,
) -> None:
    """Require all packing metadata together or none at all."""
    fields = (position_ids, seq_lens, doc_remaining)
    if any(field is not None for field in fields) and not all(field is not None for field in fields):
        raise ValueError(
            "EAGLE-3 sequence packing requires position_ids, seq_lens, and doc_remaining together; "
            f"got position_ids={'set' if position_ids is not None else 'None'}, "
            f"seq_lens={'set' if seq_lens is not None else 'None'}, "
            f"doc_remaining={'set' if doc_remaining is not None else 'None'}"
        )


def validate_eagle3_ref(ref: SampleRef) -> None:
    """Verify ``ref`` matches the EAGLE-3 schema before the consumer materializes.

    Args:
        ref: The tensor-free reference carried by the queue's lease.

    Raises:
        ValueError: if ``ref.algorithm`` is not :data:`FeatureAlgorithm.EAGLE3`,
            if any core feature is missing from the ref, or if the
            supervision encoding is neither ``logits`` alone nor the
            ``target_probs`` + ``position_mask`` pair.
    """
    if ref.algorithm is not FeatureAlgorithm.EAGLE3:
        raise ValueError(
            f"SampleRef.algorithm must be {FeatureAlgorithm.EAGLE3!r} for an EAGLE-3 consumer, got {ref.algorithm!r}"
        )
    keys = set(ref.feature_specs)
    missing = set(EAGLE3_CORE_FEATURES) - keys
    if missing:
        raise ValueError(f"EAGLE-3 ref missing required core features {sorted(missing)}; present={sorted(keys)}")
    packing_present = set(EAGLE3_PACKING_FEATURES) & keys
    if packing_present and packing_present != set(EAGLE3_PACKING_FEATURES):
        raise ValueError(
            "EAGLE-3 sequence packing requires position_ids, seq_lens, and doc_remaining together; "
            f"present={sorted(keys)}"
        )
    has_logits = "logits" in keys
    has_draft = "target_probs" in keys and "position_mask" in keys
    if has_logits == has_draft:
        raise ValueError(
            "EAGLE-3 ref must carry exactly one supervision encoding: 'logits' "
            "alone, or 'target_probs' + 'position_mask' together. "
            f"Present feature_specs={sorted(keys)}"
        )
    if ref.schema_version != EAGLE3_SCHEMA_VERSION:
        raise ValueError(f"EAGLE-3 ref schema_version must be {EAGLE3_SCHEMA_VERSION}, got {ref.schema_version}")


def eagle3_logits_tensors(
    *,
    aux_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    logits: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    doc_remaining: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Pack an EAGLE-3 colocated-path encoder's outputs into the producer's tensor dict.

    Args:
        aux_hidden_states: Tensor of shape ``[batch, sequence, hidden * num_aux_layers]``.
            ``num_aux_layers`` is the count of distinct layers the target's
            forward hooks captured (3 for the EAGLE-3 default recipe,
            concat'd along the last axis).
        input_ids: Tensor of shape ``[batch, sequence]``, ``torch.long``.
        attention_mask: Tensor of shape ``[batch, sequence]``, ``torch.long``.
        loss_mask: Tensor of shape ``[batch, sequence]``, ``torch.long``.
        logits: Tensor of shape ``[batch, sequence, vocab]``. Per the
            colocated path this is the target's full LM-head output.
        position_ids: Optional ``[batch, sequence]`` per-document positions
            when sequence packing is enabled.
        seq_lens: Optional ``[batch, max_docs]`` packed document lengths.
        doc_remaining: Optional ``[batch, sequence]`` cross-document TTT gate.

    Returns:
        A ``dict[str, torch.Tensor]`` keyed by
        :data:`EAGLE3_SUPERVISION_ENCODINGS`. Insertion order matches
        the colocated :class:`~nemo_automodel.components.speculative.eagle.target.Eagle3TargetBatch`
        field order so :meth:`SampleRef.feature_names` keeps a stable
        shape across the colocated and streaming paths.
    """
    tensors = {
        "aux_hidden_states": aux_hidden_states,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logits": logits,
    }
    if position_ids is not None:
        tensors["position_ids"] = position_ids
    if seq_lens is not None:
        tensors["seq_lens"] = seq_lens
    if doc_remaining is not None:
        tensors["doc_remaining"] = doc_remaining
    return tensors


def eagle3_logits_feature_specs(
    aux_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    logits: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    doc_remaining: torch.Tensor | None = None,
) -> dict[str, FeatureSpec]:
    """Build the per-feature :class:`FeatureSpec` map for a colocated encoder batch.

    Args: see :func:`eagle3_logits_tensors` -- one tensor per key, same
        layouts.

    Returns:
        A ``dict[str, FeatureSpec]`` keyed by feature name with the
        exact ``shape`` and ``dtype`` of each tensor. The consumer
        preallocates its receive buffer from this map before calling
        :meth:`FeatureStore.get`, mirroring how
        :func:`nemo_automodel.components.speculative.eagle.remote.protocol.encode_nccl_metadata`
        ships dtype + shape over the wire so the NCCL client can allocate.
    """
    specs = {
        "aux_hidden_states": FeatureSpec(shape=tuple(aux_hidden_states.shape), dtype=aux_hidden_states.dtype),
        "input_ids": FeatureSpec(shape=tuple(input_ids.shape), dtype=input_ids.dtype),
        "attention_mask": FeatureSpec(shape=tuple(attention_mask.shape), dtype=attention_mask.dtype),
        "loss_mask": FeatureSpec(shape=tuple(loss_mask.shape), dtype=loss_mask.dtype),
        "logits": FeatureSpec(shape=tuple(logits.shape), dtype=logits.dtype),
    }
    if position_ids is not None:
        specs["position_ids"] = FeatureSpec(shape=tuple(position_ids.shape), dtype=position_ids.dtype)
    if seq_lens is not None:
        specs["seq_lens"] = FeatureSpec(shape=tuple(seq_lens.shape), dtype=seq_lens.dtype)
    if doc_remaining is not None:
        specs["doc_remaining"] = FeatureSpec(shape=tuple(doc_remaining.shape), dtype=doc_remaining.dtype)
    return specs


__all__ = [
    "EAGLE3_CORE_FEATURES",
    "EAGLE3_DRAFT_VOCAB_SUPERVISION",
    "EAGLE3_LOGITS_SUPERVISION",
    "EAGLE3_PACKING_FEATURES",
    "EAGLE3_SCHEMA_VERSION",
    "EAGLE3_SUPERVISION_ENCODINGS",
    "eagle3_logits_feature_specs",
    "eagle3_logits_tensors",
    "validate_eagle3_packing_inputs",
    "validate_eagle3_ref",
]
