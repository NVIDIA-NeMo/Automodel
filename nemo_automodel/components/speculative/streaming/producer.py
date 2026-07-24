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

"""Streaming producer for speculative-decoding draft training.

The :class:`FeatureProducer` wraps a single target-backend forward pass
and ships its output through a :class:`FeatureStore`. The trainer-side
:class:`~nemo_automodel.components.speculative.streaming.loader.FeatureDataLoader`
re-materializes the same data as an
:class:`~nemo_automodel.components.speculative.eagle.target.Eagle3TargetBatch`
and yields it to the trainer loop.

The first producer wraps the existing
:class:`~nemo_automodel.components.speculative.eagle.target.HFEagle3TargetModel`
so the colocated path's numerical output travels bit-for-bit into the
streaming pipeline. Future producers (out-of-process SGLang, NCCL
remote) plug in behind the same ``FeatureProducer`` API without
touching the trainer.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch
from nemo_automodel.components.speculative.streaming.eagle3 import (
    EAGLE3_SCHEMA_VERSION,
    eagle3_logits_feature_specs,
    eagle3_logits_tensors,
    validate_eagle3_packing_inputs,
)
from nemo_automodel.components.speculative.streaming.refs import (
    FeatureAlgorithm,
    SampleRef,
)
from nemo_automodel.components.speculative.streaming.store import FeatureStore

logger = logging.getLogger(__name__)


def _resolve_algorithm(backend) -> FeatureAlgorithm:
    """Pick the producer-side :class:`FeatureAlgorithm` for a backend.

    Each algorithm-specific backend class declares its algorithm by name
    suffix (``*Eagle3TargetModel``, ``*DFlashTargetModel``, ...); the
    helper centralizes the dispatch so adding a new algorithm is a
    one-line change here.
    """
    name = type(backend).__name__
    if name.endswith("Eagle3TargetModel"):
        return FeatureAlgorithm.EAGLE3
    raise ValueError(
        f"FeatureProducer cannot infer FeatureAlgorithm from backend of type {name!r}; "
        f"pass algorithm=... explicitly to override"
    )


class FeatureProducer:
    """Run a target backend once and put the supervision into a :class:`FeatureStore`.

    Args:
        target_backend: an object exposing ``generate_batch(input_ids,
            attention_mask, loss_mask, ...) -> Eagle3TargetBatch`` plus
            the ``get_input_embeddings`` and ``set_vocab_mapping``
            accessors from
            :class:`~nemo_automodel.components.speculative.eagle.backend.Eagle3TargetBackend`.
        store: The :class:`FeatureStore` every ``produce`` call writes
            into.
        run_id: Stable across the whole training run; mirrored onto every
            :class:`SampleRef`.
        algorithm: Forced :class:`FeatureAlgorithm`; defaults to picking
            by the backend's runtime type.
        target_model_version: Monotonically increasing identifier of the
            target-model weights; surfaced on each :class:`SampleRef`.
        draft_weight_version: Same idea for the draft model's weights.
        sample_id_factory: Callable that produces the per-call sample id;
            defaults to ``"sample-{run_id}-{n}"``. Wrap with a hash of
            the input batch if you want stable ids across runs.

    Thread safety: ``produce`` is not reentrant with itself on the same
    producer instance; one thread should drive ``produce`` at a time.
    Multiple producers over the same store are fine; the store's lock
    serializes the puts.
    """

    def __init__(
        self,
        target_backend,
        store: FeatureStore,
        *,
        run_id: str,
        algorithm: FeatureAlgorithm | None = None,
        target_model_version: str = "0",
        draft_weight_version: str = "0",
        sample_id_factory: Callable[[int], str] | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("FeatureProducer.run_id must be a non-empty str")
        self._target = target_backend
        self._store = store
        self._run_id = run_id
        self._algorithm = algorithm if algorithm is not None else _resolve_algorithm(target_backend)
        self._target_model_version = target_model_version
        self._draft_weight_version = draft_weight_version
        self._sample_id_factory = sample_id_factory or self._default_sample_id
        self._sample_seq = 0

    def _default_sample_id(self, n: int) -> str:
        return f"sample-{self._run_id}-{n}"

    def set_vocab_mapping(self, selected_token_ids: torch.Tensor, selected_token_mask: torch.Tensor) -> None:
        """Thread the draft-vocab mapping through to the wrapped backend.

        The colocated backend ignores the mapping and projects the
        full-vocab ``logits`` trainer-side; a future draft-vocab
        encoding producer supplies the mapping here so the target can
        precompute ``target_probs`` + ``position_mask`` and the wire
        never carries a full-vocab tensor.
        """
        self._target.set_vocab_mapping(selected_token_ids, selected_token_mask)

    def get_input_embeddings(self) -> nn.Module:
        """Expose the wrapped backend's input-embedding module.

        Mirrors :meth:`Eagle3TargetBackend.get_input_embeddings` so the
        draft can seed its input embeddings from the target before the
        first streamed batch arrives.
        """
        return self._target.get_input_embeddings()

    @torch.no_grad()
    def produce(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> SampleRef:
        """Run the wrapped backend once and stash its supervision into the store.

        Args:
            input_ids: Tensor of shape ``[batch, sequence]``, ``torch.long``.
            attention_mask: Tensor of shape ``[batch, sequence]``, ``torch.long``.
            loss_mask: Tensor of shape ``[batch, sequence]``, ``torch.long``.
            position_ids: Per-document position ids when packing is on;
                ``None`` for non-packed. Carried through unchanged to
                the :class:`Eagle3TargetBatch` so the trainer's loss
                path stays bit-identical to the colocated path.
            seq_lens: Per-document sequence lengths when packing is on;
                ``None`` for non-packed.
            doc_remaining: ``[batch]`` long tensor gating cross-document
                TTT supervision when packing is on; ``None`` otherwise.

        Returns:
            A :class:`SampleRef` carrying a tensor-free description of
            the produced sample; the trainer's loader leases it from the
            queue and materializes it.

        Raises:
            MemoryError: forwarded from the store when the put would
                exceed ``max_samples`` or ``max_bytes``. Producers can
                wait for the store to drain through the queue's
                backpressure path (``put_blocks_until_below``).
        """
        validate_eagle3_packing_inputs(
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
        batch: Eagle3TargetBatch = self._target.generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
        if batch.logits is None:
            raise ValueError(
                "FeatureProducer requires a colocated backend that emits "
                "Eagle3TargetBatch.logits; got a precomputed draft-vocab encoding "
                "(target_probs + position_mask). Use a draft-vocab producer instead."
            )
        logits = batch.logits
        tensors = eagle3_logits_tensors(
            aux_hidden_states=batch.aux_hidden_states,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            loss_mask=batch.loss_mask,
            logits=logits,
            position_ids=batch.position_ids,
            seq_lens=batch.seq_lens,
            doc_remaining=batch.doc_remaining,
        )
        feature_specs = eagle3_logits_feature_specs(
            batch.aux_hidden_states,
            batch.input_ids,
            batch.attention_mask,
            batch.loss_mask,
            logits,
            batch.position_ids,
            batch.seq_lens,
            batch.doc_remaining,
        )
        self._sample_seq += 1
        sample_id = self._sample_id_factory(self._sample_seq)
        num_tokens = int(loss_mask.sum().item())
        ref = self._store.put(
            sample_id,
            tensors,
            run_id=self._run_id,
            algorithm=self._algorithm,
            schema_version=EAGLE3_SCHEMA_VERSION,
            target_model_version=self._target_model_version,
            draft_weight_version=self._draft_weight_version,
            num_tokens=num_tokens,
        )
        if ref.feature_specs != feature_specs:
            raise RuntimeError(
                f"SampleRef.feature_specs does not match what the producer just measured "
                f"(sample_id={sample_id}): ref={ref.feature_specs} measured={feature_specs}"
            )
        return ref

    def close(self) -> None:
        """Release any backend resources; the producer itself holds none."""


__all__ = ["FeatureProducer"]
