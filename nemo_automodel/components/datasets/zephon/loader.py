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

"""Thin iterable adapter with the reference aggregated opaque checkpoint protocol."""

from __future__ import annotations

import logging

# Zephon's established trusted-checkpoint wire format uses pickle. Keep this
# narrow exception until its public API provides a replacement byte serializer.
import pickle  # noqa: S403
from collections import Counter
from collections.abc import Callable, Generator, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from nemo_automodel.components.datasets.packing import (
    DEFAULT_PACKED_SEQUENCE_CONTRACT,
    BatchOutputFormat,
    PackedSequenceContract,
    build_packed_sequence_metadata,
)
from nemo_automodel.components.datasets.zephon.attention import TrainingBatch, adapt_model_inputs, enable_xdoc_attention
from nemo_automodel.components.datasets.zephon.output import convert_batch_output

if TYPE_CHECKING:
    from zephon import Pipeline, SampleBatch


logger = logging.getLogger(__name__)


def convert_text_batch(
    sample_batch: SampleBatch,
    *,
    packing_contract: PackedSequenceContract,
    ignore_index: int,
    mask_segment_starts: bool,
    allow_cross_document_attention: bool,
    output_format: BatchOutputFormat = "batched",
) -> TrainingBatch:
    """Collate text, retaining boundaries independently of the model's positions.

    Args:
        sample_batch: Packed documents with unshifted token IDs and scalar
            subdocument-reset positions [sequence] in each payload.
        packing_contract: Model-derived return requests; mask and gathered-token
            requests apply only to batched output.
        ignore_index: Loss exclusion value for shifted labels.
        mask_segment_starts: Mask targets at original subdocument boundaries.
        allow_cross_document_attention: Allow earlier subdocuments as context
            and number each packed document continuously for RoPE.
        output_format: Batched mask/metadata route or native THD lengths.

    Returns:
        Token IDs, labels, positions, padding and compact boundary/attention IDs
        of shape [batch, sequence]. Positions reset per subdocument with xdoc
        off and run continuously per batch document with xdoc on. Original
        boundaries and supervision are retained in either mode. Inputs are not
        mutated; Zephon owns token/label shifting and collation.
    """
    # THD derives kernel metadata from lengths, not batched mask/gather requests.
    if output_format == "thd":
        packing_contract = DEFAULT_PACKED_SEQUENCE_CONTRACT
    batch = sample_batch.to_training(
        tokens_field="input_ids",
        return_labels=True,
        ignore_index=ignore_index,
        mask_segment_starts=mask_segment_starts,
        return_padding_mask=True,
        return_document_ids=True,
        dtype=torch.long,
        rename_fields={"positions": "position_ids"},
        exclude_fields=("ids", "texts"),
    )
    mapped = adapt_model_inputs(cast(TrainingBatch, batch))
    if allow_cross_document_attention:
        # AutoModel lacks a model-side xdoc switch; override attention inputs here.
        mapped = enable_xdoc_attention(mapped, packing_contract=packing_contract)
    if packing_contract.return_packed_sequence_metadata:
        # These offsets index gathered tokens, unlike Zephon's padded sequence offsets.
        mapped.update(build_packed_sequence_metadata(mapped["attention_document_ids"]))
    return mapped


class ZephonDataLoader(Iterator[TrainingBatch]):
    """One owned stream; iteration never reconstructs the work source.

    ``use_distributed_checkpointing`` selects BaseRecipe's all-rank auxiliary
    state path. Deliberately does not subclass StatefulDataLoader, whose route
    checkpoints only data-owning ranks.

    The original SampleBatch keeps Zephon replay/accounting metadata unchanged.
    The consumer calls to_training and applies lightweight model-input adaptations.
    """

    use_distributed_checkpointing = True
    synchronized_stream = True

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        convert_batch: Callable[[SampleBatch], TrainingBatch],
        output_format: BatchOutputFormat = "batched",
        packing_contract: PackedSequenceContract,
    ) -> None:
        self._pipeline = pipeline
        self._convert_batch = convert_batch
        self._output_format = output_format
        self._packing_contract = packing_contract
        self._component_tokens: Counter[int] = Counter()
        self._iterator: Generator[SampleBatch, None, None] | None = None
        self._closed = False
        self._exhausted = False

    def __iter__(self) -> ZephonDataLoader:
        return self

    def __next__(self) -> TrainingBatch:
        if self._closed:
            raise RuntimeError("ZephonDataLoader is closed")
        if self._exhausted:
            raise StopIteration
        try:
            if self._iterator is None:
                self._iterator = iter(self._pipeline)
            sample_batch = next(self._iterator)
            # Convert with Zephon, apply xdoc policy, then select batched or THD output.
            # TODO(zephon): consider pipeline operations on to_training() results.
            mapped = convert_batch_output(self._convert_batch(sample_batch), output_format=self._output_format)
            # Both layouts retain the selected attention/MTP segmentation. Original
            # document_ids survive only when explicitly requested by the model.
            mapped["_packed_seq_ids"] = mapped.pop("attention_document_ids")
            if not self._packing_contract.return_document_ids:
                mapped.pop("document_ids")
            for record in sample_batch.records:
                self._component_tokens.update(record.meta.component_token_counts or {})
            # THD consumes lengths. Within batched output, only direct-ID consumers
            # expect IDs in attention_mask; the other contracts leave it unset so
            # shared model-entry hooks can build dense masks or Flash metadata.
            if self._output_format == "batched" and self._packing_contract.packed_mask_type == "document_ids":
                mapped["attention_mask"] = mapped["_packed_seq_ids"]
            return mapped
        except StopIteration:
            # Zephon retains its terminal state; the recipe checkpoints before close().
            self._exhausted = True
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release owned workers at completion, failure, or an explicit training cap."""
        if not self._closed and self._component_tokens:
            total = self._component_tokens.total()
            logger.info(
                "Zephon emitted token shares since iteration start (component IDs follow source order): %s",
                {key: count / total for key, count in self._component_tokens.items()},
            )
        self._closed = True
        iterator, self._iterator = self._iterator, None
        try:
            if iterator is not None:
                iterator.close()
        finally:
            # Older Zephon exposes iterator cleanup only; fresh DCP templates
            # do not start a runtime. Also release checkpoint-only runtimes
            # when the optional public pipeline cleanup API is available.
            close_pipeline = getattr(self._pipeline, "close", None)
            if close_pipeline is not None:
                close_pipeline()

    def state_dict_template(self) -> dict[str, bytes]:
        """Describe DCP keys without starting a fresh pipeline or calibrating it."""
        return {"zephon": b""}

    def state_dict(self) -> dict[str, bytes]:
        """Collect the complete aggregated state on every rank as one DCP value."""
        return {"zephon": pickle.dumps(self._pipeline.checkpoint())}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore trusted training checkpoint bytes before consuming the stream."""
        if self._iterator is not None:
            raise RuntimeError("Restore Zephon before beginning iteration")
        checkpoint = state.get("zephon")
        if not isinstance(checkpoint, bytes) or not checkpoint:
            raise ValueError("Zephon checkpoint requires nonempty 'zephon' bytes")
        try:
            # Only load trusted training checkpoints, as in the reference integration.
            decoded = pickle.loads(checkpoint)  # noqa: S301
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ValueError) as exc:
            raise ValueError("Zephon checkpoint contains corrupt pickle") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Zephon checkpoint payload must be a mapping")
        self._pipeline.restore(decoded)
        self._component_tokens.clear()
        self._closed = False
        self._exhausted = False
