# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Image-document encoding and field callbacks for Zephon's final converter.

Field callbacks assemble image tensors and model positions for whole-document
packs. ``SampleBatch.to_training`` owns padding, token/label tensor construction
and the next-token shift; attention policy is applied after collation.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NotRequired, cast

import torch

from nemo_automodel.components.datasets.packing import (
    DEFAULT_PACKED_SEQUENCE_CONTRACT,
    BatchOutputFormat,
    PackedSequenceContract,
)
from nemo_automodel.components.datasets.vlm.datasets import (
    VlmMediaDecodeError,
    VlmMediaTokenMismatchError,
    encode_vlm_document,
)
from nemo_automodel.components.datasets.vlm.fake_image import inject_fake_image_into_conversation
from nemo_automodel.components.datasets.vlm.loader import (
    ImagePositionState,
    VlmProcessorConfig,
    apply_vlm_chat_template,
)
from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp
from nemo_automodel.components.datasets.vlm.preprocessing import (
    collate_position_ids,
    merge_media_values,
    merge_vlm_documents,
)
from nemo_automodel.components.datasets.zephon.attention import (
    TrainingBatch,
    adapt_model_inputs,
    enable_xdoc_attention,
)
from nemo_automodel.shared.import_utils import safe_import_from
from nemo_automodel.shared.vlm_media import pad_image_tensors

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from zephon import SampleBatch
    from zephon.ops import OpContext
    from zephon.types import SampleRecord

_, BaseOp = safe_import_from("zephon.ops", "BaseOp")
_, OpTraits = safe_import_from("zephon.ops", "OpTraits")
_, SampleRecord = safe_import_from("zephon.types", "SampleRecord")
_, tombstones_for_record = safe_import_from("zephon.ops.children", "tombstones_for_record")
logger = logging.getLogger(__name__)


class VlmTrainingBatch(TrainingBatch):
    """Image training fields in addition to the text/attention batch contract.

    Pixels retain processor layouts [patches, features], [images, patches,
    features], or [images, channels, height, width]. Grids are [images, 3], sizes
    [images, 2], image positions [images, patches, 2], modality IDs [batch, sequence]
    and image counts [batch]. PP media chunks retain those layouts per microbatch.
    """

    pixel_values: NotRequired[torch.Tensor]
    image_grid_thw: NotRequired[torch.Tensor]
    image_sizes: NotRequired[torch.Tensor]
    image_position_ids: NotRequired[torch.Tensor]
    mm_token_type_ids: NotRequired[torch.Tensor]
    n_images_per_sample: NotRequired[torch.Tensor]
    _vlm_pp_media_chunks: NotRequired[dict[str, list[torch.Tensor]]]


@dataclass(frozen=True)
class VlmEncoderConfig:
    """Worker-owned processor construction for raw native conversation documents."""

    processor: VlmProcessorConfig = field(default_factory=VlmProcessorConfig)
    inject_fake_images: bool = True
    parallelism: int = 1
    chat_template: str | None = None
    on_media_error: Literal["raise", "drop"] = "raise"

    def build(self, *, pretrained_model_name_or_path: str, positions: ImagePositionState) -> BaseOp:
        """Build a Zephon operator with worker-local processor state."""
        if self.parallelism < 1:
            raise ValueError("VLM encoder parallelism must be positive.")
        if self.on_media_error not in ("raise", "drop"):
            raise ValueError("VLM on_media_error must be 'raise' or 'drop'.")
        return _EncodeImageDocuments(self, pretrained_model_name_or_path, positions)


class _EncodeImageDocuments(BaseOp):
    def __init__(self, config: VlmEncoderConfig, processor_path: str, positions: ImagePositionState) -> None:
        super().__init__()
        self._config = config
        self._processor_path = processor_path
        self._positions = positions
        self._processor: ProcessorMixin | None = None
        self._owner_pid = os.getpid()

    def setup(self, ctx: OpContext) -> None:
        """Initialize the processor once per worker; no model enters the worker."""
        super().setup(ctx)
        # In-process runners share the trainer; only cap independent worker processes.
        if os.getpid() != self._owner_pid:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
        self._processor = self._config.processor.build(pretrained_model_name_or_path=self._processor_path)
        if self._processor is None:
            raise ValueError(f"Cannot build VLM processor from {self._processor_path!r}.")
        apply_vlm_chat_template(self._processor, self._config.chat_template)

    def teardown(self) -> None:
        """Release worker state; safe after partial setup or repeated cleanup."""
        self._processor = None

    def traits(self) -> OpTraits:
        """Encoding preserves record order; opt-in decode filtering is not indexable."""
        return OpTraits(
            indexable=self._config.on_media_error == "raise",
            preserves_cursor_order=True,
            calibration_replayable=True,
            parallelism=self._config.parallelism,
        )

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Encode records while retaining metadata and bookkeeping records.

        Args:
            elems: Raw conversation records. Media inputs are image paths/PIL
                images; records have no training tensors yet.

        Returns:
            Records with unshifted IDs/labels/eligibility [sequence], scalar
            positions [sequence] or mRoPE [position_axes, sequence], processor
            pixels/grids and optional per-image patch positions. Source
            metadata/lineage is unchanged. No input is mutated.
        """
        if self._processor is None:
            raise RuntimeError("VLM processor is initialized in BaseOp.setup; initialize the worker before encoding.")
        output = []
        dropped = []
        for record in elems:
            if record.meta.is_sentinel:
                output.append(record)
                continue
            # Reject unsupported media before decoding; these are not droppable media errors.
            for message in record.payload["conversation"]:
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") not in (None, "text", "image"):
                            raise ValueError(
                                f"Zephon image conversations support only image and text content; got {item['type']!r}"
                            )
            try:
                document = encode_vlm_document(
                    record.payload, self._processor, inject_fake_images=self._config.inject_fake_images
                )
            except (VlmMediaDecodeError, VlmMediaTokenMismatchError) as exc:
                if self._config.on_media_error == "raise":
                    raise
                dropped.append((record.meta.sample_id, str(exc)))
                output.extend(tombstones_for_record(record))
                continue
            if "pixel_values_videos" in document or "video_grid_thw" in document:
                raise ValueError("The image VLM Zephon adapter does not support video documents.")
            token_types = document.get("mm_token_type_ids", torch.zeros_like(document["input_ids"]))
            document["position_ids"] = self._positions.positions(
                document["input_ids"],
                mm_token_type_ids=token_types.unsqueeze(0),
                image_grid_thw=document.get("image_grid_thw"),
                attention_mask=document["attention_mask"],
            )
            document["mm_token_type_ids"] = token_types
            output.append(SampleRecord(payload=document, meta=record.meta))
        if dropped:
            # Bound diagnostics per invocation; no worker-local counters affect replay.
            logger.warning(
                "Dropped %d VLM media documents; sample IDs/reasons (up to 8): %s", len(dropped), dropped[:8]
            )
        return output


def prepare_fake_image_document(processor: ProcessorMixin, positions: ImagePositionState) -> dict[str, object]:
    """Encode one attention-excluded leading segment for all-text packs.

    Args:
        processor: Effective parent-owned processor; its own image minimums
            determine the expanded image-token count.
        positions: Configuration-only model coordinate construction.

    Returns:
        One unshifted document with token fields [sequence], scalar/multi-axis
        positions [sequence] or [position_axes, sequence], and processor-shaped
        image media. All labels are -100 and all attention eligibility is false.
        The image and its coordinates remain intact; no training shift is applied.
    """
    document = encode_vlm_document(
        {"conversation": inject_fake_image_into_conversation([{"role": "user", "content": []}])},
        processor,
        inject_fake_images=False,
    )
    token_types = document.get("mm_token_type_ids", torch.zeros_like(document["input_ids"]))
    document["position_ids"] = positions.positions(
        document["input_ids"],
        mm_token_type_ids=token_types.unsqueeze(0),
        image_grid_thw=document.get("image_grid_thw"),
        attention_mask=document["attention_mask"],
    )
    document["mm_token_type_ids"] = token_types
    document["attention_mask"] = torch.zeros_like(document["attention_mask"])
    document["labels"] = torch.full_like(document["labels"], -100)
    return dict(document)


def merge_image_documents(
    documents: list[dict[str, object]], *, fake_document: dict[str, object] | None = None
) -> dict[str, object]:
    """Assemble complete documents chosen by ``pack_sequences`` without shifting.

    Args:
        documents: Encoded documents: IDs, labels and eligibility [sequence],
            scalar/multi-axis positions [sequence] or [position_axes, sequence],
            pixels [patches, features], [images, patches, features], or
            [images, channels, height, width]; grids [images, 3]; optional image
            positions [images, patches, 2], sizes [images, 2], and token types
            [sequence]. No document is mutated.
        fake_document: Optional pre-encoded, entirely excluded document with the
            same layouts. Prepend it only when the pack has no real image.
            Packing capacity must already reserve its complete token length.

    Returns:
        Unshifted flat token/mask/scalar-boundary tensors [tokens], model positions
        [tokens] or [position_axes, tokens], pixels/grids/image positions concatenated
        in document order, and independent processor eligibility [tokens].
        The synthetic-to-first-real transition is never supervised. Other real
        document starts retain their existing loss policy. This is a payload, not a batch.
    """
    has_image = any(
        document.get("pixel_values") is not None and document["pixel_values"].shape[0] for document in documents
    )
    fake_length = 0
    if fake_document is not None and not has_image:
        fake_length = len(fake_document["input_ids"])
        documents = [fake_document, *documents]
    # Zephon constructs shifted targets and document IDs from loss masks and positions.
    merged = merge_vlm_documents(
        documents,
        padding_idx=0,
        has_mrope=documents[0]["position_ids"].ndim == 2,
        return_labels=False,
        return_attention_mask=False,
    )
    loss_mask = torch.cat([document["labels"].ne(-100) for document in documents])
    if fake_length:
        # Prepending synthetic context must not add a target that the normal shift discards.
        loss_mask[fake_length] = False
    # Scalar resets are packing boundaries; mRoPE coordinates are never boundaries.
    return {
        "input_ids": merged["input_ids"],
        "loss_mask": loss_mask,
        "positions": torch.cat([torch.arange(length) for length in merged["seq_lens"]]),
        "model_position_ids": merged["position_ids"],
        "mm_token_type_ids": merged["mm_token_type_ids"],
        "pixel_values": merged["pixel_values"],
        "image_grid_thw": merged["image_grid_thw"],
        "image_position_ids": merged["image_position_ids"],
        "image_sizes": merged.get("image_sizes"),
        "n_images_per_sample": merged["n_images"],
        "processor_attention_mask": torch.cat([d["attention_mask"] for d in documents]).bool(),
    }


def image_pixels(values: Sequence[torch.Tensor | None], *, length: int, shifted: bool) -> torch.Tensor | None:
    """Concatenate pixel values for ``to_training(field_collators=...)``.

    Args:
        values: Optional per-document pixels [patches, features] or
            [images, patches, features] or [images, channels, height, width].
        length: Resolved unshifted padded token length; media is not token aligned.
        shifted: Whether tokens are shifted; media is never shifted.

    Returns:
        Processor-precision pixels concatenated on the leading patch/image axis, in document
        order. Spatial image tensors are zero-padded to the maximum height/width.
        When every document is text-only, return None.
    """
    present = [value for value in values if value is not None and value.shape[0]]
    if present and all(value.ndim == 4 for value in present):
        return pad_image_tensors(present)
    return merge_media_values(present) if present else None


def image_grids(values: Sequence[torch.Tensor | None], *, length: int, shifted: bool) -> torch.Tensor | None:
    """Concatenate grid fields without token slicing.

    Args:
        values: Optional image grids [images, 3], image positions [images, patches, 2],
            or original image sizes [images, 2].
        length: Unshifted padded token length; ignored for media grids.
        shifted: Token shift flag; ignored for media grids.

    Returns:
        Concatenated grids/positions with their trailing axes unchanged, preserving
        processor/image order; None when no image metadata is present.
    """
    present = [value for value in values if value is not None and value.shape[0]]
    return torch.cat(present, dim=0) if present else None


def image_counts(values: Sequence[int], *, length: int, shifted: bool) -> torch.Tensor:
    """Keep media ownership for native pipeline microbatch splitting.

    Args:
        values: Image counts for each complete packed document.
        length: Unshifted padded token length; ignored for media counts.
        shifted: Token shift flag; ignored for media counts.

    Returns:
        Integer image counts [batch] in the same order as token documents.
    """
    return torch.tensor(values, dtype=torch.long)


def model_positions(values: Sequence[torch.Tensor], *, length: int, shifted: bool) -> torch.Tensor:
    """Collate processor/model positions on the input side of the existing shift.

    Args:
        values: Unshifted positions [sequence] or [position_axes, sequence] for each document.
        length: Unshifted padded token length.
        shifted: Whether to remove the final input position with the final input token.

    Returns:
        Positions [batch, length - int(shifted)] or [position_axes, batch, length - int(shifted)]; padding is zero.
        Original mRoPE values are unchanged and input tensors are not modified.
    """
    positions = collate_position_ids(values, sequence_length=length)
    return positions[..., :-1] if shifted else positions


def _validate_flash_image_eligibility(positions: torch.Tensor, eligible: torch.Tensor) -> None:
    """Reject processor holes that contiguous Flash segments cannot represent.

    Args:
        positions: Scalar reset positions [batch, sequence] defining subdocuments.
        eligible: Boolean [batch, sequence], true for real attention-eligible tokens.
            Excluded prefixes/suffixes are representable; a gap between eligible
            tokens of the same subdocument would silently sever its attention.
    """
    document_ids = positions.eq(0).cumsum(-1)
    preceding = torch.where(eligible, document_ids, 0).cummax(-1).values
    following = torch.where(eligible, document_ids, document_ids[:, -1:] + 1).flip(-1).cummin(-1).values.flip(-1)
    holes = ~eligible & preceding.eq(document_ids) & following.eq(document_ids)
    if bool(holes.any()):
        raise ValueError(
            "HF Flash VLM packing cannot preserve attention across processor-excluded tokens inside a document. "
            "Use SDPA/eager attention. Disabling fake images is an alternative only when the native model "
            "and parallel setup do not require them (dataset.inject_fake_images: false)."
        )


def convert_image_batch(
    sample_batch: SampleBatch,
    *,
    sequence_length: int,
    pad_token_id: int,
    packing_contract: PackedSequenceContract,
    ignore_index: int = -100,
    mask_segment_starts: bool = False,
    allow_cross_document_attention: bool = False,
    pp_n_microbatches: int | None = None,
    output_format: BatchOutputFormat = "batched",
) -> VlmTrainingBatch:
    """Use Zephon's final collator, then map image inputs to AutoModel.

    Args:
        sample_batch: Zephon envelopes containing merged documents: unshifted token
            fields [tokens], mRoPE [3, tokens], pixels [patches, features] and
            grids [images, 3]. to_training selects each merged mapping directly.
        sequence_length: Shifted model sequence length.
        pad_token_id: Existing effective processor tokenizer vocabulary ID.
        packing_contract: Model return requests; mask/gather requests are batched-only.
        ignore_index: Loss exclusion value for shifted labels.
        mask_segment_starts: Zephon's existing cross-boundary label policy.
        allow_cross_document_attention: Join attention and translate each
            subdocument's model coordinates by one common offset on all axes.
        pp_n_microbatches: Split media along the native PP microbatch ownership
            after conversion. None leaves media concatenated for the ordinary forward.
        output_format: Batched mask/metadata route or native THD lengths.

    Returns:
        IDs/labels/padding/boundaries/modality IDs [batch, sequence_length], scalar
        positions [batch, sequence_length] or original mRoPE [3, batch, sequence_length],
        and document_ids/attention_document_ids [batch, sequence]. Original membership
        comes from Zephon; only attention IDs exclude processor-ineligible slots.
        The configured model entry
        hook materializes the selected attention mask/metadata on-device. Ordinary batches
        retain concatenated pixels/grids/image positions. PP batches instead carry
        native `_vlm_pp_media_chunks`, lists of media tensors per microbatch.
        Empty image fields are omitted. No inputs are mutated; Zephon owns all
        token/label shifting and padding.
    """
    # THD ignores batched mask/gather requests; its output map derives lengths.
    if output_format == "thd":
        packing_contract = DEFAULT_PACKED_SEQUENCE_CONTRACT
    batch = sample_batch.to_training(
        tokens_field="input_ids",
        dtype=torch.long,
        return_labels=True,
        return_padding_mask=True,
        return_document_ids=True,
        pad_to_length=sequence_length + 1,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        mask_segment_starts=mask_segment_starts,
        rename_fields={"positions": "position_ids"},
        extra_fields=("mm_token_type_ids", "processor_attention_mask"),
        field_collators={
            "pixel_values": image_pixels,
            "image_grid_thw": image_grids,
            "image_position_ids": image_grids,
            "image_sizes": image_grids,
            "n_images_per_sample": image_counts,
            "model_position_ids": model_positions,
        },
        exclude_fields=("ids", "texts"),
    )
    for name in ("pixel_values", "image_grid_thw", "image_position_ids", "image_sizes"):
        if batch[name] is None:
            del batch[name]
    positions = batch.pop("model_position_ids")
    eligible = batch.pop("processor_attention_mask").bool()
    if packing_contract.packed_mask_type == "flash_varlen":
        _validate_flash_image_eligibility(batch["position_ids"], eligible & ~batch["padding_mask"])
    mapped = adapt_model_inputs(cast(TrainingBatch, batch), attention_eligibility=eligible)
    mapped["position_ids"] = positions
    if allow_cross_document_attention:
        # AutoModel lacks a model-side xdoc switch; override attention inputs here.
        mapped = enable_xdoc_attention(mapped, packing_contract=packing_contract)
    if packing_contract.return_packed_sequence_metadata:
        from nemo_automodel.components.datasets.packing import build_packed_sequence_metadata

        mapped.update(build_packed_sequence_metadata(mapped["attention_document_ids"]))
    if pp_n_microbatches is not None:
        prepare_vlm_media_for_pp(mapped, batch_size=mapped["input_ids"].shape[0], n_microbatches=pp_n_microbatches)
    else:
        mapped.pop("n_images_per_sample", None)
    return cast(VlmTrainingBatch, mapped)
