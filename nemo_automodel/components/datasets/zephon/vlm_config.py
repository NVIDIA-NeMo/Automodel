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

"""Declarative image VLM construction on the shared Zephon runtime."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Literal

from nemo_automodel.components.datasets.vlm.loader import (
    VlmDataloaderBuild,
    VlmStreamBuildContext,
    apply_vlm_chat_template,
)
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonStreamDatasetConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ZephonVlmOptions(ZephonOptions):
    """Image-conversation settings alongside the shared Zephon pipeline options.

    Assistant tokens are supervised using the processor's chat template. Fake
    image segments keep vision participation for text-only packed documents.
    """

    inject_fake_images: bool = True
    on_media_error: Literal["raise", "drop"] = "raise"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_media_error not in ("raise", "drop"):
            raise ValueError("on_media_error must be 'raise' or 'drop'")


@dataclass(frozen=True, kw_only=True)
class ZephonVlmDatasetConfig(ZephonStreamDatasetConfig[VlmStreamBuildContext, VlmDataloaderBuild]):
    """Build image encoding, whole-document packing and the Zephon training loader.

    The outer VLM loader supplies its processor configuration and chat template
    as runtime dependencies; no processor or model is stored on this config.
    """

    zephon: ZephonVlmOptions = field(default_factory=ZephonVlmOptions)
    options_type: ClassVar[type[ZephonOptions]] = ZephonVlmOptions

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.output_format == "thd" and self.zephon.inject_fake_images:
            raise ValueError("THD output requires inject_fake_images=false; excluded image slots need batched output")
        if self.zephon.packing.pack_algorithm not in ("first_fit", "best_fit"):
            raise ValueError("Image VLM requires pack_algorithm=first_fit or best_fit; images cannot be split")
        if self.zephon.source.token_estimation not in (None, False):
            raise ValueError("Zephon VLM token-aware source mixing requires a media-aware estimator; use LLM mixing")
        if len(self.zephon.sources) > 1 and self.zephon.source.token_estimation is None:
            raise ValueError(
                "Set token_estimation=false for VLM source mixing; media-aware calibration is not certified"
            )

    def build(self, *, context: VlmStreamBuildContext) -> VlmDataloaderBuild:
        """Build the processor and an unsized checkpointable image training stream.

        Runtime model position state contains configuration only. Packing measures
        fully expanded processor tokens, keeps images/documents whole and leaves
        the existing next-token shift to Zephon's final converter.

        ``position_state_factory`` is invoked once in the parent; only its result
        enters workers.
        Native ``get_rope_index`` is not used or captured by Zephon workers.
        The rank-ordering context encloses processor loading only; all pipeline
        initialization, validation and checkpoint activity happens after it exits.

        Args:
            context: Runtime processor, position-state, rank and training dependencies.

        Returns:
            The complete checkpointable loader and the parent processor.
        """
        if not isinstance(context, VlmStreamBuildContext):
            raise TypeError("ZephonVlmDatasetConfig requires the VLM dataloader build context")
        self.validate_context(context)
        if context.processor_config.video_processor is not None:
            raise ValueError("The Zephon image VLM path does not support video processors")
        if self.output_format == "thd" and context.pp_n_microbatches is not None:
            raise ValueError("THD VLM pipeline media/mRoPE chunking is unsupported; use batched output for VLM PP")
        if context.pp_n_microbatches is not None and (
            context.pp_n_microbatches < 1 or context.batch_size % context.pp_n_microbatches
        ):
            raise ValueError("VLM pipeline microbatches must evenly divide the local batch size")
        if context.position_state_factory is None:
            raise ValueError("Zephon VLM requires a parent position_state_factory")
        sequence_length = self.resolve_sequence_length(packing_contract=context.packing_contract)
        positions = context.position_state_factory()
        from nemo_automodel.components.datasets.zephon.llm import resolve_token_ids
        from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader
        from nemo_automodel.components.datasets.zephon.pipeline import create_source_pipeline, finalize_pipeline
        from nemo_automodel.components.datasets.zephon.vlm import (
            VlmEncoderConfig,
            convert_image_batch,
            merge_image_documents,
            prepare_fake_image_document,
        )

        with context.dataset_build_context or nullcontext():
            processor = context.processor_config.build(
                pretrained_model_name_or_path=context.pretrained_model_name_or_path
            )
            if processor is None:
                raise ValueError(f"Cannot build VLM processor from {context.pretrained_model_name_or_path!r}")
            apply_vlm_chat_template(processor, context.chat_template)
            ids = resolve_token_ids(processor.tokenizer, pad_token_id=self.zephon.packing.pad_token_id)
            fake_document = (
                prepare_fake_image_document(processor, positions) if self.zephon.inject_fake_images else None
            )
        options = self.zephon
        fake_length = 0 if fake_document is None else len(fake_document["input_ids"])
        # Expanded image tokens and the optional fake segment share the input budget.
        pack_length = sequence_length + 1 - fake_length
        if pack_length < 1:
            raise ValueError(
                f"seq_len + 1 must exceed the reserved fake-image segment length ({fake_length}); "
                f"got seq_len={sequence_length}"
            )
        max_documents = options.packing.max_num_documents
        if max_documents is not None and options.packing.max_num_documents_scope == "batch":
            max_documents //= context.batch_size
            if not max_documents:
                raise ValueError("max_num_documents must be at least local_batch_size when scoped to the batch")
        operator = VlmEncoderConfig(
            processor=context.processor_config,
            inject_fake_images=False,
            on_media_error=self.zephon.on_media_error,
            parallelism=options.workers.tokenize_parallelism or 1,
            chat_template=context.chat_template,
        ).build(pretrained_model_name_or_path=context.pretrained_model_name_or_path, positions=positions)
        pipeline = create_source_pipeline(
            options, seed=context.seed, dp_rank=context.dp_rank, dp_world_size=context.dp_world_size
        )
        pipeline.add_op(operator).pack_sequences(
            max_length=pack_length,
            algorithm=options.packing.pack_algorithm,
            num_bins=options.packing.pack_num_bins or 64,
            max_sequences_per_bin=max_documents,
            tokens_field="input_ids",
            drop_oversized=options.packing.drop_oversized,
            pack_payloads=partial(merge_image_documents, fake_document=fake_document),
            parallelism=options.workers.pack_parallelism,
        )
        finalize_pipeline(
            pipeline,
            options,
            seed=context.seed,
            batch_size=context.batch_size,
            dp_rank=context.dp_rank,
            dp_world_size=context.dp_world_size,
        )
        loader = ZephonDataLoader(
            pipeline=pipeline,
            packing_contract=context.packing_contract,
            convert_batch=partial(
                convert_image_batch,
                output_format=self.output_format,
                sequence_length=sequence_length,
                pad_token_id=ids.pad,
                packing_contract=context.packing_contract,
                ignore_index=context.ignore_index,
                mask_segment_starts=options.training.mask_segment_starts,
                allow_cross_document_attention=options.training.allow_cross_document_attention,
                pp_n_microbatches=context.pp_n_microbatches,
            ),
            output_format=self.output_format,
        )
        logger.info(
            "Zephon stages: image conversations -> processor -> pack_sequences(%s, %s; fake reserve=%s) -> batch(%s)",
            options.packing.pack_algorithm,
            pack_length,
            fake_length,
            context.batch_size,
        )
        return VlmDataloaderBuild(dataloader=loader, processor=processor)
