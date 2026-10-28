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
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT, PackedSequenceContract
from nemo_automodel.components.datasets.vlm.loader import (
    VlmDataloaderBuild,
    VlmProcessorConfig,
    apply_vlm_chat_template,
)
from nemo_automodel.components.datasets.zephon.config import (
    ZephonLoaderConfig,
    resolve_zephon_recipe_settings,
    validate_zephon_dataset_fields,
    zephon_options_from_recipe,
)

if TYPE_CHECKING:
    from nemo_automodel.components.datasets.zephon.vlm import ImagePositionState

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ZephonVlmSourceConfig:
    """Raw AutoModel conversations containing image paths and assistant answers."""

    chat_template: str | None = None
    inject_fake_images: bool = True
    on_media_error: Literal["raise", "drop"] = "raise"

    def __post_init__(self) -> None:
        if self.on_media_error not in ("raise", "drop"):
            raise ValueError("on_media_error must be 'raise' or 'drop'")


@dataclass(frozen=True, kw_only=True)
class ZephonVlmDataloaderConfig(ZephonLoaderConfig):
    """Own image encoding, whole-document packing and shared loader construction."""

    dataset_config: ZephonVlmSourceConfig = field(default_factory=ZephonVlmSourceConfig)
    processor_config: VlmProcessorConfig = field(default_factory=VlmProcessorConfig)

    @classmethod
    def from_recipe(
        cls,
        dataset_target: object,
        dataset_values: dict[str, Any],
        loader_values: dict[str, Any],
        *,
        processor_config: VlmProcessorConfig,
        seed: int,
        packing: dict[str, Any],
    ) -> ZephonVlmDataloaderConfig:
        """Resolve raw image conversations at the Zephon integration boundary."""
        if packing.get("pack_size", 0):
            raise ValueError("Native VLM packing is incompatible with Zephon; configure dataloader.zephon")
        if dataset_target not in (
            ZephonVlmSourceConfig,
            "nemo_automodel.components.datasets.zephon.ZephonVlmSourceConfig",
        ):
            raise ValueError(
                "Zephon VLM requires dataset._target_=nemo_automodel.components.datasets.zephon.ZephonVlmSourceConfig"
            )
        values = resolve_zephon_recipe_settings(loader_values)
        validate_zephon_dataset_fields(dataset_values, config_type=ZephonVlmSourceConfig)
        return cls(
            dataset_config=ZephonVlmSourceConfig(**dataset_values),
            processor_config=processor_config,
            output_format=loader_values.get("output_format", "batched"),
            zephon=zephon_options_from_recipe(values, seed=seed),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.output_format == "thd" and self.dataset_config.inject_fake_images:
            raise ValueError("THD output requires inject_fake_images=false; excluded image slots need batched output")
        if self.zephon.packing.pack_algorithm not in ("first_fit", "best_fit"):
            raise ValueError("Image VLM requires pack_algorithm=first_fit or best_fit; images cannot be split")
        if self.processor_config.video_processor is not None:
            raise ValueError("The Zephon image VLM path does not support video processors")
        if self.zephon.source.token_estimation not in (None, False):
            raise ValueError("Zephon VLM token-aware source mixing requires a media-aware estimator; use LLM mixing")
        if len(self.zephon.sources) > 1 and self.zephon.source.token_estimation is None:
            raise ValueError(
                "Set token_estimation=false for VLM source mixing; media-aware calibration is not certified"
            )

    def build(
        self,
        *,
        pretrained_model_name_or_path: str,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        position_state_factory: Callable[[], ImagePositionState] | None = None,
        get_rope_index: Callable[..., object] | None = None,
        packing_contract: PackedSequenceContract = DEFAULT_PACKED_SEQUENCE_CONTRACT,
        dataset_build_context: AbstractContextManager[object] | None = None,
        ignore_index: int = -100,
        pp_n_microbatches: int | None = None,
        cp_size: int = 1,
    ) -> VlmDataloaderBuild:
        """Build the processor and an unsized checkpointable image training stream.

        Runtime model position state contains configuration only. Packing measures
        fully expanded processor tokens, keeps images/documents whole and leaves
        the existing next-token shift to Zephon's final converter.

        ``position_state_factory`` is invoked once in the parent; only its result
        enters workers.
        Native ``get_rope_index`` is not used or captured by Zephon workers.
        The rank-ordering context encloses processor loading only; all pipeline
        initialization, validation and checkpoint activity happens after it exits.
        """
        del get_rope_index
        if self.output_format == "thd" and pp_n_microbatches is not None:
            raise ValueError("THD VLM pipeline media/mRoPE chunking is unsupported; use batched output for VLM PP")
        self.validate_build(batch_size=batch_size, cp_size=cp_size, packing_contract=packing_contract)
        if pp_n_microbatches is not None and (pp_n_microbatches < 1 or batch_size % pp_n_microbatches):
            raise ValueError("VLM pipeline microbatches must evenly divide the local batch size")
        if position_state_factory is None:
            raise ValueError("Zephon VLM requires a parent position_state_factory")
        positions = position_state_factory()
        from nemo_automodel.components.datasets.zephon.llm import resolve_token_ids
        from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader
        from nemo_automodel.components.datasets.zephon.pipeline import create_source_pipeline, finalize_pipeline
        from nemo_automodel.components.datasets.zephon.vlm import (
            VlmEncoderConfig,
            convert_image_batch,
            merge_image_documents,
            prepare_fake_image_document,
        )

        with dataset_build_context or nullcontext():
            processor = self.processor_config.build(pretrained_model_name_or_path=pretrained_model_name_or_path)
            if processor is None:
                raise ValueError(f"Cannot build VLM processor from {pretrained_model_name_or_path!r}")
            apply_vlm_chat_template(processor, self.dataset_config.chat_template)
            ids = resolve_token_ids(processor.tokenizer, pad_token_id=self.zephon.packing.pad_token_id)
            fake_document = (
                prepare_fake_image_document(processor, positions) if self.dataset_config.inject_fake_images else None
            )
        options = self.zephon
        fake_length = 0 if fake_document is None else len(fake_document["input_ids"])
        pack_length = options.packing.seq_len + 1 - fake_length
        if pack_length < 1:
            raise ValueError(
                f"seq_len + 1 must exceed the reserved fake-image segment length ({fake_length}); "
                f"got seq_len={options.packing.seq_len}"
            )
        max_documents = options.packing.max_num_documents
        if max_documents is not None and options.packing.max_num_documents_scope == "batch":
            max_documents //= batch_size
            if not max_documents:
                raise ValueError("max_num_documents must be at least local_batch_size when scoped to the batch")
        operator = VlmEncoderConfig(
            processor=self.processor_config,
            inject_fake_images=False,
            on_media_error=self.dataset_config.on_media_error,
            parallelism=options.workers.tokenize_parallelism or 1,
            chat_template=self.dataset_config.chat_template,
        ).build(pretrained_model_name_or_path=pretrained_model_name_or_path, positions=positions)
        pipeline = create_source_pipeline(options, dp_rank=dp_rank, dp_world_size=dp_world_size)
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
        finalize_pipeline(pipeline, options, batch_size=batch_size, dp_rank=dp_rank, dp_world_size=dp_world_size)
        loader = ZephonDataLoader(
            pipeline=pipeline,
            packing_contract=packing_contract,
            convert_batch=partial(
                convert_image_batch,
                output_format=self.output_format,
                sequence_length=options.packing.seq_len,
                pad_token_id=ids.pad,
                packing_contract=packing_contract,
                ignore_index=ignore_index,
                mask_segment_starts=options.training.mask_segment_starts,
                allow_cross_document_attention=options.training.allow_cross_document_attention,
                pp_n_microbatches=pp_n_microbatches,
            ),
            output_format=self.output_format,
        )
        logger.info(
            "Zephon stages: image conversations -> processor -> pack_sequences(%s, %s; fake reserve=%s) -> batch(%s)",
            options.packing.pack_algorithm,
            pack_length,
            fake_length,
            batch_size,
        )
        return VlmDataloaderBuild(dataloader=loader, processor=processor)
