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

import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .diffusion.collate_fns import TextToImageDataloaderConfig, TextToVideoDataloaderConfig
    from .diffusion.meta_files_dataset import MetaFilesDataloaderConfig
    from .diffusion.mock_dataloader import MockWanDataloaderConfig
    from .dllm.collate import DLLMCollator
    from .dllm.corruption import (
        corrupt_all_masked,
        corrupt_blockwise,
        corrupt_mix,
        corrupt_uniform,
        corrupt_uniform_random,
    )
    from .llm.agent_chat import make_agent_chat_eval_samples
    from .llm.chat_dataset import load_openai_messages
    from .llm.dspark_cache import (
        DTYPE_MAP as DSPARK_DTYPE_MAP,
    )
    from .llm.dspark_cache import (
        build_cache_manifest,
        build_cached_dspark_dataloader,
        compute_batch_cache,
        manifest_mismatch_fields,
        read_target_weight_modules,
        tokenizer_chat_template_sha256,
        write_target_weights,
    )
    from .llm.dspark_cache import (
        existing_shard_indices as dspark_existing_shard_indices,
    )
    from .llm.dspark_cache import (
        manifest_path as dspark_manifest_path,
    )
    from .llm.dspark_cache import (
        read_manifest as dspark_read_manifest,
    )
    from .llm.dspark_cache import (
        write_manifest as dspark_write_manifest,
    )
    from .llm.dspark_cache import (
        write_shard as dspark_write_shard,
    )
    from .llm.eagle3 import (
        build_eagle3_dataloader,
        build_eagle3_token_mapping,
        load_or_build_eagle3_token_mapping,
    )
    from .llm.eagle3_cache import (
        DTYPE_MAP as EAGLE3_DTYPE_MAP,
    )
    from .llm.eagle3_cache import (
        build_cached_eagle3_dataloader,
        compress_target_probs,
        is_compressed,
        read_target_embeddings,
        write_target_embeddings,
    )
    from .llm.eagle3_cache import (
        existing_shard_indices as eagle3_existing_shard_indices,
    )
    from .llm.eagle3_cache import (
        manifest_path as eagle3_manifest_path,
    )
    from .llm.eagle3_cache import (
        read_manifest as eagle3_read_manifest,
    )
    from .llm.eagle3_cache import (
        write_manifest as eagle3_write_manifest,
    )
    from .llm.eagle3_cache import (
        write_shard as eagle3_write_shard,
    )
    from .llm.formatting_utils import has_chat_template, resolve_chat_template
    from .llm.megatron.megatron_utils import get_blend_from_list
    from .llm.megatron.sampler import MegatronSamplerConfig
    from .llm.megatron_dataset import MegatronPretrainingConfig
    from .llm.offline_cache import (
        dataloader_from_sample,
        ensure_supervision_options_match,
        resume_start_sample,
        write_cache_shards,
        write_cache_shards_distributed,
    )
    from .llm.packed_sequence import build_block_causal_additive_mask
    from .llm.retrieval_dataset import load_datasets
    from .loader import (
        DataloaderConfig,
        DatasetBuildSchedule,
        ScheduledDatasetConfig,
        make_collate_fn,
        make_dataset_config,
        make_packing_config,
    )
    from .multimodal.datasets import BagelDatasetConfig
    from .multimodal.loader import BagelDataloaderConfig
    from .utils import add_causal_masks_to_batch
    from .vlm.datasets import PreTokenizedDatasetWrapperConfig, convert_sharegpt_to_conversation
    from .vlm.dspark_collate import build_dspark_vlm_dataloader
    from .vlm.loader import VlmCollatorConfig, VlmDataloaderConfig, VlmProcessorConfig, VlmVideoProcessorConfig
    from .vlm.neat_packing_vlm import NeatPackConfig
    from .vlm.pp_media import stage_vlm_media_for_pp
    from .vlm.utils import set_image_pixel_bounds

_LAZY_ATTRS = {
    "BagelDataloaderConfig": (".multimodal.loader", "BagelDataloaderConfig"),
    "BagelDatasetConfig": (".multimodal.datasets", "BagelDatasetConfig"),
    "DLLMCollator": (".dllm.collate", "DLLMCollator"),
    "DSPARK_DTYPE_MAP": (".llm.dspark_cache", "DTYPE_MAP"),
    "DataloaderConfig": (".loader", "DataloaderConfig"),
    "DatasetBuildSchedule": (".loader", "DatasetBuildSchedule"),
    "EAGLE3_DTYPE_MAP": (".llm.eagle3_cache", "DTYPE_MAP"),
    "MegatronSamplerConfig": (".llm.megatron.sampler", "MegatronSamplerConfig"),
    "MegatronPretrainingConfig": (".llm.megatron_dataset", "MegatronPretrainingConfig"),
    "MetaFilesDataloaderConfig": (".diffusion.meta_files_dataset", "MetaFilesDataloaderConfig"),
    "MockWanDataloaderConfig": (".diffusion.mock_dataloader", "MockWanDataloaderConfig"),
    "NeatPackConfig": (".vlm.neat_packing_vlm", "NeatPackConfig"),
    "PreTokenizedDatasetWrapperConfig": (".vlm.datasets", "PreTokenizedDatasetWrapperConfig"),
    "ScheduledDatasetConfig": (".loader", "ScheduledDatasetConfig"),
    "TextToImageDataloaderConfig": (".diffusion.collate_fns", "TextToImageDataloaderConfig"),
    "TextToVideoDataloaderConfig": (".diffusion.collate_fns", "TextToVideoDataloaderConfig"),
    "VlmCollatorConfig": (".vlm.loader", "VlmCollatorConfig"),
    "VlmDataloaderConfig": (".vlm.loader", "VlmDataloaderConfig"),
    "VlmProcessorConfig": (".vlm.loader", "VlmProcessorConfig"),
    "VlmVideoProcessorConfig": (".vlm.loader", "VlmVideoProcessorConfig"),
    "add_causal_masks_to_batch": (".utils", "add_causal_masks_to_batch"),
    "build_block_causal_additive_mask": (".llm.packed_sequence", "build_block_causal_additive_mask"),
    "build_cache_manifest": (".llm.dspark_cache", "build_cache_manifest"),
    "build_cached_dspark_dataloader": (".llm.dspark_cache", "build_cached_dspark_dataloader"),
    "build_cached_eagle3_dataloader": (".llm.eagle3_cache", "build_cached_eagle3_dataloader"),
    "build_dspark_vlm_dataloader": (".vlm.dspark_collate", "build_dspark_vlm_dataloader"),
    "build_eagle3_dataloader": (".llm.eagle3", "build_eagle3_dataloader"),
    "build_eagle3_token_mapping": (".llm.eagle3", "build_eagle3_token_mapping"),
    "compress_target_probs": (".llm.eagle3_cache", "compress_target_probs"),
    "compute_batch_cache": (".llm.dspark_cache", "compute_batch_cache"),
    "convert_sharegpt_to_conversation": (".vlm.datasets", "convert_sharegpt_to_conversation"),
    "corrupt_all_masked": (".dllm.corruption", "corrupt_all_masked"),
    "corrupt_blockwise": (".dllm.corruption", "corrupt_blockwise"),
    "corrupt_mix": (".dllm.corruption", "corrupt_mix"),
    "corrupt_uniform": (".dllm.corruption", "corrupt_uniform"),
    "corrupt_uniform_random": (".dllm.corruption", "corrupt_uniform_random"),
    "dataloader_from_sample": (".llm.offline_cache", "dataloader_from_sample"),
    "dspark_existing_shard_indices": (".llm.dspark_cache", "existing_shard_indices"),
    "dspark_manifest_path": (".llm.dspark_cache", "manifest_path"),
    "dspark_read_manifest": (".llm.dspark_cache", "read_manifest"),
    "dspark_write_manifest": (".llm.dspark_cache", "write_manifest"),
    "dspark_write_shard": (".llm.dspark_cache", "write_shard"),
    "eagle3_existing_shard_indices": (".llm.eagle3_cache", "existing_shard_indices"),
    "eagle3_manifest_path": (".llm.eagle3_cache", "manifest_path"),
    "eagle3_read_manifest": (".llm.eagle3_cache", "read_manifest"),
    "eagle3_write_manifest": (".llm.eagle3_cache", "write_manifest"),
    "eagle3_write_shard": (".llm.eagle3_cache", "write_shard"),
    "ensure_supervision_options_match": (".llm.offline_cache", "ensure_supervision_options_match"),
    "get_blend_from_list": (".llm.megatron.megatron_utils", "get_blend_from_list"),
    "has_chat_template": (".llm.formatting_utils", "has_chat_template"),
    "is_compressed": (".llm.eagle3_cache", "is_compressed"),
    "load_datasets": (".llm.retrieval_dataset", "load_datasets"),
    "load_openai_messages": (".llm.chat_dataset", "load_openai_messages"),
    "load_or_build_eagle3_token_mapping": (".llm.eagle3", "load_or_build_eagle3_token_mapping"),
    "make_agent_chat_eval_samples": (".llm.agent_chat", "make_agent_chat_eval_samples"),
    "make_collate_fn": (".loader", "make_collate_fn"),
    "make_dataset_config": (".loader", "make_dataset_config"),
    "make_packing_config": (".loader", "make_packing_config"),
    "manifest_mismatch_fields": (".llm.dspark_cache", "manifest_mismatch_fields"),
    "read_target_embeddings": (".llm.eagle3_cache", "read_target_embeddings"),
    "read_target_weight_modules": (".llm.dspark_cache", "read_target_weight_modules"),
    "resolve_chat_template": (".llm.formatting_utils", "resolve_chat_template"),
    "resume_start_sample": (".llm.offline_cache", "resume_start_sample"),
    "set_image_pixel_bounds": (".vlm.utils", "set_image_pixel_bounds"),
    "stage_vlm_media_for_pp": (".vlm.pp_media", "stage_vlm_media_for_pp"),
    "tokenizer_chat_template_sha256": (".llm.dspark_cache", "tokenizer_chat_template_sha256"),
    "write_cache_shards": (".llm.offline_cache", "write_cache_shards"),
    "write_cache_shards_distributed": (".llm.offline_cache", "write_cache_shards_distributed"),
    "write_target_embeddings": (".llm.eagle3_cache", "write_target_embeddings"),
    "write_target_weights": (".llm.dspark_cache", "write_target_weights"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
