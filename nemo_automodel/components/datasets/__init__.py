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

_LAZY_ATTRS = {
    "BagelDataloaderConfig": (".multimodal.loader", "BagelDataloaderConfig"),
    "BagelDatasetConfig": (".multimodal.datasets", "BagelDatasetConfig"),
    "DLLMCollator": (".dllm.collate", "DLLMCollator"),
    "DSPARK_DTYPE_MAP": (".llm.dspark_cache", "DTYPE_MAP"),
    "DataloaderConfig": (".loader", "DataloaderConfig"),
    "DatasetBuildSchedule": (".loader", "DatasetBuildSchedule"),
    "EAGLE3_DTYPE_MAP": (".llm.eagle3_cache", "DTYPE_MAP"),
    "MegatronSamplerConfig": (".llm.megatron.sampler", "MegatronSamplerConfig"),
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
    "has_chat_template": (".llm.formatting_utils", "_has_chat_template"),
    "is_compressed": (".llm.eagle3_cache", "is_compressed"),
    "load_datasets": (".llm.retrieval_dataset", "load_datasets"),
    "load_openai_messages": (".llm.chat_dataset", "_load_openai_messages"),
    "load_or_build_eagle3_token_mapping": (".llm.eagle3", "load_or_build_eagle3_token_mapping"),
    "make_agent_chat_eval_samples": (".llm.agent_chat", "make_agent_chat_eval_samples"),
    "make_collate_fn": (".loader", "make_collate_fn"),
    "make_dataset_config": (".loader", "make_dataset_config"),
    "make_packing_config": (".loader", "make_packing_config"),
    "manifest_mismatch_fields": (".llm.dspark_cache", "manifest_mismatch_fields"),
    "read_target_embeddings": (".llm.eagle3_cache", "read_target_embeddings"),
    "read_target_weight_modules": (".llm.dspark_cache", "read_target_weight_modules"),
    "resolve_chat_template": (".llm.formatting_utils", "_resolve_chat_template"),
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
