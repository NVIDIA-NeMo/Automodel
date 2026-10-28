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

"""Local Qwen3-VL processor and tiny model retaining two DeepStack injections."""

import pytest
from transformers import Qwen3VLConfig, Qwen3VLProcessor

from tests.utils.vlm_fixtures import processor as qwen2_processor

__all__ = ["processor", "qwen2_processor", "tiny_qwen3_vl_config"]


@pytest.fixture
def processor(qwen2_processor):
    return Qwen3VLProcessor(
        tokenizer=qwen2_processor.tokenizer,
        image_processor=qwen2_processor.image_processor,
        video_processor=qwen2_processor.video_processor,
        chat_template=qwen2_processor.chat_template,
    )


def tiny_qwen3_vl_config():
    return Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=9,
        video_token_id=10,
        vision_start_token_id=11,
        vision_end_token_id=12,
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "pad_token_id": 1,
            "eos_token_id": 4,
            "bos_token_id": 2,
            "use_cache": False,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0, 1],
        },
    )
