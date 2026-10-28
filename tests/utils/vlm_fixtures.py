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

"""Small local image processor and model fixtures; no remote assets."""

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2_5_VLConfig, Qwen2_5_VLProcessor, Qwen2VLImageProcessor
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor


@pytest.fixture
def processor():
    words = [
        "<unk>",
        "<pad>",
        "<user>",
        "<assistant>",
        "<turn_end>",
        "Describe",
        "the",
        "answer",
        "image",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<turn_end>",
        additional_special_tokens=words[:5] + words[9:],
    )
    template = (
        "{% for m in messages %}{% if m.role == 'assistant' %}<assistant> {% generation %}"
        "{% for p in m.content %}{{ p.text }}{% endfor %} <turn_end>{% endgeneration %}"
        "{% else %}<user> {% for p in m.content %}{% if p.type == 'image' %}"
        "<|vision_start|><|image_pad|><|vision_end|> {% else %}{{ p.text }} {% endif %}"
        "{% endfor %}{% endif %}{% endfor %}"
    )
    tokenizer.chat_template = template
    return Qwen2_5_VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        video_processor=Qwen2VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=1024,
        ),
    )


def tiny_vlm_config():
    return Qwen2_5_VLConfig(
        image_token_id=9,
        video_token_id=10,
        vision_start_token_id=11,
        vision_end_token_id=12,
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "pad_token_id": 1,
            "eos_token_id": 4,
            "bos_token_id": 2,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "window_size": 16,
            "fullatt_block_indexes": [0],
        },
    )


@pytest.fixture
def inputs(tmp_path, processor):
    # Bound injected fake-image cost so text-only input participates in packing.
    processor.image_processor.size["longest_edge"] = 256
    processor_path = tmp_path / "processor"
    processor.save_pretrained(processor_path)
    documents = []
    for index, sizes in enumerate([[(8, 12)], [(16, 8), (8, 8)], [], [(8, 8)], [(12, 8)], [(8, 8)]]):
        media = []
        for image_index, size in enumerate(sizes):
            path = tmp_path / f"image-{index}-{image_index}.png"
            Image.new("RGB", size, color=(20 * index, 10 * image_index, 30)).save(path)
            media.append({"type": "image", "image": str(path)})
        documents.append(
            {
                "conversation": [
                    {"role": "user", "content": media + [{"type": "text", "text": "Describe image"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
                ]
            }
        )
    return str(processor_path), documents
