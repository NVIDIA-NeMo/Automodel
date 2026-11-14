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

"""Offline processor and small dense Gemma4 fixtures with the 31B architecture."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    Gemma4AudioFeatureExtractor,
    Gemma4Config,
    Gemma4ImageProcessor,
    Gemma4Processor,
    PreTrainedTokenizerFast,
)
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor


def gemma4_processor() -> Gemma4Processor:
    """Build the real image processor with a small local tokenizer."""
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
        "<image>",
        "<boi>",
        "<eoi>",
        "<audio>",
        "<boa>",
        "<eoa>",
    ]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<turn_end>",
        additional_special_tokens=words[:5] + words[9:],
        extra_special_tokens={
            "image_token": "<image>",
            "boi_token": "<boi>",
            "eoi_token": "<eoi>",
            "audio_token": "<audio>",
            "boa_token": "<boa>",
            "eoa_token": "<eoa>",
        },
    )
    template = (
        "{% for m in messages %}{% if m.role == 'assistant' %}<assistant> {% generation %}"
        "{% for p in m.content %}{{ p.text }}{% endfor %} <turn_end>{% endgeneration %}"
        "{% else %}<user> {% for p in m.content %}{% if p.type == 'image' %}<image> "
        "{% else %}{{ p.text }} {% endif %}{% endfor %}{% endif %}{% endfor %}"
    )
    tokenizer.chat_template = template
    return Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(),
        video_processor=Gemma4VideoProcessor(),
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Gemma4ImageProcessor(patch_size=2, pooling_kernel_size=3, max_soft_tokens=70),
        image_seq_length=70,
    )


def tiny_gemma4_config(*, tie_word_embeddings: bool = True) -> Gemma4Config:
    """Retain dense31B image/full/sliding structure, with explicit initial weight tying."""
    return Gemma4Config(
        dtype="float32",
        tie_word_embeddings=tie_word_embeddings,
        image_token_id=9,
        boi_token_id=10,
        eoi_token_id=11,
        vision_soft_tokens_per_image=70,
        audio_config=None,
        text_config={
            "tie_word_embeddings": tie_word_embeddings,
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "num_global_key_value_heads": 2,
            "attention_k_eq_v": True,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "enable_moe_block": False,
            "use_double_wide_mlp": False,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 8,
            "use_bidirectional_attention": "vision",
            "use_cache": False,
            "pad_token_id": 1,
            "eos_token_id": 4,
            "bos_token_id": 2,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            },
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "patch_size": 2,
            "pooling_kernel_size": 3,
            "position_embedding_size": 128,
            "default_output_length": 70,
            "use_clipped_linears": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
        },
    )
