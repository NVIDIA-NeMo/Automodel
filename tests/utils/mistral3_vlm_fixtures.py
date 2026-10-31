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

"""Offline Mistral3 processor, model config and image corpus for VLM tests."""

import json
from pathlib import Path

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Mistral3Config, PixtralImageProcessor, PixtralProcessor, PreTrainedTokenizerFast


def build_model_config() -> Mistral3Config:
    """Keep the untied Ministral3/Pixtral architecture with small CPU-test dimensions."""
    return Mistral3Config(
        tie_word_embeddings=False,
        image_token_index=10,
        spatial_merge_size=2,
        text_config={
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "sliding_window": None,
            "tie_word_embeddings": False,
            "use_cache": False,
            "pad_token_id": 1,
            "eos_token_id": 4,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "head_dim": 8,
            "image_size": 16,
            "patch_size": 2,
            "attention_dropout": 0.0,
        },
    )


def build_processor() -> PixtralProcessor:
    """Build the real Pixtral processor with a local vocabulary and chat template."""
    words = [
        "<unk>",
        "<pad>",
        "<user>",
        "<assistant>",
        "<eos>",
        "Describe",
        "the",
        "answer",
        "image",
        "other",
        "[IMG]",
        "[IMG_BREAK]",
        "[IMG_END]",
    ]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        additional_special_tokens=words[:5] + words[10:],
    )
    template = (
        "{% for m in messages %}{% if m.role == 'assistant' %}<assistant> {% generation %}"
        "{% for p in m.content %}{{ p.text }}{% endfor %} <eos>{% endgeneration %}"
        "{% else %}<user> {% for p in m.content %}{% if p.type == 'image' %}[IMG] "
        "{% else %}{{ p.text }} {% endif %}{% endfor %}{% endif %}{% endfor %}"
    )
    tokenizer.chat_template = template
    return PixtralProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 16}, patch_size=2),
        tokenizer=tokenizer,
        patch_size=2,
        spatial_merge_size=2,
        chat_template=template,
    )


def prepare(directory: Path, *, count: int = 128) -> None:
    """Write local model/processor config and finite mixed image/text JSONL shards."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    build_model_config().save_pretrained(directory / "model")
    build_processor().save_pretrained(directory / "processor")
    images = directory / "images"
    images.mkdir(exist_ok=True)
    for name, size, color in (("square", (8, 8), "blue"), ("wide", (12, 8), "red")):
        Image.new("RGB", size, color).save(images / f"{name}.png")
    documents = []
    for index in range(count):
        names = ((), ("square",), ("square", "wide"), ("wide",))[index % 4]
        content = [{"type": "image", "image": str(images / f"{name}.png")} for name in names]
        content.append({"type": "text", "text": "Describe the image"})
        documents.append(
            {
                "conversation": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
                ]
            }
        )
    corpus = directory / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "images.jsonl").write_text("".join(json.dumps(document) + "\n" for document in documents))
