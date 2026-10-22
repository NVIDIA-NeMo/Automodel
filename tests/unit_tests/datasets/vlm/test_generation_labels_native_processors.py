# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation-template labels stay aligned for processors that expand media themselves.

``ProcessorMixin.apply_chat_template`` shifts ``{% generation %}`` spans by the
``text_replacement_offsets`` a processor returns. Model-owned processors that
expand placeholders in their own ``__call__`` must return those offsets, or the
assistant mask lands on image tokens instead of the answer.
"""

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.models.muse_spark.image_processing import MuseSparkImageProcessor
from nemo_automodel.components.models.muse_spark.processing import MuseSparkProcessor
from nemo_automodel.components.models.step3p7.processing_step3 import Step3VLProcessor

_WORDS = ["<unk>", "<pad>", "<user>", "<assistant>", "<turn_end>", "Describe", "the", "answer", "image"]
_MUSE_SPECIAL = ["<|patch|>", "<|image_start|>", "<|image_end|>"]
_STEP3_SPECIAL = ["<im_patch>", "<patch_start>", "<patch_end>", "<patch_newline>", "<im_start>", "<im_end>"]
_ANSWER_TOKENS = ["the", "answer", "<turn_end>"]


def _tokenizer(special: list[str], template: str) -> PreTrainedTokenizerFast:
    vocabulary = {token: index for index, token in enumerate(_WORDS + special)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<turn_end>",
        additional_special_tokens=special,
    )
    tokenizer.chat_template = template
    return tokenizer


def _template(image_placeholder: str) -> str:
    return (
        "{% for m in messages %}"
        "{% if m.role == 'assistant' %}<assistant> {% generation %}"
        "{% for p in m.content %}{{ p.text }}{% endfor %} <turn_end>{% endgeneration %}\n"
        "{% else %}<user> {% for p in m.content %}"
        "{% if p.type == 'image' %}" + image_placeholder + " {% else %}{{ p.text }} {% endif %}"
        "{% endfor %}\n{% endif %}"
        "{% endfor %}"
    )


@pytest.fixture(params=["muse_spark", "step3"])
def processor(request, monkeypatch):
    if request.param == "muse_spark":
        template = _template("<|patch|>")
        return MuseSparkProcessor(
            image_processor=MuseSparkImageProcessor(
                patch_size=2, patch_temporal=2, downsample_factor=2, min_image_tokens=1, max_image_tokens=16
            ),
            tokenizer=_tokenizer(_MUSE_SPECIAL, template),
            chat_template=template,
            image_token="<|patch|>",
        )
    monkeypatch.setattr(ProcessorMixin, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    template = _template("<im_patch>")
    return Step3VLProcessor(tokenizer=_tokenizer(_STEP3_SPECIAL, template), chat_template=template)


def _conversation(num_images: int) -> list[dict]:
    images = [{"type": "image", "image": Image.new("RGB", (12, 8), (index, 0, 0))} for index in range(num_images)]
    return [
        {"role": "user", "content": images + [{"type": "text", "text": "Describe the image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
    ]


def _supervised_tokens(processor, labels: torch.Tensor) -> list[str]:
    return processor.tokenizer.convert_ids_to_tokens(labels[labels != -100].tolist())


@pytest.mark.parametrize("num_images", [0, 1, 2])
def test_default_collate_supervises_answer_not_image_tokens(processor, num_images):
    examples = [{"conversation": _conversation(num_images)}]
    batch = default_collate_fn(examples, processor)
    assert "assistant_masks" not in batch
    assert (batch["input_ids"] == processor.image_token_id).any() == (num_images > 0)
    assert _supervised_tokens(processor, batch["labels"]) == _ANSWER_TOKENS


@pytest.mark.parametrize("num_images", [0, 1, 2])
def test_pretokenization_supervises_answer_not_image_tokens(processor, num_images):
    dataset = PreTokenizedDatasetWrapper(
        [{"conversation": _conversation(num_images)}], processor, inject_fake_images=False, max_retries=1
    )
    sample = dataset[0]
    image_positions = sample["input_ids"] == processor.image_token_id
    assert image_positions.any() == (num_images > 0)
    assert sample["labels"][image_positions].eq(-100).all()
    assert _supervised_tokens(processor, sample["labels"]) == _ANSWER_TOKENS
    assert sample["labels"][-1] == processor.tokenizer.eos_token_id


def test_batched_samples_with_unequal_image_counts(processor):
    examples = [{"conversation": _conversation(2)}, {"conversation": _conversation(0)}]
    batch = default_collate_fn(examples, processor)
    image_counts = (batch["input_ids"] == processor.image_token_id).sum(dim=1).tolist()
    assert image_counts[0] > 0 and image_counts[1] == 0
    for row in batch["labels"]:
        assert _supervised_tokens(processor, row) == _ANSWER_TOKENS
