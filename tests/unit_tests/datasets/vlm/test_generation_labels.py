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

"""Training-template labels stay aligned after image expansion and packing."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.glm46v.processing_glm46v import Glm46VProcessor
from transformers.models.glm46v.video_processing_glm46v import Glm46VVideoProcessor

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample, _shift_sample
from nemo_automodel.components.models.glm5_next.image_processing import Glm5NextImageProcessor


@pytest.fixture
def processor():
    special = [
        "[UNK]",
        "[PAD]",
        "[gMASK]",
        "<sop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<think>",
        "</think>",
        "<|begin_of_image|>",
        "<|image|>",
        "<|end_of_image|>",
        "<|video|>",
        "<tool_call>",
        "</tool_call>",
    ]
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Describe this image. the answer reasoning weather sunny"],
        trainers.BpeTrainer(
            vocab_size=300, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="<|user|>"
    )
    template = (
        Path(__file__).parents[4] / "examples/llm_finetune/glm5_next/glm5_3_flash_training_chat_template.jinja"
    ).read_text()
    tokenizer.chat_template = template
    return Glm46VProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Glm5NextImageProcessor(min_image_tokens=1, max_image_tokens=64),
        video_processor=Glm46VVideoProcessor(),
    )


def _conversation(answer="the answer", *, reasoning=None, size=(56, 84)):
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    if reasoning is not None:
        assistant["reasoning_content"] = reasoning
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", size)},
                {"type": "text", "text": "Describe this image. the answer"},
            ],
        },
        assistant,
    ]


def _pretokenize(processor, conversations):
    dataset = PreTokenizedDatasetWrapper(
        [{"conversation": conversation} for conversation in conversations],
        processor,
        inject_fake_images=False,
        max_retries=1,
    )
    return [dataset[i] for i in range(len(dataset))]


@pytest.mark.parametrize("answer", ["the answer", "the answer ", "  the answer  ", "the " * 50, "", "the  answer"])
@pytest.mark.parametrize("reasoning", [None, "reasoning"])
def test_generation_targets_after_image_expansion(processor, answer, reasoning):
    conversation = _conversation(answer, reasoning=reasoning)
    sample = _pretokenize(processor, [conversation])[0]
    ids = sample["input_ids"].unsqueeze(0)
    assert (ids == processor.image_token_id).sum() == 6
    labels = sample["labels"].unsqueeze(0)
    expected = (reasoning or "") + "</think>" + answer.strip() + "<|user|>"
    assert processor.tokenizer.decode(labels[labels != -100]) == expected
    assert labels[ids == processor.image_token_id].eq(-100).all()
    assert labels[0, -1] == processor.tokenizer.convert_tokens_to_ids("<|user|>")
    assert (
        labels[0, : (ids[0] == processor.tokenizer.convert_tokens_to_ids("<think>")).nonzero()[0, 0] + 1].eq(-100).all()
    )


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padding_does_not_supervise_prompt_or_padding(processor, padding_side):
    processor.tokenizer.padding_side = padding_side
    # EOS and padding can share an id; only the actual turn terminator is supervised.
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    conversations = [_conversation(), _conversation("the " * 20, size=(84, 112))]
    batch = processor.apply_chat_template(
        conversations, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}
    )
    collated = default_collate_fn([{"conversation": conversation} for conversation in conversations], processor)
    labels = collated["labels"]
    assert torch.equal(collated["input_ids"], batch["input_ids"][:, :-1])
    assert labels[batch["attention_mask"][:, 1:] == 0].eq(-100).all()
    for row, answer in zip(labels, ["the answer", ("the " * 20).strip()]):
        assert processor.tokenizer.decode(row[row != -100]) == "</think>" + answer + "<|user|>"


@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_truncated_labels_match_full_sequence_slice(processor, truncation_side):
    processor.tokenizer.truncation_side = truncation_side
    conversations = [_conversation("the " * 30)]
    examples = [{"conversation": conversation} for conversation in conversations]
    full = default_collate_fn(examples, processor)
    max_length = full["input_ids"].shape[1] + 1 - 5
    actual = default_collate_fn(examples, processor, max_length=max_length)
    window = slice(5, None) if truncation_side == "left" else slice(None, -5)
    assert torch.equal(actual["labels"], full["labels"][:, window])
    assert actual["labels"].ne(-100).any()


def test_tool_and_multiturn_terminators(processor):
    conversations = [
        _conversation("the answer ")
        + [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "id": "c1", "function": {"name": "weather", "arguments": {}}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": "the answer", "reasoning_content": "reasoning"},
            {"role": "system", "content": "new rule"},
        ]
    ]
    batch = _pretokenize(processor, conversations)[0]
    labels = batch["labels"]
    assert processor.tokenizer.decode(labels[labels != -100]) == (
        "</think>the answer<|user|></think><tool_call>weather</tool_call><|observation|>"
        "reasoning</think>the answer<|system|>"
    )


def test_pretokenized_pack_preserves_each_document_stop_target(processor):
    examples = [{"conversation": _conversation("the answer ")}, {"conversation": _conversation("the " * 50)}]
    dataset = PreTokenizedDatasetWrapper(examples, processor, inject_fake_images=False, max_retries=1)
    samples = [dataset[i] for i in range(2)]
    for sample in samples:
        assert sample["labels"][-1] == processor.tokenizer.eos_token_id
        assert sample["labels"].ne(-100).any()
    shifted = [_shift_sample(sample) for sample in samples]
    packed = _build_packed_vlm_sample(
        shifted, pack_size=sum(len(s["input_ids"]) for s in shifted), padding_idx=processor.tokenizer.pad_token_id
    )
    boundary = len(shifted[0]["input_ids"])
    assert packed["labels"][boundary - 1] == processor.tokenizer.eos_token_id
    assert packed["labels"][-1] == processor.tokenizer.eos_token_id
    assert packed["labels"][boundary] == -100
    assert torch.equal(packed["labels"], torch.cat([sample["labels"] for sample in shifted]))


def test_default_collator_uses_same_targets(processor):
    examples = [{"conversation": _conversation("the answer ")}]
    batch = default_collate_fn(examples, processor)
    assert "assistant_masks" not in batch
    assert torch.equal(batch["labels"][0], _pretokenize(processor, [examples[0]["conversation"]])[0]["labels"][1:])
    assert processor.tokenizer.decode(batch["labels"][batch["labels"] != -100]) == "</think>the answer<|user|>"


def test_multiple_images_between_assistant_turns(processor):
    conversations = [_conversation("the answer ") + _conversation("the answer", size=(84, 112))]
    batch = _pretokenize(processor, conversations)[0]
    labels = batch["labels"]
    assert (batch["input_ids"] == processor.image_token_id).sum() == 18
    assert processor.tokenizer.decode(labels[labels != -100]) == "</think>the answer<|user|>" * 2
    assert labels[batch["input_ids"] == processor.image_token_id].eq(-100).all()


@pytest.mark.parametrize("named_template", [False, True])
def test_processor_template_takes_precedence_over_tokenizer_template(processor, named_template):
    processor.tokenizer.chat_template = "unused tokenizer template"
    if named_template:
        processor.chat_template = {"default": processor.chat_template, "other": "unused named template"}
    conversations = [_conversation()]
    batch = _pretokenize(processor, conversations)[0]
    labels = batch["labels"]
    assert processor.tokenizer.decode(labels[labels != -100]) == "</think>the answer<|user|>"
