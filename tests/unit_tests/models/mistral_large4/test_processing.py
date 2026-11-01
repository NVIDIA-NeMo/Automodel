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

"""Shared Mistral tokenizer, labels, multimodal collation and PP media ownership."""

import base64
import copy
from io import BytesIO
from pathlib import Path

import pytest
import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from PIL import Image

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.tokenization.tokenization_mistral_common import MistralCommonBackend
from nemo_automodel.components.datasets.vlm.collate_fns import build_labels_from_template, default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from tests.unit_tests.models.mistral_large4.test_model import make_model


@pytest.fixture
def tokenizer():
    return MistralCommonBackend(Path(__file__).parent / "tiny_tekken.json")


@pytest.mark.parametrize("effort", ["none", "high"])
@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "text"},
        {"type": "json_object"},
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
            },
        },
    ],
)
def test_settings_equal_vendor(tokenizer, effort, response_format):
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "What is 2+2?"}]
    reference = MistralTokenizer.from_file(Path(__file__).parent / "tiny_tekken.json")
    request = ChatCompletionRequest.from_openai(
        messages=messages, reasoning_effort=effort, response_format=response_format
    )
    expected = reference.encode_chat_completion(request)
    actual = tokenizer.apply_chat_template(messages, reasoning_effort=effort, response_format=response_format)
    assert actual["input_ids"] == expected.tokens
    assert actual["attention_mask"] == [1] * len(expected.tokens)
    assert (
        tokenizer.apply_chat_template(
            messages, reasoning_effort=effort, response_format=response_format, tokenize=False
        )
        == expected.text
    )


def test_registered_tokenizer_loads(tmp_path):
    MistralLarge4Config().save_pretrained(tmp_path)
    (tmp_path / "tekken.json").write_bytes((Path(__file__).parent / "tiny_tekken.json").read_bytes())
    actual = NeMoAutoTokenizer.from_pretrained(str(tmp_path))
    assert isinstance(actual, MistralCommonBackend)
    assert actual.apply_chat_template([{"role": "user", "content": "hi"}], reasoning_effort="high")["input_ids"]


def test_batch_padding_and_settings_differ(tokenizer):
    messages = [[{"role": "user", "content": "Hi"}], [{"role": "user", "content": "Give a longer answer"}]]
    actual = tokenizer.apply_chat_template(messages, padding=True, return_tensors="pt")
    assert actual["input_ids"].shape == actual["attention_mask"].shape
    assert actual["input_ids"].shape[0] == 2
    assert actual["attention_mask"][0].sum() < actual["attention_mask"][1].sum()
    assert (
        tokenizer.apply_chat_template(messages[0], reasoning_effort="none")["input_ids"]
        != tokenizer.apply_chat_template(messages[0], reasoning_effort="high")["input_ids"]
    )


def test_image_preprocessing_equal_vendor(tokenizer):
    buffer = BytesIO()
    Image.new("RGB", (56, 28), (20, 70, 130)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe."}, {"type": "image_url", "image_url": {"url": url}}],
        }
    ]
    reference = MistralTokenizer.from_file(Path(__file__).parent / "tiny_tekken.json")
    expected = reference.encode_chat_completion(
        ChatCompletionRequest.from_openai(messages=messages, reasoning_effort="high")
    )
    actual = tokenizer.apply_chat_template(messages, reasoning_effort="high", return_tensors="pt")
    assert actual["input_ids"].tolist()[0] == expected.tokens
    assert len(actual["pixel_values"]) == len(expected.images) > 0
    for pixels, ref in zip(actual["pixel_values"], expected.images, strict=True):
        torch.testing.assert_close(pixels, torch.from_numpy(ref), rtol=0, atol=0)


def test_tools_equal_vendor(tokenizer):
    messages = [{"role": "user", "content": "Look up Paris weather."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Weather lookup",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    reference = MistralTokenizer.from_file(Path(__file__).parent / "tiny_tekken.json")
    expected = reference.encode_chat_completion(
        ChatCompletionRequest.from_openai(messages=messages, tools=tools, reasoning_effort="none")
    )
    assert tokenizer.apply_chat_template(messages, tools=tools)["input_ids"] == expected.tokens


def test_vendor_enum_identity_and_unknown_request_option(tokenizer):
    from mistral_common.protocol.instruct.validator import ValidationMode

    assert tokenizer.mode is ValidationMode.test
    with pytest.raises(ValueError, match="unsupported_option"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], unsupported_option=True)


def test_standard_image_forms_preserve_pixels_and_messages(tokenizer, tmp_path):
    image = Image.new("RGB", (56, 28), (20, 70, 130))
    path = tmp_path / "image.png"
    image.save(path)
    expected = None
    for item in ({"image": image}, {"image": path}, {"path": str(path)}, {"url": path.as_uri()}):
        messages = [{"role": "user", "content": [{"type": "image", **item}, {"type": "text", "text": "Describe."}]}]
        result = tokenizer.apply_chat_template(messages, return_tensors="pt")
        assert messages[0]["content"][0] == {"type": "image", **item}
        assert result["image_sizes"].tolist() == [[28, 56]]
        if expected is not None:
            torch.testing.assert_close(result["input_ids"], expected["input_ids"], atol=0, rtol=0)
            torch.testing.assert_close(result["pixel_values"], expected["pixel_values"], atol=0, rtol=0)
        expected = result


def test_pretokenized_text_matches_collator():
    tokenizer = MistralCommonBackend(Path(__file__).parent / "tiny_tekken.json", mode="finetuning")
    answers = ["Four.", "Goodbye for now."]
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "What is two plus two?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": answers[0]}]},
        {"role": "user", "content": [{"type": "text", "text": "Bye."}]},
        {"role": "assistant", "content": [{"type": "text", "text": answers[1]}]},
    ]
    sample = PreTokenizedDatasetWrapper([{"conversation": conversation}], tokenizer, inject_fake_images=False)[0]
    batch = default_collate_fn([{"conversation": conversation}], processor=tokenizer)

    # The collator shifts once; pretokenized samples are shifted later by the packer.
    torch.testing.assert_close(sample["input_ids"][:-1], batch["input_ids"][0], rtol=0, atol=0)
    torch.testing.assert_close(sample["labels"][1:], batch["labels"][0], rtol=0, atol=0)
    supervised = tokenizer.decode(sample["labels"][sample["labels"] != -100].tolist())
    assert all(answer in supervised for answer in answers)
    assert "two plus two" not in supervised


def test_medpix_labels_equal_vendor_and_padding_is_ignored():
    path = Path(__file__).parent / "tiny_tekken.json"
    tokenizer = MistralCommonBackend(path, mode="finetuning", padding_side="right")
    oracle = MistralTokenizer.from_file(path, mode=ValidationMode.finetuning)
    prompt_oracle = MistralTokenizer.from_file(path)
    image = Image.new("RGB", (56, 28), (20, 70, 130))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    question = "The answer is lung. What organ is shown?"
    answer = "lung"
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}],
        },
        {"role": "assistant", "content": answer},
    ]
    prefix = prompt_oracle.encode_chat_completion(
        ChatCompletionRequest.from_openai(messages=messages[:1], reasoning_effort="none")
    )
    full = oracle.encode_chat_completion(ChatCompletionRequest.from_openai(messages=messages, reasoning_effort="none"))
    length = len(full.tokens) + 7
    batch = default_collate_fn(examples, processor=tokenizer, max_length=length)
    assert batch["input_ids"].shape == (1, length - 1)
    assert batch["input_ids"][0, : len(full.tokens)].tolist() == full.tokens
    assert batch["labels"][0, : len(prefix.tokens) - 1].eq(-100).all()
    assert (
        batch["labels"][0, len(prefix.tokens) - 1 : len(full.tokens) - 1].tolist() == full.tokens[len(prefix.tokens) :]
    )
    assert batch["labels"][0, len(full.tokens) - 1 :].eq(-100).all()
    assert batch["attention_mask"].sum().item() == len(full.tokens)
    for actual, expected in zip(batch["pixel_values"], full.images, strict=True):
        torch.testing.assert_close(actual, torch.from_numpy(expected).to(torch.bfloat16), rtol=0, atol=0)
    assert examples[0]["conversation"][0]["content"][0]["image"].tobytes() == image.tobytes()
    truncated = default_collate_fn(examples, processor=tokenizer, max_length=len(prefix.tokens) + 1)
    assert truncated["labels"][0, -1].item() == full.tokens[len(prefix.tokens)]
    assert truncated["labels"].ne(-100).sum().item() == 1


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_pipeline_media_preserves_whole_images_and_sample_order(microbatch_size):
    """Keep ragged [C,H,W] images intact and token/label rows [B,S] identical."""
    tokenizer = MistralCommonBackend(
        Path(__file__).parent / "tiny_tekken.json", mode="finetuning", padding_side="right"
    )
    images = [Image.new("RGB", size, (30 * i, 70, 130)) for i, size in enumerate(((56, 28), (28, 56), (56, 56)), 1)]
    examples = []
    for row_images in ([], images[:2], [], images[2:]):
        examples.append(
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": im} for im in row_images]
                        + [{"type": "text", "text": "What organ?"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "lung"}]},
                ]
            }
        )
    expected = default_collate_fn(examples, processor=tokenizer, max_length=512)
    actual = default_collate_fn(examples, processor=tokenizer, max_length=512)
    assert actual["n_images_per_sample"].tolist() == [0, 2, 0, 1]
    actual = prepare_vlm_media_for_pp(actual, batch_size=4, n_microbatches=4 // microbatch_size)
    for key in ("input_ids", "attention_mask", "labels"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert "pixel_values" not in actual
    chunks = actual["_vlm_pp_media_chunks"]["pixel_values"]
    assert [len(chunk) for chunk in chunks] == ([0, 2, 0, 1] if microbatch_size == 1 else [2, 1])
    for image, reference in zip([image for chunk in chunks for image in chunk], expected["pixel_values"], strict=True):
        torch.testing.assert_close(image, reference, atol=0, rtol=0)


@pytest.mark.parametrize("all_ignored", [False, True])
def test_shared_recipe_labels_shift_once_and_preserve_gradients(all_ignored):
    processor = MistralCommonBackend(Path(__file__).parent / "tiny_tekken.json", mode="finetuning")
    conversations = [
        [
            {"role": "user", "content": "Answer blue. What color?"},
            {"role": "assistant", "content": [{"type": "text", "text": "blue"}]},
            {"role": "user", "content": "And now?"},
            {"role": "assistant", "content": [{"type": "text", "text": "red"}]},
        ]
    ]
    full = processor.apply_chat_template(conversations, return_tensors="pt")
    labels = build_labels_from_template(full["input_ids"], conversations, processor)
    # Both assistant turns, including EOS, are supervised; repeated prompt
    # words cannot accidentally become targets.
    assert labels.ne(-100).sum() > 2
    assert labels[full["input_ids"] == processor.eos_token_id].ne(-100).all()
    batch = default_collate_fn([{"conversation": conversations[0]}], processor)
    torch.testing.assert_close(batch["labels"], labels[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(batch["input_ids"], full["input_ids"][:, :-1], atol=0, rtol=0)
    if all_ignored:
        labels.fill_(-100)
        batch["labels"].fill_(-100)
    model = make_model(vocab_size=len(processor))
    other = copy.deepcopy(model)
    expected = model(full["input_ids"], labels=labels).loss
    actual = MaskedCrossEntropy(reduction="sum")(
        other(batch["input_ids"]).logits, batch["labels"], num_label_tokens=int(batch["labels"].ne(-100).sum())
    )
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    actual.backward()
    expected.backward()
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5, msg=name)
        else:
            assert a.grad is None and b.grad is None


@pytest.mark.parametrize("as_list", [False, True])
def test_pp_image_counts_override_coincidental_batch_shape(as_list):
    from nemo_automodel.components.datasets.vlm.pp_media import chunk_vlm_media

    images = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    pixels = list(images.unbind()) if as_list else images
    sizes = torch.tensor([[4, 4], [4, 4]])
    chunks, grids = chunk_vlm_media(pixels, sizes, 2, 2, torch.tensor([0, 2]))
    assert len(chunks[0]) == 0 and len(chunks[1]) == 2
    assert grids[0].shape == (0, 2)
    torch.testing.assert_close(grids[1], sizes)
    for actual, expected in zip(chunks[1], images, strict=True):
        torch.testing.assert_close(actual, expected)
