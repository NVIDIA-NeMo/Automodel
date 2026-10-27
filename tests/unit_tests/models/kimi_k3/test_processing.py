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

"""Local K3 processor, standard collation and unshifted packing inputs."""

import base64
import json

import pytest
import torch
from PIL import Image
from tokenizers import AddedToken
from transformers import AutoProcessor

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn, pad_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.datasets.vlm.fake_image import inject_fake_image_into_conversation
from nemo_automodel.components.models.kimi_k3.processing import KimiK3ImageProcessor, KimiK3Processor
from nemo_automodel.components.models.kimi_k3.tokenization import TikTokenTokenizer


@pytest.fixture
def processor(tmp_path):
    vocab = tmp_path / "tiktoken.model"
    vocab.write_text("".join(f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)))
    names = [
        "[BOS]",
        "[EOS]",
        "[PAD]",
        "[UNK]",
        "<|media_pad|>",
        "<|media_begin|>",
        "<|media_content|>",
        "<|media_end|>",
        "<|im_end|>",
        "<|im_user|>",
        "<|im_assistant|>",
        "<|close|>",
    ]
    tokenizer = TikTokenTokenizer(
        str(vocab),
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        added_tokens_decoder={256 + i: AddedToken(name, special=True) for i, name in enumerate(names)},
    )
    return KimiK3Processor(KimiK3ImageProcessor(), tokenizer)


def _chat(image=None):
    content = [{"type": "text", "text": "Describe."}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": "Red."}]


def test_local_processor_load_save_and_checkpoint_remote_metadata(processor, tmp_path):
    from nemo_automodel._transformers import registry

    assert "kimi_k3" in registry._CUSTOM_PROCESSOR_REGISTRATIONS
    processor.save_pretrained(tmp_path)
    # Published checkpoint metadata must not cause execution of checkpoint Python.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k3"}))
    path = tmp_path / "processor_config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {"AutoProcessor": "missing_remote_code.KimiK3Processor"}
    path.write_text(json.dumps(config))
    loaded = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    assert type(loaded) is KimiK3Processor
    assert type(loaded.tokenizer) is TikTokenTokenizer
    assert type(loaded.image_processor) is KimiK3ImageProcessor
    chat = _chat(Image.new("RGB", (56, 28), "red"))
    expected = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    actual = loaded.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_native_packing_and_default_collation_share_expanded_tokens(processor):
    examples = [{"conversation": _chat(Image.new("RGB", (56, 28), "red"))}, {"conversation": _chat()}]
    wrapper = PreTokenizedDatasetWrapper(examples, processor, inject_fake_images=False, max_retries=1)
    first, second = wrapper[0], wrapper[1]
    assert int((first["input_ids"] == processor.image_token_id).sum()) == 2
    assert first["image_grid_thw"].tolist() == [[1, 2, 4]]
    expected_labels = processor.tokenizer.encode("Red.", add_special_tokens=False)
    expected_labels.append(processor.tokenizer.convert_tokens_to_ids("<|close|>"))
    assert first["labels"][first["labels"] != -100].tolist() == expected_labels
    assert torch.all(first["labels"][first["input_ids"] == processor.image_token_id] == -100)
    batch = default_collate_fn(
        examples, processor, max_length=max(len(first["input_ids"]), len(second["input_ids"])) + 2
    )
    for index, sample in enumerate([first, second]):
        n = len(sample["input_ids"]) - 1
        torch.testing.assert_close(batch["input_ids"][index, :n], sample["input_ids"][:-1])
        torch.testing.assert_close(batch["labels"][index, :n], sample["labels"][1:])
    assert batch["n_images_per_sample"].tolist() == [1, 0]


def test_standard_images_text_interface_and_image_truncation(processor):
    image = Image.new("RGB", (56, 28), "red")
    text = processor.apply_chat_template(_chat(image), tokenize=False)
    output = processor(text=[text], images=[image], return_tensors="pt")
    assert int((output["input_ids"] == processor.image_token_id).sum()) == 2
    start = int((output["input_ids"][0] == processor.image_token_id).nonzero()[0])
    # Reject cuts inside the patches and immediately before the closing marker.
    for length in (start + 1, start + 2):
        with pytest.raises(ValueError, match="image token count"):
            processor(text=text, images=image, truncation=True, max_length=length)
    processor.tokenizer.truncation_side = "left"
    begin_id = processor.tokenizer.convert_tokens_to_ids("<|media_begin|>")
    begin = output["input_ids"][0].tolist().index(begin_id)
    with pytest.raises(ValueError, match="image token count"):
        processor(text=text, images=image, truncation=True, max_length=output["input_ids"].shape[1] - begin - 1)
    processor.tokenizer.truncation_side = "right"
    with pytest.raises(ValueError, match="placeholder count"):
        processor(text=text, images=[])
    with pytest.raises(ValueError, match="video"):
        processor(text=text, videos=[image])


def _fake_and_real_batch(processor, path: str, fake_color: str) -> dict[str, torch.Tensor]:
    """Collate an injected-fake text sample and a real-image sample through a native path.

    Args:
        processor: Local K3 processor.
        path: "pretokenized" (wrapper -> pad_collate_fn) or "default_collate".
        fake_color: Fill color of the injected fake image, to perturb its pixels.

    Returns:
        Batch with input_ids/attention_mask [2, sequence], pixel_values
        [patches, 3, 14, 14] and image_grid_thw [2, 3]; row 0 owns the fake image.
    """
    from nemo_automodel.components.datasets.vlm import fake_image

    real = _chat(Image.new("RGB", (56, 28), "red"))
    original = fake_image._FAKE_IMAGE
    fake_image._FAKE_IMAGE = Image.new("RGB", (56, 56), fake_color)
    try:
        if path == "pretokenized":
            # inject_fake_images keeps its default (True), as in native recipes.
            wrapper = PreTokenizedDatasetWrapper(
                [{"conversation": _chat()}, {"conversation": real}], processor, max_retries=1
            )
            return pad_collate_fn([wrapper[0], wrapper[1]], processor)
        examples = [
            {"conversation": inject_fake_image_into_conversation(_chat()), "_injected_fake": True},
            {"conversation": real},
        ]
        return default_collate_fn(examples, processor)
    finally:
        fake_image._FAKE_IMAGE = original


@pytest.mark.parametrize("path", ["pretokenized", "default_collate"])
def test_injected_fake_images_stay_masked_through_model_training(processor, path):
    """Default fake-image injection must train; the masked fake media must not leak into text."""
    from tests.unit_tests.models.kimi_k3.test_multimodal import _tiny_vlm

    torch.manual_seed(26)
    model = _tiny_vlm(
        vocab_size=272,
        media_placeholder_token_id=processor.image_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        patch_size=14,
    )

    def run(batch):
        """Return logits [2, sequence, vocab] for a collated fake/real batch."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
        ).logits

    batch = _fake_and_real_batch(processor, path, "white")
    image_tokens = batch["input_ids"] == processor.image_token_id
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    assert image_tokens.sum(-1).tolist() == [4, 2]
    # The data contract under test: fake image tokens are masked, real ones are not.
    assert bool((batch["attention_mask"][0][image_tokens[0]] == 0).all())
    assert bool((batch["attention_mask"][1][image_tokens[1]] == 1).all())

    logits = run(batch)
    valid = batch["labels"] != -100
    assert valid[0].any() and valid[1].any()
    loss = torch.nn.functional.cross_entropy(logits[valid], batch["labels"][valid])
    loss.backward()
    assert torch.isfinite(loss)
    vision_grads = [p.grad for name, p in model.named_parameters() if name.startswith("vision_tower.")]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in vision_grads)
    assert any(grad.abs().sum() > 0 for grad in vision_grads)

    with torch.no_grad():
        other_fake = _fake_and_real_batch(processor, path, "black")
        torch.testing.assert_close(other_fake["input_ids"], batch["input_ids"], rtol=0, atol=0)
        assert not torch.equal(other_fake["pixel_values"], batch["pixel_values"])
        perturbed = run(other_fake)
        text = batch["attention_mask"][0] > 0
        # Masked fake media cannot influence any unmasked token of its sample.
        torch.testing.assert_close(perturbed[0][text], logits[0][text].detach(), rtol=0, atol=0)
        torch.testing.assert_close(perturbed[1], logits[1].detach(), rtol=0, atol=0)
        # Control: the same comparison detects a change to the real image.
        real_changed = dict(batch)
        fake_patches = int(batch["image_grid_thw"][0].prod())
        real_changed["pixel_values"] = batch["pixel_values"].clone()
        real_changed["pixel_values"][fake_patches:] = -batch["pixel_values"][fake_patches:]
        assert not torch.allclose(run(real_changed)[1], logits[1].detach())


def test_tool_call_only_assistant_message_matches_native_tokenizer(processor):
    chat = [
        {"role": "user", "content": "Inspect."},
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "inspect", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]
    expected = processor.tokenizer.apply_chat_template(chat, tokenize=True)
    output = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    assert output["input_ids"][0].tolist() == expected
    assert "pixel_values" not in output or output["pixel_values"].numel() == 0


def _tool_image_chat(reply_order: tuple[str, str]) -> list[dict[str, object]]:
    """Build a chat whose two image-bearing tool replies arrive in ``reply_order``.

    Args:
        reply_order: Tool-call ids in arrival order; call "a" returns a red image,
            call "b" a blue one of the same size.

    Returns:
        The conversation with its tool replies in the requested order.
    """
    colors = {"a": "red", "b": "blue"}
    replies = [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": [
                {"type": "image", "image": Image.new("RGB", (56, 28), colors[call_id])},
                {"type": "text", "text": f"result {call_id}"},
            ],
        }
        for call_id in reply_order
    ]
    return [
        {"role": "user", "content": "Look at both."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": f"capture_{call_id}", "arguments": {}}}
                for call_id in ("a", "b")
            ],
        },
        *replies,
        {"role": "assistant", "content": "Red then blue."},
    ]


def test_reordered_tool_result_images_follow_their_tool_calls(processor):
    """Tool replies render in tool_calls order, so their images must be collected in that order too."""
    in_order = processor.apply_chat_template(
        _tool_image_chat(("a", "b")), tokenize=True, return_dict=True, return_tensors="pt"
    )
    swapped = processor.apply_chat_template(
        _tool_image_chat(("b", "a")), tokenize=True, return_dict=True, return_tensors="pt"
    )
    assert in_order["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 4]]
    torch.testing.assert_close(swapped["input_ids"], in_order["input_ids"], rtol=0, atol=0)
    torch.testing.assert_close(swapped["image_grid_thw"], in_order["image_grid_thw"], rtol=0, atol=0)
    torch.testing.assert_close(swapped["pixel_values"], in_order["pixel_values"], rtol=0, atol=0)
    # Call "a" is rendered first and returns the red image, so the first patch is
    # red regardless of arrival order; the two images themselves differ.
    patches = int(in_order["image_grid_thw"][0].prod())
    assert not torch.equal(in_order["pixel_values"][:patches], in_order["pixel_values"][patches:])
    red = processor(processor.image_placeholder, Image.new("RGB", (56, 28), "red"), return_tensors="pt")
    torch.testing.assert_close(swapped["pixel_values"][:patches], red["pixel_values"], rtol=0, atol=0)
    # The caller's messages are not mutated by the normalization.
    chat = _tool_image_chat(("b", "a"))
    processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    assert [message["tool_call_id"] for message in chat if message["role"] == "tool"] == ["b", "a"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_chat_truncation_honors_truncation_side(processor, side):
    chat = _chat()
    full = processor.tokenizer.apply_chat_template(
        chat, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    ids, mask = list(full["input_ids"]), list(full["assistant_masks"])
    processor.tokenizer.truncation_side = side
    try:
        for limit in (0, 10, len(ids), len(ids) + 3):
            cut = processor.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=limit,
            )
            kept = slice(max(0, len(ids) - limit), None) if side == "left" else slice(0, limit)
            assert list(cut["input_ids"]) == ids[kept]
            assert list(cut["assistant_masks"]) == mask[kept]
        untruncated = processor.tokenizer.apply_chat_template(chat, tokenize=True, truncation=False, max_length=10)
        assert untruncated == ids
        processed = processor.apply_chat_template(chat, tokenize=True, truncation=True, max_length=10)
        assert processed == (ids[-10:] if side == "left" else ids[:10])
    finally:
        processor.tokenizer.truncation_side = "right"


def test_left_chat_truncation_keeps_or_rejects_whole_image_spans(processor):
    image = Image.new("RGB", (56, 28), "red")
    chat = _chat(image)
    full = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    ids = full["input_ids"][0].tolist()
    begin = ids.index(processor.tokenizer.convert_tokens_to_ids("<|media_begin|>"))
    processor.tokenizer.truncation_side = "left"
    try:
        # Cutting through <|media_begin|> drops part of the image span.
        with pytest.raises(ValueError, match="image token count"):
            processor.apply_chat_template(chat, tokenize=True, truncation=True, max_length=len(ids) - begin - 1)
        # A cut confined to the text before the span keeps the whole image.
        kept = processor.apply_chat_template(
            chat, tokenize=True, return_dict=True, return_tensors="pt", truncation=True, max_length=len(ids) - begin
        )
        assert kept["input_ids"][0].tolist() == ids[begin:]
        torch.testing.assert_close(kept["pixel_values"], full["pixel_values"], rtol=0, atol=0)
        assert kept["image_grid_thw"].tolist() == full["image_grid_thw"].tolist()
    finally:
        processor.tokenizer.truncation_side = "right"


@pytest.mark.parametrize("stage", ["before_resize", "after_resize"])
def test_image_normalization_padding_and_alpha_background(stage):
    processor = KimiK3ImageProcessor(
        media_proc_cfg={
            "transparent_bg_config": {"pattern": "white"},
            "transparent_bg_fill_stage": stage,
        }
    )
    output = processor(Image.new("RGBA", (1, 1), (255, 0, 0, 0)), return_tensors="pt")
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    patches = output["pixel_values"]
    assert patches.shape == (4, 3, 14, 14)
    torch.testing.assert_close(patches[0, :, 0, 0], torch.ones(3))
    assert torch.count_nonzero(patches == 1) == 3
    assert torch.count_nonzero(patches == -1) == patches.numel() - 3
