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

"""Real processor coverage for the shared, unshifted VLM preprocessing path."""

import pytest
import torch
from PIL import Image

from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper, encode_vlm_document
from nemo_automodel.components.datasets.vlm.preprocessing import collate_position_ids, merge_vlm_documents
from tests.utils.vlm_fixtures import processor as processor


def _document(image_sizes=((8, 12),)):
    images = [
        {"type": "image", "image": Image.new("L", size, color=20 + index)} for index, size in enumerate(image_sizes)
    ]
    return {
        "conversation": [
            {"role": "user", "content": images + [{"type": "text", "text": "Describe image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }


@pytest.mark.parametrize("image_sizes", [(), ((8, 12),), ((8, 12), (16, 8))])
def test_unshifted_encoding_matches_native_and_preserves_media(processor, image_sizes):
    document = _document(image_sizes)
    encoded = encode_vlm_document(document, processor, inject_fake_images=False)
    assert encoded["mm_token_type_ids"].shape == encoded["input_ids"].shape
    native = PreTokenizedDatasetWrapper([document], processor, inject_fake_images=False, max_retries=1)[0]
    assert native.keys() == encoded.keys()
    for key in encoded:
        torch.testing.assert_close(encoded[key], native[key], rtol=0, atol=0)
    # Each 4x4 source region becomes one language image slot after merging.
    assert encoded["input_ids"].eq(processor.image_token_id).sum().item() == sum(w * h // 16 for w, h in image_sizes)
    assert encoded["labels"][encoded["labels"] != -100].tolist() == [6, 7, 4]
    assert encoded["input_ids"][-1].item() == processor.tokenizer.eos_token_id
    assert encoded["labels"][-1].item() == processor.tokenizer.eos_token_id
    for item in document["conversation"][0]["content"][:-1]:
        assert item["image"].mode == "L"  # RGB conversion must not mutate replay input.
    if image_sizes:
        assert encoded["pixel_values"].shape[0] == sum(w * h // 4 for w, h in image_sizes)


def test_encoder_reports_oversized_without_replacement(processor):
    with pytest.raises(ValueError, match="exceeding max_length=4"):
        encode_vlm_document(_document(), processor, max_length=4, inject_fake_images=False)
    # Truncation cannot leave an image grid referring to removed image slots.
    with pytest.raises(ValueError, match="Media token mismatch"):
        encode_vlm_document(_document(), processor, max_length=4, truncate=True, inject_fake_images=False)


def test_fake_image_slots_keep_attention_and_supervision_exclusions(processor):
    encoded = encode_vlm_document(_document(()), processor)
    image_slots = encoded["input_ids"].eq(processor.image_token_id)
    assert image_slots.any()
    assert encoded["attention_mask"][image_slots].eq(0).all()
    assert encoded["labels"][image_slots].eq(-100).all()
    assert encoded["labels"][encoded["labels"] != -100].tolist() == [6, 7, 4]


@pytest.mark.parametrize("generation_mask", [False, True])
def test_processor_template_owns_multiturn_special_tokens_and_supervision(processor, generation_mask):
    from tokenizers.processors import TemplateProcessing

    # Tokenizer postprocessing would duplicate template markers if applied twice.
    processor.tokenizer.bos_token = "<user>"
    processor.tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<user> $A <turn_end>", special_tokens=[("<user>", 2), ("<turn_end>", 4)]
    )
    if not generation_mask:
        processor.chat_template = processor.chat_template.replace("{% generation %}", "").replace(
            "{% endgeneration %}", ""
        )
    document = _document(())
    document["conversation"] += [
        {"role": "user", "content": [{"type": "text", "text": "Describe image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
    ]
    expected = processor.apply_chat_template(
        [document["conversation"]], tokenize=True, return_dict=True, return_tensors="pt"
    )
    encoded = encode_vlm_document(document, processor, inject_fake_images=False)
    torch.testing.assert_close(encoded["input_ids"], expected["input_ids"][0], rtol=0, atol=0)
    assert encoded["input_ids"].eq(processor.tokenizer.eos_token_id).sum() == 2
    assert encoded["labels"][encoded["labels"] != -100].tolist() == [6, 7, 4, 6, 7, 4]


def test_processor_template_without_eos_is_not_augmented(processor):
    processor.chat_template = processor.chat_template.replace(" <turn_end>", "")
    encoded = encode_vlm_document(_document(()), processor, inject_fake_images=False)
    assert not encoded["input_ids"].eq(processor.tokenizer.eos_token_id).any()
    assert encoded["labels"][encoded["labels"] != -100].tolist() == [6, 7]


def test_unshifted_assembly_retains_terminal_targets(processor):
    documents = [
        encode_vlm_document(_document(sizes), processor, inject_fake_images=False)
        for sizes in [((8, 12),), ((16, 8), (8, 8))]
    ]
    merged = merge_vlm_documents(documents, padding_idx=processor.tokenizer.pad_token_id)
    expected_ids = torch.cat([document["input_ids"] for document in documents])
    torch.testing.assert_close(merged["input_ids"], expected_ids, rtol=0, atol=0)
    assert merged["seq_lens"] == [len(document["input_ids"]) for document in documents]
    assert merged["n_images"] == 3
    assert merged["labels"][merged["labels"] != -100].tolist() == [6, 7, 4, 6, 7, 4]
    torch.testing.assert_close(merged["image_grid_thw"], torch.tensor([[1, 6, 4], [1, 4, 8], [1, 4, 4]]))
    torch.testing.assert_close(merged["pixel_values"], torch.cat([d["pixel_values"] for d in documents]))


def test_document_assembly_can_omit_native_targets_and_attention_ids():
    documents = [
        {"input_ids": torch.tensor([2, 3]), "position_ids": torch.arange(2), "pixel_values": torch.ones(1, 3, 2, 2)},
        {"input_ids": torch.tensor([4, 5, 6]), "position_ids": torch.arange(3)},
    ]
    # Labels are deliberately absent: this consumer supplies loss masks separately.
    merged = merge_vlm_documents(documents, padding_idx=0, return_labels=False, return_attention_mask=False)
    assert "labels" not in merged and "attention_mask" not in merged
    assert merged["input_ids"].tolist() == [2, 3, 4, 5, 6]
    assert merged["position_ids"].tolist() == [0, 1, 0, 1, 2]
    assert merged["seq_lens"] == merged["seq_lens_padded"] == [2, 3]
    torch.testing.assert_close(merged["pixel_values"], documents[0]["pixel_values"])


def test_position_field_collation_preserves_mrope_and_pads_only_tail():
    first = torch.tensor([[0, 1, 2], [0, 2, 2], [0, 1, 1]])
    second = torch.tensor([[0, 1], [0, 1], [0, 2]])
    actual = collate_position_ids([first, second], sequence_length=4)
    assert actual.shape == (3, 2, 4)
    torch.testing.assert_close(actual[:, 0, :3], first)
    torch.testing.assert_close(actual[:, 1, :2], second)
    assert actual[:, 0, 3:].eq(0).all() and actual[:, 1, 2:].eq(0).all()
    with pytest.raises(ValueError, match="exceed"):
        collate_position_ids([first], sequence_length=2)


def test_spatial_assembly_uses_sizes_without_changing_native_ragged_layout():
    documents = [
        {"input_ids": [2, 3], "labels": [-100, 3], "pixel_values": torch.ones(1, 3, height, width)}
        for height, width in ((4, 4), (2, 6))
    ]
    native = merge_vlm_documents(documents, padding_idx=0)
    assert isinstance(native["pixel_values"], list)
    for actual, document in zip(native["pixel_values"], documents, strict=True):
        torch.testing.assert_close(actual, document["pixel_values"][0])
    for document in documents:
        document["image_sizes"] = torch.tensor([document["pixel_values"].shape[-2:]])
    with_sizes = merge_vlm_documents(documents, padding_idx=0)
    assert with_sizes["pixel_values"].shape == (2, 3, 4, 6)
    assert with_sizes["n_images"] == 2
    assert with_sizes["image_sizes"].tolist() == [[4, 4], [2, 6]]


def test_assembly_preserves_supplied_scalar_positions_and_alignment():
    documents = [
        {
            "input_ids": torch.tensor([2, 3, 4]),
            "labels": torch.tensor([-100, 3, 4]),
            "position_ids": torch.tensor([0, 2, 4]),
        },
        {"input_ids": [5, 6], "labels": [-100, 6], "position_ids": torch.tensor([0, 3])},
    ]
    actual = merge_vlm_documents(documents, padding_idx=1, sequence_alignment=4)
    assert actual["input_ids"].tolist() == [2, 3, 4, 1, 5, 6, 1, 1]
    assert actual["labels"].tolist() == [-100, 3, 4, -100, -100, 6, -100, -100]
    assert actual["position_ids"].tolist() == [0, 2, 4, 3, 0, 3, 2, 3]
    assert actual["attention_mask"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert actual["seq_lens"] == [3, 2]
    assert actual["seq_lens_padded"] == [4, 4]


def test_shared_media_merge_preserves_processor_precision_unless_caller_converts():
    from nemo_automodel.components.datasets.vlm.preprocessing import merge_media_values

    pixels = torch.tensor([[[1.0001, 0.5001]]], dtype=torch.float32)
    merged = merge_media_values([pixels, pixels])
    torch.testing.assert_close(merged, torch.cat((pixels, pixels)), rtol=0, atol=0)
    assert merged.dtype == torch.float32
    native = merge_media_values([pixels], dtype=torch.bfloat16)
    assert native.dtype == torch.bfloat16
    torch.testing.assert_close(pixels, torch.tensor([[[1.0001, 0.5001]]]), rtol=0, atol=0)


@pytest.mark.parametrize("use_generation_tags", [False, True])
def test_named_template_without_default_renders_and_labels_same_selection(processor, use_generation_tags):
    template = processor.chat_template
    if not use_generation_tags:
        template = template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
        # Marker derivation renders string-valued sentinel conversations.
        template = template.replace(
            "{% for p in m.content %}{{ p.text }}{% endfor %}",
            "{% if m.content is string %}{{ m.content }}{% else %}"
            "{% for p in m.content %}{{ p.text }}{% endfor %}{% endif %}",
        )
        processor.tokenizer.add_tokens(["u", "XSENTINELMARKERX"])
    processor.chat_template = {"named": template}
    # The processor-selected template must also govern fallback marker derivation.
    processor.tokenizer.chat_template = {"different": "irrelevant"}
    encoded = encode_vlm_document(_document(), processor, inject_fake_images=False)
    assert encoded["labels"][encoded["labels"] != -100].tolist() == [6, 7, 4]
    assert encoded["input_ids"].eq(processor.image_token_id).sum().item() == 6
