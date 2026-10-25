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

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
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
