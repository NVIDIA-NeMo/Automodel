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

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PixtralImageProcessor, PixtralProcessor, PreTrainedTokenizerFast

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes._typed_config import RecipeConfig


@pytest.fixture
def ministral_processor_path(tmp_path):
    tokens = ["[UNK]", "[PAD]", "[EOS]", "[IMG]", "[IMG_BREAK]", "[IMG_END]", "[USER]", "[ASSISTANT]"]
    backend = Tokenizer(
        models.WordLevel({token: i for i, token in enumerate(tokens + ["describe", "scan"])}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=tokens[3:],
        model_max_length=512,
    )
    template = (
        "{% for message in messages %}"
        "{{ '[USER] ' if message['role'] == 'user' else '[ASSISTANT] ' }}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for item in message['content'] %}"
        "{{ '[IMG]' if item['type'] == 'image' else item['text'] }} "
        "{% endfor %}{% endif %}"
        "{% if message['role'] == 'assistant' %}[EOS]{% endif %}"
        "{% endfor %}"
    )
    tokenizer.chat_template = template
    processor = PixtralProcessor(
        # Match the published Ministral checkpoints' null padding setting.
        image_processor=PixtralImageProcessor(do_pad=None, patch_size=14),
        tokenizer=tokenizer,
        patch_size=14,
        spatial_merge_size=2,
        chat_template=template,
    )
    processor.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("model_size", ["3", "8", "14"])
@pytest.mark.parametrize("loader_name", ["vlm_dataloader", "vlm_validation_dataloader"])
def test_ministral_recipe_collates_variable_resolution_images(ministral_processor_path, model_size, loader_name):
    root = Path(__file__).resolve().parents[4]
    recipe_path = root / f"examples/vlm_finetune/mistral/ministral3_{model_size}b_medpix.yaml"
    recipe = RecipeConfig(ConfigNode(yaml.safe_load(recipe_path.read_text())))
    loader = getattr(recipe, loader_name)
    processor = loader.processor_config.build(pretrained_model_name_or_path=ministral_processor_path)
    images = [Image.new("RGB", (28, 56), "red"), Image.new("RGB", (84, 28), "blue")]
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "describe"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "scan"}]},
            ]
        }
        for image in images
    ]

    batch = loader.collator.build(processor=processor)(examples)

    assert batch["pixel_values"].shape == (2, 3, 56, 84)
    assert batch["pixel_values"].dtype == torch.bfloat16
    assert batch["image_sizes"].tolist() == [[56, 28], [28, 84]]
    assert batch["n_images_per_sample"].tolist() == [1, 1]
    # Image tokens describe the unpadded image regions, with 2x2 patch merging.
    assert (batch["input_ids"] == processor.image_token_id).sum(dim=1).tolist() == [2, 3]
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] != -100).any(dim=1).all()
    for i, image in enumerate(images):
        expected = processor.image_processor(image, return_tensors="pt")["pixel_values"][0].to(torch.bfloat16)
        torch.testing.assert_close(batch["pixel_values"][i, :, : image.height, : image.width], expected, rtol=0, atol=0)
    assert torch.count_nonzero(batch["pixel_values"][0, :, :, 28:]) == 0
    assert torch.count_nonzero(batch["pixel_values"][1, :, 28:, :]) == 0
