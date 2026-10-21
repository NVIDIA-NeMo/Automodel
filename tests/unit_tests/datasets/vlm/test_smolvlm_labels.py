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

from copy import deepcopy

import pytest
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.datasets.vlm import datasets as vlm_datasets
from nemo_automodel.components.datasets.vlm.loader import VlmProcessorConfig
from nemo_automodel.recipes._typed_config import RecipeConfig

_CHECKPOINT = "HuggingFaceTB/SmolVLM-Instruct"
# Public Hugging Face checkpoint commit, matching the recipe's inspected revision.
_REVISION = "81cd9a775a4d644f2faf4e7becff4559b46b14c7"  # pragma: allowlist secret


@pytest.fixture(scope="module")
def smolvlm_processor():
    return AutoProcessor.from_pretrained(_CHECKPOINT, revision=_REVISION)


@pytest.mark.parametrize("validation", [False, True], ids=["train", "validation"])
@pytest.mark.parametrize(
    "question,answer",
    [
        ("Describe the object visible in the image.", "normal"),
        ("normal. Is this image normal?", "normal"),
        ("What is shown?", "A normal chest radiograph."),
        ("What is shown?", "No acute abnormality is seen."),
        ("Answer yes or no: is this normal?", "yes"),
    ],
)
def test_smolvlm_supervises_assistant_turn(smolvlm_processor, monkeypatch, validation, question, answer):
    """Real processor + native MedPix pipeline must supervise replies and mask prompts/padding."""
    config = RecipeConfig(load_yaml_config("examples/vlm_finetune/huggingfacetb/smolvlm_instruct_medpix_peft.yaml"))
    source = Dataset.from_dict(
        {
            "image_id": [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))],
            "question": [question, "Give a brief description of this image and its clinical findings."],
            "answer": [answer, "The image shows an unremarkable examination without any acute findings."],
        }
    )
    monkeypatch.setattr(vlm_datasets, "load_dataset", lambda *args, **kwargs: source)
    loader_config = config.vlm_validation_dataloader if validation else config.vlm_dataloader
    loader_config.shuffle = False
    loader_config.pin_memory = False
    loader_config.processor_config = VlmProcessorConfig(factory=lambda *args, **kwargs: smolvlm_processor)
    built = loader_config.build(pretrained_model_name_or_path=_CHECKPOINT, dp_rank=0, dp_world_size=1, batch_size=2)
    batch = next(iter(built.dataloader))
    conversations = [built.dataloader.dataset[i]["conversation"] for i in range(2)]
    full = smolvlm_processor.apply_chat_template(
        conversations, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}
    )
    assert torch.equal(batch["input_ids"], full["input_ids"][:, :-1])
    assert (full["attention_mask"] == 0).any(), "Exercise unequal-length padding"

    for row, conversation in enumerate(conversations):
        original = smolvlm_processor.apply_chat_template(conversation, tokenize=False)
        changed = deepcopy(conversation)
        changed[-1]["content"][0]["text"] = "UNIQUE_ALTERNATIVE_REPLY_738"
        alternative = smolvlm_processor.apply_chat_template(changed, tokenize=False)
        # Locate the reply by changing only its text. This oracle neither searches
        # for standalone answer tokens nor uses production marker/masking helpers.
        tokenizer = smolvlm_processor.tokenizer
        ids = tokenizer.encode(original, add_special_tokens=False)
        alternate_ids = tokenizer.encode(alternative, add_special_tokens=False)
        start = next(i for i, (a, b) in enumerate(zip(ids, alternate_ids)) if a != b)
        suffix = 0
        while ids[-suffix - 1] == alternate_ids[-suffix - 1]:
            suffix += 1
        # The common suffix starts at the end-of-utterance token; include it.
        assert ids[-suffix] == tokenizer.eos_token_id
        end = len(ids) - suffix + 1
        # The processor expands image placeholders; align using the unchanged
        # suffix after the reply, independently of the image-token count.
        valid_length = int(full["attention_mask"][row].sum())
        padding = full["input_ids"].shape[1] - valid_length if tokenizer.padding_side == "left" else 0
        reply_end = padding + valid_length - (len(ids) - end)
        reply_start = reply_end - (end - start)
        expected = torch.full_like(full["input_ids"][row], -100)
        expected[reply_start:reply_end] = full["input_ids"][row, reply_start:reply_end]
        assert torch.equal(batch["labels"][row], expected[1:])
