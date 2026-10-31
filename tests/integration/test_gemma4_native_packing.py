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

"""CPU native preprocessing/NEAT/model parity; no external dataset or GPU kernels."""

import copy
from types import MethodType

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import PackedDatasetWrapperConfig
from nemo_automodel.components.distributed.pipelining.hf_utils import (
    create_pipeline_forward_gemma4_text,
    create_pipeline_forward_gemma4_vlm,
    validate_hf_model_for_pipeline_support,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration
from tests.utils.gemma4_vlm_fixtures import gemma4_processor, tiny_gemma4_config


def _model(attention):
    torch.manual_seed(42)
    config = tiny_gemma4_config()
    config._attn_implementation = attention
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    return Gemma4ForConditionalGeneration(config, backend=backend).float().train()


def _conversation(size, color):
    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", size, color)},
                    {"type": "text", "text": "Describe the image"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }


def _forward(model, batch):
    inputs = {
        key: value
        for key, value in batch.items()
        if key not in ("labels", "n_images_per_sample", "n_videos_per_sample")
    }
    return model(**inputs, use_cache=False).logits


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_native_image_packing_matches_separate_logits_loss_and_gradients(attention):
    processor = gemma4_processor()
    dataset = PreTokenizedDatasetWrapper(
        [_conversation((8, 12), "red"), _conversation((12, 8), "blue")],
        processor,
        inject_fake_images=False,
        max_retries=1,
    )
    encoded = [dataset[index] for index in range(2)]
    packed_dataset = PackedDatasetWrapperConfig(pack_size=512, max_retries=1).build(
        inner_dataset=dataset,
        bins=[[0, 1]],
        padding_idx=processor.tokenizer.pad_token_id,
    )
    packed = neat_packed_vlm_collater([packed_dataset[0]])
    assert packed["pixel_values"].shape[0] == 2
    assert packed["image_position_ids"].shape[0] == 2
    assert packed["_packed_seq_ids"].max() == 2
    model = _model(attention)
    reference = copy.deepcopy(model)
    actual = _forward(model, packed)
    expected_parts = []
    for sample in encoded:
        inputs = {
            "input_ids": sample["input_ids"][:-1][None],
            "attention_mask": sample["attention_mask"][:-1][None],
            "position_ids": torch.arange(len(sample["input_ids"]) - 1)[None],
            "mm_token_type_ids": sample["mm_token_type_ids"][..., :-1].reshape(1, -1),
            "pixel_values": sample["pixel_values"].to(packed["pixel_values"].dtype),
            "image_position_ids": sample["image_position_ids"],
        }
        expected_parts.append(_forward(reference, inputs))
    expected = torch.cat(expected_parts, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    targets = torch.cat([sample["labels"][1:] for sample in encoded])[None]
    torch.testing.assert_close(packed["labels"], targets)
    actual_loss = F.cross_entropy(actual.flatten(0, 1), targets.flatten(), reduction="sum")
    expected_loss = F.cross_entropy(expected.flatten(0, 1), targets.flatten(), reduction="sum")
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=2e-5, msg=name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.vision_tower.parameters())


def test_single_document_native_neat_retains_sliding_policy():
    model = _model("sdpa")
    tokens = torch.arange(2, 26)
    sample = {
        "input_ids": tokens,
        "labels": tokens,
        "position_ids": torch.arange(24),
        "attention_mask": torch.ones(24, dtype=torch.long),
    }
    batch = neat_packed_collater([sample], packing=configure_packing("sdpa", model=model))
    assert batch["_packed_seq_ids"].tolist() == [[1] * 24]
    with torch.no_grad():
        actual = _forward(model, batch)
        expected = model(input_ids=tokens[None], attention_mask=torch.ones((1, 24)), use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_arbitrary_four_dimensional_mask_is_not_reinterpreted():
    model = _model("sdpa")
    mask = torch.zeros((1, 1, 4, 4))
    seen = []
    handle = model.model.language_model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["attention_mask"]), with_kwargs=True
    )
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[2, 3, 4, 5]]), attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert seen[0]["full_attention"] is mask
    assert seen[0]["sliding_attention"] is mask


@pytest.mark.parametrize("pipeline_wrapper", [False, True])
def test_compact_input_requires_model_hook_configuration(pipeline_wrapper):
    model = _model("sdpa")
    if pipeline_wrapper:
        model.forward = MethodType(create_pipeline_forward_gemma4_vlm(), model)
    with pytest.raises(ValueError, match="Configure packed attention before forwarding compact Gemma4 attention IDs"):
        model(input_ids=torch.tensor([[2, 3]]), _packed_seq_ids=torch.ones((1, 2), dtype=torch.long))


def test_native_tie_policy_rejects_scratch_untied_and_pipeline_aliases():
    untied_config = tiny_gemma4_config(tie_word_embeddings=False)
    assert not untied_config.tie_word_embeddings and not untied_config.text_config.tie_word_embeddings
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=False"):
        Gemma4ForConditionalGeneration(untied_config)
    model = _model("sdpa")
    assert model.model.language_model.embed_tokens.weight is model.lm_head.weight
    with pytest.raises(ValueError, match="tie_word_embeddings=True"):
        validate_hf_model_for_pipeline_support(model)


def test_pipeline_text_wrapper_preserves_model_prepared_masks_and_gradients():
    """Compare the actual decoder wrapper locally; this does not certify Gemma PP."""
    model = _model("sdpa")
    reference = model.model.language_model
    wrapped = copy.deepcopy(reference)
    wrapped.forward = MethodType(create_pipeline_forward_gemma4_text(), wrapped)
    tokens = torch.tensor([[2, 5, 6, 7, 3, 5, 8, 4]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    documents = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]])
    allowed = (documents[:, None, :, None] == documents[:, None, None, :]).tril()
    modalities = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0]])
    masks = model.prepare_document_attention_mask(allowed, modalities, documents)
    originals = {name: value.clone() for name, value in masks.items()}
    expected = reference(
        input_ids=tokens, position_ids=positions, attention_mask=masks, use_cache=False
    ).last_hidden_state
    actual = wrapped(input_ids=tokens, position_ids=positions, attention_mask=masks)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    for (name, parameter), (_, baseline) in zip(wrapped.named_parameters(), reference.named_parameters(), strict=True):
        assert (parameter.grad is None) == (baseline.grad is None), name
        if baseline.grad is not None:
            torch.testing.assert_close(parameter.grad, baseline.grad, atol=2e-5, rtol=2e-4, msg=name)
    for name, value in masks.items():
        torch.testing.assert_close(value, originals[name], atol=0, rtol=0)


@pytest.mark.parametrize("pipeline_wrapper", [False, True])
def test_flash_public_and_compact_metadata_are_rejected_before_forward(pipeline_wrapper):
    model = _model("sdpa")
    with pytest.raises(ValueError, match="Flash varlen cannot preserve"):
        configure_packing("flash_attention_2", model=model)
    model.config.text_config._attn_implementation = "flash_attention_2"
    if pipeline_wrapper:
        model.forward = MethodType(create_pipeline_forward_gemma4_vlm(), model)
    for metadata in (
        {"_packed_seq_ids": torch.ones((1, 2), dtype=torch.long)},
        {
            "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0, 2], dtype=torch.int32),
            "max_length_q": 2,
            "max_length_k": 2,
        },
    ):
        with pytest.raises(ValueError, match="Flash varlen cannot preserve"):
            model(input_ids=torch.tensor([[2, 3]]), **metadata)


def test_pipeline_vlm_wrapper_composes_native_image_policy_before_decoder():
    """Compare full-model wrappers locally; no partitioned PP support is claimed."""
    processor = gemma4_processor()
    dataset = PreTokenizedDatasetWrapper(
        [_conversation((8, 12), "red"), _conversation((12, 8), "blue")],
        processor,
        inject_fake_images=False,
        max_retries=1,
    )
    packed = PackedDatasetWrapperConfig(pack_size=255, max_retries=1).build(
        inner_dataset=dataset,
        bins=[[0, 1]],
        padding_idx=processor.tokenizer.pad_token_id,
    )
    reference = _model("sdpa")
    wrapped = copy.deepcopy(reference)
    wrapped.forward = MethodType(create_pipeline_forward_gemma4_vlm(), wrapped)
    wrapped.model.language_model.forward = MethodType(
        create_pipeline_forward_gemma4_text(), wrapped.model.language_model
    )
    contract = configure_packing("sdpa", model=reference)
    configure_packing("sdpa", model=wrapped)
    batch = neat_packed_vlm_collater([packed[0]], packing=contract)
    # The same compact hook must run before either outer forward composes masks.
    batch["_packed_seq_ids"] = batch["_packed_seq_ids"]
    del batch["attention_mask"]
    expected = _forward(reference, batch)
    actual = wrapped(
        **{
            key: value
            for key, value in batch.items()
            if key not in ("labels", "n_images_per_sample", "n_videos_per_sample")
        },
        use_cache=False,
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected_loss = F.cross_entropy(expected.flatten(0, 1), batch["labels"].flatten(), reduction="sum")
    actual_loss = F.cross_entropy(actual.flatten(0, 1), batch["labels"].flatten(), reduction="sum")
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    for (name, parameter), (_, baseline) in zip(wrapped.named_parameters(), reference.named_parameters(), strict=True):
        assert (parameter.grad is None) == (baseline.grad is None), name
        if baseline.grad is not None:
            torch.testing.assert_close(parameter.grad, baseline.grad, atol=2e-5, rtol=2e-4, msg=name)
