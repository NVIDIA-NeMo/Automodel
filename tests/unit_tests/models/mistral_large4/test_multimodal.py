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

"""Image features, complete multimodal logits/gradients, packing and checkpoint tests."""

import copy
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers import AutoConfig, PixtralVisionConfig, PixtralVisionModel

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.common.packing import get_packing_capabilities
from nemo_automodel.components.models.mistral_large4.multimodal import MistralLarge4ForConditionalGeneration
from nemo_automodel.components.models.mistral_large4.vision import MistralLarge4VisionConfig
from tests.unit_tests.models.mistral_large4.math_reference import reference_logits, tiny_config
from tests.unit_tests.models.mistral_large4.vision_reference import reference_images


def make_multimodal(*, seed=201, dtype=torch.float32):
    torch.manual_seed(seed)
    vision = MistralLarge4VisionConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=2,
        image_size=16,
    )
    config = tiny_config(
        vision_config=asdict(vision),
        dtype=dtype,
        n_routed_experts=4,
        architectures=["MistralLarge4ForConditionalGeneration"],
    )
    model = MistralLarge4ForConditionalGeneration(config)
    model.initialize_weights()
    return model


def image_batch(device="cpu"):
    """Return integer [1, tokens] IDs and differently shaped floating images [C,H,W]."""
    ids = torch.tensor([[1, 10, 10, 13, 2, 10, 12, 10, 13, 3, 4]], device=device)
    pixels = [
        torch.linspace(-1, 1, 96, device=device).reshape(3, 4, 8),
        torch.linspace(1, -1, 96, device=device).reshape(3, 8, 4),
    ]
    return ids, pixels


@pytest.mark.parametrize("seed", [201, 312, 951])
def test_multimodal_outputs_all_gradients_and_update(seed):
    model = make_multimodal(seed=seed)
    ids, images = image_batch()
    images = [x.requires_grad_() for x in images]
    oracle_pixels = [x.detach().clone().requires_grad_() for x in images]
    source = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    oracle = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in source.items()}
    image_features = reference_images(oracle, oracle_pixels, model.vision_settings)
    actual_features = model.get_image_features(images)
    for actual, expected in zip(actual_features, image_features, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    # Explicit known image positions, independent of production alignment helper.
    embeddings = F.embedding(ids[0], oracle["tok_embeddings.weight"])
    embeddings = torch.stack(
        [
            image_features[0][0]
            if i == 1
            else image_features[0][1]
            if i == 2
            else image_features[1][0]
            if i == 5
            else image_features[1][1]
            if i == 7
            else row
            for i, row in enumerate(embeddings)
        ]
    )
    expected = reference_logits(oracle, ids[0], model.config, inputs_embeds=embeddings)
    actual = model(ids, pixel_values=images).logits[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    gradients = model.state_dict_adapter.to_hf({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    for key, parameter in source.items():
        if not parameter.requires_grad:
            assert key not in gradients and oracle[key].grad is None
            continue
        expected_grad = oracle[key].grad
        if expected_grad is None:
            expected_grad = torch.zeros_like(parameter)
        torch.testing.assert_close(gradients[key], expected_grad, atol=1.6e-5, rtol=4e-5, msg=key)
        assert torch.isfinite(gradients[key]).all()
    for a, b in zip(images, oracle_pixels, strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1.6e-5, rtol=4e-5)
        assert a.grad.abs().sum() > 0
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    torch.optim.SGD(oracle.values(), lr=0.001).step()
    source = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    for key, parameter in source.items():
        torch.testing.assert_close(parameter, oracle[key], atol=2e-6, rtol=2e-5, msg=key)


def test_encoder_matches_hf_pixtral_with_reversible_native_qk_layout():
    model = make_multimodal()
    c = model.vision_settings
    config = PixtralVisionConfig(
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_channels=c.num_channels,
        patch_size=c.patch_size,
        image_size=c.image_size,
        hidden_act="silu",
        rope_parameters={"rope_type": "axial", "rope_theta": c.rope_theta},
        attn_implementation="eager",
    )
    hf = PixtralVisionModel(config)
    converted = {}
    for key, value in model.vision_encoder.state_dict().items():
        new_key = key
        for before, after in (
            ("attention.wq.", "attention.q_proj."),
            ("attention.wk.", "attention.k_proj."),
            ("attention.wv.", "attention.v_proj."),
            ("attention.wo.", "attention.o_proj."),
            ("feed_forward.w1.", "feed_forward.gate_proj."),
            ("feed_forward.w2.", "feed_forward.down_proj."),
            ("feed_forward.w3.", "feed_forward.up_proj."),
        ):
            new_key = new_key.replace(before, after)
        if ".attention.wq." in key or ".attention.wk." in key:
            dim = c.hidden_size // c.num_attention_heads
            converted_value = value.view(c.num_attention_heads, dim // 2, 2, -1).transpose(1, 2).reshape_as(value)
            restored = converted_value.view(c.num_attention_heads, 2, dim // 2, -1).transpose(1, 2).reshape_as(value)
            torch.testing.assert_close(value, restored, atol=0, rtol=0)
            value = converted_value
        converted[new_key] = value
    hf.load_state_dict(converted, strict=True)
    _, images = image_batch()
    actual = model.vision_encoder(images)
    for image, features in zip(images, actual, strict=True):
        expected = hf(image[None]).last_hidden_state[0]
        torch.testing.assert_close(features, expected, atol=1e-6, rtol=2e-5)


def test_packed_images_text_padding_and_gradient_isolation():
    model = make_multimodal()
    reference = copy.deepcopy(model)
    ids, pixels = image_batch()
    # The shared collator communicates BF16 pixels; both models consume the
    # same processed images so this checks packing, not image quantization.
    pixels = [pixel.bfloat16() for pixel in pixels]
    examples = [
        dict(input_ids=ids[0, :5], labels=ids[0, :5].clone(), pixel_values=pixels[:1]),
        dict(input_ids=ids[0, 5:], labels=ids[0, 5:].clone(), pixel_values=pixels[1:]),
        dict(input_ids=torch.tensor([7, 8, 9]), labels=torch.tensor([-100, 8, 9])),
    ]
    original_labels = [e["labels"].clone() for e in examples]
    sample = {
        "input_ids": torch.cat([e["input_ids"] for e in examples]),
        "labels": torch.cat([e["labels"] for e in examples]),
        "attention_mask": torch.cat([torch.full_like(e["input_ids"], i + 1) for i, e in enumerate(examples)]),
        "position_ids": torch.cat([torch.arange(e["input_ids"].numel()) for e in examples]),
        "pixel_values": pixels,
        "n_images": len(pixels),
    }
    capabilities = get_packing_capabilities("sdpa", model=model)
    assert capabilities.packed_mask_type == "document_ids"
    batch = neat_packed_vlm_collater([sample], packing=capabilities)
    batch.pop("mm_token_type_ids")
    batch.pop("_packed_seq_ids")
    batch.pop("n_images_per_sample")
    # Image supervision is excluded by the model's HF-loss path.
    batch["labels"] = batch["labels"].masked_fill(torch.isin(batch["input_ids"], torch.tensor([10, 12, 13])), -100)
    actual = model(**batch)
    losses, counts = [], []
    cursor = 0
    for example in examples:
        n = example["input_ids"].numel()
        labels = batch["labels"][:, cursor : cursor + n]
        expected = reference(example["input_ids"][None], labels=labels, pixel_values=example.get("pixel_values"))
        torch.testing.assert_close(actual.logits[0, cursor : cursor + n], expected.logits[0], atol=2e-6, rtol=2e-5)
        losses.append(expected.loss)
        counts.append((labels[:, 1:] != -100).sum())
        cursor += n
    expected_loss = sum(loss * count for loss, count in zip(losses, counts)) / sum(counts)
    torch.testing.assert_close(actual.loss, expected_loss)
    actual.loss.backward()
    expected_loss.backward()
    for (key, a), (_, b) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=4e-5, msg=key)
    changed = dict(batch, pixel_values=[pixels[0] * -2, pixels[1]])
    torch.testing.assert_close(model(**changed).logits[:, 5:], actual.logits[:, 5:], atol=0, rtol=0)
    for e, label in zip(examples, original_labels):
        torch.testing.assert_close(e["labels"], label, atol=0, rtol=0)
    # Left and right text padding must not consume image features or positions.
    padded = F.pad(ids, (2, 1))
    mask = F.pad(torch.ones_like(ids), (2, 1))
    torch.testing.assert_close(
        model(padded, pixel_values=pixels, attention_mask=mask).logits[:, 2:-1],
        model(ids, pixel_values=pixels).logits,
        atol=2e-6,
        rtol=2e-5,
    )
    # Whole-image validation is model-owned after generic token truncation.
    with pytest.raises(ValueError, match="complete|grid|image"):
        model(ids[:, :2], pixel_values=pixels[:1])


@pytest.mark.parametrize("case", ["missing", "extra", "row", "boundary", "masked", "partial_patch", "size", "channels"])
def test_alignment_rejects_silent_image_misassignment(case):
    model = make_multimodal()
    ids, pixels = image_batch()
    kwargs = {}
    if case == "missing":
        pixels = pixels[:1]
    if case == "extra":
        pixels = pixels + [pixels[0]]
    if case == "row":
        ids[0, 6] = 13
    if case == "boundary":
        kwargs = dict(qkv_format="thd", cu_seqlens=torch.tensor([0, 2, 11]))
    if case == "masked":
        mask = torch.ones_like(ids)
        mask[0, 2] = 0
        kwargs["attention_mask"] = mask
    if case == "partial_patch":
        pixels[0] = pixels[0][:, :, :7]
    if case == "size":
        pixels[0] = torch.zeros(3, 4, 20)
    if case == "channels":
        pixels[0] = torch.zeros(1, 4, 8)
    with pytest.raises(ValueError):
        model(ids, pixel_values=pixels, **kwargs)


def test_config_meta_initialization_and_all_weights_roundtrip(tmp_path):
    model = make_multimodal()
    original = copy.deepcopy(model.config)
    original.architectures = ["MistralLarge4ForCausalLM"]
    direct = MistralLarge4ForConditionalGeneration(original)
    assert direct.config.architectures == ["MistralLarge4ForConditionalGeneration"]
    assert original.architectures == ["MistralLarge4ForCausalLM"]
    model.config.save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    cls = ModelRegistry.get_model_cls_from_model_arch(config.architectures[0])
    assert cls is MistralLarge4ForConditionalGeneration
    with torch.device("meta"):
        restored = cls(config)
    Checkpointer.initialize_model_weights(restored, torch.device("cpu"))
    assert all(torch.isfinite(p).all() for p in restored.parameters())
    source = model.state_dict_adapter.to_hf(model.state_dict())
    save_file({k: v.contiguous() for k, v in source.items()}, tmp_path / "model.safetensors")
    restored.load_state_dict(
        restored.state_dict_adapter.from_hf(load_file(tmp_path / "model.safetensors")), strict=True
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    # Vision load parts alias final storage, including convolution's four axes.
    parts = list(restored.state_dict_adapter.iter_checkpoint_load_parts(restored.state_dict()))
    direct = [
        part
        for part in parts
        if any(
            k.startswith(("vision_encoder.", "patch_merger.", "pre_mm_projector_norm.", "vision_language_adapter."))
            for k in part.checkpoint_tensors
        )
    ]
    assert len(direct) == 24
    for part in direct:
        assert not part.temporary_checkpoint_keys
        for key, value in part.checkpoint_tensors.items():
            assert value.data_ptr() == restored.state_dict()[key].data_ptr()
            value.fill_(0.125)
            part.finish()
            assert bool((restored.state_dict()[key] == 0.125).all())
            value.copy_(source[key])
    ids, pixels = image_batch()
    torch.testing.assert_close(
        model(ids, pixel_values=pixels).logits, restored(ids, pixel_values=pixels).logits, atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_text_only_and_empty_supervision_keep_vision_connected(dtype):
    model = make_multimodal(dtype=dtype)
    ids = torch.tensor([[1, 2, 3, 4]])
    output = model(ids, labels=ids)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    for key, p in model.named_parameters():
        if key.startswith(("vision_encoder.", "vision_language_adapter.", "patch_merger.", "pre_mm_projector_norm.")):
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0, key
    model.zero_grad(set_to_none=True)
    image_ids, pixels = image_batch()
    output = model(image_ids, pixel_values=pixels, labels=torch.full_like(image_ids, -100))
    output.loss.backward()
    assert output.loss.item() == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_real_processor_variable_images_preserve_complete_grids():
    import base64
    from io import BytesIO
    from pathlib import Path

    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from PIL import Image

    from nemo_automodel._transformers.tokenization.tokenization_mistral_common import MistralCommonBackend
    from nemo_automodel.components.models.mistral_large4.multimodal import image_token_positions

    tokenizer_path = Path(__file__).parent / "tiny_tekken.json"
    tokenizer = MistralCommonBackend(tokenizer_path)
    vendor = MistralTokenizer.from_file(tokenizer_path)
    content = [{"type": "text", "text": "Compare these images."}]
    for size in ((1, 1), (29, 55), (84, 28), (2000, 40)):
        stream = BytesIO()
        Image.new("RGB", size, (25, 70, 110)).save(stream, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    messages = [{"role": "user", "content": content}]
    actual = tokenizer.apply_chat_template(messages, return_tensors="pt")
    expected = vendor.encode_chat_completion(
        ChatCompletionRequest.from_openai(messages=messages, reasoning_effort="none")
    )
    assert actual["input_ids"][0].tolist() == expected.tokens
    for left, right in zip(actual["pixel_values"], expected.images, strict=True):
        torch.testing.assert_close(left, torch.from_numpy(right), atol=0, rtol=0)
    vc = MistralLarge4VisionConfig()
    positions = image_token_positions(actual["input_ids"], actual["pixel_values"], vc)
    assert positions.numel() == sum(a * b for a, b in (vc.grid(p) for p in actual["pixel_values"]))
    assert len(actual["pixel_values"]) == 4


def test_batched_unequal_images_and_padding_match_independent_documents():
    model = make_multimodal()
    _, pixels = image_batch()
    ids = torch.tensor([[1, 10, 10, 13, 2, 0, 0], [0, 3, 10, 12, 10, 13, 4]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1]])
    output = model(ids, pixel_values=pixels, attention_mask=mask).logits
    first = model(ids[:1, :5], pixel_values=pixels[:1]).logits
    second = model(ids[1:, 1:], pixel_values=pixels[1:]).logits
    torch.testing.assert_close(output[:1, :5], first, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output[1:, 1:], second, atol=2e-6, rtol=2e-5)
    # Swapping images with the same token count but different grids must fail.
    with pytest.raises(ValueError, match="complete grid"):
        model(ids, pixel_values=pixels[::-1], attention_mask=mask)
