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

"""Real FlashAttention kernels must preserve retrieval and ordinary Mistral VLM behavior."""

import copy
from types import MethodType

import pytest
import torch
from transformers import Mistral3Config, PixtralVisionModel
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
from nemo_automodel._transformers.retrieval import BiEncoderModel, CrossEncoderModel
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Mistral3BidirectionalConfig,
    Mistral3BidirectionalModel,
    Mistral3VLBidirectionalForSequenceClassification,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA FlashAttention kernels")
@pytest.mark.parametrize("attention", ["flash_attention_2", "flash_attention_3"])
@pytest.mark.parametrize("model_kind", ["embedding", "reranker", "vlm"])
def test_mistral3_vl_flash_attention_outputs_and_gradients(attention, model_kind):
    """Mixed-size images preserve embeddings/scores, gradients, and image isolation."""
    if attention == "flash_attention_2" and not is_flash_attn_2_available():
        pytest.skip("FlashAttention 2 is an optional GPU dependency")
    if attention == "flash_attention_3" and (
        not is_flash_attn_3_available() or torch.cuda.get_device_capability()[0] != 9
    ):
        pytest.skip("native FlashAttention 3 requires its optional package and a Hopper GPU")

    torch.manual_seed(42)
    config_cls = Mistral3Config if model_kind == "vlm" else Mistral3BidirectionalConfig
    config = config_cls(
        text_config={
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 128,
            "sliding_window": None,
            "attention_dropout": 0.0,
            "use_cache": False,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
            "attention_dropout": 0.0,
        },
        image_token_index=10,
        spatial_merge_size=1,
        num_labels=1,
        pooling="avg",
        temperature=1.0,
    )
    config._attn_implementation = attention
    if model_kind == "vlm":
        config.architectures = ["Mistral3ForConditionalGeneration"]
        model_cls = _resolve_custom_model_cls_for_config(config)
        assert model_cls is not None
        model = model_cls(config)
        backbone = model.model
    elif model_kind == "reranker":
        model = CrossEncoderModel(Mistral3VLBidirectionalForSequenceClassification(config))
        backbone = model.model.model
    else:
        model = BiEncoderModel(Mistral3BidirectionalModel(config), pooling="avg", l2_normalize=True)
        backbone = model.model
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    reference = copy.deepcopy(model)
    reference_backbone = reference.model.model if model_kind == "reranker" else reference.model
    reference_backbone.set_attn_implementation("sdpa")
    # Keep the oracle on the original upstream non-FlashAttention implementation.
    reference_backbone.vision_tower.forward = MethodType(PixtralVisionModel.forward, reference_backbone.vision_tower)
    inputs = {
        "input_ids": torch.tensor([[10, 10, 10, 10, 1, 2], [10, 10, 1, 2, 3, 0]], device="cuda"),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], device="cuda"),
        "pixel_values": torch.randn(2, 3, 28, 28, device="cuda", dtype=torch.bfloat16),
        "image_sizes": torch.tensor([[28, 28], [14, 28]], device="cuda"),
    }
    actual = model(**inputs).logits if model_kind == "vlm" else model(inputs)
    expected = reference(**inputs).logits if model_kind == "vlm" else reference(inputs)
    if model_kind == "reranker":
        actual, expected = actual.logits, expected.logits
    elif model_kind == "vlm":
        # Padded query logits have no semantic meaning and differ across kernels.
        valid_tokens = inputs["attention_mask"].bool()
        actual, expected = actual[valid_tokens], expected[valid_tokens]
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.002, rtol=0.02)
    upstream_gradient = torch.randn_like(expected)
    actual.backward(upstream_gradient)
    expected.backward(upstream_gradient)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected_gradient = reference_parameters[name].grad
        assert parameter.grad is not None, name
        assert expected_gradient is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        # BF16 kernel reductions may differ by a few rounding units. Scale the
        # absolute tolerance to each gradient so small tensors cannot pass as zero.
        atol = 0.02 * expected_gradient.float().abs().max().item()
        torch.testing.assert_close(parameter.grad.float(), expected_gradient.float(), atol=atol, rtol=0.02, msg=name)
    for tower in (backbone.vision_tower, backbone.multi_modal_projector, backbone.language_model):
        assert any(parameter.grad is not None and parameter.grad.abs().max() > 0 for parameter in tower.parameters())

    model.eval()
    with torch.no_grad():
        original = model(**inputs).logits if model_kind == "vlm" else model(inputs)
        inputs["pixel_values"][1].normal_()
        changed = model(**inputs).logits if model_kind == "vlm" else model(inputs)
    if model_kind == "reranker":
        original, changed = original.logits, changed.logits
    torch.testing.assert_close(original[0], changed[0], atol=0, rtol=0)
