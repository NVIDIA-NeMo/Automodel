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

"""Real Qwen vision/text forwards preserve batching across HF DeepStack formats."""

import copy

import pytest
import torch
from transformers import PreTrainedModel, Qwen3OmniMoeThinkerConfig, Qwen3VLMoeConfig
from transformers.initialization import no_init_weights

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_omni_moe.model import Qwen3OmniMoeThinkerForConditionalGeneration
from nemo_automodel.components.models.qwen3_vl_moe.model import Qwen3VLMoeForConditionalGeneration


@pytest.mark.parametrize("family", ["vl", "omni"])
@pytest.mark.parametrize("modality", ["text", "image", "video", "mixed"])
@torch.compiler.set_stance("force_eager")
def test_real_vision_features_preserve_outputs_and_gradients(family, modality):
    """Batched media match separate real forwards/backwards without CPU JIT startup."""
    torch.manual_seed(42)
    text = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        pad_token_id=0,
        dtype="float32",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
    )
    vision = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        dtype="float32",
    )
    if family == "vl":
        config = Qwen3VLMoeConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=10,
            video_token_id=11,
            vision_start_token_id=12,
            vision_end_token_id=13,
            dtype="float32",
        )
        model_class = Qwen3VLMoeForConditionalGeneration
    else:
        config = Qwen3OmniMoeThinkerConfig(
            text_config=text,
            vision_config=vision,
            audio_config=dict(
                d_model=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                output_dim=16,
                downsample_hidden_size=8,
            ),
            image_token_id=10,
            video_token_id=11,
            audio_token_id=14,
            pad_token_id=0,
            dtype="float32",
        )
        model_class = Qwen3OmniMoeThinkerForConditionalGeneration
    config._attn_implementation = "eager"
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    # Training initialization defaults to CUDA; exercise HF buffer initialization on CPU.
    with no_init_weights():
        model = model_class(config, backend=backend).float().eval()
    PreTrainedModel.initialize_weights(model)
    # Native grouped experts are initialized by the training loader, outside these constructors.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    reference = copy.deepcopy(model)
    examples = []
    for width in (2, 4):
        tokens = [1]
        inputs = {}
        if modality in ("image", "mixed"):
            tokens += [10] * (2 * width)
            inputs["pixel_values"] = torch.randn(2 * width, 12)
            inputs["image_grid_thw"] = torch.tensor([[1, 2, width]])
        if modality in ("video", "mixed"):
            tokens += [11] * (2 * width)
            inputs["pixel_values_videos"] = torch.randn(2 * width, 12)
            inputs["video_grid_thw"] = torch.tensor([[1, 2, width]])
        tokens += [2]
        inputs["input_ids"] = torch.tensor([tokens])
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["position_ids"] = torch.arange(len(tokens)).view(1, 1, -1).expand(3, 1, -1)
        examples.append(inputs)
    length = max(example["input_ids"].shape[1] for example in examples)
    batch = {
        "input_ids": torch.zeros(2, length, dtype=torch.long),
        "attention_mask": torch.zeros(2, length, dtype=torch.long),
        "position_ids": torch.arange(length).view(1, 1, -1).expand(3, 2, -1),
    }
    for index, example in enumerate(examples):
        count = example["input_ids"].shape[1]
        batch["input_ids"][index, :count] = example["input_ids"][0]
        batch["attention_mask"][index, :count] = 1
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in examples[0]:
            batch[key] = torch.cat([example[key] for example in examples], dim=0)
    actual = model(**batch)
    actual = actual.logits if family == "vl" else actual
    loss = 0
    for index, example in enumerate(examples):
        expected = reference(**example)
        expected = expected.logits if family == "vl" else expected
        count = example["input_ids"].shape[1]
        torch.testing.assert_close(actual[index, :count], expected[0], rtol=1e-5, atol=1e-6)
        upstream = torch.randn_like(expected)
        (expected * upstream).sum().backward()
        loss = loss + (actual[index, :count] * upstream[0]).sum()
    loss.backward()
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name).grad
        if expected is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6, msg=name)
    if modality != "text":
        visual = model.model.visual if family == "vl" else model.visual
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in visual.parameters())
