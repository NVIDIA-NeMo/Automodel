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

"""Image ownership, vision lifecycle and LoRA's independent K/V residual inputs."""

import copy

import pytest
import torch
from PIL import Image

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.attention import AttentionSources, Inkling2Attention
from nemo_automodel.components.models.inkling2.config import Inkling2Config, Inkling2VisionConfig
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.processing import image_patches
from nemo_automodel.components.models.inkling2.state_dict_adapter import Inkling2StateDictAdapter
from nemo_automodel.components.models.inkling2.vision import Inkling2VisionEncoder
from nemo_automodel.shared.peft_config import PeftConfig

from .test_reference import tiny_config

pytestmark = pytest.mark.cpu_only


def vision_config() -> Inkling2VisionConfig:
    """Return small channel dimensions with the released image patch geometry."""
    return Inkling2VisionConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_ratio=2.0,
        posemb_grid_size=4,
        projection_hidden_dim=24,
    )


def test_vision_images_do_not_mix_and_materialization() -> None:
    torch.manual_seed(31)
    config = vision_config()
    with torch.device("meta"):
        model = Inkling2VisionEncoder(config, 16, dtype=torch.float32)
    model.to_empty(device="cpu")
    model.reset_parameters()
    patches = torch.randn(5, 2, 40, 40, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 2], [1, 3, 1]])
    joint = model(patches, grid)
    separate = torch.cat((model(patches[:2], grid[:1]), model(patches[2:], grid[1:])))
    torch.testing.assert_close(joint, separate)
    assert torch.isfinite(joint).all()
    joint.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    changed = patches.clone()
    changed[2:] = 0
    torch.testing.assert_close(model(changed, grid)[:2], joint[:2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="patch counts"):
        model(patches, torch.tensor([[1, 1, 1]]))


def test_processor_image_geometry_padding_and_alpha() -> None:
    patches, grid = image_patches(Image.new("RGB", (41, 79), (255, 0, 128)))
    assert grid.tolist() == [[1, 2, 2]]
    assert patches.shape == (4, 2, 40, 40, 3) and patches.dtype == torch.bfloat16
    torch.testing.assert_close(patches[:, 0], patches[:, 1], rtol=0, atol=0)
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073))
    std = torch.tensor((0.26862954, 0.2613026, 0.2757771))
    torch.testing.assert_close(patches[-1, 0, -1, -1], ((-1 / 255 - mean) / std).bfloat16(), rtol=0, atol=0)
    transparent, _ = image_patches(Image.new("RGBA", (16, 8), (0, 0, 0, 0)))
    torch.testing.assert_close(transparent[0, 0, 0, 0], ((204 / 255 - mean) / std).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(transparent[0, 0, 0, 8], ((153 / 255 - mean) / std).bfloat16(), rtol=0, atol=0)


def test_global_attention_lora_matches_merged_projection_and_gradients() -> None:
    torch.manual_seed(41)
    cfg = tiny_config()
    adapted = Inkling2Attention(cfg, 5, dtype=torch.float32, backend="eager")
    adapted.reset_parameters(0.02)
    apply_lora_to_linear_modules(
        adapted,
        PeftConfig(
            target_modules=["qkvr"], dim=3, alpha=6, dropout=0, lora_dtype="float32", use_memory_efficient_lora=False
        ),
    )
    with torch.no_grad():
        adapted.qkvr.lora_B.weight.normal_(std=0.1)
    merged = Inkling2Attention(cfg, 5, dtype=torch.float32, backend="eager")
    merged.load_state_dict({k: v for k, v in adapted.state_dict().items() if "lora_" not in k})
    with torch.no_grad():
        merged.qkvr.weight.add_(adapted.qkvr.scale * adapted.qkvr.lora_B.weight @ adapted.qkvr.lora_A.weight)
    sources = AttentionSources(*(torch.randn(2, 7, cfg.hidden_size) for _ in range(3)))
    valid, positions = torch.ones(2, 7, dtype=torch.bool), torch.arange(7).expand(2, -1)
    actual, expected = adapted(sources, valid, positions), merged(sources, valid, positions)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    grad = merged.qkvr.weight.grad
    torch.testing.assert_close(
        adapted.qkvr.lora_B.weight.grad, adapted.qkvr.scale * grad @ adapted.qkvr.lora_A.weight.T, atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(
        adapted.qkvr.lora_A.weight.grad, adapted.qkvr.scale * adapted.qkvr.lora_B.weight.T @ grad, atol=3e-6, rtol=3e-5
    )


def test_multimodal_feature_ownership_and_checkpoint_roundtrip() -> None:
    text = tiny_config()
    text.num_hidden_layers = 2
    text.local_layer_ids = [0, 1]
    text.n_layers_per_attnres_block = 2
    text.vocab_size = 200058
    text.unpadded_vocab_size = 200058
    cfg = Inkling2Config(
        text_config=text, vision_config=vision_config(), architectures=["Inkling2ForConditionalGeneration"]
    )
    model = Inkling2ForConditionalGeneration(
        cfg,
        backend=BackendConfig(
            attn="eager",
            linear="torch",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=True,
        ),
    )
    ids = torch.tensor([[1, 200005, 200005, 3]])
    mask = torch.tensor([[0, 0, 1, 0]])
    patches, grid = image_patches(Image.new("RGB", (20, 20), (23, 17, 11)))
    features = model.visual(patches, grid)
    embedded = model.embed_norm(model.embed_tokens(ids).float())
    embedded[:, 2] = features
    expected = super(Inkling2ForConditionalGeneration, model).forward(ids, inputs_embeds=embedded, logits_to_keep=1)
    captured = []
    handle = model.layers["0"].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].partial.detach().clone())
    )
    actual = model(ids, vision_patches_bthwc=patches, vision_grid_thw=grid, mm_token_type_ids=mask, logits_to_keep=1)
    handle.remove()
    torch.testing.assert_close(captured[0][:, 2], features, rtol=0, atol=0)
    torch.testing.assert_close(captured[0][:, 1], embedded[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="counts disagree"):
        model(ids, vision_patches_bthwc=patches, vision_grid_thw=grid, mm_token_type_ids=(ids == 200005).long())
    adapter = Inkling2StateDictAdapter(cfg)
    exported = dict(
        pair for key, value in model.state_dict().items() for pair in adapter.convert_single_tensor_to_hf(key, value)
    )
    assert any(key.startswith("model.visual.") for key in exported)
    restored = adapter.from_hf(exported)
    clone = copy.deepcopy(model)
    clone.load_state_dict(restored, strict=True)
    torch.testing.assert_close(
        clone(ids, vision_patches_bthwc=patches, vision_grid_thw=grid, mm_token_type_ids=mask, logits_to_keep=1),
        actual,
        rtol=0,
        atol=0,
    )
    assert model.ModelCapabilities().supports_pp


def test_native_lora_checkpoint_layout_roundtrip() -> None:
    """Shared PEFT prefixes survive while fused/expert adapter axes stay native."""
    from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
    from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM

    config = tiny_config()
    model = Inkling2ForCausalLM(config)
    apply_lora_to_linear_modules(
        model,
        PeftConfig(
            target_modules=["*.attn.qkvr", "*.attn.wo_ud", "*.mlp.experts"],
            dim=8,
            alpha=32,
            lora_dtype="float32",
            use_memory_efficient_lora=False,
        ),
    )
    state = ModelState(model, is_peft=True).state_dict()
    assert len(state) == config.num_hidden_layers * 8
    adapter = Inkling2StateDictAdapter(config)
    base_keys = adapter.get_hf_state_dict_keys(model.state_dict())
    assert base_keys and not any("lora" in key for key in base_keys)
    exported = adapter.to_hf(state)
    restored = adapter.from_hf(exported)
    assert restored.keys() == state.keys()
    for name, value in restored.items():
        assert value is state[name]
    assert any("lora_gate_and_up_A" in name for name in exported)
    assert any("qkvr.lora_B.weight" in name for name in exported)
