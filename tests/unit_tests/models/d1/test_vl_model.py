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

"""D1-3B tied readout and causal image/text gradient contracts."""

import pytest
import torch
from transformers import Lfm2VlConfig

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.conversion_mapping import get_combined_key_mapping
from nemo_automodel.components.models.d1.decision_head import D1TokenReadoutConfig
from nemo_automodel.components.models.d1.vl_model import Lfm2VlForConditionalGeneration


def tiny_config():
    return Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        image_token_id=31,
        tie_word_embeddings=True,
        projector_hidden_size=64,
        projector_use_layernorm=False,
        projector_bias=True,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": False,
            "pad_token_id": 0,
            "use_cache": False,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "num_patches": 16,
            "vision_use_head": False,
        },
    )


def test_tie_identity_and_native_registry():
    model = Lfm2VlForConditionalGeneration(tiny_config())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert ModelRegistry.get_model_cls_from_model_arch("Lfm2VlForConditionalGeneration") is type(model)
    model.to(torch.bfloat16)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.model.language_model.rotary_emb.inv_freq.dtype == torch.float32


def test_checkpoint_conversion_preserves_released_parameter_names():
    model = Lfm2VlForConditionalGeneration(tiny_config())
    mapping = get_combined_key_mapping("lfm2_vl", {}, model=model)
    names = set(model.state_dict())
    converted = {mapping(name) if callable(mapping) else mapping.get(name, name) for name in names}
    assert converted == names


def test_causal_future_tokens_do_not_change_past():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lfm2VlForConditionalGeneration(tiny_config()).eval().to(device)
    ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    changed = torch.tensor([[1, 2, 3, 7, 8]], device=device)
    a = model(ids, use_cache=False).logits
    b = model(changed, use_cache=False).logits
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(a[:, 4], b[:, 4])


def test_image_context_has_gradients_and_frozen_adapter_preserves_tie():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lfm2VlForConditionalGeneration(tiny_config()).eval().to(device)
    pixels = torch.randn(1, 16, 768, requires_grad=True, device=device)
    ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3]], device=device)
    output = model(
        ids,
        pixel_values=pixels,
        spatial_shapes=torch.tensor([[4, 4]], device=device),
        pixel_attention_mask=torch.ones(1, 16, dtype=torch.int32, device=device),
        use_cache=False,
    )
    output.logits[:, -1, :3].sum().backward()
    assert pixels.grad.abs().sum() > 0
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    adapter = D1TokenReadoutConfig(64, (2, 3, 4)).build(model.get_output_embeddings().weight)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
    torch.nn.functional.cross_entropy(adapter(hidden, ((0, 1), (2,))), torch.tensor([1], device=device)).backward()
    assert adapter.delta_weight.grad.abs().sum() > 0
    assert model.get_output_embeddings().weight.grad is None
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


def test_rejects_untying_and_invalid_logit_loss_slicing():
    config = tiny_config()
    config.tie_word_embeddings = False
    with pytest.raises(NotImplementedError, match="tie_word_embeddings=False"):
        Lfm2VlForConditionalGeneration(config)
    model = Lfm2VlForConditionalGeneration(tiny_config())
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="logits_to_keep"):
        model(ids, labels=ids, logits_to_keep=1)
