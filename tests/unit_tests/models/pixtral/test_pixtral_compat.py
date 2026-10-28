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

"""Exercise the Pixtral backport through every AutoModel owner.

CI reruns this file with real Transformers 5.18.0 in addition to the project pin;
faking a version string alone cannot test its changed rotary-position contract.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Mistral3Config, PixtralVisionModel

from nemo_automodel.components.models.ministral_bidirectional.model import (
    Mistral3BidirectionalConfig,
    Mistral3BidirectionalModel,
)
from nemo_automodel.components.models.mistral3_vlm.model import Mistral3FP8VLMForConditionalGeneration
from nemo_automodel.components.models.pixtral import compat


@pytest.fixture(params=["retrieval", "mistral3", "mistral4"])
def vision_tower_factory(request):
    def make_tower(attention="eager"):
        text_config = dict(
            model_type="ministral3",
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=64,
        )
        vision_config = dict(
            model_type="pixtral",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=24,
            patch_size=8,
            attention_dropout=0.0,
        )
        config_cls = Mistral3BidirectionalConfig if request.param == "retrieval" else Mistral3Config
        config = config_cls(text_config=text_config, vision_config=vision_config, spatial_merge_size=1)
        config._attn_implementation = attention
        if request.param == "retrieval":
            return Mistral3BidirectionalModel(config).vision_tower
        if request.param == "mistral3":
            return Mistral3FP8VLMForConditionalGeneration(config).model.vision_tower

        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.mistral4.configuration import Mistral4Config
        from nemo_automodel.components.models.mistral4.model import Mistral3ForConditionalGeneration

        # Only exercise the real vision tower and its constructor binding here.
        config.text_config = Mistral4Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=0,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            max_position_embeddings=64,
        )
        backend = BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            rope_fusion=False,
            enable_hf_state_dict_adapter=False,
        )
        return Mistral3ForConditionalGeneration(config, backend=backend).model.vision_tower

    return make_tower


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("infer_sizes", [False, True])
def test_pixtral_backport_preserves_outputs_and_gradients(vision_tower_factory, attention, infer_sizes):
    """Mixed rectangular images retain upstream outputs, captured states, and gradients."""
    torch.manual_seed(42)
    tower = vision_tower_factory(attention).eval()
    stock = PixtralVisionModel(copy.deepcopy(tower.config)).eval()
    stock.load_state_dict(tower.state_dict())
    pixels = torch.randn(2, 3, 24, 24)
    sizes = None if infer_sizes else torch.tensor([[24, 16], [8, 24]])
    actual = tower(pixels, image_sizes=sizes, output_hidden_states=True)
    expected = stock(pixels, image_sizes=sizes, output_hidden_states=True)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=0, rtol=0)
    assert len(actual.hidden_states) == len(expected.hidden_states)
    for actual_hidden, expected_hidden in zip(actual.hidden_states, expected.hidden_states):
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=0, rtol=0)
    gradient = torch.randn_like(expected.last_hidden_state)
    actual.last_hidden_state.backward(gradient)
    expected.last_hidden_state.backward(gradient)
    for (name, parameter), (stock_name, stock_parameter) in zip(tower.named_parameters(), stock.named_parameters()):
        assert name == stock_name
        assert parameter.grad is not None, name
        assert stock_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, stock_parameter.grad, atol=1e-6, rtol=1e-5, msg=name)


# 5.19.1 is a future-version example, not a claim that it contains HF #49373.
@pytest.mark.parametrize("version", ["5.15.1", "5.17.0", "5.18.0", "5.19.0", "5.19.1"])
def test_pixtral_backport_is_instance_and_version_local(monkeypatch, vision_tower_factory, version):
    """All owners bind the fix only on affected releases; unrelated HF models stay native."""
    monkeypatch.setattr(compat, "transformers", SimpleNamespace(__version__=version))
    tower = vision_tower_factory()
    assert (tower.forward.__func__ is PixtralVisionModel.forward) == (version not in {"5.17.0", "5.18.0", "5.19.0"})
    stock = PixtralVisionModel(tower.config)
    assert stock.forward.__func__ is PixtralVisionModel.forward
