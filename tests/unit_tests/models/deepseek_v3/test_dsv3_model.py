# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import patch

import pytest
import torch

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM, DeepseekV3Model


class TestDeepseekV3ModelUpdates:
    def test_from_pretrained_classmethod(self):
        """Ensure classmethod from_pretrained builds config then delegates to from_config."""
        cfg = DeepseekV3Config(vocab_size=100, hidden_size=64, num_attention_heads=4, num_hidden_layers=1,
                               intermediate_size=128, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16)

        with patch("transformers.models.deepseek_v3.configuration_deepseek_v3.DeepseekV3Config.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = cfg

            with patch.object(DeepseekV3ForCausalLM, "from_config", wraps=DeepseekV3ForCausalLM.from_config) as mock_from_config:
                model = DeepseekV3ForCausalLM.from_pretrained("deepseek/model")
                assert isinstance(model, DeepseekV3ForCausalLM)
                mock_from_pretrained.assert_called_once_with("deepseek/model")
                called_cfg = mock_from_config.call_args[0][0]
                assert called_cfg is cfg

    def test_modelclass_export_exists(self):
        """Ensure ModelClass pointer is defined and points to class."""
        from nemo_automodel.components.models.deepseek_v3 import model as dsv3_mod

        assert hasattr(dsv3_mod, "ModelClass")
        assert dsv3_mod.ModelClass is DeepseekV3ForCausalLM


class TestDeepseekV3ModelInputsEmbeds:
    """Tests for inputs_embeds support in DeepseekV3Model."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return DeepseekV3Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=128,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
        )

    @pytest.fixture
    def backend(self):
        """Create a backend config for testing."""
        return BackendConfig(linear="torch", rms_norm="torch", attn="sdpa", rope_fusion=False)

    def test_forward_with_inputs_embeds_instead_of_input_ids(self, small_config, backend):
        """Test DeepseekV3Model accepts inputs_embeds instead of input_ids."""
        model = DeepseekV3Model(small_config, backend)
        model.eval()

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Should work with inputs_embeds and no input_ids
        with torch.no_grad():
            output = model(input_ids=None, inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_forward_raises_when_both_input_ids_and_inputs_embeds(self, small_config, backend):
        """Test DeepseekV3Model raises error when both input_ids and inputs_embeds provided."""
        model = DeepseekV3Model(small_config, backend)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        inputs_embeds = torch.randn(batch_size, seq_len, small_config.hidden_size)

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            model(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def test_forward_raises_when_neither_input_ids_nor_inputs_embeds(self, small_config, backend):
        """Test DeepseekV3Model raises error when neither input_ids nor inputs_embeds provided."""
        model = DeepseekV3Model(small_config, backend)

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            model(input_ids=None, inputs_embeds=None)
