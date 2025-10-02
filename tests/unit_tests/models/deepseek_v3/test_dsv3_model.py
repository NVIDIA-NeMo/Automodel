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

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM


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


