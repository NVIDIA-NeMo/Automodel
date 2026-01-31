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

import sys
import types
import importlib.util
from unittest.mock import patch, MagicMock

# Mock fast_hadamard_transform before importing deepseek_v32 modules
if 'fast_hadamard_transform' not in sys.modules:
    mock_hadamard = types.ModuleType('fast_hadamard_transform')
    mock_hadamard.__spec__ = importlib.util.spec_from_loader('fast_hadamard_transform', loader=None)
    mock_hadamard.hadamard_transform = lambda x, scale: x
    sys.modules['fast_hadamard_transform'] = mock_hadamard

from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM


class TestDeepseekV32ModelUpdates:
    def test_from_pretrained_classmethod(self):
        """Ensure classmethod from_pretrained builds config then delegates to from_config."""
        cfg = DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=128,
            moe_intermediate_size=64,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
            qk_head_dim=32,
            kv_lora_rank=32,
            q_lora_rank=64,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
        )

        with patch.object(DeepseekV32Config, "from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = cfg

            with patch.object(DeepseekV32ForCausalLM, "from_config", wraps=DeepseekV32ForCausalLM.from_config) as mock_from_config:
                model = DeepseekV32ForCausalLM.from_pretrained("deepseek/model")
                assert isinstance(model, DeepseekV32ForCausalLM)
                mock_from_pretrained.assert_called_once_with("deepseek/model")
                called_cfg = mock_from_config.call_args[0][0]
                assert called_cfg is cfg

    def test_modelclass_export_exists(self):
        """Ensure ModelClass pointer is defined and points to class."""
        from nemo_automodel.components.models.deepseek_v32 import model as dsv32_mod

        assert hasattr(dsv32_mod, "ModelClass")
        assert dsv32_mod.ModelClass is DeepseekV32ForCausalLM


class TestDeepseekV32Config:
    def test_config_defaults(self):
        """Test that config has expected default values."""
        cfg = DeepseekV32Config()

        # V3.2 specific defaults
        assert cfg.q_lora_rank == 1536
        assert cfg.kv_lora_rank == 512
        assert cfg.qk_nope_head_dim == 128
        assert cfg.qk_rope_head_dim == 64
        assert cfg.v_head_dim == 128

        # Indexer defaults
        assert cfg.index_n_heads == 64
        assert cfg.index_head_dim == 128
        assert cfg.index_topk == 2048

        # Model type
        assert cfg.model_type == "deepseek_v32"

    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        cfg = DeepseekV32Config(
            hidden_size=256,
            q_lora_rank=128,
            index_topk=512,
        )

        assert cfg.hidden_size == 256
        assert cfg.q_lora_rank == 128
        assert cfg.index_topk == 512
