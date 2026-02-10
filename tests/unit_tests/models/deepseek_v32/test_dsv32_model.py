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

    def test_config_rope_scaling(self):
        """Test that config handles rope_scaling parameter."""
        rope_scaling = {
            "factor": 2.0,
            "mscale": 1.0,
            "original_max_position_embeddings": 4096
        }
        cfg = DeepseekV32Config(rope_scaling=rope_scaling)

        assert cfg.rope_scaling == rope_scaling
        assert cfg.rope_scaling["factor"] == 2.0

    def test_config_keys_to_ignore_at_inference(self):
        """Test that keys_to_ignore_at_inference is set correctly."""
        cfg = DeepseekV32Config()
        assert cfg.keys_to_ignore_at_inference == ["past_key_values"]


class TestDeepseekV32Block:
    def create_moe_config(self):
        """Create a valid MoEConfig for tests."""
        from nemo_automodel.components.moe.config import MoEConfig
        return MoEConfig(
            dim=64,
            inter_dim=128,
            moe_inter_dim=64,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.001,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            norm_topk_prob=True,
        )

    def test_block_with_dense_layer(self):
        """Test DeepseekV32Block initialization with dense layer (layer_idx < first_k_dense_replace)."""
        from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32Block
        from nemo_automodel.components.models.common import BackendConfig

        cfg = DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=4,
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
            first_k_dense_replace=3,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
        )

        moe_config = self.create_moe_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        # Layer 0 should be dense (0 < 3)
        block = DeepseekV32Block(layer_idx=0, config=cfg, moe_config=moe_config, backend=backend)

        assert block.layer_idx == 0
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')

        # Check that MLP is dense (not MoE)
        from nemo_automodel.components.moe.layers import MLP
        assert isinstance(block.mlp, MLP)

    def test_block_with_moe_layer(self):
        """Test DeepseekV32Block initialization with MoE layer (layer_idx >= first_k_dense_replace)."""
        from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32Block
        from nemo_automodel.components.models.common import BackendConfig

        cfg = DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=4,
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
            first_k_dense_replace=3,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
        )

        moe_config = self.create_moe_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        # Layer 3 should be MoE (3 >= 3)
        block = DeepseekV32Block(layer_idx=3, config=cfg, moe_config=moe_config, backend=backend)

        assert block.layer_idx == 3

        # Check that MLP is MoE (not dense)
        from nemo_automodel.components.moe.layers import MoE
        assert isinstance(block.mlp, MoE)


class TestDeepseekV32Model:
    def test_model_initialization(self):
        """Test DeepseekV32Model initialization."""
        from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32Model
        from nemo_automodel.components.models.common import BackendConfig

        cfg = DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
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

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
        model = DeepseekV32Model(cfg, backend)

        assert hasattr(model, 'embed_tokens')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'freqs_cis')
        assert len(model.layers) == 2

    def test_model_initialization_with_moe_config(self):
        """Test DeepseekV32Model initialization with explicit MoE config."""
        from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32Model
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.moe.config import MoEConfig

        cfg = DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
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

        moe_config = MoEConfig(
            dim=64,
            inter_dim=128,
            moe_inter_dim=64,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.001,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            norm_topk_prob=True,
        )

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
        model = DeepseekV32Model(cfg, backend, moe_config=moe_config)

        assert model.moe_config == moe_config


class TestDeepseekV32ForCausalLM:
    def test_from_config_with_explicit_backend(self):
        """Test from_config with explicit backend parameter."""
        from nemo_automodel.components.models.common import BackendConfig

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

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
        model = DeepseekV32ForCausalLM.from_config(cfg, backend=backend)

        assert isinstance(model, DeepseekV32ForCausalLM)
        assert model.backend == backend

    def test_from_config_with_moe_config(self):
        """Test from_config with explicit MoE config."""
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.moe.config import MoEConfig

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

        moe_config = MoEConfig(
            dim=64,
            inter_dim=128,
            moe_inter_dim=64,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.001,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            norm_topk_prob=True,
        )

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
        model = DeepseekV32ForCausalLM.from_config(cfg, moe_config=moe_config, backend=backend)

        assert isinstance(model, DeepseekV32ForCausalLM)
        assert model.model.moe_config == moe_config

    def test_init_with_state_dict_adapter_disabled(self):
        """Test initialization without state dict adapter."""
        from nemo_automodel.components.models.common import BackendConfig

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

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)
        model = DeepseekV32ForCausalLM(cfg, backend=backend)

        assert not hasattr(model, 'state_dict_adapter')

    def test_init_with_state_dict_adapter_enabled(self):
        """Test initialization with state dict adapter enabled."""
        from nemo_automodel.components.models.common import BackendConfig

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

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=True)
        model = DeepseekV32ForCausalLM(cfg, backend=backend)

        assert hasattr(model, 'state_dict_adapter')
        from nemo_automodel.components.models.deepseek_v32.state_dict_adapter import DeepSeekV32StateDictAdapter
        assert isinstance(model.state_dict_adapter, DeepSeekV32StateDictAdapter)

    def test_model_has_lm_head(self):
        """Test that model has lm_head for language modeling."""
        from nemo_automodel.components.models.common import BackendConfig

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

        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
        model = DeepseekV32ForCausalLM(cfg, backend=backend)

        assert hasattr(model, 'lm_head')
        # lm_head should project from hidden_size to vocab_size
        assert model.lm_head.in_features == cfg.hidden_size
        assert model.lm_head.out_features == cfg.vocab_size
