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

from unittest.mock import Mock, patch

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.models.gpt_oss.model import Block, GptOssForCausalLM, GptOssModel
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def gpt_config():
    return GptOssConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=None,
        layer_types=["full_attention", "sliding_attention"],
        num_local_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        }
    )


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=128,
        inter_dim=256,
        moe_inter_dim=256,
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.01,
        norm_topk_prob=False,
        expert_bias=True,
        router_bias=True,
        expert_activation="quick_geglu",
        activation_alpha=1.702,
        activation_limit=7.0,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestBlock:
    """Test Block (transformer layer) module."""

    def test_block_init(self, gpt_config, moe_config, backend_config):
        """Test Block initialization."""
        layer_idx = 0
        block = Block(layer_idx, gpt_config, moe_config, backend_config)

        assert hasattr(block, "self_attn")
        assert hasattr(block, "mlp")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert isinstance(block.mlp, MoE)

    def test_block_init_sliding_attention(self, gpt_config, moe_config, backend_config):
        """Test Block initialization with sliding attention."""
        layer_idx = 1  # This should use sliding attention based on layer_types
        block = Block(layer_idx, gpt_config, moe_config, backend_config)

        # Verify sliding window is set correctly in attention
        assert block.self_attn.sliding_window == gpt_config.sliding_window

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_shape_preservation(self, gpt_config, moe_config, backend_config, device):
        """Test that Block forward preserves input shape."""
        block = Block(0, gpt_config, moe_config, backend_config)
        block = block.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)
        freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, device=device)

        with patch.object(block.self_attn.attn_module, "__call__") as mock_attn, \
             patch.object(block.mlp, "forward") as mock_mlp:
            # Mock attention output
            mock_attn.return_value = torch.randn(
                batch_size, gpt_config.num_attention_heads, seq_len, gpt_config.head_dim, device=device
            )
            # Mock MLP output (return just tensor, not tuple)
            mock_mlp.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            output = block(x, freqs_cis=freqs_cis)

            assert output.shape == x.shape
            assert output.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_with_attention_mask(self, gpt_config, moe_config, backend_config, device):
        """Test Block forward with attention mask."""
        block = Block(0, gpt_config, moe_config, backend_config)
        block = block.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)
        freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens

        with patch.object(block.self_attn.attn_module, "__call__") as mock_attn, \
             patch.object(block.mlp, "forward") as mock_mlp:
            mock_attn.return_value = torch.randn(
                batch_size, gpt_config.num_attention_heads, seq_len, gpt_config.head_dim, device=device
            )
            mock_mlp.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            output = block(x, freqs_cis=freqs_cis, attention_mask=attention_mask)

            # Verify that MLP received the correct padding mask
            mock_mlp.assert_called_once()
            args, kwargs = mock_mlp.call_args
            # Check if padding_mask is in call args (could be positional or keyword)
            if len(args) > 1:
                padding_mask = args[1]
            else:
                padding_mask = kwargs.get("padding_mask")
            assert padding_mask is not None

    def test_mlp_handling_regular_mlp(self, gpt_config, backend_config, device):
        """Test _mlp method with regular MLP."""
        # Create a config that would result in regular MLP
        moe_config = MoEConfig(
            dim=128, inter_dim=256, moe_inter_dim=256, n_routed_experts=0, n_shared_experts=1, n_activated_experts=1,
            n_expert_groups=1, n_limited_groups=1, train_gate=True, gate_bias_update_factor=0,
            score_func="softmax", route_scale=1.0, aux_loss_coeff=0.01, norm_topk_prob=False,
            expert_bias=True, router_bias=True, expert_activation="quick_geglu",
            activation_alpha=1.702, activation_limit=7.0,
        )

        block = Block(0, gpt_config, moe_config, backend_config)

        # Manually replace with regular MLP for testing
        block.mlp = MLP(dim=128, inter_dim=256, backend="torch")
        block = block.to(device)

        x = torch.randn(2, 8, 128, device=device)
        output = block._mlp(x, padding_mask=None)

        assert output.shape == x.shape
        assert output.device == device

    def test_init_weights(self, gpt_config, moe_config, backend_config, device):
        """Test Block weight initialization."""
        block = Block(0, gpt_config, moe_config, backend_config)

        with patch.object(block.input_layernorm, "reset_parameters") as mock_input_norm, \
             patch.object(block.post_attention_layernorm, "reset_parameters") as mock_post_norm, \
             patch.object(block.self_attn, "init_weights") as mock_attn_init, \
             patch.object(block.mlp, "init_weights") as mock_mlp_init:

            block.init_weights(device)

            mock_input_norm.assert_called_once()
            mock_post_norm.assert_called_once()
            mock_attn_init.assert_called_once_with(device)
            mock_mlp_init.assert_called_once_with(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGptOssModel:
    """Test GptOssModel."""

    def test_gpt_oss_model_init(self, gpt_config, backend_config):
        """Test GptOssModel initialization."""
        model = GptOssModel(gpt_config, backend_config)

        assert model.config == gpt_config
        assert model.backend == backend_config
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")
        assert hasattr(model, "rotary_emb")
        assert len(model.layers) == gpt_config.num_hidden_layers

    def test_gpt_oss_model_init_with_custom_moe_config(self, gpt_config, moe_config, backend_config):
        """Test GptOssModel initialization with custom MoE config."""
        model = GptOssModel(gpt_config, backend_config, moe_config=moe_config)

        assert model.moe_config == moe_config

    def test_embedding_dimensions(self, gpt_config, backend_config):
        """Test embedding layer dimensions."""
        model = GptOssModel(gpt_config, backend_config)

        assert model.embed_tokens.num_embeddings == gpt_config.vocab_size
        assert model.embed_tokens.embedding_dim == gpt_config.hidden_size

    def test_rotary_embedding_configuration(self, gpt_config, backend_config):
        """Test rotary embedding configuration."""
        model = GptOssModel(gpt_config, backend_config)

        assert model.rotary_emb.head_dim == gpt_config.head_dim
        # The initial_context_length comes from rope_scaling config, which defaults to max_seq_len
        # But rope_scaling might use different values, so let's check the actual rope_scaling used
        rope_scaling = getattr(gpt_config, "rope_scaling", None) or {
            "original_max_position_embeddings": model.max_seq_len,
        }
        expected_initial_length = rope_scaling["original_max_position_embeddings"]
        assert model.rotary_emb.initial_context_length == expected_initial_length

    def test_forward_shape_correctness(self, gpt_config, backend_config, device):
        """Test forward pass output shape."""
        model = GptOssModel(gpt_config, backend_config)
        model = model.to(device)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len), device=device)

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq") as mock_rope:
            # Mock rotary embedding computation
            mock_rope.return_value = (1.0, torch.randn(16, device=device))

            # Mock each layer's forward pass
            for layer in model.layers.values():
                with patch.object(layer, "forward") as mock_layer:
                    mock_layer.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            output = model(input_ids)

            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device

    def test_forward_with_position_ids(self, gpt_config, backend_config, device):
        """Test forward pass with custom position IDs."""
        model = GptOssModel(gpt_config, backend_config)
        model = model.to(device)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq") as mock_rope:
            mock_rope.return_value = (1.0, torch.randn(16, device=device))

            for layer in model.layers.values():
                with patch.object(layer, "forward") as mock_layer:
                    mock_layer.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            output = model(input_ids, position_ids=position_ids)

            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)

    def test_forward_with_padding_mask(self, gpt_config, backend_config, device):
        """Test forward pass with padding mask."""
        model = GptOssModel(gpt_config, backend_config)
        model = model.to(device)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len), device=device)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        padding_mask[:, -2:] = True  # Mask last 2 tokens

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq") as mock_rope:
            mock_rope.return_value = (1.0, torch.randn(16, device=device))

            for layer in model.layers.values():
                with patch.object(layer, "forward") as mock_layer:
                    mock_layer.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            output = model(input_ids, padding_mask=padding_mask)

            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)

    def test_init_weights(self, gpt_config, backend_config, device):
        """Test model weight initialization."""
        model = GptOssModel(gpt_config, backend_config)

        original_embed_weight = model.embed_tokens.weight.clone()

        with patch.object(model.norm, "reset_parameters") as mock_norm:
            for layer in model.layers.values():
                with patch.object(layer, "init_weights") as mock_layer_init:
                    pass

            model.init_weights(device)

            mock_norm.assert_called_once()
            # Verify embeddings changed
            assert not torch.equal(model.embed_tokens.weight, original_embed_weight)
            # Verify device was set on rotary embedding
            assert model.rotary_emb.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGptOssForCausalLM:
    """Test GptOssForCausalLM."""

    def test_from_config_with_string(self, gpt_config, backend_config):
        """Test from_config class method with string path."""
        with patch("transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = gpt_config

            model = GptOssForCausalLM.from_config("test-model", backend=backend_config)

            mock_from_pretrained.assert_called_once_with("test-model", trust_remote_code=False)
            assert isinstance(model, GptOssForCausalLM)

    def test_from_config_with_config_object(self, gpt_config, backend_config):
        """Test from_config class method with config object."""
        model = GptOssForCausalLM.from_config(gpt_config, backend=backend_config)

        assert isinstance(model, GptOssForCausalLM)
        assert model.config == gpt_config

    def test_gpt_oss_for_causal_lm_init(self, gpt_config, backend_config):
        """Test GptOssForCausalLM initialization."""
        model = GptOssForCausalLM(gpt_config, backend=backend_config)

        assert model.config == gpt_config
        assert model.backend == backend_config
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")
        assert isinstance(model.model, GptOssModel)

    def test_lm_head_dimensions(self, gpt_config, backend_config):
        """Test language modeling head dimensions."""
        model = GptOssForCausalLM(gpt_config, backend=backend_config)

        assert model.lm_head.in_features == gpt_config.hidden_size
        assert model.lm_head.out_features == gpt_config.vocab_size
        assert not hasattr(model.lm_head, "bias") or model.lm_head.bias is None

    def test_forward_output_shape(self, gpt_config, backend_config, device):
        """Test forward pass output shape."""
        model = GptOssForCausalLM(gpt_config, backend=backend_config)
        model = model.to(device)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model:
            mock_model.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            output = model(input_ids)

            assert output.shape == (batch_size, seq_len, gpt_config.vocab_size)
            assert output.device == device

    def test_initialize_weights(self, gpt_config, backend_config, device):
        """Test weight initialization."""
        model = GptOssForCausalLM(gpt_config, backend=backend_config)

        original_lm_head_weight = model.lm_head.weight.clone()

        with patch.object(model.model, "init_weights") as mock_model_init:
            model.initialize_weights(device, dtype=torch.float32)

            mock_model_init.assert_called_once_with(buffer_device=device)
            # Verify LM head weights changed
            assert not torch.equal(model.lm_head.weight, original_lm_head_weight)
            # Verify model was moved to correct dtype
            assert model.lm_head.weight.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialize_weights_gpu_specific(self, gpt_config, backend_config):
        """Test GPU-specific weight initialization."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        model = GptOssForCausalLM(gpt_config, backend=backend_config)

        with patch.object(model.model, "init_weights") as mock_model_init:
            model.initialize_weights(device, dtype=torch.bfloat16)

            mock_model_init.assert_called_once_with(buffer_device=device)
            # After initialization, move model to GPU and verify dtype
            model = model.to(device)
            assert model.lm_head.weight.device == device
            assert model.lm_head.weight.dtype == torch.bfloat16
            # Verify rotary embedding device is set correctly
            assert model.model.rotary_emb.device == device

    def test_state_dict_adapter_creation(self, gpt_config, backend_config):
        """Test state dict adapter creation when enabled."""
        backend_config.enable_hf_state_dict_adapter = True
        model = GptOssForCausalLM(gpt_config, backend=backend_config)

        assert hasattr(model, "state_dict_adapter")

    def test_forward_kwargs_passing(self, gpt_config, backend_config, device):
        """Test that forward passes kwargs correctly to underlying model."""
        model = GptOssForCausalLM(gpt_config, backend=backend_config)
        model = model.to(device)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with patch.object(model.model, "forward") as mock_model:
            mock_model.return_value = torch.randn(batch_size, seq_len, gpt_config.hidden_size, device=device)

            model(input_ids, attention_mask=attention_mask, custom_kwarg="test")

            # Verify model.forward was called with all arguments
            mock_model.assert_called_once()
            args, kwargs = mock_model.call_args
            assert "attention_mask" in kwargs
            assert "custom_kwarg" in kwargs
            assert kwargs["custom_kwarg"] == "test"
