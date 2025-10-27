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

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

from nemo_automodel.components.models.glm4_moe.model import Block, Glm4MoeForCausalLM, Glm4MoeModel
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def glm_config():
    return Glm4MoeConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=2,  # First 2 layers are dense, rest are MoE
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_qk_norm=True,
        partial_rotary_factor=0.5,
        attention_bias=False,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def moe_config():
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
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
    )


class TestBlock:
    def test_block_uses_mlp_for_dense_layers(self, glm_config, moe_config, backend_config):
        """First k layers (< first_k_dense_replace) should use MLP"""
        block = Block(layer_idx=0, config=glm_config, moe_config=moe_config, backend=backend_config)

        assert isinstance(block.mlp, MLP)
        assert hasattr(block, "self_attn")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")

    def test_block_uses_moe_for_sparse_layers(self, glm_config, moe_config, backend_config):
        """Layers >= first_k_dense_replace should use MoE"""
        block = Block(layer_idx=2, config=glm_config, moe_config=moe_config, backend=backend_config)

        assert isinstance(block.mlp, MoE)

    def test_block_stores_layer_idx(self, glm_config, moe_config, backend_config):
        layer_idx = 3
        block = Block(layer_idx=layer_idx, config=glm_config, moe_config=moe_config, backend=backend_config)

        assert block.layer_idx == layer_idx

    def test_forward_pass_calls_attention_and_mlp(self, glm_config, backend_config, device):
        block = Block(layer_idx=0, config=glm_config, moe_config=magic_moe_config(glm_config), backend=backend_config)
        block = block.to(device)

        batch, seq_len = 2, 4
        x = torch.randn(batch, seq_len, glm_config.hidden_size, device=device)
        freqs_cis = torch.randn(batch, seq_len, int(glm_config.head_dim * glm_config.partial_rotary_factor), device=device)

        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn, \
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp:
            out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()

    def test_forward_builds_padding_mask_from_attention(self, glm_config, backend_config, device):
        block = Block(layer_idx=0, config=glm_config, moe_config=magic_moe_config(glm_config), backend=backend_config)
        block = block.to(device)

        x = torch.randn(1, 3, glm_config.hidden_size, device=device)
        freqs_cis = torch.randn(1, 3, int(glm_config.head_dim * glm_config.partial_rotary_factor), device=device)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)

        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn, \
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp:
            block(x, freqs_cis=freqs_cis, attention_mask=attention_mask)

        mock_attn.assert_called_once()
        _, kwargs = mock_mlp.call_args
        padding_mask = kwargs.get("padding_mask")
        assert padding_mask is not None
        torch.testing.assert_close(padding_mask, attention_mask.logical_not())

    def test_forward_uses_provided_padding_mask(self, glm_config, backend_config, device):
        """Test that if padding_mask is provided, it's used directly"""
        block = Block(layer_idx=0, config=glm_config, moe_config=magic_moe_config(glm_config), backend=backend_config)
        block = block.to(device)

        x = torch.randn(1, 3, glm_config.hidden_size, device=device)
        freqs_cis = torch.randn(1, 3, int(glm_config.head_dim * glm_config.partial_rotary_factor), device=device)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)
        padding_mask = torch.tensor([[0, 0, 1]], dtype=torch.bool, device=device)

        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn, \
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp:
            block(x, freqs_cis=freqs_cis, attention_mask=attention_mask, padding_mask=padding_mask)

        _, kwargs = mock_mlp.call_args
        received_padding_mask = kwargs.get("padding_mask")
        torch.testing.assert_close(received_padding_mask, padding_mask)

    def test_mlp_wrapper_handles_mlp_instance(self, glm_config, backend_config):
        block = Block(layer_idx=0, config=glm_config, moe_config=magic_moe_config(glm_config), backend=backend_config)
        x = torch.randn(2, 4, glm_config.hidden_size).to(torch.bfloat16)

        out = block._mlp(x, padding_mask=None)

        assert out.shape == x.shape

    def test_mlp_wrapper_handles_moe_instance(self, glm_config, backend_config):
        block = Block(layer_idx=2, config=glm_config, moe_config=magic_moe_config(glm_config), backend=backend_config)
        x = torch.randn(2, 4, glm_config.hidden_size).to(torch.bfloat16)
        padding_mask = torch.zeros(2, 4, dtype=torch.bool)

        with patch.object(block.mlp, "forward", return_value=torch.zeros_like(x)) as mock_moe:
            out = block._mlp(x, padding_mask=padding_mask)

        mock_moe.assert_called_once_with(x, padding_mask)
        assert out.shape == x.shape

    def test_init_weights_resets_sublayers(self, glm_config, backend_config):
        block = Block(layer_idx=0, config=glm_config, moe_config=magic_moe_config(glm_config), backend=backend_config)

        with patch.object(block.input_layernorm, "reset_parameters") as mock_in, \
            patch.object(block.post_attention_layernorm, "reset_parameters") as mock_post, \
            patch.object(block.self_attn, "init_weights") as mock_attn, \
            patch.object(block.mlp, "init_weights") as mock_mlp:
            block.init_weights(torch.device("cpu"))

        mock_in.assert_called_once()
        mock_post.assert_called_once()
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()


class TestGlm4MoeModel:
    def test_model_initialization_sets_components(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)

        assert model.config == glm_config
        assert model.backend == backend_config
        assert len(model.layers) == glm_config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == glm_config.vocab_size
        assert model.rotary_emb.head_dim == glm_config.head_dim

    def test_model_initializes_moe_config_with_sigmoid_scoring(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)

        assert hasattr(model, "moe_config")
        assert model.moe_config.dim == glm_config.hidden_size
        assert model.moe_config.n_routed_experts == glm_config.n_routed_experts
        assert model.moe_config.n_shared_experts == glm_config.n_shared_experts
        assert model.moe_config.n_activated_experts == glm_config.num_experts_per_tok
        assert model.moe_config.score_func == "sigmoid"  # GLM4 uses sigmoid
        assert model.moe_config.softmax_before_topk is False
        assert model.moe_config.route_scale == glm_config.routed_scaling_factor

    def test_model_initializes_moe_config_with_expert_groups(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)

        assert model.moe_config.n_expert_groups == glm_config.n_group
        assert model.moe_config.n_limited_groups == glm_config.topk_group

    def test_model_accepts_custom_moe_config(self, glm_config, backend_config, moe_config):
        model = Glm4MoeModel(glm_config, backend=backend_config, moe_config=moe_config)

        assert model.moe_config == moe_config

    def test_model_uses_partial_rotary_factor(self, glm_config, backend_config):
        glm_config.partial_rotary_factor = 0.75
        model = Glm4MoeModel(glm_config, backend=backend_config)

        assert model.rotary_emb.partial_rotary_factor == 0.75

    def test_forward_runs_all_layers(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)

        batch, seq_len = 2, 5
        input_ids = torch.randint(0, glm_config.vocab_size, (batch, seq_len))
        freqs_mock = MagicMock(return_value=(1.0, torch.ones(glm_config.head_dim // 2)))

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq", freqs_mock):
            with patch.object(Block, "forward", side_effect=lambda *_, **__: torch.randn(batch, seq_len, glm_config.hidden_size)) as mock_block:
                out = model(input_ids)

        assert out.shape == (batch, seq_len, glm_config.hidden_size)
        assert mock_block.call_count == glm_config.num_hidden_layers

    def test_forward_generates_position_ids_if_not_provided(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)
        batch, seq_len = 2, 4
        input_ids = torch.randint(0, glm_config.vocab_size, (batch, seq_len))

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq", return_value=(1.0, torch.ones(glm_config.head_dim // 2))):
            with patch.object(Block, "forward", side_effect=lambda *_, **kwargs: torch.randn(batch, seq_len, glm_config.hidden_size)):
                with patch("nemo_automodel.components.models.glm4_moe.model.position_ids_to_freqs_cis") as mock_freqs:
                    mock_freqs.return_value = torch.randn(batch, seq_len, int(glm_config.head_dim * glm_config.partial_rotary_factor))
                    out = model(input_ids)

        # Verify position_ids_to_freqs_cis was called
        mock_freqs.assert_called_once()
        call_args = mock_freqs.call_args
        position_ids = call_args[0][1]
        assert position_ids.shape == (batch, seq_len)
        expected_pos_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch, -1)
        torch.testing.assert_close(position_ids, expected_pos_ids)

    def test_forward_accepts_position_ids(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, glm_config.vocab_size, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq", return_value=(1.0, torch.ones(glm_config.head_dim // 2))):
            with patch.object(Block, "forward", return_value=torch.zeros(batch, seq_len, glm_config.hidden_size)):
                out = model(input_ids, position_ids=position_ids)

        assert out.shape == (batch, seq_len, glm_config.hidden_size)

    def test_forward_computes_freqs_cis_from_rotary_emb(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)
        batch, seq_len = 1, 3
        input_ids = torch.randint(0, glm_config.vocab_size, (batch, seq_len))

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq", return_value=(1.0, torch.ones(glm_config.head_dim // 2))):
            with patch("nemo_automodel.components.models.glm4_moe.model.position_ids_to_freqs_cis") as mock_freqs:
                mock_freqs.return_value = torch.randn(batch, seq_len, int(glm_config.head_dim * glm_config.partial_rotary_factor))
                with patch.object(Block, "forward", return_value=torch.zeros(batch, seq_len, glm_config.hidden_size)):
                    model(input_ids)

        mock_freqs.assert_called_once()
        assert mock_freqs.call_args[0][0] == model.rotary_emb

    def test_init_weights_updates_embeddings_and_layers(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)
        original = model.embed_tokens.weight.clone()

        with patch.object(model.norm, "reset_parameters") as mock_norm, \
            patch.object(Block, "init_weights") as mock_layer_init:
            model.init_weights(torch.device("cpu"))

        mock_norm.assert_called_once()
        assert not torch.equal(model.embed_tokens.weight, original)
        assert mock_layer_init.call_count == glm_config.num_hidden_layers

    def test_init_weights_updates_rotary_emb_device(self, glm_config, backend_config):
        model = Glm4MoeModel(glm_config, backend=backend_config)
        device = torch.device("cpu")

        with patch.object(model.norm, "reset_parameters"), \
            patch.object(Block, "init_weights"):
            model.init_weights(buffer_device=device)

        assert model.rotary_emb.device == device


class TestGlm4MoeForCausalLM:
    def test_forward_returns_logits(self, glm_config, backend_config, device):
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)
        model = model.to(device)

        batch, seq_len = 2, 6
        input_ids = torch.randint(0, glm_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward", return_value=torch.randn(batch, seq_len, glm_config.hidden_size, device=device).to(torch.bfloat16)):
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, glm_config.vocab_size)

    def test_forward_with_thd_format_squeezes_input(self, glm_config, backend_config, device):
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)
        model = model.to(device)

        batch, seq_len = 1, 5
        input_ids = torch.randint(0, glm_config.vocab_size, (batch, seq_len), device=device)

        with patch("nemo_automodel.components.models.glm4_moe.model.squeeze_input_for_thd") as mock_squeeze, \
             patch.object(model.model, "forward", return_value=torch.randn(seq_len, glm_config.hidden_size, device=device).to(torch.bfloat16)):
            mock_squeeze.return_value = (input_ids.squeeze(0), None, None, {"qkv_format": "thd"})
            logits = model(input_ids, qkv_format="thd")

        mock_squeeze.assert_called_once()
        # Output should be unsqueezed back to batch dimension
        assert logits.shape == (batch, seq_len, glm_config.vocab_size)

    def test_initialize_weights_invokes_submodules(self, glm_config, backend_config):
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)
        original = model.lm_head.weight.clone()

        with patch.object(model.model, "init_weights") as mock_init:
            model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        mock_init.assert_called_once()
        assert not torch.equal(model.lm_head.weight, original)
        assert model.lm_head.weight.dtype == torch.float32

    def test_initialize_weights_uses_scaled_std_for_lm_head(self, glm_config, backend_config):
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)

        with patch.object(model.model, "init_weights"), \
             patch("torch.nn.init.trunc_normal_") as mock_trunc:
            model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        # Check that trunc_normal_ was called with scaled std
        mock_trunc.assert_called()
        call_args = mock_trunc.call_args
        assert call_args[1]["std"] == glm_config.hidden_size ** -0.5

    def test_initialize_weights_sets_e_score_correction_bias_for_moe_layers(self, glm_config, backend_config):
        """GLM4 MoE initializes e_score_correction_bias for MoE layers"""
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)
        device = torch.device("cpu")

        with patch.object(model.model, "init_weights"):
            model.initialize_weights(buffer_device=device, dtype=torch.float32)

        # Check that MoE layers (>= first_k_dense_replace) have e_score_correction_bias
        for layer_idx, layer in enumerate(model.model.layers.values()):
            if isinstance(layer.mlp, MoE):
                assert layer_idx >= glm_config.first_k_dense_replace
                assert hasattr(layer.mlp.gate, "e_score_correction_bias")
                assert layer.mlp.gate.e_score_correction_bias.shape == (glm_config.n_routed_experts,)
                assert layer.mlp.gate.e_score_correction_bias.dtype == torch.float32
                torch.testing.assert_close(
                    layer.mlp.gate.e_score_correction_bias,
                    torch.zeros(glm_config.n_routed_experts, dtype=torch.float32)
                )

    def test_initialize_weights_updates_rotary_emb_device_after_dtype_move(self, glm_config, backend_config):
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)
        device = torch.device("cpu")

        with patch.object(model.model, "init_weights"):
            model.initialize_weights(buffer_device=device, dtype=torch.float32)

        assert model.model.rotary_emb.device == device

    def test_state_dict_adapter_created_when_enabled(self, glm_config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)

        assert hasattr(model, "state_dict_adapter")

    def test_state_dict_adapter_not_created_when_disabled(self, glm_config, backend_config):
        backend_config.enable_hf_state_dict_adapter = False
        model = Glm4MoeForCausalLM(glm_config, backend=backend_config)

        assert not hasattr(model, "state_dict_adapter")


class TestGlm4MoeModelClassmethods:
    def test_from_config_creates_model(self, glm_config, backend_config):
        model = Glm4MoeForCausalLM.from_config(glm_config, backend=backend_config)

        assert isinstance(model, Glm4MoeForCausalLM)
        assert model.config == glm_config
        assert model.backend == backend_config

    def test_from_pretrained_classmethod(self):
        """Ensure classmethod from_pretrained builds config then delegates to from_config."""
        cfg = Glm4MoeConfig(
            vocab_size=128,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            head_dim=16,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            first_k_dense_replace=1,
            moe_intermediate_size=64,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=False,
            use_qk_norm=True,
            partial_rotary_factor=0.5,
            attention_bias=False,
        )

        with patch("transformers.models.glm4_moe.configuration_glm4_moe.Glm4MoeConfig.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = cfg

            with patch.object(Glm4MoeForCausalLM, "from_config", wraps=Glm4MoeForCausalLM.from_config) as mock_from_config:
                model = Glm4MoeForCausalLM.from_pretrained("glm4_moe/model")
                assert isinstance(model, Glm4MoeForCausalLM)
                mock_from_pretrained.assert_called_once_with("glm4_moe/model")
                called_cfg = mock_from_config.call_args[0][0]
                assert called_cfg is cfg

    def test_modelclass_export_exists(self):
        """Ensure ModelClass pointer is defined and points to class."""
        from nemo_automodel.components.models.glm4_moe import model as glm_mod

        assert hasattr(glm_mod, "ModelClass")
        assert glm_mod.ModelClass is Glm4MoeForCausalLM


def magic_moe_config(config: Glm4MoeConfig) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
    )
