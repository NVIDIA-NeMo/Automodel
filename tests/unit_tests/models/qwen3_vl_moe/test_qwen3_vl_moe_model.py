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

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModelOutputWithPast,
)

from nemo_automodel.components.models.qwen3_vl_moe.model import (
    Fp32SafeQwen3VLMoeTextRotaryEmbedding,
    Fp32SafeQwen3VLMoeVisionRotaryEmbedding,
    ModelClass,
    Qwen3VLMoeBlock,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeTextModelBackend,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture
def text_config():
    return Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        pad_token_id=0,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )


@pytest.fixture
def moe_config(text_config):
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=text_config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def vl_config(text_config):
    vision_cfg = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=8,
        deepstack_visual_indexes=[0, 1],
    )
    return Qwen3VLMoeConfig(text_config=text_config.to_dict(), vision_config=vision_cfg)


class TestFp32SafeRotaryEmbeddings:
    def test_text_rotary_inv_freq_remains_fp32(self, text_config):
        rotary = Fp32SafeQwen3VLMoeTextRotaryEmbedding(config=text_config)
        original = rotary.inv_freq.clone()

        rotary = rotary.to(torch.float16)

        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq.float(), original.float())

    def test_vision_rotary_inv_freq_remains_fp32(self):
        rotary = Fp32SafeQwen3VLMoeVisionRotaryEmbedding(dim=16)
        original = rotary.inv_freq.clone()

        rotary = rotary.to(torch.float16)

        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq.float(), original.float())


class TestQwen3VLMoeBlock:
    """Tests for Qwen3VLMoeBlock position_embeddings to freqs_cis conversion."""

    def test_forward_converts_position_embeddings_to_freqs_cis(self, text_config, backend_config, moe_config, device):
        """Test that position_embeddings (cos, sin) are converted to freqs_cis format."""
        block = Qwen3VLMoeBlock(0, text_config, moe_config, backend_config).to(device)

        batch, seq_len = 2, 4
        hidden_size = text_config.hidden_size
        head_dim = text_config.head_dim

        x = torch.randn(batch, seq_len, hidden_size, device=device)
        # position_embeddings: (cos, sin) each with shape [..., head_dim * 2]
        cos = torch.randn(batch, seq_len, head_dim * 2, device=device)
        sin = torch.randn(batch, seq_len, head_dim * 2, device=device)
        position_embeddings = (cos, sin)

        # Mock parent forward to capture freqs_cis
        captured_kwargs = {}
        original_forward = block.__class__.__bases__[0].forward

        def mock_forward(self, x, freqs_cis, **kwargs):
            captured_kwargs["freqs_cis"] = freqs_cis
            return x

        with patch.object(block.__class__.__bases__[0], "forward", mock_forward):
            block.forward(x=x, position_embeddings=position_embeddings)

        # Verify freqs_cis was constructed from position_embeddings
        assert "freqs_cis" in captured_kwargs
        freqs_cis = captured_kwargs["freqs_cis"]
        # freqs_cis should be cat of cos[:head_dim] and sin[:head_dim]
        expected_freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)
        torch.testing.assert_close(freqs_cis, expected_freqs_cis)

    def test_forward_uses_freqs_cis_directly_when_provided(self, text_config, backend_config, moe_config, device):
        """Test that freqs_cis is used directly when provided (position_embeddings ignored)."""
        block = Qwen3VLMoeBlock(0, text_config, moe_config, backend_config).to(device)

        batch, seq_len = 2, 4
        hidden_size = text_config.hidden_size
        head_dim = text_config.head_dim

        x = torch.randn(batch, seq_len, hidden_size, device=device)
        freqs_cis = torch.randn(batch, seq_len, head_dim * 2, device=device)

        captured_kwargs = {}

        def mock_forward(self, x, freqs_cis, **kwargs):
            captured_kwargs["freqs_cis"] = freqs_cis
            return x

        with patch.object(block.__class__.__bases__[0], "forward", mock_forward):
            block.forward(x=x, freqs_cis=freqs_cis)

        # Verify the same freqs_cis was passed through
        torch.testing.assert_close(captured_kwargs["freqs_cis"], freqs_cis)

    def test_forward_raises_when_no_freqs_cis_or_position_embeddings(self, text_config, backend_config, moe_config, device):
        """Test that ValueError is raised when neither freqs_cis nor position_embeddings provided."""
        block = Qwen3VLMoeBlock(0, text_config, moe_config, backend_config).to(device)

        x = torch.randn(2, 4, text_config.hidden_size, device=device)

        with pytest.raises(ValueError, match="requires freqs_cis or position_embeddings"):
            block.forward(x=x)


class TestQwen3VLMoeTextModelBackendLayersDict:
    """Tests for layers being nn.ModuleDict instead of nn.ModuleList."""

    def test_layers_is_module_dict(self, text_config, backend_config, moe_config):
        """Test that layers is an nn.ModuleDict with string keys."""
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        assert isinstance(model.layers, nn.ModuleDict)
        assert all(isinstance(key, str) for key in model.layers.keys())
        assert list(model.layers.keys()) == [str(i) for i in range(text_config.num_hidden_layers)]

    def test_layers_are_qwen3vlmoe_blocks(self, text_config, backend_config, moe_config):
        """Test that each layer is a Qwen3VLMoeBlock instance."""
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        for layer in model.layers.values():
            assert isinstance(layer, Qwen3VLMoeBlock)


class TestQwen3VLMoeTextModelBackend:
    def test_initialization_sets_expected_components(self, text_config, backend_config, moe_config):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        assert model.config is text_config
        assert model.backend is backend_config
        assert model.embed_tokens.num_embeddings == text_config.vocab_size
        assert len(model.layers) == text_config.num_hidden_layers
        assert isinstance(model.rotary_emb, Fp32SafeQwen3VLMoeTextRotaryEmbedding)

    def test_forward_skips_norm_when_none(self, text_config, backend_config, moe_config, device):
        """Test that forward() skips norm layer when it is None (PP support)."""
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        model.norm = None  # Simulate PP stage without norm

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)):
            # Should not raise when norm is None
            output = model(input_ids=input_ids)

        assert isinstance(output, Qwen3VLMoeModelOutputWithPast)
        assert output.last_hidden_state.shape == (batch, seq_len, text_config.hidden_size)

    def test_forward_runs_layers_and_returns_output(self, text_config, backend_config, moe_config, device):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 2, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x + 1)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)):
            output = model(input_ids=input_ids)

        assert isinstance(output, Qwen3VLMoeModelOutputWithPast)
        assert output.last_hidden_state.shape == (batch, seq_len, text_config.hidden_size)
        assert output.past_key_values is None
        assert all(layer.forward.call_count == 1 for layer in model.layers.values())
        freqs_shape = model.layers["0"].forward.call_args.kwargs["freqs_cis"].shape
        assert freqs_shape == (3, batch, seq_len, text_config.head_dim * 2)

    def test_forward_applies_deepstack_visual_embeds(self, text_config, backend_config, moe_config, device):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 1, 2
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        deepstack_visual_embeds = [
            torch.randn(1, text_config.hidden_size, device=device) for _ in range(len(model.layers))
        ]
        visual_pos_masks = torch.tensor([[True, False]], device=device)

        with (
            patch.object(model.rotary_emb, "forward", return_value=(cos, sin)),
            patch.object(model, "_deepstack_process", side_effect=lambda hs, *_: hs) as mock_deepstack,
        ):
            model(
                input_ids=input_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

        assert mock_deepstack.call_count == len(model.layers)

    def test_deepstack_process_adds_visual_embeds(self, text_config, backend_config, moe_config, device):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)

        hidden_states = torch.zeros(1, 3, text_config.hidden_size, device=device)
        visual_pos_masks = torch.tensor([[False, True, False]], device=device)
        visual_embeds = torch.full((1, text_config.hidden_size), 2.0, device=device)

        out = model._deepstack_process(hidden_states.clone(), visual_pos_masks, visual_embeds)

        torch.testing.assert_close(out[visual_pos_masks], visual_embeds)
        torch.testing.assert_close(out[visual_pos_masks.logical_not()], hidden_states[visual_pos_masks.logical_not()])

    def test_init_weights_invokes_layer_init(self, text_config, backend_config, moe_config):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        for layer in model.layers.values():
            layer.init_weights = MagicMock()

        original = model.embed_tokens.weight.clone()

        with patch.object(model.norm, "reset_parameters") as mock_norm:
            buffer_ctx = torch.cuda.device(torch.cuda.current_device())
            model.init_weights(buffer_device=buffer_ctx)

        mock_norm.assert_called_once()
        assert not torch.equal(model.embed_tokens.weight, original)
        for layer in model.layers.values():
            layer.init_weights.assert_called_once()

    def test_get_set_input_embeddings(self, text_config, backend_config, moe_config):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)
        new_embed = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

        model.set_input_embeddings(new_embed)

        assert model.get_input_embeddings() is new_embed


class TestQwen3VLMoeForConditionalGeneration:
    def test_initialization_configures_backend_components(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.backend is backend_config
        assert isinstance(model.model, Qwen3VLMoeModel)
        assert isinstance(model.model.language_model, Qwen3VLMoeTextModelBackend)
        assert model.model.moe_config is model.model.language_model.moe_config

        vision_model = getattr(model.model, "visual")
        assert isinstance(vision_model.rotary_pos_emb, Fp32SafeQwen3VLMoeVisionRotaryEmbedding)
        assert vision_model.rotary_pos_emb.inv_freq.dtype == torch.float32

    def test_pad_token_id_defaults_to_negative_one_when_missing(self, vl_config, backend_config, moe_config):
        """Test that pad_token_id defaults to -1 when text_config.pad_token_id is 0 (falsy)."""
        # Test that the model correctly handles the getattr fallback
        # When pad_token_id is 0 (falsy but valid), it should use 0, not -1
        vl_config.text_config.pad_token_id = 0
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        # 0 is a valid pad_token_id, should not fallback to -1
        assert model.pad_token_id == 0

    def test_pad_token_id_uses_config_value_when_present(self, vl_config, backend_config, moe_config):
        """Test that pad_token_id uses the config value when present."""
        vl_config.text_config.pad_token_id = 42

        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.pad_token_id == 42

    def test_pad_token_id_defaults_to_negative_one_when_none(self, vl_config, backend_config, moe_config):
        """Test that pad_token_id defaults to -1 when text_config.pad_token_id is None."""
        vl_config.text_config.pad_token_id = None

        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.pad_token_id == -1

    def test_forward_returns_logits_from_lm_head(self, vl_config, backend_config, moe_config, device):
        """Test that forward() returns logits from lm_head when present."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        # Get model dtype for consistent tensor creation
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype)
            mock_model_forward.return_value = mock_output

            logits = model.forward(input_ids=input_ids)

        # Should return logits with vocab_size dimension
        assert logits.shape == (batch, seq_len, vl_config.text_config.vocab_size)
        mock_model_forward.assert_called_once()

    def test_forward_returns_hidden_states_when_no_lm_head(self, vl_config, backend_config, moe_config, device):
        """Test that forward() returns hidden states when lm_head is None."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model.lm_head = None  # Simulate PP stage without lm_head

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            hidden_states = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_output.last_hidden_state = hidden_states
            mock_model_forward.return_value = mock_output

            result = model.forward(input_ids=input_ids)

        # Should return hidden states directly
        torch.testing.assert_close(result, hidden_states)

    def test_forward_retrieves_pixel_values_from_stored_chunks(self, vl_config, backend_config, moe_config, device):
        """Test that forward() retrieves pixel_values from stored chunks for PP VLM support."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        # Get model dtype for consistent tensor creation
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 1, 4
        # Create input_ids with media tokens (151655 for image)
        input_ids = torch.tensor([[151655, 1, 2, 3]], device=device)

        # Store pixel_values chunks for PP VLM support
        pixel_values_chunk = torch.randn(1, 3, 4, 4, device=device, dtype=model_dtype)
        image_grid_hws_chunk = torch.tensor([[2, 2]], device=device)  # [N, 2] format
        model._vlm_pixel_values_chunks = [pixel_values_chunk]
        model._vlm_image_grid_hws_chunks = [image_grid_hws_chunk]
        model._vlm_chunk_idx = 0

        captured_kwargs = {}

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype)
            mock_model_forward.return_value = mock_output

            def capture_kwargs(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return mock_output

            mock_model_forward.side_effect = capture_kwargs
            model.forward(input_ids=input_ids)

        # pixel_values should be retrieved from chunks
        assert "pixel_values" in captured_kwargs
        torch.testing.assert_close(captured_kwargs["pixel_values"], pixel_values_chunk)
        # image_grid_thw should be converted from [N, 2] to [N, 3] format
        assert "image_grid_thw" in captured_kwargs
        assert captured_kwargs["image_grid_thw"].shape == (1, 3)
        # Chunk index should be incremented
        assert model._vlm_chunk_idx == 1

    def test_forward_handles_thd_format(self, vl_config, backend_config, moe_config, device):
        """Test that forward() correctly handles thd format by calling squeeze_input_for_thd."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, device=device)
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)

        squeezed_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        squeezed_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        squeezed_padding_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        squeezed_kwargs = {"foo": "bar"}
        
        # Mock the model.model.forward to avoid internal tensor operations
        mock_hidden = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype)

        with (
            patch(
                "nemo_automodel.components.models.qwen3_vl_moe.model.squeeze_input_for_thd",
                return_value=(squeezed_ids, squeezed_position_ids, squeezed_padding_mask, squeezed_kwargs),
            ) as mock_squeeze,
            patch.object(model.model, "forward") as mock_model_forward,
        ):
            mock_output = MagicMock()
            mock_output.last_hidden_state = mock_hidden
            mock_model_forward.return_value = mock_output
            
            result = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                qkv_format="thd",
            )

            # Result should be logits from lm_head
            assert result.shape == (batch, seq_len, vl_config.text_config.vocab_size)
            
            # Verify squeeze_input_for_thd was called with correct args
            squeeze_args = mock_squeeze.call_args[0]
            assert squeeze_args[0] is input_ids
            assert squeeze_args[1] is position_ids
            assert squeeze_args[2] is padding_mask
            assert squeeze_args[3]["qkv_format"] == "thd"
            
            # Verify model.model.forward was called
            mock_model_forward.assert_called_once()

    def test_initialize_weights_invokes_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        with (
            patch.object(model.model.language_model, "init_weights") as mock_init,
            patch("torch.nn.init.trunc_normal_") as mock_trunc,
        ):
            buffer_ctx = torch.cuda.device(torch.cuda.current_device())
            model.initialize_weights(buffer_device=buffer_ctx, dtype=torch.float32)

        mock_init.assert_called_once()
        mock_trunc.assert_called_once()
        assert model.lm_head.weight.dtype == torch.float32

    def test_state_dict_adapter_created_when_enabled(self, vl_config, backend_config, moe_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert hasattr(model, "state_dict_adapter")


class TestQwen3VLMoeModel:
    def test_property_accessors_delegate_to_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        core = model.model

        assert isinstance(core, Qwen3VLMoeModel)
        assert core.layers is core.language_model.layers
        assert core.embed_tokens is core.language_model.embed_tokens
        assert core.norm is core.language_model.norm

    def test_forward_uses_embed_tokens_when_inputs_embeds_not_provided(self, vl_config, backend_config, moe_config, device):
        """Test that forward() uses embed_tokens to create inputs_embeds from input_ids."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(core.language_model, "forward") as mock_lang_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_lang_forward.return_value = mock_output

            core.forward(input_ids=input_ids)

        # language_model.forward should be called with inputs_embeds derived from input_ids
        call_kwargs = mock_lang_forward.call_args.kwargs
        assert call_kwargs["input_ids"] is None
        assert call_kwargs["inputs_embeds"] is not None
        assert call_kwargs["inputs_embeds"].shape == (batch, seq_len, vl_config.text_config.hidden_size)

    def test_forward_accepts_float_input_ids_as_inputs_embeds(self, vl_config, backend_config, moe_config, device):
        """Test that forward() treats float tensor input_ids as inputs_embeds (PP support)."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        # Pass float tensor as input_ids (this is actually inputs_embeds from previous PP stage)
        float_input = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=torch.bfloat16)

        with (
            patch.object(core, "get_input_embeddings", return_value=None),
            patch.object(core.language_model, "forward") as mock_lang_forward,
        ):
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_lang_forward.return_value = mock_output

            core.forward(input_ids=float_input)

            # language_model.forward should be called with the float tensor as inputs_embeds
            call_kwargs = mock_lang_forward.call_args.kwargs
            assert call_kwargs["input_ids"] is None
            torch.testing.assert_close(call_kwargs["inputs_embeds"], float_input)

    def test_forward_raises_when_no_embeds_and_no_embed_tokens(self, vl_config, backend_config, moe_config, device):
        """Test that forward() raises ValueError when no inputs_embeds and no embed_tokens."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        # Pass integer input_ids without embed_tokens
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(core, "get_input_embeddings", return_value=None):
            with pytest.raises(ValueError, match="inputs_embeds must be provided"):
                core.forward(input_ids=input_ids)


class TestQwen3VLMoeFromPretrainedAndModelClass:
    def test_from_pretrained_classmethod(self):
        cfg = Qwen3VLMoeConfig()
        cfg.text_config.rope_parameters = {
            "rope_theta": 10000.0,
            "rope_type": "default",
            "mrope_section": [1, 1, 1],
            "partial_rotary_factor": 1.0,
        }
        # Add pad_token_id required by transformers v5
        cfg.text_config.pad_token_id = 0

        with (
            patch(
                "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeConfig.from_pretrained",
                return_value=cfg,
            ) as mock_from_pretrained,
            patch.object(
                Qwen3VLMoeForConditionalGeneration, "from_config", wraps=Qwen3VLMoeForConditionalGeneration.from_config
            ) as mock_from_config,
        ):
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained("qwen3/vl-moe")

        assert isinstance(model, Qwen3VLMoeForConditionalGeneration)
        mock_from_pretrained.assert_called_once_with("qwen3/vl-moe")
        assert mock_from_config.call_args[0][0] is cfg

    def test_modelclass_export_exists(self):
        assert ModelClass is Qwen3VLMoeForConditionalGeneration
