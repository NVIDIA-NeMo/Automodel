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
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeModelOutputWithPast,
)

from nemo_automodel.components.models.qwen3_5_moe.model import (
    Fp32SafeQwen3_5MoeTextRotaryEmbedding,
    Fp32SafeQwen3_5MoeVisionRotaryEmbedding,
    ModelClass,
    Qwen3_5MoeBlock,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeTextModelBackend,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.models.common import BackendConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture
def text_config():
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        # All full_attention layers for testing simplicity
        layer_types=["full_attention", "full_attention"],
    )


@pytest.fixture
def text_config_with_linear(text_config):
    """Config that includes a linear_attention layer for testing Qwen3_5MoeBlock."""
    return Qwen3_5MoeTextConfig(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        num_hidden_layers=2,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        intermediate_size=64,
        moe_intermediate_size=text_config.moe_intermediate_size,
        shared_expert_intermediate_size=text_config.shared_expert_intermediate_size,
        num_experts=text_config.num_experts,
        num_experts_per_tok=text_config.num_experts_per_tok,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=["full_attention", "linear_attention"],
    )


@pytest.fixture
def moe_config(text_config):
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.hidden_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=1,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=text_config.shared_expert_intermediate_size,
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
    )
    return Qwen3_5MoeConfig(text_config=text_config.to_dict(), vision_config=vision_cfg)


# ---------------------------------------------------------------------------
# Fp32-safe rotary embedding tests
# ---------------------------------------------------------------------------
class TestFp32SafeRotaryEmbeddings:
    def test_text_rotary_inv_freq_remains_fp32(self, text_config):
        rotary = Fp32SafeQwen3_5MoeTextRotaryEmbedding(config=text_config)
        original = rotary.inv_freq.clone()

        rotary = rotary.to(torch.float16)

        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq.float(), original.float())

    def test_vision_rotary_inv_freq_remains_fp32(self):
        rotary = Fp32SafeQwen3_5MoeVisionRotaryEmbedding(dim=16)
        original = rotary.inv_freq.clone()

        rotary = rotary.to(torch.float16)

        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq.float(), original.float())


# ---------------------------------------------------------------------------
# Qwen3_5MoeBlock tests
# ---------------------------------------------------------------------------
class TestQwen3_5MoeBlock:
    def test_full_attention_block_has_self_attn(self, text_config, moe_config, backend_config):
        block = Qwen3_5MoeBlock(0, text_config, moe_config, backend_config)

        assert hasattr(block, "self_attn")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")

    def test_linear_attention_block_has_native_gated_delta_net(self, text_config_with_linear, moe_config, backend_config):
        """Layer 1 is linear_attention — should use HF Qwen3_5MoeGatedDeltaNet."""
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

        block = Qwen3_5MoeBlock(1, text_config_with_linear, moe_config, backend_config)

        assert hasattr(block, "linear_attn")
        assert isinstance(block.linear_attn, Qwen3_5MoeGatedDeltaNet)

    def test_init_weights_full_attention(self, text_config, moe_config, backend_config):
        block = Qwen3_5MoeBlock(0, text_config, moe_config, backend_config)

        with (
            patch.object(block.input_layernorm, "reset_parameters") as mock_in_norm,
            patch.object(block.post_attention_layernorm, "reset_parameters") as mock_post_norm,
            patch.object(block.self_attn, "init_weights") as mock_attn,
            patch.object(block.mlp, "init_weights") as mock_mlp,
        ):
            block.init_weights(torch.device("cpu"))

        mock_in_norm.assert_called_once()
        mock_post_norm.assert_called_once()
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()

    def test_init_weights_linear_attention_initializes_projections(self, text_config_with_linear, moe_config, backend_config):
        block = Qwen3_5MoeBlock(1, text_config_with_linear, moe_config, backend_config)

        with (
            patch.object(block.input_layernorm, "reset_parameters"),
            patch.object(block.post_attention_layernorm, "reset_parameters"),
            patch.object(block.mlp, "init_weights"),
            patch("torch.nn.init.trunc_normal_") as mock_trunc,
            patch.object(block.linear_attn.norm, "reset_parameters") as mock_norm_reset,
        ):
            block.init_weights(torch.device("cpu"))

        # 5 linear projections should be initialized: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
        assert mock_trunc.call_count == 5
        mock_norm_reset.assert_called_once()


# ---------------------------------------------------------------------------
# TextModelBackend tests
# ---------------------------------------------------------------------------
class TestQwen3_5MoeTextModelBackendLayersDict:
    def test_layers_is_module_dict(self, text_config, backend_config, moe_config):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        assert isinstance(model.layers, nn.ModuleDict)
        assert all(isinstance(key, str) for key in model.layers.keys())
        assert list(model.layers.keys()) == [str(i) for i in range(text_config.num_hidden_layers)]

    def test_layers_are_qwen3_5_moe_blocks(self, text_config, backend_config, moe_config):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        for layer in model.layers.values():
            assert isinstance(layer, Qwen3_5MoeBlock)


class TestQwen3_5MoeTextModelBackend:
    def test_initialization_sets_expected_components(self, text_config, backend_config, moe_config):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        assert model.config is text_config
        assert model.backend is backend_config
        assert model.embed_tokens.num_embeddings == text_config.vocab_size
        assert len(model.layers) == text_config.num_hidden_layers
        assert isinstance(model.rotary_emb, Fp32SafeQwen3_5MoeTextRotaryEmbedding)

    def test_moe_config_defaults_when_not_provided(self, text_config, backend_config):
        """MoE config should be created automatically from text_config when not provided."""
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config)

        assert model.moe_config is not None
        assert model.moe_config.n_routed_experts == text_config.num_experts
        assert model.moe_config.n_activated_experts == text_config.num_experts_per_tok
        assert model.moe_config.n_shared_experts == 1
        assert model.moe_config.shared_expert_gate is True

    def test_forward_raises_on_kv_cache(self, text_config, backend_config, moe_config, device):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)

        input_ids = torch.randint(0, text_config.vocab_size, (1, 4), device=device)

        with pytest.raises(NotImplementedError, match="KV cache is not supported"):
            model(input_ids=input_ids, past_key_values=MagicMock())

    def test_forward_raises_on_use_cache(self, text_config, backend_config, moe_config, device):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)

        input_ids = torch.randint(0, text_config.vocab_size, (1, 4), device=device)

        with pytest.raises(NotImplementedError, match="KV cache is not supported"):
            model(input_ids=input_ids, use_cache=True)

    def test_forward_skips_norm_when_none(self, text_config, backend_config, moe_config, device):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        model.norm = None

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)):
            output = model(input_ids=input_ids)

        assert isinstance(output, Qwen3_5MoeModelOutputWithPast)
        assert output.last_hidden_state.shape == (batch, seq_len, text_config.hidden_size)

    def test_forward_runs_layers_and_returns_output(self, text_config, backend_config, moe_config, device):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 2, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x + 1)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)):
            output = model(input_ids=input_ids)

        assert isinstance(output, Qwen3_5MoeModelOutputWithPast)
        assert output.last_hidden_state.shape == (batch, seq_len, text_config.hidden_size)
        assert output.past_key_values is None
        assert output.rope_deltas is None
        assert all(layer.forward.call_count == 1 for layer in model.layers.values())

    def test_forward_handles_4d_position_ids(self, text_config, backend_config, moe_config, device):
        """4D position_ids [text, T, H, W] should strip the text dim, keeping [T, H, W]."""
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 1, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)
        # 4D position_ids: [4, batch, seq_len]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(4, batch, -1).to(device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)) as mock_rotary:
            model(input_ids=input_ids, position_ids=position_ids)

        # rotary_emb should receive [3, batch, seq_len] position_ids (text dim stripped)
        call_args = mock_rotary.call_args
        passed_pos = call_args[0][1]
        assert passed_pos.shape[0] == 3

    def test_forward_handles_2d_position_ids(self, text_config, backend_config, moe_config, device):
        """2D position_ids [batch, seq_len] should be expanded to [3, batch, seq_len]."""
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 1, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)
        position_ids = torch.arange(seq_len).unsqueeze(0).to(device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)) as mock_rotary:
            model(input_ids=input_ids, position_ids=position_ids)

        call_args = mock_rotary.call_args
        passed_pos = call_args[0][1]
        assert passed_pos.shape[0] == 3

    def test_init_weights_invokes_layer_init(self, text_config, backend_config, moe_config):
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

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
        model = Qwen3_5MoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)
        new_embed = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

        model.set_input_embeddings(new_embed)

        assert model.get_input_embeddings() is new_embed


# ---------------------------------------------------------------------------
# Qwen3_5MoeModel (VL wrapper) tests
# ---------------------------------------------------------------------------
class TestQwen3_5MoeModel:
    def test_property_accessors_delegate_to_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        core = model.model

        assert isinstance(core, Qwen3_5MoeModel)
        assert core.layers is core.language_model.layers
        assert core.embed_tokens is core.language_model.embed_tokens
        assert core.norm is core.language_model.norm

    def test_forward_uses_embed_tokens_when_inputs_embeds_not_provided(self, vl_config, backend_config, moe_config, device):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(core.language_model, "forward") as mock_lang_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_lang_forward.return_value = mock_output

            core.forward(input_ids=input_ids)

        call_kwargs = mock_lang_forward.call_args.kwargs
        assert call_kwargs["input_ids"] is None
        assert call_kwargs["inputs_embeds"] is not None
        assert call_kwargs["inputs_embeds"].shape == (batch, seq_len, vl_config.text_config.hidden_size)

    def test_forward_accepts_float_input_ids_as_inputs_embeds(self, vl_config, backend_config, moe_config, device):
        """Float tensor input_ids are treated as inputs_embeds (pipeline-parallel support)."""
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        float_input = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=torch.bfloat16)

        with (
            patch.object(core, "get_input_embeddings", return_value=None),
            patch.object(core.language_model, "forward") as mock_lang_forward,
        ):
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_lang_forward.return_value = mock_output

            core.forward(input_ids=float_input)

        call_kwargs = mock_lang_forward.call_args.kwargs
        assert call_kwargs["input_ids"] is None
        torch.testing.assert_close(call_kwargs["inputs_embeds"], float_input)

    def test_forward_raises_when_no_embeds_and_no_embed_tokens(self, vl_config, backend_config, moe_config, device):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (2, 3), device=device)

        with patch.object(core, "get_input_embeddings", return_value=None):
            with pytest.raises(ValueError, match="inputs_embeds must be provided"):
                core.forward(input_ids=input_ids)


# ---------------------------------------------------------------------------
# Top-level conditional generation model tests
# ---------------------------------------------------------------------------
class TestQwen3_5MoeForConditionalGeneration:
    def test_initialization_configures_backend_components(self, vl_config, backend_config, moe_config):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.backend is backend_config
        assert isinstance(model.model, Qwen3_5MoeModel)
        assert isinstance(model.model.language_model, Qwen3_5MoeTextModelBackend)
        assert model.model.moe_config is model.model.language_model.moe_config

        vision_model = getattr(model.model, "visual")
        assert isinstance(vision_model.rotary_pos_emb, Fp32SafeQwen3_5MoeVisionRotaryEmbedding)
        assert vision_model.rotary_pos_emb.inv_freq.dtype == torch.float32

    def test_pad_token_id_uses_config_value(self, vl_config, backend_config, moe_config):
        vl_config.text_config.pad_token_id = 42
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        assert model.pad_token_id == 42

    def test_pad_token_id_defaults_to_negative_one_when_none(self, vl_config, backend_config, moe_config):
        vl_config.text_config.pad_token_id = None
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        assert model.pad_token_id == -1

    def test_forward_returns_logits_from_lm_head(self, vl_config, backend_config, moe_config, device):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype
            )
            mock_model_forward.return_value = mock_output

            logits = model.forward(input_ids=input_ids)

        assert logits.shape == (batch, seq_len, vl_config.text_config.vocab_size)
        mock_model_forward.assert_called_once()

    def test_forward_returns_hidden_states_when_no_lm_head(self, vl_config, backend_config, moe_config, device):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model.lm_head = None

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            hidden_states = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_output.last_hidden_state = hidden_states
            mock_model_forward.return_value = mock_output

            result = model.forward(input_ids=input_ids)

        torch.testing.assert_close(result, hidden_states)

    def test_forward_retrieves_pixel_values_from_stored_chunks(self, vl_config, backend_config, moe_config, device):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 1, 4
        input_ids = torch.tensor([[vl_config.image_token_id, 1, 2, 3]], device=device)

        pixel_values_chunk = torch.randn(1, 3, 4, 4, device=device, dtype=model_dtype)
        image_grid_hws_chunk = torch.tensor([[2, 2]], device=device)
        model._vlm_pixel_values_chunks = [pixel_values_chunk]
        model._vlm_image_grid_hws_chunks = [image_grid_hws_chunk]
        model._vlm_chunk_idx = 0

        captured_kwargs = {}

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype
            )

            def capture_kwargs(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return mock_output

            mock_model_forward.side_effect = capture_kwargs
            model.forward(input_ids=input_ids)

        assert "pixel_values" in captured_kwargs
        torch.testing.assert_close(captured_kwargs["pixel_values"], pixel_values_chunk)
        assert "image_grid_thw" in captured_kwargs
        assert captured_kwargs["image_grid_thw"].shape == (1, 3)
        assert model._vlm_chunk_idx == 1

    def test_forward_handles_thd_format(self, vl_config, backend_config, moe_config, device):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
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

        mock_hidden = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype)

        with (
            patch(
                "nemo_automodel.components.models.qwen3_5_moe.model.squeeze_input_for_thd",
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

        assert result.shape == (batch, seq_len, vl_config.text_config.vocab_size)
        squeeze_args = mock_squeeze.call_args[0]
        assert squeeze_args[0] is input_ids
        assert squeeze_args[1] is position_ids
        assert squeeze_args[2] is padding_mask
        assert squeeze_args[3]["qkv_format"] == "thd"

    def test_initialize_weights_invokes_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

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
        model = Qwen3_5MoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert hasattr(model, "state_dict_adapter")


# ---------------------------------------------------------------------------
# from_pretrained / ModelClass export tests
# ---------------------------------------------------------------------------
class TestQwen3_5MoeFromPretrainedAndModelClass:
    def test_from_pretrained_classmethod(self):
        cfg = Qwen3_5MoeConfig()
        cfg.text_config.pad_token_id = 0

        with (
            patch(
                "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeConfig.from_pretrained",
                return_value=cfg,
            ) as mock_from_pretrained,
            patch.object(
                Qwen3_5MoeForConditionalGeneration, "from_config", wraps=Qwen3_5MoeForConditionalGeneration.from_config
            ) as mock_from_config,
        ):
            model = Qwen3_5MoeForConditionalGeneration.from_pretrained("qwen3_5/moe")

        assert isinstance(model, Qwen3_5MoeForConditionalGeneration)
        mock_from_pretrained.assert_called_once_with("qwen3_5/moe")
        assert mock_from_config.call_args[0][0] is cfg

    def test_modelclass_export_exists(self):
        assert ModelClass is Qwen3_5MoeForConditionalGeneration
