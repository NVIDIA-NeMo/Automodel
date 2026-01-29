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
from nemo_automodel.components.moe.layers import MoEConfig
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
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 1]},
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

    def test_forward_applies_deepstack_visual_embeds(
        self, text_config, backend_config, moe_config, device
    ):
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

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)), patch.object(
            model, "_deepstack_process", side_effect=lambda hs, *_: hs
        ) as mock_deepstack:
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
        torch.testing.assert_close(
            out[visual_pos_masks.logical_not()], hidden_states[visual_pos_masks.logical_not()]
        )

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

    def test_forward_handles_thd_format(self, vl_config, backend_config, moe_config, device):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, device=device)
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)

        squeezed_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        squeezed_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        squeezed_padding_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        squeezed_kwargs = {"foo": "bar"}

        with patch(
            "nemo_automodel.components.models.qwen3_vl_moe.model.squeeze_input_for_thd",
            return_value=(squeezed_ids, squeezed_position_ids, squeezed_padding_mask, squeezed_kwargs),
        ) as mock_squeeze, patch.object(
            HFQwen3VLMoeForConditionalGeneration, "forward", return_value="sentinel"
        ) as mock_super:
            result = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                qkv_format="thd",
            )

        assert result == "sentinel"
        squeeze_args = mock_squeeze.call_args[0]
        assert squeeze_args[0] is input_ids
        assert squeeze_args[1] is position_ids
        assert squeeze_args[2] is padding_mask
        assert squeeze_args[3]["qkv_format"] == "thd"

        super_kwargs = mock_super.call_args.kwargs
        assert super_kwargs["attention_mask"] is None
        assert super_kwargs["padding_mask"] is squeezed_padding_mask
        assert super_kwargs["position_ids"] is squeezed_position_ids
        assert super_kwargs["input_ids"] is squeezed_ids
        assert super_kwargs["foo"] == "bar"

    def test_initialize_weights_invokes_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        with patch.object(model.model.language_model, "init_weights") as mock_init, patch(
            "torch.nn.init.trunc_normal_"
        ) as mock_trunc:
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


class TestQwen3VLMoeFromPretrainedAndModelClass:
    def test_from_pretrained_classmethod(self):
        cfg = Qwen3VLMoeConfig()
        cfg.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 1]}

        with patch(
            "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeConfig.from_pretrained",
            return_value=cfg,
        ) as mock_from_pretrained, patch.object(
            Qwen3VLMoeForConditionalGeneration, "from_config", wraps=Qwen3VLMoeForConditionalGeneration.from_config
        ) as mock_from_config:
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained("qwen3/vl-moe")

        assert isinstance(model, Qwen3VLMoeForConditionalGeneration)
        mock_from_pretrained.assert_called_once_with("qwen3/vl-moe")
        assert mock_from_config.call_args[0][0] is cfg

    def test_modelclass_export_exists(self):
        assert ModelClass is Qwen3VLMoeForConditionalGeneration
