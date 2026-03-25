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
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4ForConditionalGeneration,
    Gemma4Gate,
    Gemma4MoE,
    Gemma4MoEDecoderLayer,
    Gemma4MoEModel,
    Gemma4MoETextModelBackend,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _make_text_config(**overrides):
    """Build a minimal Gemma4TextConfig for unit tests."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=256,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        expert_intermediate_size=64,
        layer_types=["full_attention", "sliding_attention"] * 2,
        sliding_window=128,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _make_gemma4_config(**text_overrides):
    """Build a Gemma4Config wrapping a Gemma4TextConfig."""
    text_cfg = _make_text_config(**text_overrides)
    return Gemma4Config(text_config=text_cfg)


def _make_moe_config(text_config=None):
    """Build the MoEConfig that matches the default test text config."""
    tc = text_config or _make_text_config()
    return MoEConfig(
        dim=tc.hidden_size,
        inter_dim=tc.intermediate_size,
        moe_inter_dim=tc.expert_intermediate_size,
        n_routed_experts=tc.num_experts,
        n_shared_experts=0,
        n_activated_experts=tc.top_k_experts,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_activation="geglu",
        softmax_before_topk=False,
    )


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def text_config():
    return _make_text_config()


@pytest.fixture
def gemma4_config():
    return _make_gemma4_config()


@pytest.fixture
def dense_config():
    return _make_gemma4_config(enable_moe_block=False)


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
def moe_config(text_config):
    return _make_moe_config(text_config)


# ---------------------------------------------------------------------------
# Gemma4Gate tests
# ---------------------------------------------------------------------------
class TestGemma4Gate:
    def test_init_creates_expected_submodules(self, text_config):
        gate = Gemma4Gate(text_config)

        assert hasattr(gate, "norm")
        assert hasattr(gate, "proj")
        assert hasattr(gate, "scale")
        assert hasattr(gate, "root_size")
        assert gate.topk == text_config.top_k_experts
        assert gate.num_experts == text_config.num_experts

    def test_proj_output_features_match_num_experts(self, text_config):
        gate = Gemma4Gate(text_config)
        assert gate.proj.out_features == text_config.num_experts
        assert gate.proj.in_features == text_config.hidden_size

    def test_root_size_value(self, text_config):
        gate = Gemma4Gate(text_config)
        expected = text_config.hidden_size ** -0.5
        torch.testing.assert_close(gate.root_size, torch.tensor(expected))

    def test_scale_initialized_to_ones(self, text_config):
        gate = Gemma4Gate(text_config)
        torch.testing.assert_close(gate.scale, torch.ones(text_config.hidden_size))

    def test_forward_output_shapes(self, text_config):
        gate = Gemma4Gate(text_config)
        batch, seq = 2, 8
        x = torch.randn(batch, seq, text_config.hidden_size)

        weights, indices, aux_loss = gate(x)

        assert weights.shape == (batch, seq, text_config.top_k_experts)
        assert indices.shape == (batch, seq, text_config.top_k_experts)
        assert aux_loss is None

    def test_forward_weights_are_normalized(self, text_config):
        gate = Gemma4Gate(text_config)
        x = torch.randn(2, 4, text_config.hidden_size)

        weights, _, _ = gate(x)

        weight_sums = weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-5, rtol=1e-5)

    def test_forward_weights_are_non_negative(self, text_config):
        gate = Gemma4Gate(text_config)
        x = torch.randn(3, 6, text_config.hidden_size)

        weights, _, _ = gate(x)

        assert (weights >= 0).all()

    def test_forward_indices_within_expert_range(self, text_config):
        gate = Gemma4Gate(text_config)
        x = torch.randn(2, 4, text_config.hidden_size)

        _, indices, _ = gate(x)

        assert (indices >= 0).all()
        assert (indices < text_config.num_experts).all()

    def test_init_weights_is_noop(self, text_config):
        gate = Gemma4Gate(text_config)
        scale_before = gate.scale.clone()
        gate.init_weights(torch.device("cpu"))
        torch.testing.assert_close(gate.scale, scale_before)


# ---------------------------------------------------------------------------
# Gemma4MoE tests
# ---------------------------------------------------------------------------
class TestGemma4MoE:
    def test_is_moe_subclass(self, moe_config, backend_config, text_config):
        moe = Gemma4MoE(moe_config, backend_config, text_config)
        assert isinstance(moe, MoE)

    def test_gate_is_gemma4_gate(self, moe_config, backend_config, text_config):
        moe = Gemma4MoE(moe_config, backend_config, text_config)
        assert isinstance(moe.gate, Gemma4Gate)

    def test_gate_topk_matches_config(self, moe_config, backend_config, text_config):
        moe = Gemma4MoE(moe_config, backend_config, text_config)
        assert moe.gate.topk == text_config.top_k_experts

    def test_gate_num_experts_matches_config(self, moe_config, backend_config, text_config):
        moe = Gemma4MoE(moe_config, backend_config, text_config)
        assert moe.gate.num_experts == text_config.num_experts


# ---------------------------------------------------------------------------
# Gemma4MoEDecoderLayer tests
# ---------------------------------------------------------------------------
class TestGemma4MoEDecoderLayer:
    def test_init_creates_submodules(self, text_config, moe_config, backend_config):
        layer = Gemma4MoEDecoderLayer(text_config, layer_idx=0, moe_config=moe_config, backend=backend_config)

        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "mlp")
        assert hasattr(layer, "moe")
        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "post_attention_layernorm")
        assert hasattr(layer, "pre_feedforward_layernorm")
        assert hasattr(layer, "post_feedforward_layernorm")
        assert hasattr(layer, "pre_feedforward_layernorm_2")
        assert hasattr(layer, "post_feedforward_layernorm_1")
        assert hasattr(layer, "post_feedforward_layernorm_2")

    def test_stores_layer_idx(self, text_config, moe_config, backend_config):
        layer = Gemma4MoEDecoderLayer(text_config, layer_idx=2, moe_config=moe_config, backend=backend_config)
        assert layer.layer_idx == 2

    def test_attention_type_from_config(self, text_config, moe_config, backend_config):
        for idx, expected_type in enumerate(text_config.layer_types):
            layer = Gemma4MoEDecoderLayer(text_config, layer_idx=idx, moe_config=moe_config, backend=backend_config)
            assert layer.attention_type == expected_type

    def test_all_layers_have_layer_scalar_buffer(self, text_config, moe_config, backend_config):
        for idx in range(text_config.num_hidden_layers):
            layer = Gemma4MoEDecoderLayer(text_config, layer_idx=idx, moe_config=moe_config, backend=backend_config)
            assert layer.layer_scalar is not None
            torch.testing.assert_close(layer.layer_scalar, torch.ones(1))

    def test_moe_is_gemma4_moe_instance(self, text_config, moe_config, backend_config):
        layer = Gemma4MoEDecoderLayer(text_config, layer_idx=0, moe_config=moe_config, backend=backend_config)
        assert isinstance(layer.moe, Gemma4MoE)

    def test_forward_output_shape(self, text_config, moe_config, backend_config, device):
        layer = Gemma4MoEDecoderLayer(text_config, layer_idx=0, moe_config=moe_config, backend=backend_config)
        layer = layer.to(device).to(torch.bfloat16)

        batch, seq = 2, 4
        x = torch.randn(batch, seq, text_config.hidden_size, device=device, dtype=torch.bfloat16)
        pos_emb = (
            torch.randn(batch, seq, text_config.head_dim // 2, device=device, dtype=torch.bfloat16),
            torch.randn(batch, seq, text_config.head_dim // 2, device=device, dtype=torch.bfloat16),
        )

        with (
            patch.object(layer.self_attn, "forward", return_value=(torch.zeros_like(x), None)),
            patch.object(layer.moe, "forward", return_value=torch.zeros_like(x)),
        ):
            out = layer(x, position_embeddings=pos_emb)

        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Gemma4MoETextModelBackend tests
# ---------------------------------------------------------------------------
class TestGemma4MoETextModelBackend:
    def test_init_creates_components(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)

        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")
        assert hasattr(model, "rotary_emb")

    def test_layer_count_matches_config(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        assert len(model.layers) == text_config.num_hidden_layers

    def test_layers_are_moduledict(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        assert isinstance(model.layers, torch.nn.ModuleDict)

    def test_layer_keys_are_string_indices(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        expected_keys = [str(i) for i in range(text_config.num_hidden_layers)]
        assert list(model.layers.keys()) == expected_keys

    def test_all_layers_are_moe_decoder_layers(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        for layer in model.layers.values():
            assert isinstance(layer, Gemma4MoEDecoderLayer)

    def test_moe_config_auto_created(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)

        assert model.moe_config.dim == text_config.hidden_size
        assert model.moe_config.n_routed_experts == text_config.num_experts
        assert model.moe_config.n_activated_experts == text_config.top_k_experts
        assert model.moe_config.moe_inter_dim == text_config.expert_intermediate_size
        assert model.moe_config.expert_activation == "geglu"

    def test_moe_config_accepts_override(self, text_config, backend_config, moe_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config, moe_config=moe_config)
        assert model.moe_config is moe_config

    def test_embed_tokens_dimensions(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        assert model.embed_tokens.num_embeddings == text_config.vocab_size

    def test_get_input_embeddings(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        assert model.get_input_embeddings() is model.embed_tokens

    def test_set_input_embeddings(self, text_config, backend_config):
        model = Gemma4MoETextModelBackend(text_config, backend=backend_config)
        new_emb = torch.nn.Embedding(100, text_config.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.embed_tokens is new_emb


# ---------------------------------------------------------------------------
# Gemma4ForConditionalGeneration tests
# ---------------------------------------------------------------------------
class TestGemma4ForConditionalGeneration:
    def test_moe_init_replaces_language_model(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)

        assert isinstance(model.model, Gemma4MoEModel)
        assert isinstance(model.model.language_model, Gemma4MoETextModelBackend)

    def test_dense_init_keeps_hf_model(self, dense_config, backend_config):
        model = Gemma4ForConditionalGeneration(dense_config, backend=backend_config)

        assert not isinstance(model.model.language_model, Gemma4MoETextModelBackend)

    def test_moe_stores_backend(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        assert model.backend is backend_config

    def test_moe_exposes_moe_config(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        assert hasattr(model.model, "moe_config")
        assert model.model.moe_config is model.model.language_model.moe_config

    def test_state_dict_adapter_created_when_enabled(self, gemma4_config):
        backend = BackendConfig(
            linear="torch", attn="sdpa", rms_norm="torch",
            experts="torch", dispatcher="torch",
            enable_hf_state_dict_adapter=True,
        )
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend)
        assert hasattr(model, "state_dict_adapter")

    def test_state_dict_adapter_not_created_when_disabled(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        assert not hasattr(model, "state_dict_adapter")

    def test_text_config_dict_override_applied(self):
        cfg = _make_gemma4_config()
        override = {"use_cache": False}
        model = Gemma4ForConditionalGeneration(
            cfg,
            backend=BackendConfig(
                linear="torch", attn="sdpa", rms_norm="torch",
                experts="torch", dispatcher="torch",
                enable_hf_state_dict_adapter=False,
            ),
            text_config=override,
        )
        text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
        assert text_cfg.use_cache is False

    def test_forward_moe_path_returns_logits(self, gemma4_config, backend_config, device):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)

        batch, seq = 2, 6
        text_config = gemma4_config.text_config
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=device)

        with patch.object(
            model.model.language_model,
            "forward",
            return_value=MagicMock(
                last_hidden_state=torch.randn(batch, seq, text_config.hidden_size, device=device, dtype=torch.bfloat16)
            ),
        ):
            logits = model(input_ids)

        assert logits.shape == (batch, seq, text_config.vocab_size)

    def test_forward_applies_logit_softcapping(self, backend_config, device):
        cfg = _make_gemma4_config(final_logit_softcapping=30.0)
        model = Gemma4ForConditionalGeneration(cfg, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)

        text_config = cfg.text_config
        batch, seq = 1, 4
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=device)

        large_hidden = torch.randn(batch, seq, text_config.hidden_size, device=device, dtype=torch.bfloat16) * 100

        with patch.object(
            model.model.language_model,
            "forward",
            return_value=MagicMock(last_hidden_state=large_hidden),
        ):
            logits = model(input_ids)

        assert logits.abs().max() <= 30.0 + 1e-2

    def test_forward_generates_cache_position(self, gemma4_config, backend_config, device):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)

        text_config = gemma4_config.text_config
        batch, seq = 1, 4
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=device)

        captured = {}

        def capture_forward(**kwargs):
            captured["cache_position"] = kwargs.get("cache_position")
            return MagicMock(
                last_hidden_state=torch.randn(batch, seq, text_config.hidden_size, device=device, dtype=torch.bfloat16)
            )

        with patch.object(model.model.language_model, "forward", side_effect=capture_forward):
            model(input_ids, cache_position=None)

        assert captured["cache_position"] is not None
        torch.testing.assert_close(
            captured["cache_position"],
            torch.arange(seq, device=device),
        )

    def test_initialize_weights_dense_only_casts_dtype(self, dense_config, backend_config):
        model = Gemma4ForConditionalGeneration(dense_config, backend=backend_config)
        model.initialize_weights(dtype=torch.float32)
        for p in model.parameters():
            assert p.dtype == torch.float32

    def test_initialize_weights_moe_calls_init_weights(self, gemma4_config, backend_config, device):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)

        init_calls = []
        for layer in model.model.language_model.layers.values():
            original_init = layer.moe.init_weights

            def make_tracker(orig):
                def tracker(buf_dev):
                    init_calls.append(buf_dev)
                    return orig(buf_dev)
                return tracker

            layer.moe.init_weights = make_tracker(original_init)

        model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)

        assert len(init_calls) == gemma4_config.text_config.num_hidden_layers

    def test_initialize_weights_moe_casts_dtype(self, gemma4_config, backend_config, device):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        model.initialize_weights(buffer_device=device, dtype=torch.float32)
        for p in model.parameters():
            assert p.dtype == torch.float32


# ---------------------------------------------------------------------------
# Classmethods tests
# ---------------------------------------------------------------------------
class TestGemma4ForConditionalGenerationClassmethods:
    def test_from_config_creates_model(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration.from_config(gemma4_config, backend=backend_config)
        assert isinstance(model, Gemma4ForConditionalGeneration)

    def test_from_pretrained_classmethod(self):
        cfg = _make_gemma4_config()
        backend = BackendConfig(
            linear="torch", attn="sdpa", rms_norm="torch",
            experts="torch", dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )

        with patch(
            "transformers.models.gemma4.configuration_gemma4.Gemma4Config.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.return_value = cfg

            with patch.object(
                Gemma4ForConditionalGeneration, "from_config",
                wraps=Gemma4ForConditionalGeneration.from_config,
            ) as mock_from_config:
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    "gemma4/model", backend=backend,
                )
                assert isinstance(model, Gemma4ForConditionalGeneration)
                mock_from_pretrained.assert_called_once_with("gemma4/model")
                called_cfg = mock_from_config.call_args[0][0]
                assert called_cfg is cfg

    def test_model_class_export_exists(self):
        from nemo_automodel.components.models.gemma4_moe import model as gemma4_mod

        assert hasattr(gemma4_mod, "ModelClass")
        assert gemma4_mod.ModelClass is Gemma4ForConditionalGeneration


# ---------------------------------------------------------------------------
# Gemma4MoEModel wrapper tests
# ---------------------------------------------------------------------------
class TestGemma4MoEModel:
    def test_layers_property(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        assert model.model.layers is model.model.language_model.layers

    def test_embed_tokens_property(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        assert model.model.embed_tokens is model.model.language_model.embed_tokens

    def test_norm_property(self, gemma4_config, backend_config):
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        assert model.model.norm is model.model.language_model.norm
