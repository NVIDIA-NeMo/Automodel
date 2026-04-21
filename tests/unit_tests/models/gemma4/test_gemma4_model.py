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
        max_position_embeddings=256,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
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
        moe_inter_dim=tc.moe_intermediate_size,
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
        expected = text_config.hidden_size**-0.5
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

    def test_forward_routes_gate_input_to_gate(self, moe_config, backend_config, text_config, device):
        # HF Gemma4 sends the raw residual to the router and the normalized
        # input to the experts. When gate_input is provided, the gate must
        # receive it (not the normalized x).
        moe = Gemma4MoE(moe_config, backend_config, text_config).to(device).to(torch.bfloat16)

        batch, seq = 2, 4
        dim = text_config.hidden_size
        num_tokens = batch * seq
        topk = text_config.top_k_experts

        weights = torch.ones(num_tokens, topk, device=device, dtype=torch.bfloat16)
        indices = torch.zeros(num_tokens, topk, device=device, dtype=torch.long)
        expert_out = torch.zeros(num_tokens, dim, device=device, dtype=torch.bfloat16)

        normed = torch.randn(batch, seq, dim, device=device, dtype=torch.bfloat16)
        raw = torch.randn(batch, seq, dim, device=device, dtype=torch.bfloat16)

        # Patch .forward rather than the submodule itself — nn.Module blocks
        # assigning a MagicMock as a child module.
        with (
            patch.object(moe.gate, "forward", return_value=(weights, indices, None)) as mock_gate,
            patch.object(moe.experts, "forward", return_value=expert_out) as mock_experts,
        ):
            moe(normed, gate_input=raw)

        # gate receives the raw residual (not the normed input)
        gate_input_arg = mock_gate.call_args[0][0]
        torch.testing.assert_close(gate_input_arg, raw.view(-1, dim))
        # experts still receive the normed input
        expert_input_arg = mock_experts.call_args[0][0]
        torch.testing.assert_close(expert_input_arg, normed.view(-1, dim))

    def test_forward_without_gate_input_preserves_prior_behavior(self, moe_config, backend_config, text_config, device):
        # When gate_input is omitted, the gate must receive x — preserving
        # the original single-input contract for any non-Gemma4 caller.
        moe = Gemma4MoE(moe_config, backend_config, text_config).to(device).to(torch.bfloat16)

        batch, seq = 2, 4
        dim = text_config.hidden_size
        num_tokens = batch * seq
        topk = text_config.top_k_experts

        weights = torch.ones(num_tokens, topk, device=device, dtype=torch.bfloat16)
        indices = torch.zeros(num_tokens, topk, device=device, dtype=torch.long)
        expert_out = torch.zeros(num_tokens, dim, device=device, dtype=torch.bfloat16)

        x = torch.randn(batch, seq, dim, device=device, dtype=torch.bfloat16)

        with (
            patch.object(moe.gate, "forward", return_value=(weights, indices, None)) as mock_gate,
            patch.object(moe.experts, "forward", return_value=expert_out),
        ):
            moe(x)

        gate_input_arg = mock_gate.call_args[0][0]
        torch.testing.assert_close(gate_input_arg, x.view(-1, dim))


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

    def test_moe_receives_unnormed_residual_as_gate_input(self, text_config, moe_config, backend_config, device):
        # Regression guard for the Gemma4 MoE double-normalization bug:
        # the decoder layer must pass the raw post-attention residual (not
        # pre_feedforward_layernorm_2(x)) to the gate. See upstream issue
        # #1852 — double-norm caused gen_kl_error 0.116 on
        # Gemma4-26B-A4B-it GRPO; after the fix, 0.0011.
        layer = Gemma4MoEDecoderLayer(text_config, layer_idx=0, moe_config=moe_config, backend=backend_config)
        layer = layer.to(device).to(torch.bfloat16)

        batch, seq = 2, 4
        x = torch.randn(batch, seq, text_config.hidden_size, device=device, dtype=torch.bfloat16)
        pos_emb = (
            torch.randn(batch, seq, text_config.head_dim // 2, device=device, dtype=torch.bfloat16),
            torch.randn(batch, seq, text_config.head_dim // 2, device=device, dtype=torch.bfloat16),
        )
        # Sentinel distinguishable by value — what pre_feedforward_layernorm_2 returns.
        sentinel = torch.full_like(x, 7.0)

        with (
            patch.object(layer.self_attn, "forward", return_value=(torch.zeros_like(x), None)),
            patch.object(layer.pre_feedforward_layernorm_2, "forward", return_value=sentinel),
            patch.object(layer.moe, "forward", return_value=torch.zeros_like(x)) as mock_moe,
        ):
            layer(x, position_embeddings=pos_emb)

        # Positional moe_input is the normalized sentinel from pre_feedforward_layernorm_2.
        moe_input_arg = mock_moe.call_args[0][0]
        torch.testing.assert_close(moe_input_arg, sentinel)

        # gate_input kwarg must be passed AND differ from the normalized sentinel —
        # i.e. it is the raw post-attention residual, not the layernormed moe_input.
        assert "gate_input" in mock_moe.call_args.kwargs
        gate_input = mock_moe.call_args.kwargs["gate_input"]
        assert not torch.equal(gate_input, sentinel)


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
        assert model.moe_config.moe_inter_dim == text_config.moe_intermediate_size
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
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
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
                linear="torch",
                attn="sdpa",
                rms_norm="torch",
                experts="torch",
                dispatcher="torch",
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

    def test_dense_forward_cp_preserves_mm_token_type_ids(self, dense_config, backend_config, device):
        """Dense Gemma4 should preserve caller-provided mm_token_type_ids."""
        from unittest.mock import patch

        if not (torch.distributed.is_available() and torch.cuda.is_available()
                and torch.distributed.is_initialized()):
            pytest.skip("distributed process group not initialized")

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        # Single-rank mesh: simulates a CP rank without needing multiple processes.
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("cp",))

        model = Gemma4ForConditionalGeneration(dense_config, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)

        batch, seq = 1, 4
        text_config = dense_config.text_config if hasattr(dense_config, "text_config") else dense_config
        raw_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=device)
        # Wrap input_ids as a DTensor (as make_cp_batch_and_ctx would do under CP).
        input_ids_dt = DTensor.from_local(raw_ids, device_mesh=mesh, placements=[Shard(1)])
        mm_token_type_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)

        captured = {}

        def _fake_super_forward(**kwargs):
            captured["mm_token_type_ids"] = kwargs.get("mm_token_type_ids")
            return MagicMock(logits=torch.zeros(batch, seq, text_config.vocab_size, device=device))

        with patch.object(
            model.__class__.__bases__[1],  # HFGemma4ForConditionalGeneration
            "forward",
            side_effect=_fake_super_forward,
        ):
            model(input_ids_dt, mm_token_type_ids=mm_token_type_ids)

        torch.testing.assert_close(captured["mm_token_type_ids"], mm_token_type_ids)

    def test_moe_forward_cp_nulls_mm_token_type_ids_for_mask(self, gemma4_config, backend_config, device):
        """Under CP (DTensor inputs_embeds) Gemma4MoETextModelBackend must not pass
        mm_token_type_ids to create_causal_mask_mapping."""
        from unittest.mock import patch

        if not (torch.distributed.is_available() and torch.cuda.is_available()
                and torch.distributed.is_initialized()):
            pytest.skip("distributed process group not initialized")

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("cp",))

        # Use a config with use_bidirectional_attention == "vision" so the
        # create_causal_mask_mapping branch is taken.
        cfg = _make_gemma4_config(use_bidirectional_attention="vision")
        model = Gemma4ForConditionalGeneration(cfg, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)
        text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg

        batch, seq = 1, 4
        raw_embeds = torch.randn(batch, seq, text_cfg.hidden_size, device=device, dtype=torch.bfloat16)
        embeds_dt = DTensor.from_local(raw_embeds, device_mesh=mesh, placements=[Shard(1)])
        mm_token_type_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)

        captured = {}

        def _fake_mask_mapping(**kwargs):
            captured["mm_token_type_ids"] = kwargs.get("mm_token_type_ids")
            return {lt: None for lt in set(text_cfg.layer_types)}

        backend = model.model.language_model
        with patch(
            "transformers.models.gemma4.modeling_gemma4.create_causal_mask_mapping",
            side_effect=_fake_mask_mapping,
        ):
            try:
                backend.forward(inputs_embeds=embeds_dt, mm_token_type_ids=mm_token_type_ids)
            except Exception:
                pass  # forward may fail after mask; we only care about the captured kwarg

        assert captured.get("mm_token_type_ids") is None, (
            "mm_token_type_ids must be nulled when inputs_embeds is a DTensor (CP active)"
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

    def test_prepare_inputs_embeds_for_cp_text_only(self, gemma4_config, backend_config, device):
        """prepare_inputs_embeds_for_cp with pixel_values=None must return plain
        text embeddings with no image-feature merge path taken."""
        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)

        text_config = gemma4_config.text_config
        batch, seq = 2, 6
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=device)

        embeds = model.prepare_inputs_embeds_for_cp(input_ids, pixel_values=None)

        assert embeds.shape == (batch, seq, text_config.hidden_size)
        assert embeds.dtype == torch.bfloat16
        # With no pixel_values the function should be equivalent to the raw
        # embed_tokens lookup (no masked_scatter path).
        expected = model.get_input_embeddings()(input_ids)
        torch.testing.assert_close(embeds, expected)

    def test_prepare_inputs_embeds_for_cp_scatters_image_features(self, gemma4_config, backend_config, device):
        """prepare_inputs_embeds_for_cp with pixel_values must masked_scatter
        image features into positions marked by mm_token_type_ids == 1."""
        from unittest.mock import MagicMock, patch

        model = Gemma4ForConditionalGeneration(gemma4_config, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)
        text_config = gemma4_config.text_config

        batch, seq, n_image_tokens = 1, 8, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=device)
        # Mark the first three positions as image tokens.
        mm_token_type_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)
        mm_token_type_ids[:, :n_image_tokens] = 1

        # Sentinel image features we can identify after the scatter.
        sentinel = torch.full(
            (n_image_tokens, text_config.hidden_size),
            7.5,
            device=device,
            dtype=torch.bfloat16,
        )
        fake_vision_output = MagicMock()
        fake_vision_output.pooler_output = sentinel
        pixel_values = torch.randn(batch, 3, 32, 32, device=device, dtype=torch.bfloat16)

        with patch.object(model.model, "get_image_features", return_value=fake_vision_output):
            embeds = model.prepare_inputs_embeds_for_cp(
                input_ids,
                pixel_values=pixel_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        assert embeds.shape == (batch, seq, text_config.hidden_size)
        # Image-token slots should carry the sentinel; the rest should equal
        # the plain text embedding of those positions.
        torch.testing.assert_close(embeds[0, :n_image_tokens], sentinel)
        plain = model.get_input_embeddings()(input_ids)
        torch.testing.assert_close(embeds[0, n_image_tokens:], plain[0, n_image_tokens:])

    def test_prepare_model_inputs_for_cp_builds_per_layer_inputs(self, backend_config, device):
        """CP pre-processing must build per-layer inputs from pad-masked token ids."""
        cfg = _make_gemma4_config(
            enable_moe_block=False,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=256,
            pad_token_id=0,
        )
        model = Gemma4ForConditionalGeneration(cfg, backend=backend_config)
        model = model.to(device).to(torch.bfloat16)
        text_config = cfg.text_config

        batch, seq, n_image_tokens = 1, 6, 2
        input_ids = torch.randint(4, text_config.vocab_size, (batch, seq), device=device)
        mm_token_type_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)
        mm_token_type_ids[:, :n_image_tokens] = 1

        sentinel = torch.full(
            (n_image_tokens, text_config.hidden_size),
            3.5,
            device=device,
            dtype=torch.bfloat16,
        )
        fake_vision_output = MagicMock()
        fake_vision_output.pooler_output = sentinel
        pixel_values = torch.randn(batch, 3, 32, 32, device=device, dtype=torch.bfloat16)

        with patch.object(model.model, "get_image_features", return_value=fake_vision_output):
            prepared = model.prepare_model_inputs_for_cp(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        assert set(prepared) == {"inputs_embeds", "per_layer_inputs"}
        assert prepared["inputs_embeds"].shape == (batch, seq, text_config.hidden_size)
        assert prepared["per_layer_inputs"].shape == (
            batch,
            seq,
            text_config.num_hidden_layers,
            text_config.hidden_size_per_layer_input,
        )

        llm_input_ids = input_ids.masked_fill(mm_token_type_ids == 1, model.pad_token_id)
        expected_per_layer_inputs = model.model.language_model.embed_tokens_per_layer(llm_input_ids).reshape(
            batch,
            seq,
            text_config.num_hidden_layers,
            text_config.hidden_size_per_layer_input,
        )
        torch.testing.assert_close(prepared["per_layer_inputs"], expected_per_layer_inputs)
        torch.testing.assert_close(prepared["inputs_embeds"][0, :n_image_tokens], sentinel)


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
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )

        with patch(
            "transformers.models.gemma4.configuration_gemma4.Gemma4Config.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.return_value = cfg

            with patch.object(
                Gemma4ForConditionalGeneration,
                "from_config",
                wraps=Gemma4ForConditionalGeneration.from_config,
            ) as mock_from_config:
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    "gemma4/model",
                    backend=backend,
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
