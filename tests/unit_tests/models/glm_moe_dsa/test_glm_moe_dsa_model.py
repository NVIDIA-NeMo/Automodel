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

import importlib.util
import sys
import types
from unittest.mock import patch

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

# Mock fast_hadamard_transform before importing deepseek_v32 modules
try:
    import fast_hadamard_transform  # noqa: F401
except ImportError:
    if "fast_hadamard_transform" not in sys.modules:
        mock_hadamard = types.ModuleType("fast_hadamard_transform")
        mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
        mock_hadamard.hadamard_transform = lambda x, scale: x
        sys.modules["fast_hadamard_transform"] = mock_hadamard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.model import Block, GlmMoeDsaForCausalLM, GlmMoeDsaModel
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def config():
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
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


def _make_moe_config(cfg: GlmMoeDsaConfig) -> MoEConfig:
    return MoEConfig(
        dim=cfg.hidden_size,
        inter_dim=cfg.intermediate_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        n_expert_groups=cfg.n_group,
        n_limited_groups=cfg.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=cfg.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=cfg.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
    )


class TestBlock:
    def test_block_uses_mlp_for_dense_layers(self, config, backend_config):
        block = Block(layer_idx=0, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        assert isinstance(block.mlp, MLP)
        assert hasattr(block, "self_attn")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")

    def test_block_uses_moe_for_sparse_layers(self, config, backend_config):
        block = Block(layer_idx=2, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        assert isinstance(block.mlp, MoE)

    def test_block_stores_layer_idx(self, config, backend_config):
        block = Block(layer_idx=3, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        assert block.layer_idx == 3

    def test_forward_pass_calls_attention_and_mlp(self, config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        block = block.to(device)

        batch, seq_len = 2, 4
        x = torch.randn(batch, seq_len, config.hidden_size, device=device)
        freqs_cis = torch.randn(batch, seq_len, config.qk_rope_head_dim // 2, device=device)

        with (
            patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn,
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp,
        ):
            out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()

    def test_forward_builds_padding_mask_from_attention(self, config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        block = block.to(device)

        x = torch.randn(1, 3, config.hidden_size, device=device)
        freqs_cis = torch.randn(1, 3, config.qk_rope_head_dim // 2, device=device)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)

        with (
            patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)),
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp,
        ):
            block(x, freqs_cis=freqs_cis, attention_mask=attention_mask)

        _, kwargs = mock_mlp.call_args
        padding_mask = kwargs.get("padding_mask")
        assert padding_mask is not None
        torch.testing.assert_close(padding_mask, attention_mask.logical_not())

    def test_forward_uses_provided_padding_mask(self, config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        block = block.to(device)

        x = torch.randn(1, 3, config.hidden_size, device=device)
        freqs_cis = torch.randn(1, 3, config.qk_rope_head_dim // 2, device=device)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)
        padding_mask = torch.tensor([[0, 0, 1]], dtype=torch.bool, device=device)

        with (
            patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)),
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp,
        ):
            block(x, freqs_cis=freqs_cis, attention_mask=attention_mask, padding_mask=padding_mask)

        _, kwargs = mock_mlp.call_args
        received_padding_mask = kwargs.get("padding_mask")
        torch.testing.assert_close(received_padding_mask, padding_mask)

    def test_mlp_wrapper_handles_mlp_instance(self, config, backend_config):
        block = Block(layer_idx=0, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        x = torch.randn(2, 4, config.hidden_size).to(torch.bfloat16)
        out = block._mlp(x, padding_mask=None)
        assert out.shape == x.shape

    def test_mlp_wrapper_handles_moe_instance(self, config, backend_config):
        block = Block(layer_idx=2, config=config, moe_config=_make_moe_config(config), backend=backend_config)
        x = torch.randn(2, 4, config.hidden_size).to(torch.bfloat16)
        padding_mask = torch.zeros(2, 4, dtype=torch.bool)

        with patch.object(block.mlp, "forward", return_value=torch.zeros_like(x)) as mock_moe:
            out = block._mlp(x, padding_mask=padding_mask)

        mock_moe.assert_called_once_with(x, padding_mask)
        assert out.shape == x.shape

    def test_init_weights_resets_sublayers(self, config, backend_config):
        block = Block(layer_idx=0, config=config, moe_config=_make_moe_config(config), backend=backend_config)

        with (
            patch.object(block.input_layernorm, "reset_parameters") as mock_in,
            patch.object(block.post_attention_layernorm, "reset_parameters") as mock_post,
            patch.object(block.self_attn, "init_weights") as mock_attn,
            patch.object(block.mlp, "init_weights") as mock_mlp,
        ):
            block.init_weights(torch.device("cpu"))

        mock_in.assert_called_once()
        mock_post.assert_called_once()
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()


class TestGlmMoeDsaModel:
    def test_model_initialization_sets_components(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)

        assert model.config == config
        assert model.backend == backend_config
        assert len(model.layers) == config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == config.vocab_size

    def test_model_initializes_moe_config_with_sigmoid_scoring(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)

        assert hasattr(model, "moe_config")
        assert model.moe_config.dim == config.hidden_size
        assert model.moe_config.n_routed_experts == config.n_routed_experts
        assert model.moe_config.n_shared_experts == config.n_shared_experts
        assert model.moe_config.n_activated_experts == config.num_experts_per_tok
        assert model.moe_config.score_func == "sigmoid"
        assert model.moe_config.softmax_before_topk is False
        assert model.moe_config.route_scale == config.routed_scaling_factor

    def test_model_initializes_moe_config_with_expert_groups(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)

        assert model.moe_config.n_expert_groups == config.n_group
        assert model.moe_config.n_limited_groups == config.topk_group

    def test_model_accepts_custom_moe_config(self, config, backend_config):
        moe_config = _make_moe_config(config)
        model = GlmMoeDsaModel(config, backend=backend_config, moe_config=moe_config)

        assert model.moe_config == moe_config

    def test_model_precomputes_freqs(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)

        assert hasattr(model, "freqs")
        assert model.freqs is not None
        assert model.qk_rope_head_dim == config.qk_rope_head_dim

    def test_model_extracts_rope_theta_from_rope_parameters(self, config, backend_config):
        with patch(
            "nemo_automodel.components.models.glm_moe_dsa.model.precompute_freqs_cis"
        ) as mock_precompute:
            mock_precompute.return_value = torch.randn(10)
            GlmMoeDsaModel(config, backend=backend_config)

        mock_precompute.assert_called_once()
        call_kwargs = mock_precompute.call_args[1]
        assert call_kwargs["rope_theta"] == config.rope_parameters["rope_theta"]

    def test_forward_runs_all_layers(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)

        batch, seq_len = 2, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

        with patch.object(
            Block, "forward", side_effect=lambda *_, **__: torch.randn(batch, seq_len, config.hidden_size)
        ) as mock_block:
            out = model(input_ids)

        assert out.shape == (batch, seq_len, config.hidden_size)
        assert mock_block.call_count == config.num_hidden_layers

    def test_forward_generates_position_ids_if_not_provided(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)
        batch, seq_len = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

        with patch(
            "nemo_automodel.components.models.glm_moe_dsa.model.freqs_cis_from_position_ids"
        ) as mock_freqs:
            mock_freqs.return_value = torch.randn(batch, seq_len, config.qk_rope_head_dim // 2)
            with patch.object(
                Block, "forward", side_effect=lambda *_, **__: torch.randn(batch, seq_len, config.hidden_size)
            ):
                model(input_ids)

        mock_freqs.assert_called_once()
        position_ids = mock_freqs.call_args[0][0]
        assert position_ids.shape == (batch, seq_len)
        expected_pos_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch, -1)
        torch.testing.assert_close(position_ids, expected_pos_ids)

    def test_forward_accepts_position_ids(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)

        with patch.object(Block, "forward", return_value=torch.zeros(batch, seq_len, config.hidden_size)):
            out = model(input_ids, position_ids=position_ids)

        assert out.shape == (batch, seq_len, config.hidden_size)

    def test_init_weights_updates_embeddings_and_layers(self, config, backend_config):
        model = GlmMoeDsaModel(config, backend=backend_config)
        original = model.embed_tokens.weight.clone()

        with (
            patch.object(model.norm, "reset_parameters") as mock_norm,
            patch.object(Block, "init_weights") as mock_layer_init,
        ):
            model.init_weights(torch.device("cpu"))

        mock_norm.assert_called_once()
        assert not torch.equal(model.embed_tokens.weight, original)
        assert mock_layer_init.call_count == config.num_hidden_layers


class TestGlmMoeDsaForCausalLM:
    def test_forward_returns_logits(self, config, backend_config, device):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config).to(device)

        batch, seq_len = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with patch.object(
            model.model,
            "forward",
            return_value=torch.randn(batch, seq_len, config.hidden_size, device=device).to(torch.bfloat16),
        ):
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_forward_with_thd_format_squeezes_input(self, config, backend_config, device):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config).to(device)

        batch, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with (
            patch("nemo_automodel.components.models.glm_moe_dsa.model.squeeze_input_for_thd") as mock_squeeze,
            patch.object(
                model.model,
                "forward",
                return_value=torch.randn(seq_len, config.hidden_size, device=device).to(torch.bfloat16),
            ),
        ):
            mock_squeeze.return_value = (input_ids.squeeze(0), None, None, {"qkv_format": "thd"})
            logits = model(input_ids, qkv_format="thd")

        mock_squeeze.assert_called_once()
        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_initialize_weights_invokes_submodules(self, config, backend_config):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)
        original = model.lm_head.weight.clone()

        with patch.object(model.model, "init_weights") as mock_init:
            model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        mock_init.assert_called_once()
        assert not torch.equal(model.lm_head.weight, original)
        assert model.lm_head.weight.dtype == torch.float32

    def test_initialize_weights_uses_scaled_std_for_lm_head(self, config, backend_config):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)

        with patch.object(model.model, "init_weights"), patch("torch.nn.init.trunc_normal_") as mock_trunc:
            model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        mock_trunc.assert_called()
        call_args = mock_trunc.call_args
        assert call_args[1]["std"] == config.hidden_size**-0.5

    def test_initialize_weights_sets_e_score_correction_bias_for_moe_layers(self, config, backend_config):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)
        device = torch.device("cpu")

        with patch.object(model.model, "init_weights"):
            model.initialize_weights(buffer_device=device, dtype=torch.float32)

        for layer_idx, layer in enumerate(model.model.layers.values()):
            if isinstance(layer.mlp, MoE):
                assert config.mlp_layer_types[layer_idx] == "sparse"
                assert hasattr(layer.mlp.gate, "e_score_correction_bias")
                assert layer.mlp.gate.e_score_correction_bias.shape == (config.n_routed_experts,)
                assert layer.mlp.gate.e_score_correction_bias.dtype == torch.float32
                torch.testing.assert_close(
                    layer.mlp.gate.e_score_correction_bias,
                    torch.zeros(config.n_routed_experts, dtype=torch.float32),
                )

    def test_state_dict_adapter_created_when_enabled(self, config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)
        assert hasattr(model, "state_dict_adapter")

    def test_state_dict_adapter_not_created_when_disabled(self, config, backend_config):
        backend_config.enable_hf_state_dict_adapter = False
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)
        assert not hasattr(model, "state_dict_adapter")

    def test_get_set_input_embeddings(self, config, backend_config):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)
        assert model.get_input_embeddings() is model.model.embed_tokens

        new_embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        model.set_input_embeddings(new_embed)
        assert model.get_input_embeddings() is new_embed

    def test_get_set_output_embeddings(self, config, backend_config):
        model = GlmMoeDsaForCausalLM(config, backend=backend_config)
        assert model.get_output_embeddings() is model.lm_head

        new_lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        model.set_output_embeddings(new_lm_head)
        assert model.get_output_embeddings() is new_lm_head


class TestGlmMoeDsaClassmethods:
    def test_from_config_creates_model(self, config, backend_config):
        model = GlmMoeDsaForCausalLM.from_config(config, backend=backend_config)

        assert isinstance(model, GlmMoeDsaForCausalLM)
        assert model.config == config
        assert model.backend == backend_config

    def test_from_pretrained_classmethod(self, config):
        with patch(
            "transformers.models.glm_moe_dsa.configuration_glm_moe_dsa.GlmMoeDsaConfig.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.return_value = config

            with patch.object(
                GlmMoeDsaForCausalLM, "from_config", wraps=GlmMoeDsaForCausalLM.from_config
            ) as mock_from_config:
                model = GlmMoeDsaForCausalLM.from_pretrained("zai-org/GLM-5")
                assert isinstance(model, GlmMoeDsaForCausalLM)
                mock_from_pretrained.assert_called_once_with("zai-org/GLM-5")
                called_cfg = mock_from_config.call_args[0][0]
                assert called_cfg is config

    def test_modelclass_export_exists(self):
        from nemo_automodel.components.models.glm_moe_dsa import model as dsa_mod

        assert hasattr(dsa_mod, "ModelClass")
        assert dsa_mod.ModelClass is GlmMoeDsaForCausalLM
