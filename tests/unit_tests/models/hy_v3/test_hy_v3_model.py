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

"""Unit tests for the HYV3 Block / HYV3Model / HYV3ForCausalLM layers."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v3.config import HYV3Config
from nemo_automodel.components.models.hy_v3.model import (
    Block,
    HYV3ForCausalLM,
    HYV3Model,
    ModelClass,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


HIDDEN = 64
INTER = 128
MOE_INTER = 64
N_HEADS = 8
N_KV = 2
HEAD_DIM = 16
N_EXPERTS = 4


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def config():
    return HYV3Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        num_experts=N_EXPERTS,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
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
        gate_precision="float32",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        enable_fsdp_optimizations=False,
    )


@pytest.fixture
def moe_config(config):
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=config.num_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
    )


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class TestBlock:
    def test_dense_layer_uses_mlp_when_idx_below_first_k_dense(self, config, moe_config, backend_config):
        """layer_idx < first_k_dense_replace -> dense MLP, not MoE."""
        config.first_k_dense_replace = 1
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MLP)
        assert not isinstance(block.mlp, MoE)

    def test_moe_layer_uses_moe_when_idx_at_or_above_first_k_dense(self, config, moe_config, backend_config):
        config.first_k_dense_replace = 1
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MoE)

    def test_first_k_dense_replace_higher_threshold(self, config, moe_config, backend_config):
        """If first_k_dense_replace=3, layers 0-2 are dense and 3+ are MoE."""
        config.first_k_dense_replace = 3
        for i in (0, 1, 2):
            block = Block(layer_idx=i, config=config, moe_config=moe_config, backend=backend_config)
            assert isinstance(block.mlp, MLP), f"layer {i} should be dense"
        block = Block(layer_idx=3, config=config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MoE)

    def test_block_has_required_submodules(self, config, moe_config, backend_config):
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config)
        assert hasattr(block, "self_attn")
        assert hasattr(block, "mlp")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert block.layer_idx == 1

    def test_forward_residual_calls_attn_then_mlp(self, config, moe_config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config).to(device)
        bsz, seq = 2, 4
        x = torch.randn(bsz, seq, HIDDEN, device=device, dtype=torch.bfloat16)
        freqs = torch.zeros(1, seq, HEAD_DIM, device=device)
        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn, \
             patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp:
            out = block(x, freqs_cis=freqs)
        assert out.shape == x.shape
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()

    def test_padding_mask_built_from_attention_mask(self, config, moe_config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config).to(device)
        x = torch.randn(1, 3, HIDDEN, device=device, dtype=torch.bfloat16)
        freqs = torch.zeros(1, 3, HEAD_DIM, device=device)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)
        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)), \
             patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp:
            block(x, freqs_cis=freqs, attention_mask=mask)
        _, kwargs = mock_mlp.call_args
        torch.testing.assert_close(kwargs["padding_mask"], mask.logical_not())

    def test_mlp_wrapper_dense_path(self, config, moe_config, backend_config, device):
        config.first_k_dense_replace = 1
        block = (
            Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)
            .to(device)
            .to(torch.bfloat16)
        )
        x = torch.randn(2, 4, HIDDEN, device=device, dtype=torch.bfloat16)
        out = block._mlp(x, padding_mask=None)
        assert out.shape == x.shape

    def test_init_weights_invokes_subcomponents(self, config, moe_config, backend_config, device):
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config).to(device)
        with patch.object(block.input_layernorm, "reset_parameters") as in_norm, \
             patch.object(block.post_attention_layernorm, "reset_parameters") as post_norm, \
             patch.object(block.self_attn, "init_weights") as attn_init, \
             patch.object(block.mlp, "init_weights") as mlp_init:
            block.init_weights(buffer_device=device)
        in_norm.assert_called_once()
        post_norm.assert_called_once()
        attn_init.assert_called_once()
        mlp_init.assert_called_once()


# ---------------------------------------------------------------------------
# HYV3Model
# ---------------------------------------------------------------------------


class TestHYV3Model:
    def test_construction_sets_components(self, config, backend_config):
        model = HYV3Model(config, backend=backend_config)
        assert len(model.layers) == config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == config.vocab_size
        assert model.norm is not None
        assert model.rotary_emb.head_dim == config.head_dim
        assert isinstance(model.moe_config, MoEConfig)

    def test_dense_then_moe_layer_structure(self, config, backend_config):
        config.first_k_dense_replace = 1
        config.num_hidden_layers = 3
        model = HYV3Model(config, backend=backend_config)
        assert isinstance(model.layers["0"].mlp, MLP)
        assert isinstance(model.layers["1"].mlp, MoE)
        assert isinstance(model.layers["2"].mlp, MoE)

    def test_moe_config_inferred_from_config(self, config, backend_config):
        model = HYV3Model(config, backend=backend_config)
        mc = model.moe_config
        assert mc.dim == config.hidden_size
        assert mc.moe_inter_dim == config.moe_intermediate_size
        assert mc.n_routed_experts == config.num_experts
        assert mc.n_activated_experts == config.num_experts_per_tok
        assert mc.n_shared_experts == config.num_shared_experts
        assert mc.score_func == "sigmoid"
        assert mc.expert_activation == "swiglu"

    def test_moe_overrides_take_effect(self, config, backend_config):
        model = HYV3Model(config, backend=backend_config, moe_overrides={"score_func": "softmax", "route_scale": 2.0})
        assert model.moe_config.score_func == "softmax"
        assert model.moe_config.route_scale == 2.0

    def test_explicit_moe_config_passes_through(self, config, backend_config, moe_config):
        model = HYV3Model(config, backend=backend_config, moe_config=moe_config)
        assert model.moe_config is moe_config

    def test_explicit_moe_config_and_overrides_conflict(self, config, backend_config, moe_config):
        with pytest.raises(ValueError, match="Cannot pass both"):
            HYV3Model(config, backend=backend_config, moe_config=moe_config, moe_overrides={"score_func": "softmax"})

    def test_forward_runs_all_layers(self, config, backend_config, device):
        model = HYV3Model(config, backend=backend_config).to(device)
        bsz, seq = 1, 4
        input_ids = torch.randint(0, config.vocab_size, (bsz, seq), device=device)
        with patch.object(Block, "forward", side_effect=lambda x=None, **kw: x if x is not None else kw["x"]) as mock_block:
            out = model(input_ids)
        assert out.shape == (bsz, seq, HIDDEN)
        assert mock_block.call_count == config.num_hidden_layers

    def test_forward_accepts_explicit_position_ids(self, config, backend_config, device):
        model = HYV3Model(config, backend=backend_config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=device)
        position_ids = torch.arange(4, device=device).unsqueeze(0)
        with patch.object(Block, "forward", side_effect=lambda x=None, **kw: x if x is not None else kw["x"]):
            out = model(input_ids, position_ids=position_ids)
        assert out.shape == (1, 4, HIDDEN)

    def test_init_weights_resets_layers_norm_embeddings(self, config, backend_config, device):
        model = HYV3Model(config, backend=backend_config).to(device)
        embed_before = model.embed_tokens.weight.detach().clone()
        with patch.object(model.norm, "reset_parameters") as mock_norm, \
             patch.object(Block, "init_weights") as mock_layer_init:
            model.init_weights(buffer_device=device)
        mock_norm.assert_called_once()
        assert mock_layer_init.call_count == config.num_hidden_layers
        # Embedding weights are re-initialized.
        assert not torch.equal(model.embed_tokens.weight.detach(), embed_before)


# ---------------------------------------------------------------------------
# HYV3ForCausalLM
# ---------------------------------------------------------------------------


class TestHYV3ForCausalLM:
    def test_construction_attaches_model_and_lm_head(self, config, backend_config):
        model = HYV3ForCausalLM(config, backend=backend_config)
        assert isinstance(model.model, HYV3Model)
        assert model.lm_head.weight.shape == (config.vocab_size, config.hidden_size)
        assert model.config is config

    def test_state_dict_adapter_attached_when_enabled(self, config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = HYV3ForCausalLM(config, backend=backend_config)
        from nemo_automodel.components.models.hy_v3.state_dict_adapter import HYV3StateDictAdapter

        assert hasattr(model, "state_dict_adapter")
        assert isinstance(model.state_dict_adapter, HYV3StateDictAdapter)

    def test_state_dict_adapter_not_attached_when_disabled(self, config, backend_config):
        backend_config.enable_hf_state_dict_adapter = False
        model = HYV3ForCausalLM(config, backend=backend_config)
        assert not hasattr(model, "state_dict_adapter")

    def test_default_backend_built_when_omitted(self, config):
        model = HYV3ForCausalLM(config)
        assert isinstance(model.backend, BackendConfig)

    def test_get_set_input_embeddings(self, config, backend_config):
        model = HYV3ForCausalLM(config, backend=backend_config)
        new_emb = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.get_input_embeddings() is new_emb

    def test_get_set_output_embeddings(self, config, backend_config):
        model = HYV3ForCausalLM(config, backend=backend_config)
        new_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        model.set_output_embeddings(new_head)
        assert model.get_output_embeddings() is new_head

    def test_forward_returns_logits_shape(self, config, backend_config, device):
        model = HYV3ForCausalLM(config, backend=backend_config).to(device)
        bsz, seq = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (bsz, seq), device=device)
        fake_hidden = torch.randn(bsz, seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        with patch.object(model.model, "forward", return_value=fake_hidden):
            logits = model(input_ids)
        assert logits.shape == (bsz, seq, config.vocab_size)

    def test_initialize_weights_invokes_inner_init(self, config, backend_config, device):
        model = HYV3ForCausalLM(config, backend=backend_config).to(device)
        with patch.object(model.model, "init_weights") as mock_init:
            model.initialize_weights(buffer_device=device, dtype=torch.float32)
        mock_init.assert_called_once()
        assert model.lm_head.weight.dtype == torch.float32

    def test_update_moe_gate_bias_no_op_when_factor_zero(self, config, backend_config, device):
        """gate_bias_update_factor defaults to 0.0; update_moe_gate_bias must NOT call
        gate.update_bias() when the factor is zero (the bug fixed in 564ff4f2)."""
        model = HYV3ForCausalLM(config, backend=backend_config).to(device)
        for layer in model.model.layers.values():
            if isinstance(layer.mlp, MoE):
                with patch.object(layer.mlp.gate, "update_bias") as mock:
                    model.update_moe_gate_bias()
                    mock.assert_not_called()

    def test_update_moe_gate_bias_calls_when_factor_positive(self, config, backend_config, device):
        model = HYV3ForCausalLM(
            config, backend=backend_config, moe_overrides={"gate_bias_update_factor": 1e-3}
        ).to(device)
        called = 0
        for layer in model.model.layers.values():
            if isinstance(layer.mlp, MoE):
                with patch.object(layer.mlp.gate, "update_bias") as mock:
                    model.update_moe_gate_bias()
                    if mock.called:
                        called += 1
        assert called > 0

    def test_from_config_classmethod_passes_through(self, config, backend_config):
        model = HYV3ForCausalLM.from_config(config, backend=backend_config)
        assert isinstance(model, HYV3ForCausalLM)

    def test_from_pretrained_resolves_config_then_delegates(self, config, backend_config):
        with patch("transformers.AutoConfig.from_pretrained", return_value=config) as mock_acfg, \
             patch.object(HYV3ForCausalLM, "from_config", wraps=HYV3ForCausalLM.from_config) as mock_fc:
            model = HYV3ForCausalLM.from_pretrained("tencent/Hy3-preview", backend=backend_config)
        mock_acfg.assert_called_once()
        mock_fc.assert_called_once()
        assert isinstance(model, HYV3ForCausalLM)

    def test_modelclass_alias(self):
        assert ModelClass is HYV3ForCausalLM


# ---------------------------------------------------------------------------
# Module-level export
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_init_exports_hyv3_for_causal_lm(self):
        from nemo_automodel.components.models.hy_v3 import HYV3ForCausalLM as exported

        assert exported is HYV3ForCausalLM

    def test_module_class_pointer(self):
        from nemo_automodel.components.models.hy_v3 import model as mod

        assert mod.ModelClass is HYV3ForCausalLM
