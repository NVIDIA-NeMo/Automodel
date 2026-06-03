# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the Hy-MT2 Block / HyMT2Model / HyMT2ForCausalLM layers."""

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_mt2.config import HyMT2Config
from nemo_automodel.components.models.hy_mt2.model import (
    Block,
    HyMT2ForCausalLM,
    HyMT2Model,
    ModelClass,
    _resolve_score_func,
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
    return HyMT2Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        expert_hidden_dim=MOE_INTER,
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
        rms_norm_eps=1e-5,
        router_scaling_factor=2.826,
        route_norm=True,
        moe_router_use_sigmoid=True,
        moe_router_enable_expert_bias=True,
        enable_lm_head_fp32=True,
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
        route_scale=config.router_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
    )


class TestResolveScoreFunc:
    def test_default_is_sigmoid(self):
        # Config without the flag falls back to sigmoid (Hy-MT2 default).
        class _NoFlag:
            pass

        assert _resolve_score_func(_NoFlag()) == "sigmoid"

    def test_true_maps_to_sigmoid(self):
        class _Cfg:
            moe_router_use_sigmoid = True

        assert _resolve_score_func(_Cfg()) == "sigmoid"

    def test_false_maps_to_softmax(self):
        class _Cfg:
            moe_router_use_sigmoid = False

        assert _resolve_score_func(_Cfg()) == "softmax"


class TestBlock:
    def test_dense_layer_uses_mlp_when_idx_below_first_k_dense(self, config, moe_config, backend_config):
        config.first_k_dense_replace = 1
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MLP)

    def test_moe_layer_uses_moe_when_idx_at_or_above_first_k_dense(self, config, moe_config, backend_config):
        config.first_k_dense_replace = 1
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MoE)

    def test_block_has_required_submodules(self, config, moe_config, backend_config):
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config)
        assert hasattr(block, "self_attn")
        assert hasattr(block, "mlp")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert block.layer_idx == 1


class TestHyMT2Model:
    def test_construction_sets_components(self, config, backend_config):
        model = HyMT2Model(config, backend=backend_config)
        assert len(model.layers) == config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == config.vocab_size
        assert model.norm is not None
        assert model.rotary_emb.head_dim == config.head_dim
        assert isinstance(model.moe_config, MoEConfig)

    def test_dense_then_moe_layer_structure(self, config, backend_config):
        config.first_k_dense_replace = 1
        config.num_hidden_layers = 3
        model = HyMT2Model(config, backend=backend_config)
        assert isinstance(model.layers["0"].mlp, MLP)
        assert isinstance(model.layers["1"].mlp, MoE)
        assert isinstance(model.layers["2"].mlp, MoE)

    def test_moe_config_inferred_from_config(self, config, backend_config):
        model = HyMT2Model(config, backend=backend_config)
        mc = model.moe_config
        assert mc.dim == config.hidden_size
        assert mc.moe_inter_dim == config.moe_intermediate_size
        assert mc.n_routed_experts == config.num_experts
        assert mc.n_activated_experts == config.num_experts_per_tok
        assert mc.n_shared_experts == config.num_shared_experts
        assert mc.score_func == "sigmoid"  # because moe_router_use_sigmoid=True
        assert mc.expert_activation == "swiglu"
        assert mc.route_scale == config.router_scaling_factor
        assert mc.norm_topk_prob is True  # because route_norm=True
        assert mc.force_e_score_correction_bias is True

    def test_score_func_follows_use_sigmoid_flag(self, config, backend_config):
        config.moe_router_use_sigmoid = False
        model = HyMT2Model(config, backend=backend_config)
        assert model.moe_config.score_func == "softmax"

    def test_expert_hidden_dim_preferred_over_moe_intermediate(self, config, backend_config):
        # When both are set, expert_hidden_dim wins for the expert MLP dim.
        config.expert_hidden_dim = 32
        config.moe_intermediate_size = 999  # would be wrong if used
        model = HyMT2Model(config, backend=backend_config)
        assert model.moe_config.moe_inter_dim == 32

    def test_moe_overrides_take_effect(self, config, backend_config):
        model = HyMT2Model(config, backend=backend_config, moe_overrides={"score_func": "softmax", "route_scale": 1.5})
        assert model.moe_config.score_func == "softmax"
        assert model.moe_config.route_scale == 1.5

    def test_explicit_moe_config_and_overrides_conflict(self, config, backend_config, moe_config):
        with pytest.raises(ValueError, match="Cannot pass both"):
            HyMT2Model(config, backend=backend_config, moe_config=moe_config, moe_overrides={"score_func": "softmax"})


class TestHyMT2ForCausalLM:
    def test_model_class_alias(self):
        assert ModelClass is HyMT2ForCausalLM

    def test_construction(self, config, backend_config):
        model = HyMT2ForCausalLM(config, backend=backend_config)
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")
        assert model.config is config
        assert model._enable_lm_head_fp32 is True

    def test_enable_lm_head_fp32_default_false_without_config_flag(self, backend_config):
        # When the config does not declare the flag, default to False.
        class _Cfg:
            vocab_size = 32
            hidden_size = HIDDEN
            intermediate_size = INTER
            moe_intermediate_size = MOE_INTER
            num_hidden_layers = 1
            num_attention_heads = N_HEADS
            num_key_value_heads = N_KV
            head_dim = HEAD_DIM
            num_experts = N_EXPERTS
            num_experts_per_tok = 2
            num_shared_experts = 1
            first_k_dense_replace = 1
            max_position_embeddings = 128
            rope_theta = 10000.0
            # PretrainedConfig populates ``rope_parameters`` from ``rope_theta``
            # in its ``__init__``; this bare mock skips that, so declare it
            # explicitly to match what ``get_rope_config`` reads.
            rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
            rms_norm_eps = 1e-5
            torch_dtype = "bfloat16"
            attention_bias = False
            qk_norm = True
            route_norm = False
            router_scaling_factor = 1.0
            moe_router_enable_expert_bias = False
            moe_router_use_sigmoid = True

        model = HyMT2ForCausalLM(_Cfg(), backend=backend_config)
        assert model._enable_lm_head_fp32 is False

    def test_lm_head_fp32_upcast_when_weight_promoted(self, config, backend_config, device):
        """When the parallelizer has promoted lm_head.weight to fp32 (the
        ``distributed.moe.lm_head_precision: float32`` path), the in-model
        fallback feeds the bf16 hidden state up to fp32, runs lm_head, and
        casts logits back to bf16."""
        model = HyMT2ForCausalLM(config, backend=backend_config).to(device).to(torch.bfloat16)
        # Simulate the parallelizer's promotion of lm_head.weight to fp32.
        model.lm_head = model.lm_head.to(torch.float32)
        bf16_hidden = torch.randn(1, 4, HIDDEN, device=device, dtype=torch.bfloat16)
        with patch.object(model.model, "forward", return_value=bf16_hidden):
            input_ids = torch.randint(0, config.vocab_size, (1, 4), device=device)
            logits = model(input_ids).logits
        # Output dtype must be the input dtype (bf16), not fp32.
        assert logits.dtype == torch.bfloat16

    def test_lm_head_no_upcast_when_weight_is_bf16(self, config, backend_config, device):
        """If the parallelizer did NOT promote lm_head.weight, the model must
        fall through to ``self.lm_head(hidden)`` without trying to upcast,
        to avoid the dtype mismatch (fp32 input vs bf16 weight) and the
        ``F.linear`` DTensor mixing crash that the prior implementation hit."""
        model = HyMT2ForCausalLM(config, backend=backend_config).to(device).to(torch.bfloat16)
        # lm_head.weight is bf16 (no promotion). Even though enable_lm_head_fp32
        # is True on the config, the in-model path must NOT activate.
        assert model.lm_head.weight.dtype == torch.bfloat16
        assert model._enable_lm_head_fp32 is True
        bf16_hidden = torch.randn(1, 4, HIDDEN, device=device, dtype=torch.bfloat16)
        with patch.object(model.model, "forward", return_value=bf16_hidden):
            input_ids = torch.randint(0, config.vocab_size, (1, 4), device=device)
            logits = model(input_ids).logits
        assert logits.dtype == torch.bfloat16

    def test_lm_head_no_upcast_when_disabled(self, config, backend_config, device):
        config.enable_lm_head_fp32 = False
        model = HyMT2ForCausalLM(config, backend=backend_config).to(device).to(torch.bfloat16)
        bf16_hidden = torch.randn(1, 4, HIDDEN, device=device, dtype=torch.bfloat16)
        with patch.object(model.model, "forward", return_value=bf16_hidden):
            input_ids = torch.randint(0, config.vocab_size, (1, 4), device=device)
            logits = model(input_ids).logits
        assert logits.dtype == torch.bfloat16

    def test_get_set_input_embeddings(self, config, backend_config):
        model = HyMT2ForCausalLM(config, backend=backend_config)
        emb = model.get_input_embeddings()
        assert emb is model.model.embed_tokens
        new_emb = torch.nn.Embedding(8, HIDDEN)
        model.set_input_embeddings(new_emb)
        assert model.get_input_embeddings() is new_emb

    def test_get_set_output_embeddings(self, config, backend_config):
        model = HyMT2ForCausalLM(config, backend=backend_config)
        assert model.get_output_embeddings() is model.lm_head
        new_head = torch.nn.Linear(HIDDEN, 8, bias=False)
        model.set_output_embeddings(new_head)
        assert model.get_output_embeddings() is new_head
