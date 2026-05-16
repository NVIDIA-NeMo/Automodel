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

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

# Skip module if HF doesn't have the configurations available (older transformers).
pytest.importorskip("transformers.models.ernie4_5")
pytest.importorskip("transformers.models.ernie4_5_moe")

from transformers.models.ernie4_5.configuration_ernie4_5 import Ernie4_5Config
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie4_5.model import (
    Ernie4_5_MoeForCausalLM,
    Ernie4_5_MoeModel,
    Ernie4_5Attention,
    Ernie4_5Block,
    Ernie4_5ForCausalLM,
    Ernie4_5Model,
    Ernie4_5MoeBlock,
    ModelClass,
)
from nemo_automodel.components.models.ernie4_5.state_dict_adapter import (
    Ernie4_5_MoeStateDictAdapter,
    Ernie4_5StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE


@pytest.fixture
def dense_config():
    return Ernie4_5Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,
        tie_word_embeddings=True,
        pad_token_id=0,
    )


@pytest.fixture
def moe_hf_config():
    """Small Ernie4.5-MoE config: 4 layers, layers 1..3 are MoE (interval=1)."""
    return Ernie4_5_MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        moe_intermediate_size=16,
        moe_k=2,
        moe_num_experts=4,
        # Use 0 shared experts so MoE.forward stays on CPU (the shared-experts
        # path eagerly allocates torch.cuda.Stream, which crashes on CPU-only CI).
        moe_num_shared_experts=0,
        moe_layer_start_index=1,
        moe_layer_end_index=3,
        moe_layer_interval=1,
        router_aux_loss_coef=0.001,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


# ---------------------------------------------------------------------------
# Ernie4_5Attention
# ---------------------------------------------------------------------------
class TestErnie4_5Attention:
    def test_projection_shapes(self, dense_config, backend_config):
        attn = Ernie4_5Attention(dense_config, backend_config)
        assert attn.num_heads == dense_config.num_attention_heads
        assert attn.num_kv_heads == dense_config.num_key_value_heads
        assert attn.head_dim == dense_config.head_dim
        assert attn.q_proj.weight.shape == (
            dense_config.num_attention_heads * dense_config.head_dim,
            dense_config.hidden_size,
        )
        assert attn.k_proj.weight.shape == (
            dense_config.num_key_value_heads * dense_config.head_dim,
            dense_config.hidden_size,
        )
        assert attn.v_proj.weight.shape == (
            dense_config.num_key_value_heads * dense_config.head_dim,
            dense_config.hidden_size,
        )
        assert attn.o_proj.weight.shape == (
            dense_config.hidden_size,
            dense_config.num_attention_heads * dense_config.head_dim,
        )


# ---------------------------------------------------------------------------
# Dense decoder block
# ---------------------------------------------------------------------------
class TestErnie4_5Block:
    def test_uses_mlp(self, dense_config, backend_config):
        block = Ernie4_5Block(dense_config, backend_config)
        assert isinstance(block.mlp, MLP)
        assert hasattr(block, "self_attn")
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")


# ---------------------------------------------------------------------------
# MoE decoder block: dense-vs-MoE selection
# ---------------------------------------------------------------------------
class TestErnie4_5MoeBlock:
    @pytest.fixture
    def moe_config(self, moe_hf_config):
        return MoEConfig(
            dim=moe_hf_config.hidden_size,
            inter_dim=moe_hf_config.intermediate_size,
            moe_inter_dim=moe_hf_config.moe_intermediate_size,
            n_routed_experts=moe_hf_config.moe_num_experts,
            n_shared_experts=moe_hf_config.moe_num_shared_experts,
            n_activated_experts=moe_hf_config.moe_k,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax_with_bias",
            route_scale=1.0,
            aux_loss_coeff=moe_hf_config.router_aux_loss_coef,
            norm_topk_prob=True,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            shared_expert_inter_dim=moe_hf_config.moe_intermediate_size,
            shared_expert_activation="swiglu",
            force_e_score_correction_bias=True,
        )

    def test_layer_before_start_is_dense(self, moe_hf_config, moe_config, backend_config):
        """layer_idx=0 < moe_layer_start_index=1, so this block uses MLP."""
        block = Ernie4_5MoeBlock(0, moe_hf_config, moe_config, backend_config)
        assert isinstance(block.mlp, MLP)

    def test_layer_in_range_is_moe(self, moe_hf_config, moe_config, backend_config):
        """layer_idx=2 ∈ [1, 3] and (2+1)%1==0 → MoE."""
        block = Ernie4_5MoeBlock(2, moe_hf_config, moe_config, backend_config)
        assert isinstance(block.mlp, MoE)

    def test_layer_skipped_by_interval_is_dense(self, moe_hf_config, moe_config, backend_config):
        """With moe_layer_interval=2, only odd-after-+1 indices become MoE."""
        moe_hf_config.moe_layer_interval = 2
        # layer 1: (1+1)%2 == 0 → MoE
        block_moe = Ernie4_5MoeBlock(1, moe_hf_config, moe_config, backend_config)
        assert isinstance(block_moe.mlp, MoE)
        # layer 2: (2+1)%2 != 0 → MLP
        block_dense = Ernie4_5MoeBlock(2, moe_hf_config, moe_config, backend_config)
        assert isinstance(block_dense.mlp, MLP)


# ---------------------------------------------------------------------------
# Dense model
# ---------------------------------------------------------------------------
class TestErnie4_5Model:
    def test_structure(self, dense_config, backend_config):
        model = Ernie4_5Model(dense_config, backend_config)
        assert model.config is dense_config
        assert model.backend is backend_config
        assert len(model.layers) == dense_config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == dense_config.vocab_size
        assert model.embed_tokens.embedding_dim == dense_config.hidden_size


# ---------------------------------------------------------------------------
# MoE model
# ---------------------------------------------------------------------------
class TestErnie4_5_MoeModel:
    def test_moe_config_defaults_from_hf_config(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeModel(moe_hf_config, backend_config)
        assert hasattr(model, "moe_config")
        cfg = model.moe_config
        assert cfg.dim == moe_hf_config.hidden_size
        assert cfg.inter_dim == moe_hf_config.intermediate_size
        assert cfg.moe_inter_dim == moe_hf_config.moe_intermediate_size
        assert cfg.n_routed_experts == moe_hf_config.moe_num_experts
        assert cfg.n_shared_experts == moe_hf_config.moe_num_shared_experts
        assert cfg.n_activated_experts == moe_hf_config.moe_k
        assert cfg.aux_loss_coeff == moe_hf_config.router_aux_loss_coef
        assert cfg.score_func == "softmax_with_bias"
        assert cfg.norm_topk_prob is True
        assert cfg.expert_bias == moe_hf_config.use_bias
        assert cfg.router_bias is False
        assert cfg.force_e_score_correction_bias is True

    def test_moe_overrides_apply(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeModel(
            moe_hf_config,
            backend_config,
            moe_overrides={"aux_loss_coeff": 0.05, "norm_topk_prob": False},
        )
        assert model.moe_config.aux_loss_coeff == 0.05
        assert model.moe_config.norm_topk_prob is False

    def test_accepts_explicit_moe_config(self, moe_hf_config, backend_config):
        cfg = MoEConfig(
            dim=moe_hf_config.hidden_size,
            inter_dim=moe_hf_config.intermediate_size,
            moe_inter_dim=moe_hf_config.moe_intermediate_size,
            n_routed_experts=moe_hf_config.moe_num_experts,
            n_shared_experts=0,
            n_activated_experts=moe_hf_config.moe_k,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=True,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
        )
        model = Ernie4_5_MoeModel(moe_hf_config, backend_config, moe_config=cfg)
        assert model.moe_config is cfg

    def test_rejects_both_moe_config_and_overrides(self, moe_hf_config, backend_config):
        cfg = MoEConfig(
            dim=32,
            inter_dim=64,
            moe_inter_dim=16,
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=True,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
        )
        with pytest.raises(ValueError, match="Cannot pass both"):
            Ernie4_5_MoeModel(moe_hf_config, backend_config, moe_config=cfg, moe_overrides={"aux_loss_coeff": 0.1})

    def test_structure(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeModel(moe_hf_config, backend_config)
        assert len(model.layers) == moe_hf_config.num_hidden_layers
        # Layer 0 below start_index → MLP; layer 1..3 → MoE
        assert isinstance(model.layers["0"].mlp, MLP)
        assert isinstance(model.layers["1"].mlp, MoE)
        assert isinstance(model.layers["2"].mlp, MoE)
        assert isinstance(model.layers["3"].mlp, MoE)


# ---------------------------------------------------------------------------
# Causal-LM heads
# ---------------------------------------------------------------------------
class TestErnie4_5ForCausalLM:
    def test_from_config_constructs(self, dense_config, backend_config):
        model = Ernie4_5ForCausalLM.from_config(dense_config, backend=backend_config)
        assert isinstance(model, Ernie4_5ForCausalLM)
        assert model.config is dense_config

    def test_from_pretrained_uses_classmethod(self, dense_config, backend_config):
        with patch(
            "transformers.models.ernie4_5.configuration_ernie4_5.Ernie4_5Config.from_pretrained",
            return_value=dense_config,
        ) as mock_from_pretrained:
            model = Ernie4_5ForCausalLM.from_pretrained("baidu/ernie-4.5", backend=backend_config)
        mock_from_pretrained.assert_called_once_with("baidu/ernie-4.5")
        assert isinstance(model, Ernie4_5ForCausalLM)

    def test_lm_head_shape(self, dense_config, backend_config):
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        assert model.lm_head.weight.shape == (dense_config.vocab_size, dense_config.hidden_size)
        assert model.vocab_size == dense_config.vocab_size

    def test_lm_head_tied(self, dense_config, backend_config):
        dense_config.tie_word_embeddings = True
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_lm_head_untied(self, dense_config, backend_config):
        dense_config.tie_word_embeddings = False
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        assert model.lm_head.weight is not model.model.embed_tokens.weight

    def test_tie_weights_rebinds(self, dense_config, backend_config):
        dense_config.tie_word_embeddings = True
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        # Replace lm_head with a fresh tensor; tie_weights should restore the tie.
        model.lm_head.weight = torch.nn.Parameter(torch.zeros_like(model.lm_head.weight))
        assert model.lm_head.weight is not model.model.embed_tokens.weight
        model.tie_weights()
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_input_output_embeddings_accessors(self, dense_config, backend_config):
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        assert model.get_input_embeddings() is model.model.embed_tokens
        assert model.get_output_embeddings() is model.lm_head

        new_emb = torch.nn.Embedding(dense_config.vocab_size, dense_config.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.model.embed_tokens is new_emb

        new_head = torch.nn.Linear(dense_config.hidden_size, dense_config.vocab_size, bias=False)
        model.set_output_embeddings(new_head)
        assert model.lm_head is new_head

    def test_state_dict_adapter_off_by_default(self, dense_config, backend_config):
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        assert not hasattr(model, "state_dict_adapter")

    def test_state_dict_adapter_when_enabled(self, dense_config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        assert isinstance(model.state_dict_adapter, Ernie4_5StateDictAdapter)


class TestErnie4_5_MoeForCausalLM:
    def test_from_config_constructs(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeForCausalLM.from_config(moe_hf_config, backend=backend_config)
        assert isinstance(model, Ernie4_5_MoeForCausalLM)
        assert model.config is moe_hf_config

    def test_from_pretrained_uses_classmethod(self, moe_hf_config, backend_config):
        with patch(
            "transformers.models.ernie4_5_moe.configuration_ernie4_5_moe.Ernie4_5_MoeConfig.from_pretrained",
            return_value=moe_hf_config,
        ) as mock_from_pretrained:
            model = Ernie4_5_MoeForCausalLM.from_pretrained("baidu/ernie-4.5-moe", backend=backend_config)
        mock_from_pretrained.assert_called_once_with("baidu/ernie-4.5-moe")
        assert isinstance(model, Ernie4_5_MoeForCausalLM)

    def test_lm_head_shape(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        assert model.lm_head.weight.shape == (moe_hf_config.vocab_size, moe_hf_config.hidden_size)

    def test_lm_head_tied(self, moe_hf_config, backend_config):
        moe_hf_config.tie_word_embeddings = True
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_lm_head_untied(self, moe_hf_config, backend_config):
        moe_hf_config.tie_word_embeddings = False
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        assert model.lm_head.weight is not model.model.embed_tokens.weight

    def test_state_dict_adapter_off_by_default(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        assert not hasattr(model, "state_dict_adapter")

    def test_state_dict_adapter_when_enabled(self, moe_hf_config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        assert isinstance(model.state_dict_adapter, Ernie4_5_MoeStateDictAdapter)

    def test_moe_overrides_threaded_through(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeForCausalLM(
            moe_hf_config,
            backend=backend_config,
            moe_overrides={"aux_loss_coeff": 0.123},
        )
        assert model.model.moe_config.aux_loss_coeff == 0.123

    def test_input_output_embeddings_accessors(self, moe_hf_config, backend_config):
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        assert model.get_input_embeddings() is model.model.embed_tokens
        assert model.get_output_embeddings() is model.lm_head

    def test_tie_weights_rebinds(self, moe_hf_config, backend_config):
        moe_hf_config.tie_word_embeddings = True
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        model.lm_head.weight = torch.nn.Parameter(torch.zeros_like(model.lm_head.weight))
        assert model.lm_head.weight is not model.model.embed_tokens.weight
        model.tie_weights()
        assert model.lm_head.weight is model.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Forward-pass shape tests (CPU)
# ---------------------------------------------------------------------------
class TestForwardShapes:
    """Run a tiny forward pass through the dense and MoE causal-LM heads.

    These tests catch regressions in input/position-id handling, embedding
    wiring, and the lm_head output shape across qkv_format variants. They
    deliberately use a tiny config so they run on CPU in CI without GPUs.
    """

    def _dense_model(self, dense_config, backend_config):
        torch.manual_seed(0)
        model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
        return model.to(torch.float32).eval()

    def _moe_model(self, moe_hf_config, backend_config):
        torch.manual_seed(0)
        model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
        return model.to(torch.float32).eval()

    def test_dense_forward_bshd_shape(self, dense_config, backend_config):
        model = self._dense_model(dense_config, backend_config)
        batch, seq = 2, 4
        input_ids = torch.randint(0, dense_config.vocab_size, (batch, seq))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (batch, seq, dense_config.vocab_size)

    def test_dense_forward_thd_shape(self, dense_config, backend_config):
        """thd path: 1-D position_ids must not crash dense Ernie4_5Model.forward."""
        model = self._dense_model(dense_config, backend_config)
        seq = 6
        input_ids = torch.randint(0, dense_config.vocab_size, (1, seq))
        # squeeze_input_for_thd unconditionally squeezes position_ids, so callers
        # must supply one when using qkv_format="thd".
        position_ids = torch.arange(seq).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids, position_ids=position_ids, qkv_format="thd")
        # thd output is unsqueezed back to batch dimension.
        assert logits.shape == (1, seq, dense_config.vocab_size)

    def test_dense_forward_accepts_explicit_position_ids(self, dense_config, backend_config):
        model = self._dense_model(dense_config, backend_config)
        batch, seq = 1, 5
        input_ids = torch.randint(0, dense_config.vocab_size, (batch, seq))
        position_ids = torch.arange(seq).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids, position_ids=position_ids)
        assert logits.shape == (batch, seq, dense_config.vocab_size)

    def test_dense_forward_logits_to_keep_int(self, dense_config, backend_config):
        model = self._dense_model(dense_config, backend_config)
        batch, seq = 1, 8
        input_ids = torch.randint(0, dense_config.vocab_size, (batch, seq))
        with torch.no_grad():
            logits = model(input_ids, logits_to_keep=2)
        assert logits.shape == (batch, 2, dense_config.vocab_size)

    def test_moe_forward_bshd_shape(self, moe_hf_config, backend_config):
        model = self._moe_model(moe_hf_config, backend_config)
        batch, seq = 1, 4
        input_ids = torch.randint(0, moe_hf_config.vocab_size, (batch, seq))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (batch, seq, moe_hf_config.vocab_size)

    def test_moe_forward_thd_shape(self, moe_hf_config, backend_config):
        model = self._moe_model(moe_hf_config, backend_config)
        seq = 4
        input_ids = torch.randint(0, moe_hf_config.vocab_size, (1, seq))
        # squeeze_input_for_thd requires position_ids to be a tensor.
        position_ids = torch.arange(seq).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids, position_ids=position_ids, qkv_format="thd")
        assert logits.shape == (1, seq, moe_hf_config.vocab_size)


# ---------------------------------------------------------------------------
# Layer equivalence against HF reference
# ---------------------------------------------------------------------------
class TestLayerEquivalence:
    """Numerical equivalence between the rewritten NeMo layers and HF references.

    The model-onboarding skill requires every rewritten layer to be compared
    against the original HF implementation. These tests run on CPU in fp32
    with matched weights.
    """

    @pytest.fixture
    def equiv_config(self):
        cfg = Ernie4_5Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            use_bias=False,
            tie_word_embeddings=True,
            pad_token_id=0,
        )
        cfg._attn_implementation = "eager"
        cfg.torch_dtype = torch.float32
        return cfg

    def test_rope_equivalence(self, equiv_config):
        """NeMo rope_utils must rotate q/k identically to HF's reference path."""
        from transformers.models.ernie4_5.modeling_ernie4_5 import (
            Ernie4_5RotaryEmbedding as HFRotary,
        )
        from transformers.models.ernie4_5.modeling_ernie4_5 import (
            apply_rotary_pos_emb as hf_apply_rope,
        )

        from nemo_automodel.components.models.ernie4_5.rope_utils import (
            Ernie4_5RotaryEmbedding as NeMoRotary,
        )
        from nemo_automodel.components.models.ernie4_5.rope_utils import (
            apply_rotary_pos_emb as nemo_apply_rope,
        )

        torch.manual_seed(0)
        batch, seq, heads, hdim = 1, 6, 2, equiv_config.head_dim
        q_bshd = torch.randn(batch, seq, heads, hdim, dtype=torch.float32)
        k_bshd = torch.randn(batch, seq, heads, hdim, dtype=torch.float32)
        position_ids = torch.arange(seq).unsqueeze(0)

        # HF path: rotary returns concat-of-freqs cos/sin; apply expects bhsd q/k.
        hf_rotary = HFRotary(equiv_config)
        cos_h, sin_h = hf_rotary(q_bshd, position_ids)
        q_hf, k_hf = hf_apply_rope(q_bshd.transpose(1, 2), k_bshd.transpose(1, 2), cos_h, sin_h)
        q_hf = q_hf.transpose(1, 2)
        k_hf = k_hf.transpose(1, 2)

        # NeMo path: rotary returns already-interleaved cos/sin; apply uses bshd q/k.
        nemo_rotary = NeMoRotary(equiv_config)
        cos_n, sin_n = nemo_rotary(q_bshd, position_ids)
        q_nm, k_nm = nemo_apply_rope(q_bshd, k_bshd, cos_n, sin_n)

        torch.testing.assert_close(q_nm, q_hf, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_nm, k_hf, atol=1e-5, rtol=1e-5)

    def test_attention_equivalence(self, equiv_config, backend_config):
        """NeMo Ernie4_5Attention must match HF reference attention with shared weights."""
        from transformers.models.ernie4_5.modeling_ernie4_5 import (
            Ernie4_5Attention as HFAttention,
        )
        from transformers.models.ernie4_5.modeling_ernie4_5 import (
            Ernie4_5RotaryEmbedding as HFRotary,
        )

        from nemo_automodel.components.models.ernie4_5.rope_utils import (
            Ernie4_5RotaryEmbedding as NeMoRotary,
        )

        torch.manual_seed(0)
        batch, seq = 1, 4
        hidden = torch.randn(batch, seq, equiv_config.hidden_size, dtype=torch.float32)
        position_ids = torch.arange(seq).unsqueeze(0)

        hf_attn = HFAttention(equiv_config, layer_idx=0).to(torch.float32).eval()
        hf_rotary = HFRotary(equiv_config)
        nemo_attn = Ernie4_5Attention(equiv_config, backend_config).to(torch.float32).eval()

        # Share projection weights so both layers compute on identical q/k/v subspaces.
        with torch.no_grad():
            nemo_attn.q_proj.weight.copy_(hf_attn.q_proj.weight)
            nemo_attn.k_proj.weight.copy_(hf_attn.k_proj.weight)
            nemo_attn.v_proj.weight.copy_(hf_attn.v_proj.weight)
            nemo_attn.o_proj.weight.copy_(hf_attn.o_proj.weight)

        # Additive causal mask for HF eager attention (-inf above diagonal).
        upper_triangle = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        causal_mask = torch.zeros(1, 1, seq, seq, dtype=torch.float32).masked_fill(upper_triangle, float("-inf"))

        cos_h, sin_h = hf_rotary(hidden, position_ids)
        with torch.no_grad():
            hf_out, _ = hf_attn(
                hidden,
                position_embeddings=(cos_h, sin_h),
                attention_mask=causal_mask,
            )

        nemo_rotary = NeMoRotary(equiv_config)
        cos_n, sin_n = nemo_rotary(hidden, position_ids)
        with torch.no_grad():
            nemo_out = nemo_attn(hidden, position_embeddings=(cos_n, sin_n))

        torch.testing.assert_close(hf_out, nemo_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
class TestModelClassExport:
    def test_modelclass_points_to_moe_lm(self):
        assert ModelClass is Ernie4_5_MoeForCausalLM
