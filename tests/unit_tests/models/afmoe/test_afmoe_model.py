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

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.afmoe.config import AfmoeConfig
from nemo_automodel.components.models.afmoe.model import AfmoeForCausalLM, AfmoeModel, Block, _build_moe_config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, Gate, MoE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def tiny_config():
    return AfmoeConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_dense_layers=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        score_func="sigmoid",
        route_norm=True,
        route_scale=2.0,
        global_attn_every_n_layers=2,
        sliding_window=64,
        mup_enabled=False,
        n_group=1,
        topk_group=1,
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


class TestBlock:
    def test_dense_layer_uses_mlp(self, tiny_config, backend_config):
        moe_config = _build_moe_config(tiny_config)
        block = Block(layer_idx=0, config=tiny_config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MLP)

    def test_moe_layer_uses_moe(self, tiny_config, backend_config):
        moe_config = _build_moe_config(tiny_config)
        block = Block(layer_idx=1, config=tiny_config, moe_config=moe_config, backend=backend_config)
        assert isinstance(block.mlp, MoE)

    def test_has_four_norms(self, tiny_config, backend_config):
        moe_config = _build_moe_config(tiny_config)
        block = Block(layer_idx=0, config=tiny_config, moe_config=moe_config, backend=backend_config)
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert hasattr(block, "pre_mlp_layernorm")
        assert hasattr(block, "post_mlp_layernorm")

    def test_forward_shape(self, tiny_config, backend_config, device):
        moe_config = _build_moe_config(tiny_config)
        block = Block(layer_idx=0, config=tiny_config, moe_config=moe_config, backend=backend_config).to(device)

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, tiny_config.hidden_size, device=device)
        freqs_cis = torch.randn(batch, seq_len, tiny_config.head_dim, device=device)

        with (
            patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn,
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp,
        ):
            out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()


class TestAfmoeModel:
    def test_initialization(self, tiny_config, backend_config):
        model = AfmoeModel(tiny_config, backend=backend_config)

        assert model.config == tiny_config
        assert len(model.layers) == tiny_config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == tiny_config.vocab_size

    def test_layer_types_correct(self, tiny_config, backend_config):
        model = AfmoeModel(tiny_config, backend=backend_config)

        # layer 0: sliding, layer 1: full, layer 2: sliding, layer 3: full
        assert model.layers["0"].self_attn.is_local_attention is True
        assert model.layers["1"].self_attn.is_local_attention is False
        assert model.layers["2"].self_attn.is_local_attention is True
        assert model.layers["3"].self_attn.is_local_attention is False

    def test_dense_vs_moe_layers(self, tiny_config, backend_config):
        model = AfmoeModel(tiny_config, backend=backend_config)

        # layer 0 is dense (num_dense_layers=1), layers 1-3 are MoE
        assert isinstance(model.layers["0"].mlp, MLP)
        assert isinstance(model.layers["1"].mlp, MoE)
        assert isinstance(model.layers["2"].mlp, MoE)
        assert isinstance(model.layers["3"].mlp, MoE)


class TestAfmoeForCausalLM:
    def test_forward_returns_logits(self, tiny_config, backend_config, device):
        model = AfmoeForCausalLM(tiny_config, backend=backend_config).to(device)

        batch, seq_len = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(
            model.model,
            "forward",
            return_value=torch.randn(batch, seq_len, tiny_config.hidden_size, device=device, dtype=torch.bfloat16),
        ):
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, tiny_config.vocab_size)

    def test_state_dict_adapter_created(self, tiny_config):
        backend = BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=True,
        )
        model = AfmoeForCausalLM(tiny_config, backend=backend)
        assert hasattr(model, "state_dict_adapter")

    def test_modelclass_export(self):
        from nemo_automodel.components.models.afmoe import model as afmoe_mod

        assert hasattr(afmoe_mod, "ModelClass")
        assert afmoe_mod.ModelClass is AfmoeForCausalLM

    def test_from_pretrained_classmethod(self, tiny_config):
        with patch.object(AfmoeConfig, "from_pretrained", return_value=tiny_config):
            with patch.object(AfmoeForCausalLM, "from_config", wraps=AfmoeForCausalLM.from_config) as mock:
                model = AfmoeForCausalLM.from_pretrained("arcee-ai/Trinity-Large-Thinking")
                assert isinstance(model, AfmoeForCausalLM)
                mock.assert_called_once()


class TestBuildMoeConfig:
    def test_fields_mapped_correctly(self, tiny_config):
        moe_cfg = _build_moe_config(tiny_config)

        assert moe_cfg.dim == tiny_config.hidden_size
        assert moe_cfg.inter_dim == tiny_config.intermediate_size
        assert moe_cfg.moe_inter_dim == tiny_config.moe_intermediate_size
        assert moe_cfg.n_routed_experts == tiny_config.num_experts
        assert moe_cfg.n_shared_experts == tiny_config.num_shared_experts
        assert moe_cfg.n_activated_experts == tiny_config.num_experts_per_tok
        assert moe_cfg.score_func == "sigmoid"
        assert moe_cfg.route_scale == tiny_config.route_scale
        assert moe_cfg.norm_topk_prob is True
        assert moe_cfg.force_e_score_correction_bias is True


class TestDualNormParity:
    def test_manual_trace_matches_forward(self, tiny_config, backend_config, device):
        """Manual 4-norm residual trace must be bit-identical to Block.forward()."""
        torch.manual_seed(42)
        moe_config = _build_moe_config(tiny_config)
        block = Block(layer_idx=0, config=tiny_config, moe_config=moe_config, backend=backend_config).to(device)
        block.eval()

        batch, seq_len = 1, 4
        x = torch.randn(batch, seq_len, tiny_config.hidden_size, device=device, dtype=torch.bfloat16)
        if backend_config.rope_fusion:
            freqs_cis = torch.randn(seq_len, 1, 1, tiny_config.head_dim, device=device)
        else:
            freqs_cis = torch.randn(batch, seq_len, tiny_config.head_dim, device=device)

        with torch.no_grad():
            # Manual trace: attention sublayer
            residual = x
            h = block.input_layernorm(x)
            h = block.self_attn(h, freqs_cis=freqs_cis)
            h = block.post_attention_layernorm(h)
            after_attn = residual + h

            # Manual trace: MLP sublayer
            residual = after_attn
            h = block.pre_mlp_layernorm(after_attn)
            h = block._mlp(h, padding_mask=None)
            h = block.post_mlp_layernorm(h)
            expected = residual + h

            # Block forward
            actual = block(x, freqs_cis=freqs_cis)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestMoeRoutingParity:
    def test_sigmoid_norm_scale(self, device):
        """Manual sigmoid -> topk -> normalize -> scale must match Gate.forward()."""
        torch.manual_seed(42)

        moe_config = MoEConfig(
            dim=64,
            inter_dim=128,
            moe_inter_dim=32,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            score_func="sigmoid",
            route_scale=2.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=True,
            force_e_score_correction_bias=True,
            dtype=torch.bfloat16,
        )

        gate = Gate(moe_config).to(device)
        torch.manual_seed(123)
        gate.weight.data = torch.randn(4, 64, device=device, dtype=torch.bfloat16)

        x = torch.randn(8, 64, device=device, dtype=torch.bfloat16)  # 8 tokens
        token_mask = torch.ones(8, dtype=torch.bool, device=device)

        with torch.no_grad():
            weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Manual reference: sigmoid -> bias -> topk -> gather original -> normalize -> scale
        with torch.no_grad():
            scores = torch.sigmoid(x @ gate.weight.data.T)  # [8, 4]
            original_scores = scores.clone()
            biased = scores + gate.e_score_correction_bias  # zeros, no-op
            manual_idx = torch.topk(biased, 2, dim=-1)[1]
            manual_w = original_scores.gather(1, manual_idx)
            manual_w = manual_w / (manual_w.sum(dim=-1, keepdim=True) + 1e-20)
            manual_w = manual_w * 2.0

        assert torch.equal(indices, manual_idx), "Expert indices mismatch"
        torch.testing.assert_close(weights, manual_w, rtol=1e-3, atol=1e-3)
        assert aux_loss is None
