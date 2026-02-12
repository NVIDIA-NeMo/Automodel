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

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.glm4_moe_lite.model import (
    Block,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteModel,
    ModelClass,
)
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.models.common.utils import BackendConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@dataclass
class MockGlm4MoeLiteConfig:
    """Mock config combining GLM4 MoE config with MLA-specific fields."""

    # Basic model config
    vocab_size: int = 256
    hidden_size: int = 64
    num_hidden_layers: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 256
    rope_theta: float = 5000.0
    torch_dtype: str = "bfloat16"

    # MLA config
    num_attention_heads: int = 4
    q_lora_rank: int = 16
    kv_lora_rank: int = 8
    qk_nope_head_dim: int = 8
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16
    rope_scaling: dict = None

    # MoE config
    intermediate_size: int = 128
    moe_intermediate_size: int = 64
    n_routed_experts: int = 4
    n_shared_experts: int = 0
    num_experts_per_tok: int = 2
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True

    # Layer types (first layer dense, rest sparse)
    mlp_layer_types: list = None

    def __post_init__(self):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def config():
    return MockGlm4MoeLiteConfig()


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def moe_config(config):
    return make_moe_config(config)


def make_moe_config(config: MockGlm4MoeLiteConfig) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
    )


class TestBlock:
    def test_block_initializes_with_mla_attention(self, config, moe_config, backend_config):
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config)

        from nemo_automodel.components.models.deepseek_v3.layers import MLA

        assert isinstance(block.self_attn, MLA)

    def test_block_initializes_moe_for_sparse_layer(self, config, moe_config, backend_config):
        # Layer 1 is sparse based on mlp_layer_types
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=backend_config)

        assert isinstance(block.mlp, MoE)
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")

    def test_block_initializes_mlp_for_dense_layer(self, config, moe_config, backend_config):
        # Layer 0 is dense based on mlp_layer_types
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)

        assert isinstance(block.mlp, MLP)

    def test_forward_pass_calls_attention_and_mlp(self, config, moe_config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)
        block = block.to(device)

        batch, seq_len = 2, 4
        x = torch.randn(batch, seq_len, config.hidden_size, device=device)
        # freqs_cis for MLA (needs proper shape for fused rope)
        freqs_cis = torch.randn(seq_len, 1, 1, config.qk_rope_head_dim * 2, device=device)

        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn, patch.object(
            block, "_mlp", return_value=torch.zeros_like(x)
        ) as mock_mlp:
            out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()

    def test_forward_builds_padding_mask_from_attention(self, config, moe_config, backend_config, device):
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)
        block = block.to(device)

        x = torch.randn(1, 3, config.hidden_size, device=device)
        freqs_cis = torch.randn(3, 1, 1, config.qk_rope_head_dim * 2, device=device)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)

        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn, patch.object(
            block, "_mlp", return_value=torch.zeros_like(x)
        ) as mock_mlp:
            block(x, freqs_cis=freqs_cis, attention_mask=attention_mask)

        mock_attn.assert_called_once()
        _, kwargs = mock_mlp.call_args
        padding_mask = kwargs.get("padding_mask")
        assert padding_mask is not None
        torch.testing.assert_close(padding_mask, attention_mask.logical_not())

    def test_mlp_wrapper_handles_mlp_instance(self, config, moe_config, backend_config):
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)
        # Layer 0 uses MLP
        x = torch.randn(2, 4, config.hidden_size).to(torch.bfloat16)

        out = block._mlp(x, padding_mask=None)

        assert out.shape == x.shape

    def test_init_weights_resets_sublayers(self, config, moe_config, backend_config):
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=backend_config)

        with patch.object(block.input_layernorm, "reset_parameters") as mock_in, patch.object(
            block.post_attention_layernorm, "reset_parameters"
        ) as mock_post, patch.object(block.self_attn, "init_weights") as mock_attn, patch.object(
            block.mlp, "init_weights"
        ) as mock_mlp:
            block.init_weights(torch.device("cpu"))

        mock_in.assert_called_once()
        mock_post.assert_called_once()
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()


class TestGlm4MoeLiteModel:
    def test_model_initialization_sets_components(self, config, backend_config):
        model = Glm4MoeLiteModel(config, backend=backend_config)

        assert model.config == config
        assert model.backend == backend_config
        assert len(model.layers) == config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == config.vocab_size
        assert model.qk_rope_head_dim == config.qk_rope_head_dim

    def test_model_creates_correct_moe_config(self, config, backend_config):
        model = Glm4MoeLiteModel(config, backend=backend_config)

        assert model.moe_config.dim == config.hidden_size
        assert model.moe_config.n_routed_experts == config.n_routed_experts
        assert model.moe_config.n_activated_experts == config.num_experts_per_tok
        assert model.moe_config.score_func == "sigmoid"

    def test_forward_runs_all_layers(self, config, backend_config):
        model = Glm4MoeLiteModel(config, backend=backend_config)

        batch, seq_len = 2, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

        with patch.object(
            Block, "forward", side_effect=lambda *_, **__: torch.randn(batch, seq_len, config.hidden_size)
        ) as mock_block:
            out = model(input_ids)

        assert out.shape == (batch, seq_len, config.hidden_size)
        assert mock_block.call_count == config.num_hidden_layers

    def test_forward_accepts_position_ids(self, config, backend_config):
        model = Glm4MoeLiteModel(config, backend=backend_config)
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)

        with patch.object(
            Block, "forward", return_value=torch.zeros(batch, seq_len, config.hidden_size)
        ):
            out = model(input_ids, position_ids=position_ids)

        assert out.shape == (batch, seq_len, config.hidden_size)

    def test_init_weights_updates_embeddings_and_layers(self, config, backend_config):
        model = Glm4MoeLiteModel(config, backend=backend_config)
        original = model.embed_tokens.weight.clone()

        with patch.object(model.norm, "reset_parameters") as mock_norm, patch.object(
            Block, "init_weights"
        ) as mock_layer_init:
            model.init_weights(torch.device("cpu"))

        mock_norm.assert_called_once()
        assert not torch.equal(model.embed_tokens.weight, original)
        assert mock_layer_init.call_count == config.num_hidden_layers


class TestGlm4MoeLiteForCausalLM:
    def test_forward_returns_logits(self, config, backend_config, device):
        model = Glm4MoeLiteForCausalLM(config, backend=backend_config)
        model = model.to(device)

        batch, seq_len = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with patch.object(
            model.model,
            "forward",
            return_value=torch.randn(batch, seq_len, config.hidden_size, device=device).to(torch.bfloat16),
        ):
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_forward_handles_thd_format(self, config, backend_config, device):
        model = Glm4MoeLiteForCausalLM(config, backend=backend_config)
        model = model.to(device)

        batch, seq_len = 1, 4
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)

        with patch.object(
            model.model,
            "forward",
            return_value=torch.randn(seq_len, config.hidden_size, device=device).to(torch.bfloat16),
        ):
            logits = model(input_ids, position_ids=position_ids, padding_mask=padding_mask, qkv_format="thd")

        # thd format should add batch dimension back
        assert logits.shape == (1, seq_len, config.vocab_size)

    def test_initialize_weights_invokes_submodules(self, config, backend_config):
        model = Glm4MoeLiteForCausalLM(config, backend=backend_config)
        original = model.lm_head.weight.clone()

        with patch.object(model.model, "init_weights") as mock_init:
            model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        mock_init.assert_called_once()
        assert not torch.equal(model.lm_head.weight, original)
        assert model.lm_head.weight.dtype == torch.float32

    def test_state_dict_adapter_created_when_enabled(self, config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Glm4MoeLiteForCausalLM(config, backend=backend_config)

        assert hasattr(model, "state_dict_adapter")

    def test_from_config_classmethod(self, config, backend_config):
        model = Glm4MoeLiteForCausalLM.from_config(config, backend=backend_config)

        assert isinstance(model, Glm4MoeLiteForCausalLM)
        assert model.config == config


class TestModelClass:
    def test_modelclass_export_exists(self):
        from nemo_automodel.components.models.glm4_moe_lite import model as glm4_moe_lite_mod

        assert hasattr(glm4_moe_lite_mod, "ModelClass")
        assert glm4_moe_lite_mod.ModelClass is Glm4MoeLiteForCausalLM

    def test_modelclass_equals_glm4moeliteforclm(self):
        assert ModelClass is Glm4MoeLiteForCausalLM
