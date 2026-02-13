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

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class MockNemotronV3Config:
    """Mock configuration for NemotronV3 model."""

    def __init__(self, **overrides):
        # Attention configuration
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.hidden_size = 256
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP/MoE configuration
        self.intermediate_size = 512
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"

        # Mamba configuration
        self.mamba_num_heads = 4
        self.mamba_head_dim = 32
        self.ssm_state_size = 16
        self.n_groups = 1
        self.chunk_size = 256
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # General configuration
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 2
        self.vocab_size = 100
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False

        # Layer types for hybrid architecture - using only attention and mlp for tests
        self.layers_block_type = ["attention", "mlp"]

        # MoE configuration
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = 128
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = 128

        # Apply overrides
        for key, value in overrides.items():
            setattr(self, key, value)


class TestNemotronV3Model:
    """Test NemotronV3Model base model."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            enable_deepep=False,
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    def test_model_init(self, config, backend):
        """Test NemotronV3Model initialization."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.config == config
        assert model.backend == backend
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")

    def test_model_layers_count(self, config, backend):
        """Test that model has correct number of layers."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert len(model.layers) == config.num_hidden_layers

    def test_model_layer_types(self, config, backend):
        """Test that model layers have correct types."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.layers["0"].block_type == "attention"
        assert model.layers["1"].block_type == "mlp"

    def test_model_embedding_dimensions(self, config, backend):
        """Test that embeddings have correct dimensions."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.embed_tokens.num_embeddings == config.vocab_size
        assert model.embed_tokens.embedding_dim == config.hidden_size

    def test_model_forward_shape(self, config, backend):
        """Test model forward pass produces correct shape."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_with_mask(self, config, backend):
        """Test model forward pass with attention mask."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(input_ids, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_with_causal_mask_mapping(self, config, backend):
        """Test model forward pass with causal mask mapping."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Create causal mask mapping
        causal_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float("-inf")
        )
        causal_mask_mapping = {"full_attention": causal_mask}

        output = model(input_ids, causal_mask_mapping=causal_mask_mapping)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_with_inputs_embeds(self, config, backend):
        """Test model forward pass with inputs_embeds instead of input_ids."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = model(inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_inputs_embeds_bypasses_embedding(self, config, backend):
        """Test that inputs_embeds bypasses the embedding layer."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        # Should work even with input_ids=None (the default)
        output = model(input_ids=None, inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_inputs_embeds_takes_precedence(self, config, backend):
        """Test that inputs_embeds takes precedence over input_ids when both provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        # When both are provided, inputs_embeds should be used (input_ids ignored)
        output = model(input_ids, inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_no_input_ids_no_inputs_embeds_raises(self, config, backend):
        """Test that ValueError is raised when neither input_ids nor inputs_embeds is provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        with pytest.raises(ValueError, match="input_ids must be provided if inputs_embeds is not provided"):
            model(input_ids=None)

    def test_model_forward_inputs_embeds_with_mask(self, config, backend):
        """Test model forward pass with inputs_embeds and attention mask."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_moe_config_creation(self, config, backend):
        """Test that model creates MoE config correctly."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        # Add MoE layer
        config.layers_block_type = ["moe", "attention"]

        model = NemotronV3Model(config, backend=backend)

        assert model.moe_config is not None
        assert model.moe_config.n_routed_experts == config.n_routed_experts
        assert model.moe_config.n_activated_experts == config.num_experts_per_tok
        assert model.moe_config.expert_activation == "relu2"
        assert model.moe_config.shared_expert_activation == "relu2"

    def test_model_custom_moe_config(self, config, backend):
        """Test that model uses custom MoE config when provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        custom_moe_config = MoEConfig(
            n_routed_experts=16,
            n_shared_experts=2,
            n_activated_experts=4,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

        config.layers_block_type = ["moe", "attention"]

        model = NemotronV3Model(config, backend=backend, moe_config=custom_moe_config)

        assert model.moe_config == custom_moe_config
        assert model.moe_config.n_routed_experts == 16

    def test_model_dtype(self, config, backend):
        """Test that model uses correct dtype."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        # Embeddings should be bfloat16
        assert model.embed_tokens.weight.dtype == torch.bfloat16


class TestNemotronHForCausalLM:
    """Test NemotronHForCausalLM (causal LM wrapper)."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            enable_deepep=False,
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    def test_causal_lm_init(self, config, backend):
        """Test NemotronHForCausalLM initialization."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)

        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")
        assert model.config == config

    def test_causal_lm_lm_head_shape(self, config, backend):
        """Test that lm_head has correct shape."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)

        assert model.lm_head.in_features == config.hidden_size
        assert model.lm_head.out_features == config.vocab_size

    def test_causal_lm_forward_shape(self, config, backend):
        """Test causal LM forward pass produces logits with correct shape."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_causal_lm_forward_float32_logits(self, config, backend):
        """Test that logits are computed in float32."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        assert logits.dtype == torch.float32

    def test_causal_lm_forward_with_inputs_embeds(self, config, backend):
        """Test causal LM forward pass with inputs_embeds."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        logits = model(inputs_embeds=inputs_embeds)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert logits.dtype == torch.float32

    def test_causal_lm_forward_no_input_ids_no_inputs_embeds_raises(self, config, backend):
        """Test that ValueError is raised when neither input_ids nor inputs_embeds is provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        with pytest.raises(ValueError, match="input_ids must be provided if inputs_embeds is not provided"):
            model()

    def test_causal_lm_from_config(self, config, backend):
        """Test from_config classmethod."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM.from_config(config, backend=backend)

        assert isinstance(model, NemotronHForCausalLM)
        assert model.config == config

    def test_causal_lm_state_dict_adapter_disabled(self, config, backend):
        """Test that state_dict_adapter is not created when disabled."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        backend.enable_hf_state_dict_adapter = False
        model = NemotronHForCausalLM(config, backend=backend)

        assert not hasattr(model, "state_dict_adapter")

    def test_causal_lm_state_dict_adapter_enabled(self, config, backend):
        """Test that state_dict_adapter is created when enabled."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        backend.enable_hf_state_dict_adapter = True
        model = NemotronHForCausalLM(config, backend=backend)

        assert hasattr(model, "state_dict_adapter")

    def test_model_class_export(self):
        """Test that ModelClass is exported correctly."""
        from nemo_automodel.components.models.nemotron_v3.model import (
            ModelClass,
            NemotronHForCausalLM,
        )

        assert ModelClass is NemotronHForCausalLM


class TestNemotronV3ModelWithMoE:
    """Test NemotronV3Model with MoE layers."""

    @pytest.fixture
    def config(self):
        config = MockNemotronV3Config()
        config.layers_block_type = ["moe", "attention"]
        return config

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            enable_deepep=False,
            fake_balanced_gate=True,  # Use fake balanced gate for deterministic testing
            enable_hf_state_dict_adapter=False,
        )

    def test_moe_model_init(self, config, backend):
        """Test model with MoE layer initializes correctly."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        # First layer should be MoE
        assert model.layers["0"].block_type == "moe"
        assert hasattr(model.layers["0"].mixer, "experts")
        assert hasattr(model.layers["0"].mixer, "shared_experts")

    @skip_if_no_gpu
    def test_moe_model_forward(self, config, backend):
        """Test MoE model forward pass."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_moe_uses_relu2_activation(self, config, backend):
        """Test that MoE experts use relu2 activation."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.moe_config.expert_activation == "relu2"

        # Experts should use non-gated activation (up_projs instead of gate_and_up_projs)
        experts = model.layers["0"].mixer.experts
        # Non-gated experts have different weight shapes
        # gate_and_up_projs for non-gated has shape [n_experts, dim, inter_dim] instead of [n_experts, dim, 2*inter_dim]
        assert experts.gate_and_up_projs.shape[2] == config.moe_intermediate_size

    def test_moe_shared_experts_relu2(self, config, backend):
        """Test that shared experts use relu2 activation."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        shared_experts = model.layers["0"].mixer.shared_experts

        # Shared experts should not have gate_proj (relu2 is non-gated)
        assert shared_experts.gate_proj is None
        assert shared_experts.up_proj is not None
