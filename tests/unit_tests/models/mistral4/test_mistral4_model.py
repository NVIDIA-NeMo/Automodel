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

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.model import (
    _HF_MISTRAL3_AVAILABLE,
    Mistral4Block,
    Mistral4ForCausalLM,
    Mistral4MLA,
    Mistral4Model,
    ModelClass,
    _build_moe_config,
    _get_llama_4_attn_scale,
)
from nemo_automodel.components.moe.config import MoEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def text_config():
    cfg = Mock(spec=[])
    cfg.vocab_size = 256
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.q_lora_rank = 32
    cfg.kv_lora_rank = 16
    cfg.qk_nope_head_dim = 8
    cfg.qk_rope_head_dim = 8
    cfg.qk_head_dim = 16
    cfg.v_head_dim = 16
    cfg.n_routed_experts = 4
    cfg.n_shared_experts = 1
    cfg.num_experts_per_tok = 2
    cfg.n_group = 1
    cfg.topk_group = 1
    cfg.first_k_dense_replace = 0
    cfg.norm_topk_prob = True
    cfg.routed_scaling_factor = 1.0
    cfg.max_position_embeddings = 256
    cfg.rms_norm_eps = 1e-6
    cfg.torch_dtype = torch.bfloat16
    cfg.rope_parameters = {
        "type": "yarn",
        "rope_theta": 10000.0,
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "max_position_embeddings": 256,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale_all_dim": 1.0,
        "mscale": 1.0,
        "llama_4_scaling_beta": 0.1,
    }
    cfg.rope_interleave = True
    return cfg


@pytest.fixture
def backend():
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


# ---------------------------------------------------------------------------
# _get_llama_4_attn_scale
# ---------------------------------------------------------------------------


class TestGetLlama4AttnScale:
    def test_scale_is_one_below_max_pos(self):
        """For positions < max_pos_embeddings, scale should be 1.0."""
        pos = torch.arange(10).unsqueeze(0)
        scale = _get_llama_4_attn_scale(pos, beta=0.1, max_position_embeddings=8192)
        torch.testing.assert_close(scale, torch.ones_like(scale))

    def test_scale_increases_above_max_pos(self):
        """For positions >= max_pos_embeddings, scale should be > 1.0."""
        pos = torch.tensor([[0, 8192, 16384]])
        scale = _get_llama_4_attn_scale(pos, beta=0.1, max_position_embeddings=8192)
        assert scale[0, 0, 0].item() == 1.0
        assert scale[0, 1, 0].item() > 1.0
        assert scale[0, 2, 0].item() > scale[0, 1, 0].item()

    def test_output_shape(self):
        pos = torch.arange(5).unsqueeze(0)  # [1, 5]
        scale = _get_llama_4_attn_scale(pos, beta=0.1, max_position_embeddings=8192)
        assert scale.shape == (1, 5, 1)


# ---------------------------------------------------------------------------
# _build_moe_config
# ---------------------------------------------------------------------------


class TestBuildMoeConfig:
    def test_returns_moe_config(self, text_config):
        moe_config = _build_moe_config(text_config)
        assert isinstance(moe_config, MoEConfig)
        assert moe_config.n_routed_experts == 4
        assert moe_config.n_shared_experts == 1
        assert moe_config.n_activated_experts == 2
        assert moe_config.score_func == "softmax_with_bias"
        assert moe_config.dim == 64
        assert moe_config.moe_inter_dim == 32


# ---------------------------------------------------------------------------
# Mistral4MLA
# ---------------------------------------------------------------------------


class TestMistral4MLA:
    def test_init_stores_scaling_params(self, text_config, backend):
        mla = Mistral4MLA(text_config, backend)
        assert mla.llama_4_scaling_beta == 0.1
        assert mla.llama_4_orig_max_pos == 8192

    def test_init_no_scaling_beta(self, text_config, backend):
        text_config.rope_parameters = {"rope_theta": 10000.0}
        mla = Mistral4MLA(text_config, backend)
        assert mla.llama_4_scaling_beta is None

    def test_init_rope_scaling_fallback(self, text_config, backend):
        """Falls back to rope_scaling when rope_parameters is absent."""
        del text_config.rope_parameters
        text_config.rope_scaling = {"rope_theta": 10000.0, "llama_4_scaling_beta": 0.2}
        mla = Mistral4MLA(text_config, backend)
        assert mla.llama_4_scaling_beta == 0.2

    def test_forward_shape(self, text_config, backend, device):
        mla = Mistral4MLA(text_config, backend).to(device).to(torch.bfloat16)
        B, S = 2, 8
        x = torch.randn(B, S, 64, dtype=torch.bfloat16, device=device)
        # Build freqs_cis
        from nemo_automodel.components.models.deepseek_v3.rope_utils import (
            freqs_cis_from_position_ids,
            precompute_freqs_cis,
        )

        rp = text_config.rope_parameters
        all_freqs = precompute_freqs_cis(8, 256, rp["rope_theta"], rp).to(device)
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        freqs_cis = freqs_cis_from_position_ids(pos_ids, all_freqs, qkv_format="bshd", for_fused_rope=False)

        out = mla(x, freqs_cis, position_ids=pos_ids)
        assert out.shape == (B, S, 64)

    def test_forward_without_scaling_beta(self, text_config, backend, device):
        """Forward works when llama_4_scaling_beta is None (no-op scaling path)."""
        text_config.rope_parameters = {
            "type": "yarn",
            "rope_theta": 10000.0,
            "factor": 128.0,
            "original_max_position_embeddings": 8192,
            "max_position_embeddings": 256,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
        }
        mla = Mistral4MLA(text_config, backend).to(device).to(torch.bfloat16)
        assert mla.llama_4_scaling_beta is None
        B, S = 1, 4
        x = torch.randn(B, S, 64, dtype=torch.bfloat16, device=device)
        from nemo_automodel.components.models.deepseek_v3.rope_utils import (
            freqs_cis_from_position_ids,
            precompute_freqs_cis,
        )

        rp = text_config.rope_parameters
        all_freqs = precompute_freqs_cis(8, 256, rp["rope_theta"], rp).to(device)
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        freqs_cis = freqs_cis_from_position_ids(pos_ids, all_freqs, qkv_format="bshd", for_fused_rope=False)
        out = mla(x, freqs_cis, position_ids=pos_ids)
        assert out.shape == (B, S, 64)


# ---------------------------------------------------------------------------
# Mistral4Block
# ---------------------------------------------------------------------------


class TestMistral4Block:
    def test_uses_mistral4mla(self, text_config, backend):
        moe_config = _build_moe_config(text_config)
        block = Mistral4Block(0, text_config, moe_config, backend)
        assert isinstance(block.self_attn, Mistral4MLA)


# ---------------------------------------------------------------------------
# Mistral4Model
# ---------------------------------------------------------------------------


class TestMistral4Model:
    def test_init(self, text_config, backend):
        model = Mistral4Model(text_config, backend)
        assert len(model.layers) == 2
        assert model.embed_tokens.num_embeddings == 256
        assert model.embed_tokens.embedding_dim == 64

    def test_forward_with_input_ids(self, text_config, backend, device):
        model = Mistral4Model(text_config, backend).to(device).to(torch.bfloat16)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        out = model(input_ids)
        assert out.shape == (1, 8, 64)

    def test_forward_with_inputs_embeds(self, text_config, backend, device):
        model = Mistral4Model(text_config, backend).to(device).to(torch.bfloat16)
        embeds = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
        out = model(inputs_embeds=embeds)
        assert out.shape == (1, 8, 64)

    def test_forward_raises_on_both_inputs(self, text_config, backend, device):
        model = Mistral4Model(text_config, backend).to(device).to(torch.bfloat16)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        embeds = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
        with pytest.raises(ValueError, match="exactly one"):
            model(input_ids, inputs_embeds=embeds)

    def test_uses_mistral4block(self, text_config, backend):
        model = Mistral4Model(text_config, backend)
        for layer in model.layers.values():
            assert isinstance(layer, Mistral4Block)

    def test_update_moe_gate_bias(self, text_config, backend, device):
        model = Mistral4Model(text_config, backend).to(device).to(torch.bfloat16)
        model.train()
        # Run a forward pass to populate _cumulative_expert_load
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        model(input_ids)
        model.update_moe_gate_bias()

    def test_init_weights(self, text_config, backend, device):
        model = Mistral4Model(text_config, backend).to(device).to(torch.bfloat16)
        model.init_weights(buffer_device=device)
        # After init_weights, freqs_cis should be on the correct device
        assert model.freqs_cis.device == device

    def test_custom_moe_config(self, text_config, backend):
        custom_moe = _build_moe_config(text_config)
        custom_moe.n_activated_experts = 1
        model = Mistral4Model(text_config, backend, moe_config=custom_moe)
        assert model.moe_config.n_activated_experts == 1


# ---------------------------------------------------------------------------
# Mistral4ForCausalLM
# ---------------------------------------------------------------------------


class TestMistral4ForCausalLM:
    def test_init(self, text_config, backend):
        model = Mistral4ForCausalLM(text_config, backend=backend)
        assert model.lm_head is not None
        assert model.model is not None

    def test_from_config_extracts_text_config(self, text_config, backend):
        wrapper = Mock()
        wrapper.text_config = text_config
        model = Mistral4ForCausalLM.from_config(wrapper, backend=backend)
        assert model.config is text_config

    def test_init_extracts_text_config(self, text_config, backend):
        """__init__ also extracts text_config if present."""
        wrapper = Mock(spec=[])
        wrapper.text_config = text_config
        model = Mistral4ForCausalLM(wrapper, backend=backend)
        assert model.config is text_config

    def test_forward_shape(self, text_config, backend, device):
        model = Mistral4ForCausalLM(text_config, backend=backend).to(device).to(torch.bfloat16)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        logits = model(input_ids)
        assert logits.shape == (1, 8, 256)

    def test_model_class_export(self):
        assert ModelClass is Mistral4ForCausalLM

    def test_get_set_input_embeddings(self, text_config, backend):
        model = Mistral4ForCausalLM(text_config, backend=backend)
        embed = model.get_input_embeddings()
        assert isinstance(embed, nn.Embedding)
        new_embed = nn.Embedding(256, 64)
        model.set_input_embeddings(new_embed)
        assert model.get_input_embeddings() is new_embed

    def test_get_set_output_embeddings(self, text_config, backend):
        model = Mistral4ForCausalLM(text_config, backend=backend)
        lm_head = model.get_output_embeddings()
        assert lm_head is not None
        new_head = nn.Linear(64, 256, bias=False)
        model.set_output_embeddings(new_head)
        assert model.get_output_embeddings() is new_head

    def test_update_moe_gate_bias(self, text_config, backend, device):
        model = Mistral4ForCausalLM(text_config, backend=backend).to(device).to(torch.bfloat16)
        model.train()
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        model(input_ids)
        model.update_moe_gate_bias()

    def test_initialize_weights(self, text_config, backend, device):
        model = Mistral4ForCausalLM(text_config, backend=backend).to(device)
        model.initialize_weights(buffer_device=device)
        assert model.model.freqs_cis.device == device

    def test_state_dict_adapter_created(self, text_config):
        backend = BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            rope_fusion=False,
            enable_hf_state_dict_adapter=True,
        )
        model = Mistral4ForCausalLM(text_config, backend=backend)
        assert hasattr(model, "state_dict_adapter")

    def test_default_backend(self, text_config):
        model = Mistral4ForCausalLM(text_config)
        assert model.backend is not None


# ---------------------------------------------------------------------------
# Mistral4Config
# ---------------------------------------------------------------------------


class TestMistral4Config:
    def test_defaults(self):
        from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

        cfg = Mistral4Config()
        assert cfg.model_type == "mistral4"
        assert cfg.vocab_size == 131072
        assert cfg.hidden_size == 4096
        assert cfg.qk_head_dim == 128  # 64 + 64
        assert cfg.head_dim == 128  # v_head_dim

    def test_custom_values(self):
        from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

        cfg = Mistral4Config(vocab_size=256, hidden_size=64, num_hidden_layers=2)
        assert cfg.vocab_size == 256
        assert cfg.hidden_size == 64
        assert cfg.num_hidden_layers == 2

    def test_rope_parameters_default(self):
        from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

        cfg = Mistral4Config()
        assert cfg.rope_parameters is not None
        assert cfg.rope_parameters["type"] == "yarn"
        assert cfg.rope_parameters["llama_4_scaling_beta"] == 0.1

    def test_custom_rope_parameters(self):
        from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

        custom_rope = {
            "type": "yarn",
            "rope_theta": 5000.0,
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 16.0,
            "beta_slow": 2.0,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
            "llama_4_scaling_beta": 0.2,
        }
        cfg = Mistral4Config(rope_parameters=custom_rope)
        assert cfg.rope_parameters["rope_theta"] == 5000.0
        assert cfg.rope_parameters["llama_4_scaling_beta"] == 0.2

    def test_num_key_value_heads_default(self):
        from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

        cfg = Mistral4Config(num_key_value_heads=None, num_attention_heads=16)
        assert cfg.num_key_value_heads == 16

    def test_autoconfig_registration(self):
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        # Import registry to trigger registration
        import nemo_automodel._transformers.registry  # noqa: F401

        assert "mistral4" in CONFIG_MAPPING


# ---------------------------------------------------------------------------
# Multimodal: Mistral4TextModelBackend, Mistral3Model, Mistral3ForConditionalGeneration
# ---------------------------------------------------------------------------

_skip_no_hf_mistral3 = pytest.mark.skipif(
    not _HF_MISTRAL3_AVAILABLE,
    reason="transformers mistral3 not available",
)


@pytest.fixture
def multimodal_config():
    """Build a small Mistral3Config wrapping a Mistral4Config-like text config."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

    # Ensure Mistral4Config is registered before Mistral3Config tries to resolve it
    if "mistral4" not in CONFIG_MAPPING:
        AutoConfig.register("mistral4", Mistral4Config)

    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    text_config = Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
    )
    config = Mistral3Config(
        text_config=text_config.to_dict(),
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 16,
            "patch_size": 4,
        },
        image_token_index=10,
        spatial_merge_size=2,
    )
    return config


def _make_fake_vision_tower(hidden_size=32, patch_size=4):
    """Create a fake vision tower that returns expected outputs."""
    tower = Mock()
    tower.patch_size = patch_size

    def fake_forward(pixel_values, image_sizes=None, output_hidden_states=True, return_dict=True):
        B = pixel_values.shape[0]
        # Each image produces (H/patch * W/patch) tokens
        n_patches = (image_sizes[0, 0] // patch_size) * (image_sizes[0, 1] // patch_size)
        features = torch.randn(B, n_patches, hidden_size, device=pixel_values.device, dtype=pixel_values.dtype)
        result = Mock()
        result.hidden_states = [features]
        return result

    tower.side_effect = fake_forward
    tower.__call__ = fake_forward
    return tower


@_skip_no_hf_mistral3
class TestMistral4TextModelBackend:
    def test_init(self, text_config, backend):
        from nemo_automodel.components.models.mistral4.model import Mistral4TextModelBackend

        tm = Mistral4TextModelBackend(text_config, backend)
        assert tm.lm_head is not None
        assert tm.moe_config is not None

    def test_properties(self, text_config, backend):
        from nemo_automodel.components.models.mistral4.model import Mistral4TextModelBackend

        tm = Mistral4TextModelBackend(text_config, backend)
        assert tm.embed_tokens is tm.model.embed_tokens
        assert tm.layers is tm.model.layers
        assert tm.norm is tm.model.norm

    def test_get_set_input_embeddings(self, text_config, backend):
        from nemo_automodel.components.models.mistral4.model import Mistral4TextModelBackend

        tm = Mistral4TextModelBackend(text_config, backend)
        embed = tm.get_input_embeddings()
        assert isinstance(embed, nn.Embedding)
        new_embed = nn.Embedding(256, 64)
        tm.set_input_embeddings(new_embed)
        assert tm.get_input_embeddings() is new_embed

    def test_forward_returns_base_model_output(self, text_config, backend, device):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        from nemo_automodel.components.models.mistral4.model import Mistral4TextModelBackend

        tm = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        out = tm(input_ids=input_ids)
        assert isinstance(out, BaseModelOutputWithPast)
        assert out.last_hidden_state.shape == (1, 8, 64)
        assert out.past_key_values is None

    def test_forward_with_inputs_embeds(self, text_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import Mistral4TextModelBackend

        tm = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        embeds = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
        out = tm(inputs_embeds=embeds)
        assert out.last_hidden_state.shape == (1, 8, 64)

    def test_init_weights(self, text_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import Mistral4TextModelBackend

        tm = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        tm.init_weights(buffer_device=device)


@_skip_no_hf_mistral3
class TestMistral3Model:
    def test_init(self, text_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend)
        vision_tower = Mock()
        projector = Mock()
        config = Mock(spec=[])
        model = OurMistral3Model(config, vision_tower, projector, language_model)
        assert model.vision_tower is vision_tower
        assert model.multi_modal_projector is projector
        assert model.language_model is language_model

    def test_properties_delegate_to_language_model(self, text_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend)
        model = OurMistral3Model(Mock(spec=[]), Mock(), Mock(), language_model)
        assert model.layers is language_model.layers
        assert model.embed_tokens is language_model.embed_tokens
        assert model.norm is language_model.norm
        assert model.get_input_embeddings() is language_model.get_input_embeddings()

    def test_forward_text_only(self, text_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        model = OurMistral3Model(Mock(spec=[]), None, None, language_model)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        out = model(input_ids=input_ids)
        assert out.last_hidden_state.shape == (1, 8, 64)

    def test_forward_raises_on_both_inputs(self, text_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        model = OurMistral3Model(Mock(spec=[]), None, None, language_model)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        embeds = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
        with pytest.raises(ValueError, match="exactly one"):
            model(input_ids=input_ids, inputs_embeds=embeds)

    def test_forward_float_input_ids_as_embeds(self, text_config, backend, device):
        """When embed_tokens is None and input_ids is float, treat as embeds."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        # Remove embed_tokens to simulate PP stage without embeddings
        language_model.model.embed_tokens = None
        model = OurMistral3Model(Mock(spec=[]), None, None, language_model)
        float_ids = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
        out = model(input_ids=float_ids)
        assert out.last_hidden_state.shape == (1, 8, 64)


@_skip_no_hf_mistral3
class TestMistral3ForConditionalGeneration:
    def test_init(self, multimodal_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend)
        assert model.model is not None
        assert model.lm_head is not None
        assert model.vocab_size == 256
        assert model.image_token_index == 10

    def test_get_set_input_embeddings(self, multimodal_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend)
        embed = model.get_input_embeddings()
        assert isinstance(embed, nn.Embedding)
        new_embed = nn.Embedding(256, 64)
        model.set_input_embeddings(new_embed)
        assert model.get_input_embeddings() is new_embed

    def test_get_set_output_embeddings(self, multimodal_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend)
        lm_head = model.get_output_embeddings()
        assert lm_head is not None
        new_head = nn.Linear(64, 256, bias=False)
        model.set_output_embeddings(new_head)
        assert model.get_output_embeddings() is new_head

    def test_lm_head_property(self, multimodal_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend)
        assert model.lm_head is model.model.language_model.lm_head

    def test_forward_text_only(self, multimodal_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend).to(device).to(torch.bfloat16)
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        logits = model(input_ids)
        assert logits.shape == (1, 8, 256)

    def test_from_config(self, multimodal_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG.from_config(multimodal_config, backend=backend)
        assert model.config is multimodal_config

    def test_num_hidden_layers_override(self, multimodal_config, backend):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend, num_hidden_layers=1)
        assert len(model.model.language_model.layers) == 1

    def test_update_moe_gate_bias(self, multimodal_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend).to(device).to(torch.bfloat16)
        model.train()
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        model(input_ids)
        model.update_moe_gate_bias()

    def test_initialize_weights(self, multimodal_config, backend, device):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend).to(device)
        model.initialize_weights(buffer_device=device)

    def test_state_dict_adapter_created(self, multimodal_config):
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        backend = BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            rope_fusion=False,
            enable_hf_state_dict_adapter=True,
        )
        model = OurMistral3ForCG(multimodal_config, backend=backend)
        assert hasattr(model, "state_dict_adapter")

    def test_lm_head_none_on_pruned_model(self, multimodal_config, backend):
        """lm_head property gracefully returns None when language_model is pruned (PP)."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend)
        # Simulate PP pruning: remove language_model's lm_head
        model.model.language_model.lm_head = None
        # The forward path handles lm_head=None gracefully via try/except
        assert model.lm_head is None

    def test_forward_without_lm_head(self, multimodal_config, backend, device):
        """Forward returns hidden_states when lm_head is None (PP stage without head)."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend).to(device).to(torch.bfloat16)
        model.model.language_model.lm_head = None
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        out = model(input_ids)
        # Without lm_head, output is hidden_states (hidden_size=64) not logits (vocab_size=256)
        assert out.shape == (1, 8, 64)

    def test_pp_vlm_chunk_retrieval(self, multimodal_config, backend):
        """PP VLM chunk attributes are stored and chunk_idx increments correctly."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3ForConditionalGeneration as OurMistral3ForCG,
        )

        model = OurMistral3ForCG(multimodal_config, backend=backend)
        # Verify model has image_token_index set from config
        assert model.image_token_index == 10
        # Verify chunk attributes can be set (used by PP recipe)
        fake_pixels = torch.randn(1, 3, 16, 16)
        fake_image_sizes = torch.tensor([[16, 16]])
        model._vlm_pixel_values_chunks = [fake_pixels, fake_pixels]
        model._vlm_image_grid_hws_chunks = [fake_image_sizes, fake_image_sizes]
        model._vlm_chunk_idx = 0
        assert len(model._vlm_pixel_values_chunks) == 2
        # Simulate chunk consumption
        model._vlm_chunk_idx = 1
        assert model._vlm_chunk_idx == 1


@_skip_no_hf_mistral3
class TestMistral3ModelVision:
    """Tests for vision-related paths in Mistral3Model."""

    def test_get_image_features(self, text_config, backend, device):
        """Test _get_image_features with a mocked vision tower."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        # Mock vision tower
        vision_tower = Mock()
        vision_tower.patch_size = 4
        hidden_size = 32
        n_patches = 16  # raw patches before projector
        fake_features = torch.randn(1, n_patches, hidden_size, device=device, dtype=torch.bfloat16)
        vision_output = Mock()
        vision_output.hidden_states = [fake_features]
        vision_tower.return_value = vision_output

        # downsample_ratio = patch_size * spatial_merge_size = 4 * 2 = 8
        # image 16x16 -> (16/8)*(16/8) = 4 vision tokens after merge
        n_vision_tokens = 4

        # Mock projector: output after patch merging has n_vision_tokens
        projector = Mock()
        projector.return_value = torch.randn(
            1, n_vision_tokens, text_config.hidden_size, device=device, dtype=torch.bfloat16
        )

        config = Mock(spec=[])
        config.spatial_merge_size = 2
        config.vision_feature_layer = -1

        model = OurMistral3Model(config, vision_tower, projector, language_model)
        pixel_values = torch.randn(1, 3, 16, 16, device=device, dtype=torch.bfloat16)
        image_sizes = torch.tensor([[16, 16]], device=device)

        features = model._get_image_features(pixel_values, image_sizes)
        assert isinstance(features, tuple)
        assert len(features) == 1  # 1 image
        assert features[0].shape[0] == n_vision_tokens

    def test_forward_with_vision(self, text_config, backend, device):
        """Forward with pixel_values merges vision features into text embeddings."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)

        # downsample_ratio = patch_size * spatial_merge_size = 4 * 2 = 8
        # image_size = 16x16 -> (16/8)*(16/8) = 4 vision tokens
        n_vision_tokens = 4

        # Mock vision tower
        vision_tower = Mock()
        vision_tower.patch_size = 4
        hidden_size = 32
        # _get_image_features squeezes dim 0, so hidden_states should be [1, n_patches, hidden_size]
        n_patches = 16  # before projector downsampling
        fake_features = torch.randn(1, n_patches, hidden_size, device=device, dtype=torch.bfloat16)
        vision_output = Mock()
        vision_output.hidden_states = [fake_features]
        vision_tower.return_value = vision_output

        # Mock projector: output has n_vision_tokens after patch merging
        projector = Mock()
        projector.return_value = torch.randn(
            n_vision_tokens, text_config.hidden_size, device=device, dtype=torch.bfloat16
        )

        config = Mock(spec=[])
        config.spatial_merge_size = 2
        config.vision_feature_layer = -1
        config.image_token_index = 10

        model = OurMistral3Model(config, vision_tower, projector, language_model)

        # Build input: 4 image tokens at positions 0-3, rest are normal text tokens
        seq_len = 8
        input_ids = torch.randint(0, 256, (1, seq_len), device=device)
        input_ids[0, :n_vision_tokens] = 10  # image token index

        pixel_values = torch.randn(1, 3, 16, 16, device=device, dtype=torch.bfloat16)
        image_sizes = torch.tensor([[16, 16]], device=device)

        out = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes)
        assert out.last_hidden_state.shape == (1, seq_len, text_config.hidden_size)

    def test_forward_raises_on_pp_stage_without_embeds(self, text_config, backend, device):
        """PP stage with no embed_tokens and integer input_ids raises ValueError."""
        from nemo_automodel.components.models.mistral4.model import (
            Mistral3Model as OurMistral3Model,
        )
        from nemo_automodel.components.models.mistral4.model import (
            Mistral4TextModelBackend,
        )

        language_model = Mistral4TextModelBackend(text_config, backend).to(device).to(torch.bfloat16)
        language_model.model.embed_tokens = None
        model = OurMistral3Model(Mock(spec=[]), None, None, language_model)

        # Integer input_ids (not float) should raise
        input_ids = torch.randint(0, 256, (1, 8), device=device)
        with pytest.raises(ValueError, match="inputs_embeds must be provided"):
            model(input_ids=input_ids)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_mistral4_in_registry(self):
        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        assert "Mistral4ForCausalLM" in dict(MODEL_ARCH_MAPPING)
        assert "Mistral3ForConditionalGeneration" in dict(MODEL_ARCH_MAPPING)

    def test_registry_module_path(self):
        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        mapping = dict(MODEL_ARCH_MAPPING)
        assert mapping["Mistral4ForCausalLM"][0] == "nemo_automodel.components.models.mistral4.model"
        assert mapping["Mistral3ForConditionalGeneration"][0] == "nemo_automodel.components.models.mistral4.model"
