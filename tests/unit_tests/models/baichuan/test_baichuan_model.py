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


import pytest
import torch
from transformers.cache_utils import DynamicCache

from nemo_automodel.components.models.baichuan.configuration import BaichuanConfig
from nemo_automodel.components.models.baichuan.model import (
    MLP,
    Attention,
    BaichuanForCausalLM,
    BaichuanModel,
    DecoderLayer,
    ModelClass,
    NormHead,
    RMSNorm,
    RotaryEmbedding,
    _apply_rotary_pos_emb,
    _expand_mask,
    _make_causal_mask,
    _rotate_half,
)


def _tiny_config(**overrides) -> BaichuanConfig:
    defaults = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        use_cache=False,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return BaichuanConfig(**defaults)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestBaichuanConfig:
    def test_default_values(self):
        cfg = BaichuanConfig()
        assert cfg.vocab_size == 125696
        assert cfg.hidden_size == 4096
        assert cfg.intermediate_size == 11008
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 32
        assert cfg.hidden_act == "silu"
        assert cfg.max_position_embeddings == 4096
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.model_type == "baichuan"
        assert cfg.z_loss_weight == 0

    def test_custom_values(self):
        cfg = _tiny_config(z_loss_weight=0.01)
        assert cfg.vocab_size == 32
        assert cfg.hidden_size == 16
        assert cfg.z_loss_weight == 0.01


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(16, eps=1e-6)
        x = torch.randn(2, 4, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_bf16_input(self):
        norm = RMSNorm(16, eps=1e-6)
        norm.weight.data = norm.weight.data.to(torch.bfloat16)
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16


class TestRotaryEmbedding:
    def test_cached_shapes(self):
        dim, max_pos = 8, 64
        rope = RotaryEmbedding(dim, max_position_embeddings=max_pos)
        x = torch.randn(2, 2, 10, dim)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (1, 1, 10, dim)
        assert sin.shape == (1, 1, 10, dim)

    def test_extends_cache_beyond_max(self):
        rope = RotaryEmbedding(8, max_position_embeddings=16)
        x = torch.randn(1, 1, 32, 8)
        cos, sin = rope(x, seq_len=32)
        assert cos.shape[-2] == 32
        assert rope.max_seq_len_cached == 32

    def test_device_migration(self):
        rope = RotaryEmbedding(8, max_position_embeddings=16)
        x = torch.randn(1, 1, 8, 8)
        cos, sin = rope(x, seq_len=8)
        assert cos.device == x.device


class TestRotateHalf:
    def test_output(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = _rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected)


class TestApplyRotaryPosEmb:
    def test_shapes_preserved(self):
        bsz, n_heads, seq_len, head_dim = 2, 2, 4, 8
        q = torch.randn(bsz, n_heads, seq_len, head_dim)
        k = torch.randn(bsz, n_heads, seq_len, head_dim)
        cos = torch.randn(1, 1, seq_len, head_dim)
        sin = torch.randn(1, 1, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        q_out, k_out = _apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestMakeCausalMask:
    def test_shape_and_causality(self):
        mask = _make_causal_mask((2, 4), torch.float32, torch.device("cpu"))
        assert mask.shape == (2, 1, 4, 4)
        assert mask[0, 0, 0, 1] == torch.finfo(torch.float32).min
        assert mask[0, 0, 1, 0] == 0.0

    def test_with_past_kv(self):
        mask = _make_causal_mask((1, 3), torch.float32, torch.device("cpu"), past_key_values_length=2)
        assert mask.shape == (1, 1, 3, 5)


class TestExpandMask:
    def test_2d_mask(self):
        mask = torch.ones(2, 4)
        out = _expand_mask(mask, torch.float32)
        assert out.shape == (2, 1, 4, 4)

    def test_3d_mask(self):
        mask = torch.ones(2, 4, 4)
        out = _expand_mask(mask, torch.float32)
        assert out.shape == (2, 1, 4, 4)

    def test_zero_positions_masked(self):
        mask = torch.tensor([[1, 1, 0, 1]])
        out = _expand_mask(mask, torch.float32)
        assert out[0, 0, 0, 2] == torch.finfo(torch.float32).min


class TestMLP:
    def test_forward_shape(self):
        mlp = MLP(hidden_size=16, intermediate_size=32, hidden_act="silu")
        x = torch.randn(2, 4, 16)
        out = mlp(x)
        assert out.shape == x.shape


class TestNormHead:
    def test_forward_shape(self):
        head = NormHead(hidden_size=16, vocab_size=32)
        x = torch.randn(2, 4, 16)
        out = head(x)
        assert out.shape == (2, 4, 32)

    def test_training_normalizes_weight(self):
        head = NormHead(hidden_size=16, vocab_size=32)
        head.train()
        x = torch.randn(1, 1, 16)
        _ = head(x)
        assert head.first_flag is True

    def test_eval_caches_normalized_weight(self):
        head = NormHead(hidden_size=16, vocab_size=32)
        head.eval()
        x = torch.randn(1, 1, 16)
        _ = head(x)
        assert head.first_flag is False
        _ = head(x)


class TestAttention:
    def test_forward_shape(self):
        cfg = _tiny_config()
        attn = Attention(cfg)
        bsz, seq_len = 2, 4
        hidden = torch.randn(bsz, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        out, _, _ = attn(hidden, position_ids=position_ids)
        assert out.shape == (bsz, seq_len, cfg.hidden_size)

    def test_invalid_head_dim_raises(self):
        with pytest.raises(ValueError, match="hidden_size must be divisible"):
            Attention(_tiny_config(hidden_size=15, num_attention_heads=4))

    def test_with_past_key_value(self):
        cfg = _tiny_config()
        attn = Attention(cfg)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        bsz, seq_len, past_len = 1, 1, 3
        hidden = torch.randn(bsz, seq_len, cfg.hidden_size)
        position_ids = torch.tensor([[past_len]])
        past_kv = (
            torch.randn(bsz, cfg.num_attention_heads, past_len, head_dim),
            torch.randn(bsz, cfg.num_attention_heads, past_len, head_dim),
        )
        out, _, new_kv = attn(hidden, position_ids=position_ids, past_key_value=past_kv, use_cache=True)
        assert out.shape == (bsz, seq_len, cfg.hidden_size)
        assert new_kv[0].shape[-2] == past_len + seq_len


class TestDecoderLayer:
    def test_forward_shape(self):
        cfg = _tiny_config()
        layer = DecoderLayer(cfg)
        bsz, seq_len = 2, 4
        hidden = torch.randn(bsz, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        (out,) = layer(hidden, position_ids=position_ids)
        assert out.shape == hidden.shape

    def test_use_cache_returns_kv(self):
        cfg = _tiny_config()
        layer = DecoderLayer(cfg)
        bsz, seq_len = 1, 4
        hidden = torch.randn(bsz, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        outputs = layer(hidden, position_ids=position_ids, use_cache=True)
        assert len(outputs) == 2
        assert outputs[1] is not None


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------
class TestBaichuanPreTrainedModel:
    def test_init_weights_linear(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        linear = torch.nn.Linear(4, 4, bias=True)
        linear.bias.data.fill_(99.0)
        model._init_weights(linear)
        assert linear.bias.data.abs().max() == 0.0

    def test_init_weights_embedding(self):
        cfg = _tiny_config(pad_token_id=0)
        model = BaichuanForCausalLM(cfg)
        emb = torch.nn.Embedding(8, 4, padding_idx=0)
        model._init_weights(emb)
        assert emb.weight.data[0].abs().max() == 0.0

    def test_gradient_checkpointing_setter(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        model._set_gradient_checkpointing(model.model, value=True)
        assert model.model.gradient_checkpointing is True


class TestBaichuanModel:
    def test_init_components(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        assert model.embed_tokens.num_embeddings == cfg.vocab_size
        assert len(model.layers) == cfg.num_hidden_layers

    def test_get_set_input_embeddings(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        new_emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.get_input_embeddings() is new_emb

    def test_forward_with_input_ids(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        model.eval()
        bsz, seq_len = 2, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.last_hidden_state.shape == (bsz, seq_len, cfg.hidden_size)

    def test_forward_with_inputs_embeds(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        model.eval()
        bsz, seq_len = 1, 3
        embeds = torch.randn(bsz, seq_len, cfg.hidden_size)
        with torch.no_grad():
            out = model(inputs_embeds=embeds)
        assert out.last_hidden_state.shape == (bsz, seq_len, cfg.hidden_size)

    def test_forward_raises_on_both_inputs(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        with pytest.raises(ValueError, match="cannot specify both"):
            model(
                input_ids=torch.zeros(1, 2, dtype=torch.long),
                inputs_embeds=torch.randn(1, 2, cfg.hidden_size),
            )

    def test_forward_raises_on_no_inputs(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        with pytest.raises(ValueError, match="have to specify either"):
            model()

    def test_forward_output_hidden_states(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        assert out.hidden_states is not None
        assert len(out.hidden_states) == cfg.num_hidden_layers + 1

    def test_forward_return_dict_false(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(input_ids=input_ids, return_dict=False)
        assert isinstance(out, tuple)

    def test_forward_with_explicit_position_ids(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        model.eval()
        bsz, seq_len = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=input_ids, position_ids=position_ids)
        assert out.last_hidden_state.shape == (bsz, seq_len, cfg.hidden_size)

    def test_forward_single_token_no_causal_mask(self):
        cfg = _tiny_config()
        model = BaichuanModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 1))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.last_hidden_state.shape == (1, 1, cfg.hidden_size)


class TestBaichuanForCausalLM:
    def test_forward_logits_shape(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        model.eval()
        bsz, seq_len = 2, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (bsz, seq_len, cfg.vocab_size)

    def test_forward_with_labels_computes_loss(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        model.train()
        bsz, seq_len = 2, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.dim() == 0

    def test_forward_with_z_loss(self):
        cfg = _tiny_config(z_loss_weight=0.1)
        model = BaichuanForCausalLM(cfg)
        model.train()
        bsz, seq_len = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None

    def test_forward_return_dict_false(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(input_ids=input_ids, return_dict=False)
        assert isinstance(out, tuple)
        assert out[0].shape[-1] == cfg.vocab_size

    def test_forward_return_dict_false_with_labels(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        model.train()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        labels = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model(input_ids=input_ids, labels=labels, return_dict=False)
        assert isinstance(out, tuple)
        assert out[0].dim() == 0  # scalar loss

    def test_get_set_input_embeddings(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        new_emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.get_input_embeddings() is new_emb

    def test_get_set_output_embeddings(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        new_head = NormHead(cfg.hidden_size, cfg.vocab_size)
        model.set_output_embeddings(new_head)
        assert model.get_output_embeddings() is new_head

    def test_get_set_decoder(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        decoder = model.get_decoder()
        assert isinstance(decoder, BaichuanModel)
        new_decoder = BaichuanModel(cfg)
        model.set_decoder(new_decoder)
        assert model.get_decoder() is new_decoder

    def test_model_class_alias(self):
        assert ModelClass is BaichuanForCausalLM

    def test_is_hf_checkpointing_mixin(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

        assert isinstance(model, HFCheckpointingMixin)


class TestPrepareInputsForGeneration:
    def test_basic(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        attention_mask = torch.ones(1, 4)
        inputs = model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
        assert "input_ids" in inputs
        assert inputs["position_ids"] is not None

    def test_with_past(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        past = tuple(
            (
                torch.randn(1, cfg.num_attention_heads, 3, head_dim),
                torch.randn(1, cfg.num_attention_heads, 3, head_dim),
            )
            for _ in range(cfg.num_hidden_layers)
        )
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        attention_mask = torch.ones(1, 4)
        inputs = model.prepare_inputs_for_generation(input_ids, past_key_values=past, attention_mask=attention_mask)
        assert inputs["input_ids"].shape[1] == 1

    def test_with_inputs_embeds_no_past(self):
        cfg = _tiny_config()
        model = BaichuanForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        embeds = torch.randn(1, 4, cfg.hidden_size)
        inputs = model.prepare_inputs_for_generation(input_ids, inputs_embeds=embeds)
        assert "inputs_embeds" in inputs
        assert "input_ids" not in inputs


class TestDynamicCacheCompat:
    """Regression test for DynamicCache incompatibility (baichuan_2_7b_squad_vllm_deploy)."""

    def test_forward_with_dynamic_cache(self):
        cfg = _tiny_config(use_cache=True)
        model = BaichuanModel(cfg)
        model.eval()
        bsz, seq_len = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

        # First forward to populate cache
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        legacy_cache = out.past_key_values

        # Convert legacy cache to DynamicCache (simulates what GenerationMixin does)
        dynamic_cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(legacy_cache):
            dynamic_cache.update(key, value, layer_idx)

        # Second forward with DynamicCache — this was the failing path
        next_token = torch.randint(0, cfg.vocab_size, (bsz, 1))
        with torch.no_grad():
            out2 = model(input_ids=next_token, past_key_values=dynamic_cache, use_cache=True)
        assert out2.last_hidden_state.shape == (bsz, 1, cfg.hidden_size)


class TestReorderCache:
    def test_reorders_correctly(self):
        past = tuple((torch.randn(3, 2, 4, 8), torch.randn(3, 2, 4, 8)) for _ in range(2))
        beam_idx = torch.tensor([2, 0, 1])
        reordered = BaichuanForCausalLM._reorder_cache(past, beam_idx)
        assert len(reordered) == 2
        for layer_past in reordered:
            assert layer_past[0].shape == (3, 2, 4, 8)
        assert torch.allclose(reordered[0][0][0], past[0][0][2])
