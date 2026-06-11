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

"""Coverage for gemma4 MoE model context-parallel paths and helpers.

Complements test_gemma4_model.py by exercising the CP-prep methods, the
packed-mask builder branches, the SDPA-GQA override context manager, and the
CP/vision branches of the forward pass.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe import model as gm
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    _build_packed_gemma4_causal_mask_mapping,
    _force_repeat_kv_for_sdpa,
)


def _text_config(**overrides):
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


def _cfg(**text_overrides):
    return Gemma4Config(text_config=_text_config(**text_overrides))


def _backend():
    return BackendConfig(
        linear="torch", attn="sdpa", rms_norm="torch", experts="torch",
        dispatcher="torch", fake_balanced_gate=False, enable_hf_state_dict_adapter=False,
    )


# ---------------------------------------------------------------------------
# _force_repeat_kv_for_sdpa
# ---------------------------------------------------------------------------
def test_force_repeat_kv_swaps_and_restores():
    from transformers.integrations import sdpa_attention

    original = sdpa_attention.use_gqa_in_sdpa
    with _force_repeat_kv_for_sdpa():
        assert sdpa_attention.use_gqa_in_sdpa(None, None) is False
    assert sdpa_attention.use_gqa_in_sdpa is original


def test_force_repeat_kv_import_error_is_silent(monkeypatch):
    # Remove the cached attribute AND null the submodule so the inner
    # `from transformers.integrations import sdpa_attention` raises ImportError.
    import transformers.integrations as ti

    monkeypatch.delattr(ti, "sdpa_attention", raising=False)
    monkeypatch.setitem(sys.modules, "transformers.integrations.sdpa_attention", None)
    with _force_repeat_kv_for_sdpa():
        pass  # must not raise


# ---------------------------------------------------------------------------
# _build_packed_gemma4_causal_mask_mapping
# ---------------------------------------------------------------------------
def test_packed_mask_rejects_non_2d_packed_ids():
    with pytest.raises(ValueError, match="2D"):
        _build_packed_gemma4_causal_mask_mapping(
            torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
            dtype=torch.float32, sliding_window=None,
        )


def test_packed_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        _build_packed_gemma4_causal_mask_mapping(
            torch.ones(1, 4, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long),
            dtype=torch.float32, sliding_window=None,
        )


def test_packed_mask_rejects_both_additive_and_block():
    with pytest.raises(ValueError, match="Only one of"):
        _build_packed_gemma4_causal_mask_mapping(
            torch.ones(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long),
            dtype=torch.float32, sliding_window=None, as_additive=True, as_block_mask=True,
        )


def test_packed_mask_additive_form_has_zero_and_neg_inf():
    packed = torch.tensor([[1, 1, 1, 1]])
    mm = torch.tensor([[0, 1, 1, 0]])
    out = _build_packed_gemma4_causal_mask_mapping(
        packed, mm, dtype=torch.float32, sliding_window=2, as_additive=True,
    )
    full = out["full_attention"]
    assert full.shape == (1, 1, 4, 4)
    assert full[0, 0, 0, 0].item() == 0.0  # allowed -> 0.0
    assert full[0, 0, 0, 3].item() == torch.finfo(torch.float32).min  # masked -> -inf


def test_packed_mask_block_mask_form_returns_block_masks():
    packed = torch.tensor([[1, 1, 1, 1]])
    mm = torch.tensor([[0, 1, 1, 0]])
    out = _build_packed_gemma4_causal_mask_mapping(
        packed, mm, dtype=torch.float32, sliding_window=2, as_block_mask=True,
    )
    # create_block_mask returns BlockMask objects for both layer types
    assert set(out) == {"full_attention", "sliding_attention"}
    assert type(out["full_attention"]).__name__ == "BlockMask"
    assert type(out["sliding_attention"]).__name__ == "BlockMask"


def test_packed_mask_default_form_returns_bool_4d():
    packed = torch.tensor([[1, 1, 2, 2]])  # two documents
    mm = torch.zeros(1, 4, dtype=torch.long)
    out = _build_packed_gemma4_causal_mask_mapping(packed, mm, dtype=torch.float32, sliding_window=None)
    full = out["full_attention"]
    assert full.shape == (1, 1, 4, 4) and full.dtype == torch.bool
    # cross-document attention is disallowed
    assert full[0, 0, 2, 0].item() is False


# ---------------------------------------------------------------------------
# _get_text_pad_token_id (called unbound with a lightweight fake self)
# ---------------------------------------------------------------------------
def _pad_id(**self_attrs):
    fake = SimpleNamespace(**self_attrs)
    return Gemma4ForConditionalGeneration._get_text_pad_token_id(fake)


def test_pad_token_id_uses_instance_attr():
    assert _pad_id(pad_token_id=7, config=SimpleNamespace(text_config=None, eos_token_id=None)) == 7


def test_pad_token_id_falls_back_to_config_text_config():
    cfg = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=3), eos_token_id=None)
    assert _pad_id(pad_token_id=-1, config=cfg) == 3


def test_pad_token_id_falls_back_to_eos_list():
    cfg = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=None), eos_token_id=[9, 10])
    assert _pad_id(pad_token_id=None, config=cfg) == 9


def test_pad_token_id_raises_when_unresolvable():
    cfg = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=None), eos_token_id=None)
    with pytest.raises(ValueError, match="valid pad_token_id"):
        _pad_id(pad_token_id=-1, config=cfg)


# ---------------------------------------------------------------------------
# _get_special_image_mask (unbound with fake self)
# ---------------------------------------------------------------------------
def test_special_image_mask_from_mm_token_type_ids():
    fake = SimpleNamespace(config=SimpleNamespace(image_token_id=42))
    mm = torch.tensor([[0, 1, 0, 1]])
    out = Gemma4ForConditionalGeneration._get_special_image_mask(fake, torch.zeros(1, 4, dtype=torch.long), mm)
    assert out.tolist() == [[False, True, False, True]]


def test_special_image_mask_from_image_token_id():
    fake = SimpleNamespace(config=SimpleNamespace(image_token_id=42))
    ids = torch.tensor([[1, 42, 3, 42]])
    out = Gemma4ForConditionalGeneration._get_special_image_mask(fake, ids, None)
    assert out.tolist() == [[False, True, False, True]]


# ---------------------------------------------------------------------------
# __init__: pad_token_id derived from a list-valued eos_token_id
# ---------------------------------------------------------------------------
def test_init_derives_pad_token_id_from_eos_list():
    cfg = _cfg(pad_token_id=None, eos_token_id=[5, 6])
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend())
    assert model.pad_token_id == 5


# ---------------------------------------------------------------------------
# prepare_model_inputs_for_cp / prepare_inputs_embeds_for_cp
# ---------------------------------------------------------------------------
def test_prepare_model_inputs_requires_input_ids():
    model = Gemma4ForConditionalGeneration(_cfg(), backend=_backend())
    with pytest.raises(ValueError, match="requires input_ids"):
        model.prepare_model_inputs_for_cp(input_ids=None)


def test_prepare_inputs_embeds_for_cp_delegates():
    cfg = _cfg()
    cfg.image_token_id = 42
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    ids = torch.tensor([[1, 42, 3, 4]])
    # prepare_inputs_embeds_for_cp returns just the inputs_embeds tensor
    embeds = model.prepare_inputs_embeds_for_cp(input_ids=ids)
    expected = model.prepare_model_inputs_for_cp(input_ids=ids)["inputs_embeds"]
    assert isinstance(embeds, torch.Tensor)
    assert embeds.shape == expected.shape


def test_forward_pre_embed_only_routes_to_prepare(monkeypatch):
    cfg = _cfg()
    cfg.image_token_id = 42
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    sentinel = {"inputs_embeds": torch.zeros(1)}
    monkeypatch.setattr(model, "prepare_model_inputs_for_cp", lambda **kw: sentinel)
    out = model(input_ids=torch.tensor([[1, 2, 3]]), _pre_embed_only=True)
    assert out is sentinel


# ---------------------------------------------------------------------------
# forward: MoE CP with pixel_values is rejected
# ---------------------------------------------------------------------------
def test_forward_moe_cp_pixel_values_raises():
    model = Gemma4ForConditionalGeneration(_cfg(), backend=_backend()).to(torch.bfloat16)
    model._cp_enabled = True
    with pytest.raises(NotImplementedError, match="pixel_values requires"):
        model(input_ids=torch.tensor([[1, 2, 3, 4]]), pixel_values=torch.randn(1, 3, 8, 8))


# ---------------------------------------------------------------------------
# forward: dense CP branches
# ---------------------------------------------------------------------------
def test_forward_dense_cp_pixel_values_raises():
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend()).to(torch.bfloat16)
    model._cp_enabled = True
    with pytest.raises(NotImplementedError, match="pixel_values requires"):
        model(input_ids=torch.tensor([[1, 2, 3, 4]]), pixel_values=torch.randn(1, 3, 8, 8))


def test_forward_dense_cp_requires_inputs():
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend()).to(torch.bfloat16)
    model._cp_enabled = True
    with pytest.raises(ValueError, match="requires either input_ids or inputs_embeds"):
        model(input_ids=None)


def test_forward_dense_cp_embeds_from_ids_with_softcapping():
    cfg = _cfg(enable_moe_block=False, final_logit_softcapping=30.0)
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    model._cp_enabled = True
    batch, seq = 1, 4
    hidden = torch.randn(batch, seq, cfg.text_config.hidden_size, dtype=torch.bfloat16)
    with mock.patch.object(
        model.model.language_model, "forward",
        return_value=SimpleNamespace(last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None),
    ) as fwd:
        out = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    # input_ids were embedded before being passed to the text model
    assert fwd.call_args.kwargs["inputs_embeds"] is not None
    assert out.logits.shape == (batch, seq, cfg.text_config.vocab_size)
    # softcapping bounds logits within +/- cap
    assert out.logits.abs().max().item() <= 30.0 + 1e-3


def test_forward_dense_train_vision_fills_mm_token_type_ids():
    cfg = _cfg(enable_moe_block=False, use_bidirectional_attention="vision")
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    model.train()
    captured = {}

    def fake_super_forward(*args, **kwargs):
        captured["mm"] = kwargs.get("mm_token_type_ids")
        return SimpleNamespace(logits=torch.zeros(1, 4, cfg.text_config.vocab_size))

    # Patch the HF parent forward so we only exercise the pre-delegation branch.
    with mock.patch(
        "transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration.forward",
        side_effect=fake_super_forward,
    ):
        model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert captured["mm"] is not None  # zeros tensor was synthesized
    assert torch.equal(captured["mm"], torch.zeros(1, 4, dtype=torch.long))


# ---------------------------------------------------------------------------
# Gemma4MoETextModelBackend.forward: CP and vision-bidirectional mask branches
# ---------------------------------------------------------------------------
def _moe_backend(**text_overrides):
    model = Gemma4ForConditionalGeneration(_cfg(**text_overrides), backend=_backend()).to(torch.bfloat16)
    return model.model.language_model


def test_backend_forward_cp_enabled_sets_sdpa_and_null_masks():
    backend = _moe_backend()
    seq = 4
    embeds = torch.randn(1, seq, 64, dtype=torch.bfloat16)
    out = backend(
        inputs_embeds=embeds,
        cp_enabled=True,
        position_ids=torch.arange(seq).unsqueeze(0),
        mm_token_type_ids=torch.zeros(1, seq, dtype=torch.long),
    )
    # cp path forces the HF dispatcher to SDPA so the CP hook can intercept it
    assert backend.config._attn_implementation == "sdpa"
    assert out.last_hidden_state.shape == (1, seq, 64)


def test_backend_forward_not_cp_pops_packed_seq_ids():
    backend = _moe_backend()
    seq = 4
    embeds = torch.randn(1, seq, 64, dtype=torch.bfloat16)
    # passing _packed_seq_ids with cp disabled exercises the pop branch
    out = backend(
        inputs_embeds=embeds,
        cp_enabled=False,
        position_ids=torch.arange(seq).unsqueeze(0),
        _packed_seq_ids=torch.ones(1, seq, dtype=torch.long),
    )
    assert out.last_hidden_state.shape == (1, seq, 64)


def test_backend_forward_vision_packed_builds_mask_mapping():
    backend = _moe_backend(use_bidirectional_attention="vision")
    seq = 4
    embeds = torch.randn(1, seq, 64, dtype=torch.bfloat16)
    out = backend(
        inputs_embeds=embeds,
        cp_enabled=False,
        position_ids=torch.arange(seq).unsqueeze(0),
        mm_token_type_ids=torch.tensor([[0, 1, 1, 0]]),
        _packed_seq_ids=torch.ones(1, seq, dtype=torch.long),
    )
    assert out.last_hidden_state.shape == (1, seq, 64)


def test_backend_forward_vision_without_mm_synthesizes_zeros():
    backend = _moe_backend(use_bidirectional_attention="vision")
    seq = 4
    embeds = torch.randn(1, seq, 64, dtype=torch.bfloat16)
    # No mm_token_type_ids and no _packed_seq_ids -> synthesize zeros + HF mask mapping
    out = backend(
        inputs_embeds=embeds,
        cp_enabled=False,
        position_ids=torch.arange(seq).unsqueeze(0),
    )
    assert out.last_hidden_state.shape == (1, seq, 64)


# ---------------------------------------------------------------------------
# Gemma4MoEDecoderLayer.forward: flex kernel_options / CP padding_mask branches
# ---------------------------------------------------------------------------
def test_decoder_forward_flex_kernel_options_and_padding_branches():
    from unittest.mock import patch

    from nemo_automodel.components.models.gemma4_moe.model import Gemma4MoEDecoderLayer
    from nemo_automodel.components.moe.layers import MoEConfig

    tc = _text_config()
    mc = MoEConfig(
        dim=tc.hidden_size, inter_dim=tc.intermediate_size, moe_inter_dim=tc.moe_intermediate_size,
        n_routed_experts=tc.num_experts, n_shared_experts=0, n_activated_experts=tc.top_k_experts,
        n_expert_groups=0, n_limited_groups=0, train_gate=True, gate_bias_update_factor=0.0,
        score_func="softmax", route_scale=1.0, aux_loss_coeff=0.0, norm_topk_prob=True,
        expert_activation="geglu", softmax_before_topk=False,
    )
    layer = Gemma4MoEDecoderLayer(tc, layer_idx=0, moe_config=mc, backend=_backend()).to(torch.bfloat16)
    # Drive the flex_attention + head_dim>256 + CP-hook branches in the layer forward.
    layer.config._attn_implementation = "flex_attention"
    layer.self_attn.head_dim = 300
    layer.self_attn._cp_uses_attention_hook = True

    b, s = 1, 4
    x = torch.randn(b, s, tc.hidden_size, dtype=torch.bfloat16)
    pos = (
        torch.randn(b, s, tc.head_dim // 2, dtype=torch.bfloat16),
        torch.randn(b, s, tc.head_dim // 2, dtype=torch.bfloat16),
    )
    padding_mask = torch.tensor([[False, False, True, True]])
    captured = {}

    def fake_attn(**kw):
        captured.update(kw)
        return torch.zeros_like(x), None

    with (
        patch.object(layer.self_attn, "forward", side_effect=fake_attn),
        patch.object(layer.moe, "forward", return_value=torch.zeros_like(x)),
    ):
        out = layer(x, position_embeddings=pos, padding_mask=padding_mask)

    assert "kernel_options" in captured  # flex + head_dim>256 path (line 279)
    assert captured.get("padding_mask") is not None  # CP-hook padding_mask (line 293)
    assert out.shape == x.shape  # flex masked_fill on padded rows ran (line 307)


# ---------------------------------------------------------------------------
# _prepare_per_layer_inputs_for_cp + per_layer_inputs threading
# ---------------------------------------------------------------------------
def test_prepare_per_layer_inputs_returns_none_without_per_layer():
    fake = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(hidden_size_per_layer_input=0)))
    out = Gemma4ForConditionalGeneration._prepare_per_layer_inputs_for_cp(
        fake, torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 4, dtype=torch.bool)
    )
    assert out is None


def test_prepare_per_layer_inputs_returns_per_layer_tensor():
    lm = SimpleNamespace(
        hidden_size_per_layer_input=8,
        get_per_layer_inputs=lambda ids, _x: torch.full((1, 4, 8), 5.0),
    )
    fake = SimpleNamespace(
        model=SimpleNamespace(language_model=lm),
        _get_text_pad_token_id=lambda: 0,
    )
    out = Gemma4ForConditionalGeneration._prepare_per_layer_inputs_for_cp(
        fake, torch.tensor([[1, 2, 3, 4]]), torch.tensor([[False, True, False, False]])
    )
    assert out.shape == (1, 4, 8)


def test_prepare_model_inputs_threads_per_layer_inputs(monkeypatch):
    cfg = _cfg()
    cfg.image_token_id = 42
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    monkeypatch.setattr(model, "_prepare_per_layer_inputs_for_cp", lambda ids, mask: torch.zeros(1, 4, 8))
    prepared = model.prepare_model_inputs_for_cp(input_ids=torch.tensor([[1, 42, 3, 4]]))
    assert "per_layer_inputs" in prepared
    assert prepared["per_layer_inputs"].shape == (1, 4, 8)
