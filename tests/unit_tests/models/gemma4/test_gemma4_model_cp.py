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
packed-mask builder branches, and the CP/vision branches of the forward pass.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    HFGemma4ForConditionalGeneration,
    _build_packed_gemma4_causal_mask_mapping,
    _Gemma4KVShareHolder,
    _kv_sharing_active,
)

# Run only on the GPU job and build everything under the CUDA device context.
# HF weight init (_init_weights -> torch.Tensor.normal_) on bf16 CPU tensors has a
# large fixed per-call cost (~12s for even this tiny model, ~150x slower than GPU),
# which made this file ~149s on the CPU unit-test job. Constructing on-device keeps
# that init on the GPU.
pytestmark = pytest.mark.run_only_on("GPU")


@pytest.fixture(autouse=True)
def _build_on_cuda():
    """Build the models/tensors in these GPU-only tests on CUDA (see note above)."""
    with torch.device("cuda"):
        yield


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
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


# ---------------------------------------------------------------------------
# _build_packed_gemma4_causal_mask_mapping
# ---------------------------------------------------------------------------
def test_packed_mask_rejects_non_2d_packed_ids():
    with pytest.raises(ValueError, match="2D"):
        _build_packed_gemma4_causal_mask_mapping(
            torch.ones(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            dtype=torch.float32,
            sliding_window=None,
        )


def test_packed_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        _build_packed_gemma4_causal_mask_mapping(
            torch.ones(1, 4, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            dtype=torch.float32,
            sliding_window=None,
        )


def test_packed_mask_rejects_both_additive_and_block():
    with pytest.raises(ValueError, match="Only one of"):
        _build_packed_gemma4_causal_mask_mapping(
            torch.ones(1, 4, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            dtype=torch.float32,
            sliding_window=None,
            as_additive=True,
            as_block_mask=True,
        )


def test_packed_mask_additive_form_has_zero_and_neg_inf():
    packed = torch.tensor([[1, 1, 1, 1]])
    mm = torch.tensor([[0, 1, 1, 0]])
    out = _build_packed_gemma4_causal_mask_mapping(
        packed,
        mm,
        dtype=torch.float32,
        sliding_window=2,
        as_additive=True,
    )
    full = out["full_attention"]
    assert full.shape == (1, 1, 4, 4)
    assert full[0, 0, 0, 0].item() == 0.0  # allowed -> 0.0
    assert full[0, 0, 0, 3].item() == torch.finfo(torch.float32).min  # masked -> -inf


def test_packed_mask_block_mask_form_returns_block_masks():
    packed = torch.tensor([[1, 1, 1, 1]])
    mm = torch.tensor([[0, 1, 1, 0]])
    out = _build_packed_gemma4_causal_mask_mapping(
        packed,
        mm,
        dtype=torch.float32,
        sliding_window=2,
        as_block_mask=True,
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
# prepare_model_inputs_for_cp
# ---------------------------------------------------------------------------
def test_prepare_model_inputs_requires_input_ids():
    model = Gemma4ForConditionalGeneration(_cfg(), backend=_backend())
    with pytest.raises(ValueError, match="requires input_ids"):
        model.prepare_model_inputs_for_cp({"input_ids": None})


def test_forward_pre_embed_only_routes_to_prepare(monkeypatch):
    cfg = _cfg()
    cfg.image_token_id = 42
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    sentinel = {"inputs_embeds": torch.zeros(1)}
    monkeypatch.setattr(model, "prepare_model_inputs_for_cp", lambda *a, **kw: sentinel)
    # The dispatcher hands the whole batch dict through the _cp_batch kwarg.
    out = model(_pre_embed_only=True, _cp_batch={"input_ids": torch.tensor([[1, 2, 3]])})
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
        model.model.language_model,
        "forward",
        return_value=SimpleNamespace(
            last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None
        ),
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
        dim=tc.hidden_size,
        inter_dim=tc.intermediate_size,
        moe_inter_dim=tc.moe_intermediate_size,
        n_routed_experts=tc.num_experts,
        n_shared_experts=0,
        n_activated_experts=tc.top_k_experts,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_activation="geglu",
        softmax_before_topk=False,
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
    prepared = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[1, 42, 3, 4]])})
    assert "per_layer_inputs" in prepared
    assert prepared["per_layer_inputs"].shape == (1, 4, 8)


def test_prepare_model_inputs_removes_consumed_keys_from_batch(monkeypatch):
    """The hook mutates ``batch`` in place, dropping the raw inputs it consumed.
    ``mm_token_type_ids`` is popped from the input but re-emitted in the output dict."""
    cfg = _cfg()
    cfg.image_token_id = 42
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    monkeypatch.setattr(model, "_prepare_per_layer_inputs_for_cp", lambda ids, mask: None)
    batch = {
        "input_ids": torch.tensor([[1, 42, 3, 4]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]]),
    }
    prepared = model.prepare_model_inputs_for_cp(batch)
    assert prepared["input_ids"] is None
    assert prepared["pixel_values"] is None
    # mm_token_type_ids is consumed but re-emitted as an output, so the real
    # value (not the None consumed-marker) wins in the returned dict.
    assert prepared["mm_token_type_ids"] is not None


# ---------------------------------------------------------------------------
# get_capabilities: dense 31B supports CP; E2B/E4B (audio) does not; MoE does
# ---------------------------------------------------------------------------
def test_get_capabilities_plain_dense_supports_cp():
    caps = Gemma4ForConditionalGeneration.get_capabilities(_cfg(enable_moe_block=False))
    assert caps.supports_cp is True
    assert caps.supports_tp is True
    assert caps.supports_pp is True
    assert caps.supports_ep is False


def test_get_capabilities_dense_audio_variant_supports_cp():
    # E2B/E4B: dense + audio_config present -> CP supported (kv-sharing +
    # per-layer-inputs flow through the model-owned ring). TP/PP/EP stay off.
    cfg = _cfg(enable_moe_block=False)
    cfg.audio_config = {}  # non-None marks the dense+audio variant (E2B/E4B)
    caps = Gemma4ForConditionalGeneration.get_capabilities(cfg)
    assert caps.supports_cp is True
    assert caps.supports_tp is False
    assert caps.supports_pp is False
    assert caps.supports_ep is False


def test_get_capabilities_moe_supports_cp_and_ep():
    caps = Gemma4ForConditionalGeneration.get_capabilities(_cfg(enable_moe_block=True))
    assert caps.supports_cp is True
    assert caps.supports_ep is True
    assert caps.supports_tp is False


# ---------------------------------------------------------------------------
# Dense __init__ attaches the model-owned ring to each self-attention module
# ---------------------------------------------------------------------------
def test_dense_init_attaches_ring_to_self_attention():
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend())
    hooked = [m for m in model.modules() if getattr(m, "_cp_manual_metadata_keys", None)]
    # one per dense decoder layer
    assert len(hooked) == model.config.text_config.num_hidden_layers
    for m in hooked:
        assert m._cp_manual_metadata_keys == (
            "mm_token_type_ids",
            "_packed_seq_ids",
            "padding_mask",
            "_gemma4_vision_group_ids",
        )
        assert callable(m.setup_cp_attention)


# ---------------------------------------------------------------------------
# setup_cp_attention: model-level seam, idempotent, fans out to submodules
# ---------------------------------------------------------------------------
def test_setup_cp_attention_sets_flag_and_calls_submodules():
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend())
    calls = {"n": 0}
    seen_mesh = []
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "setup_cp_attention"):

            def _stub(mesh, _calls=calls, _seen=seen_mesh):
                _calls["n"] += 1
                _seen.append(mesh)

            m.setup_cp_attention = _stub
    mesh = object()
    model.setup_cp_attention(mesh)
    assert model._cp_enabled is True
    assert calls["n"] == model.config.text_config.num_hidden_layers
    assert all(s is mesh for s in seen_mesh)


def test_setup_cp_attention_idempotent():
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend())
    calls = {"n": 0}
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "setup_cp_attention"):
            m.setup_cp_attention = lambda mesh, _c=calls: _c.__setitem__("n", _c["n"] + 1)
    model.setup_cp_attention(object())
    first = calls["n"]
    assert first > 0
    # second call returns early via the _cp_enabled guard -> no extra submodule calls
    model.setup_cp_attention(object())
    assert calls["n"] == first


# ---------------------------------------------------------------------------
# _cp_shard_batch: installs the ring (idempotent) then delegates to the sharder
# ---------------------------------------------------------------------------
def test_cp_shard_batch_installs_ring_then_delegates(monkeypatch):
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend())
    installed = {"mesh": None}
    monkeypatch.setattr(model, "setup_cp_attention", lambda mesh: installed.__setitem__("mesh", mesh))

    sentinel = ("ctx", {"sharded": True})
    seen = {}

    def fake_shard(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0, **kwargs):
        seen.update(
            cp_mesh=cp_mesh, tp_mesh=tp_mesh, batch=batch, loss_mask=loss_mask, padding_token_id=padding_token_id
        )
        return sentinel

    monkeypatch.setattr(
        "nemo_automodel.components.models.gemma4_moe.model.make_contiguous_shard_cp_batch_and_ctx",
        fake_shard,
    )

    cp_mesh, tp_mesh, batch = object(), object(), {"input_ids": torch.tensor([[1, 2]])}
    out = model._cp_shard_batch(cp_mesh, tp_mesh, batch, loss_mask="lm", padding_token_id=7)
    assert out is sentinel
    assert installed["mesh"] is cp_mesh  # ring installed with the CP submesh
    assert seen["cp_mesh"] is cp_mesh and seen["tp_mesh"] is tp_mesh
    assert seen["loss_mask"] == "lm" and seen["padding_token_id"] == 7


def test_prepare_model_inputs_attaches_cp_shard_batch_fn():
    model = Gemma4ForConditionalGeneration(_cfg(enable_moe_block=False), backend=_backend()).to(torch.bfloat16)
    prepared = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[1, 2, 3, 4]])})
    # The model attaches its own bound batch-sharding callable (model-owned install seam).
    # shard_batch is the model's bound sharding callable, wrapped so it
    # records its shard facts on the sharder it belongs to.
    assert prepared["cp_sharder"].shard_batch.func == model._cp_shard_batch
    assert prepared["cp_sharder"].shard_batch.keywords["record_on"] is prepared["cp_sharder"]


# ---------------------------------------------------------------------------
# Dense CP forward stashes CP-sharded metadata on the ring-hooked self_attn modules
# ---------------------------------------------------------------------------
def test_forward_dense_cp_stashes_metadata_on_ring_modules():
    cfg = _cfg(enable_moe_block=False, final_logit_softcapping=30.0)
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    model.setup_cp_attention(object())  # installs ring -> _cp_uses_attention_hook on self_attn

    batch, seq = 1, 4
    hidden = torch.randn(batch, seq, cfg.text_config.hidden_size, dtype=torch.bfloat16)
    packed = torch.tensor([[1, 1, 2, 2]])
    with mock.patch.object(
        model.model.language_model,
        "forward",
        return_value=SimpleNamespace(
            last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None
        ),
    ):
        model(input_ids=torch.tensor([[1, 2, 3, 4]]), _packed_seq_ids=packed)

    hooked = [m for m in model.modules() if getattr(m, "_cp_uses_attention_hook", False)]
    assert hooked, "expected ring-hooked self_attn modules"
    for m in hooked:
        meta = m._cp_dense_metadata
        assert set(meta) == {"mm_token_type_ids", "padding_mask", "_packed_seq_ids", "_gemma4_vision_group_ids"}
        assert torch.equal(meta["_packed_seq_ids"], packed)


def test_forward_dense_cp_injects_kv_share_holder():
    # CP dense path + kv-sharing active -> the cache-free _Gemma4KVShareHolder is
    # injected and threaded to the language model (mirrors the non-CP path).
    cfg = _cfg(enable_moe_block=False, num_kv_shared_layers=2)
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    model.setup_cp_attention(object())  # flips _cp_enabled -> CP forward branch

    hidden = torch.randn(1, 4, cfg.text_config.hidden_size, dtype=torch.bfloat16)
    captured = {}

    def fake_lm_forward(*args, **kwargs):
        captured["pkv"] = kwargs.get("past_key_values")
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None)

    with mock.patch.object(model.model.language_model, "forward", side_effect=fake_lm_forward):
        model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert isinstance(captured["pkv"], _Gemma4KVShareHolder)


# ---------------------------------------------------------------------------
# _Gemma4KVShareHolder / _kv_sharing_active
# (cache-free kv-sharing under use_cache=False / CP, gated to E2B/E4B)
# ---------------------------------------------------------------------------
def test_kv_share_holder_is_cache_free_passthrough():
    h = _Gemma4KVShareHolder()
    assert h.shared_layers == {}
    # Satisfies HF's `past_key_values is not None` gate without acting like a cache:
    # zero cache offset and a pass-through update (no per-token accumulation).
    assert h.get_seq_length() == 0
    assert h.get_seq_length(0, foo=1) == 0
    assert h.get_mask_sizes(13) == (13, 0)
    assert h.get_mask_sizes(7, layer_idx=3) == (7, 0)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out_k, out_v = h.update(k, v, layer_idx=0)
    assert out_k is k and out_v is v


def test_kv_sharing_active_thresholds():
    assert _kv_sharing_active(SimpleNamespace(num_kv_shared_layers=18)) is True
    assert _kv_sharing_active(SimpleNamespace(num_kv_shared_layers=1)) is True
    assert _kv_sharing_active(SimpleNamespace(num_kv_shared_layers=0)) is False
    assert _kv_sharing_active(SimpleNamespace(num_kv_shared_layers=None)) is False
    assert _kv_sharing_active(SimpleNamespace()) is False  # attribute absent -> not kv-sharing


# ---------------------------------------------------------------------------
# Non-CP dense forward: kv-share holder injection (gated to E2B/E4B)
# ---------------------------------------------------------------------------
def test_forward_dense_noncp_injects_kv_share_holder():
    # kv-sharing active + dense + NOT cp-enabled -> the non-CP dense path passes a
    # cache-free _Gemma4KVShareHolder as past_key_values to the HF super().forward.
    cfg = _cfg(enable_moe_block=False, num_kv_shared_layers=2)
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    captured = {}

    def fake_super_forward(_self, *args, **kwargs):
        captured["pkv"] = kwargs.get("past_key_values")
        return "OUT"

    with mock.patch.object(HFGemma4ForConditionalGeneration, "forward", fake_super_forward):
        out = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert out == "OUT"
    assert isinstance(captured["pkv"], _Gemma4KVShareHolder)


def test_forward_dense_noncp_no_holder_without_kv_sharing():
    # No kv-sharing -> no holder injected (preserves prior behavior; 31B/MoE path).
    cfg = _cfg(enable_moe_block=False, num_kv_shared_layers=0)
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.bfloat16)
    captured = {}

    def fake_super_forward(_self, *args, **kwargs):
        captured["pkv"] = kwargs.get("past_key_values")
        return "OUT"

    with mock.patch.object(HFGemma4ForConditionalGeneration, "forward", fake_super_forward):
        model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert captured["pkv"] is None
