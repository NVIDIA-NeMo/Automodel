# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Inkling 2 DSpark draft: FP32 parity against a per-block FP64 oracle, released layout, contract checks.

The oracle restates the released drafter one block at a time from the serving math
(TokenSpeed ``inkling2_mtp.py`` / vLLM ``inkling2/dspark.py``): explicit visible-key
lists per block, complex-number RoPE, full ``rnn_proj`` matmuls, and the FP64 fold of
``hidden_out`` into the LM head that the serving engines use.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components.models.inkling2 import dspark as inkling2_dspark
from nemo_automodel.components.models.inkling2.dspark import (
    Inkling2DSparkConfig,
    Inkling2DSparkModel,
    Inkling2DSparkReleasedConfig,
)
from nemo_automodel.components.speculative.dspark.core import DSparkTrainerModule

pytestmark = pytest.mark.cpu_only

# Three draft layers (0 and 2 local, 1 global), a three-slot window that the
# anchors below cross, four-slot blocks (three proposals), and padded vocab.
_MTP_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 48,
    "max_position_embeddings": 1024,
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 3,
    "block_size": 4,
    "markov_rank": 6,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [2, 3, 4],
}
_VOCAB, _REAL_VOCAB, _SOFTCAP, _TARGET_LAYERS = 40, 37, 5.0, 4
_SEQ, _BATCH = 16, 2
# Row 0: anchors 1 (first legal), 6 (window cut), 11 (window cut), and one padding block.
# Row 1: 3 (window cut), 9, 13 (label runs past the sequence end), 4.
_ANCHORS = torch.tensor([[1, 6, 11, 0], [3, 4, 9, 13]])
_KEEP = torch.tensor([[True, True, True, False], [True, True, True, True]])


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=_TARGET_LAYERS,
        vocab_size=_VOCAB,
        unpadded_vocab_size=_REAL_VOCAB,
        final_logit_softcapping=_SOFTCAP,
    )


def _options(**overrides) -> SimpleNamespace:
    options = SimpleNamespace(
        num_draft_layers=3,
        target_layer_ids=[1, 2, 3],
        block_size=3,
        num_anchors=4,
        mask_token_id=0,
        markov_rank=6,
        markov_head_type="rnn_hidden",
        confidence_head_alpha=0.0,
        confidence_head_with_markov=False,
    )
    for key, value in overrides.items():
        if not hasattr(options, key):
            raise ValueError(f"options has no field {key!r}")
        setattr(options, key, value)
    return options


def _build(seed: int = 0) -> Inkling2DSparkModel:
    torch.manual_seed(seed)
    model = Inkling2DSparkConfig.from_options(_text_config(), _MTP_CONFIG, _options()).build()
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Non-trivial norms, mixes and biases so every term of the math is exercised.
            if name.endswith("norm.weight") or "norm_ctx" in name:
                param.uniform_(0.5, 1.5)
            elif name in ("fuse_weight", "markov_head.rnn_proj.bias", "markov_head.hidden_out.bias"):
                param.normal_(0.0, 1.0)
    embed = torch.nn.Embedding(_VOCAB, 32)
    embed.embed_norm = torch.nn.Module()
    embed.embed_norm.weight = torch.nn.Parameter(torch.rand(32) + 0.5)
    lm_head = torch.nn.Linear(32, _VOCAB, bias=False)
    model.initialize_embeddings_and_head(embed_tokens=embed, lm_head=lm_head, freeze=True)
    return model


def _inputs(seed: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    loss_mask = torch.ones(_BATCH, _SEQ)
    loss_mask[0, 0] = 0
    loss_mask[1, 12] = 0  # truncates the row-1 block anchored at 9 after two proposals
    return {
        "input_ids": torch.randint(0, _REAL_VOCAB, (_BATCH, _SEQ), generator=gen),
        "target_hidden_states": torch.randn(_BATCH, _SEQ, 3 * 32, generator=gen),
        "loss_mask": loss_mask,
        "target_last_hidden_states": torch.randn(_BATCH, _SEQ, 32, generator=gen),
    }


@pytest.fixture
def fixed_anchors(monkeypatch):
    def _sample(*, seq_len, loss_mask, num_anchors, device, doc_remaining=None):
        return _ANCHORS.clone(), _KEEP.clone()

    monkeypatch.setattr(inkling2_dspark, "sample_anchor_positions", _sample)


# --------------------------------------------------------------------------- oracle


def _oracle_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _oracle_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotate pairs (x[i], x[i + d/2]) as complex numbers by exp(1j * pos * theta^(-2i/d))."""
    half = x.shape[-1] // 2
    freqs = theta ** (-torch.arange(half, dtype=torch.float64) * 2 / x.shape[-1])
    angle = positions.to(torch.float64)[:, None] * freqs
    rotation = torch.polar(torch.ones_like(angle), angle)  # [len, half]
    while rotation.dim() < x.dim():
        rotation = rotation.unsqueeze(1)  # broadcast over heads
    z = torch.complex(x[..., :half], x[..., half:]) * rotation
    return torch.cat((z.real, z.imag), dim=-1)


def _oracle_block(p: dict[str, torch.Tensor], ids: torch.Tensor, taps: torch.Tensor, anchor: int) -> torch.Tensor:
    """Logits of one block's proposals, shape [proposals, real_vocab], in FP64."""
    cfg = _MTP_CONFIG
    eps, theta, slots = cfg["rms_norm_eps"], cfg["rope_theta"], cfg["block_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    alpha = torch.softmax(p["fuse_weight"], dim=-1)

    def mix(consumer: int, rows: torch.Tensor) -> torch.Tensor:
        return (alpha[consumer][None, :, None] * taps[rows]).sum(1)

    x = torch.stack(
        [_oracle_rms(p["embed_tokens.weight"][ids[anchor]], p["embed_norm_weight"], eps)]
        + [p["mask_embed"]] * (slots - 1)
    )
    block_pos = torch.arange(anchor, anchor + slots)
    for layer in range(cfg["num_hidden_layers"]):
        pre = f"layers.{layer}."
        first = max(0, anchor - cfg["sliding_window_size"]) if layer in cfg["local_layer_ids"] else 0
        ctx_pos = torch.arange(first, anchor)
        ctx = _oracle_rms(mix(layer, ctx_pos), p[pre + "dflare_norm.weight"], eps)
        h = _oracle_rms(x, p[pre + "input_layernorm.weight"], eps)

        def heads(t, n):
            return t.view(t.shape[0], n, hd)

        q = _oracle_rope(
            _oracle_rms(heads(h @ p[pre + "self_attn.q_proj.weight"].T, nh), p[pre + "self_attn.q_norm.weight"], eps),
            block_pos,
            theta,
        )
        k_b = _oracle_rope(
            _oracle_rms(heads(h @ p[pre + "self_attn.k_proj.weight"].T, nkv), p[pre + "self_attn.k_norm.weight"], eps),
            block_pos,
            theta,
        )
        v_b = heads(h @ p[pre + "self_attn.v_proj.weight"].T, nkv)
        k_c = _oracle_rope(
            _oracle_rms(
                heads(ctx @ p[pre + "self_attn.k_proj_ctx.weight"].T, nkv), p[pre + "self_attn.k_norm_ctx.weight"], eps
            ),
            ctx_pos,
            theta,
        )
        v_c = heads(ctx @ p[pre + "self_attn.v_proj_ctx.weight"].T, nkv)
        k, v = torch.cat((k_c, k_b)), torch.cat((v_c, v_b))
        out = []
        for head in range(nh):
            kv_head = head // (nh // nkv)
            scores = q[:, head] @ k[:, kv_head].T / math.sqrt(hd)
            out.append(torch.softmax(scores, dim=-1) @ v[:, kv_head])
        x = x + torch.cat(out, dim=-1) @ p[pre + "self_attn.o_proj.weight"].T
        h = _oracle_rms(x, p[pre + "post_attention_layernorm.weight"], eps)
        gate = h @ p[pre + "mlp.gate_proj.weight"].T
        x = x + (gate * torch.sigmoid(gate) * (h @ p[pre + "mlp.up_proj.weight"].T)) @ p[pre + "mlp.down_proj.weight"].T
    rows = _oracle_rms(x, p["norm.weight"], eps)[1:]

    # Markov head, serving form: FP64 fold of hidden_out into the LM head.
    head_w = p["lm_head.weight"]
    fold_weight = head_w @ p["markov_head.hidden_out.weight"]
    fold_bias = head_w @ p["markov_head.hidden_out.bias"]
    f_rnn = _oracle_rms(mix(len(alpha) - 1, torch.tensor([anchor - 1]))[0], p["markov_head.dflare_norm.weight"], eps)
    state = torch.zeros(cfg["markov_rank"], dtype=torch.float64)
    previous = ids[anchor]
    logits = []
    for step in range(slots - 1):
        z = torch.cat((state, p["markov_head.markov_w1.weight"][previous], rows[step], f_rnn))
        gate, candidate, output = (p["markov_head.rnn_proj.weight"] @ z + p["markov_head.rnn_proj.bias"]).chunk(3)
        gate = torch.sigmoid(gate)
        state = gate * state + (1 - gate) * torch.tanh(candidate)
        raw = head_w @ rows[step] + fold_bias + fold_weight @ torch.tanh(output)
        logits.append((_SOFTCAP * torch.tanh(raw / _SOFTCAP))[:_REAL_VOCAB])
        previous = ids[anchor + step + 1] if anchor + step + 1 < len(ids) else ids[-1]
    return torch.stack(logits)


def _oracle_params(model: Inkling2DSparkModel) -> dict[str, torch.Tensor]:
    params = {name: p.detach().double().requires_grad_(p.requires_grad) for name, p in model.named_parameters()}
    params["embed_norm_weight"] = model.embed_norm_weight.double()
    return params


# --------------------------------------------------------------------------- parity


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.double() - expected).abs().max().item()
    return diff, diff / max(expected.abs().max().item(), 1e-30)


def test_forward_and_backward_match_the_fp64_oracle(fixed_anchors):
    model = _build()
    inputs = _inputs()
    out = model(**inputs)
    assert out.draft_logits.shape == (_BATCH, 4, 3, _REAL_VOCAB)
    assert out.confidence_pred is None

    params = _oracle_params(model)
    weights = torch.randn(out.draft_logits.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    oracle_loss = 0.0
    loss = 0.0
    for b in range(_BATCH):
        for n in range(4):
            if not _KEEP[b, n]:
                continue
            anchor = int(_ANCHORS[b, n])
            taps = inputs["target_hidden_states"][b].double().view(_SEQ, 3, 32)
            expected = _oracle_block(params, inputs["input_ids"][b], taps, anchor)
            actual = out.draft_logits[b, n]
            abs_err, rel_err = _max_errors(actual, expected.detach())
            assert abs_err < 2e-5 and rel_err < 2e-6, (b, n, abs_err, rel_err)
            oracle_loss = oracle_loss + (expected * weights[b, n]).sum()
            loss = loss + (actual.double() * weights[b, n]).sum()
    oracle_loss.backward()
    loss.backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, name
            continue
        abs_err, rel_err = _max_errors(param.grad, params[name].grad)
        assert rel_err < 1e-5, (name, abs_err, rel_err)


def test_teacher_logits_are_the_softcapped_real_vocab_target_logits(fixed_anchors):
    model = _build()
    inputs = _inputs()
    out = model(**inputs)
    head = model.lm_head.weight.double()
    for b in range(_BATCH):
        for n in range(4):
            if not _KEEP[b, n]:
                continue
            anchor = int(_ANCHORS[b, n])
            for j in range(3):
                row = min(anchor + j, _SEQ - 2)  # the row whose next-token distribution scores proposal j
                raw = head @ inputs["target_last_hidden_states"][b, row].double()
                expected = (_SOFTCAP * torch.tanh(raw / _SOFTCAP))[:_REAL_VOCAB]
                if anchor + j + 1 < _SEQ:
                    torch.testing.assert_close(
                        out.aligned_target_logits[b, n, j].double(), expected, atol=1e-5, rtol=1e-5
                    )


def test_labels_and_eval_mask_follow_the_proposal_rows(fixed_anchors):
    model = _build()
    inputs = _inputs()
    out = model(**inputs)
    ids = inputs["input_ids"]
    # Proposal j of the block anchored at a predicts token a + j + 1.
    assert torch.equal(out.target_ids[0, 1], ids[0, 7:10])
    assert out.eval_mask[0, 1].all()
    # Loss mask zero at 12 stops the block anchored at 9 after proposals 10 and 11.
    assert out.eval_mask[1, 2].tolist() == [True, True, False]
    # The block anchored at 13 runs past the sequence end after 14 and 15.
    assert out.eval_mask[1, 3].tolist() == [True, True, False]
    assert not out.eval_mask[0, 3].any() and not out.block_keep_mask[0, 3]


def test_context_outside_the_anchored_window_is_invisible_to_local_layers():
    model = _build()
    inputs = _inputs()
    window = _MTP_CONFIG["sliding_window_size"]
    global_mask, local_mask = model._attention_masks(_ANCHORS, _KEEP, _SEQ, torch.float32)
    for b in range(_BATCH):
        for n in range(4):
            anchor = int(_ANCHORS[b, n])
            rows = slice(4 * n, 4 * n + 4)
            local_ctx = (local_mask[b, 0, rows, :_SEQ] == 0).all(dim=0)
            global_ctx = (global_mask[b, 0, rows, :_SEQ] == 0).all(dim=0)
            if not _KEEP[b, n]:
                assert not local_ctx.any() and not global_ctx.any()
                continue
            # Every row of a block sees the same context: [anchor - window, anchor) locally, [0, anchor) globally.
            assert torch.equal((local_mask[b, 0, rows, :_SEQ] == 0), local_ctx.expand(4, -1))
            assert local_ctx.nonzero().flatten().tolist() == list(range(max(0, anchor - window), anchor))
            assert global_ctx.nonzero().flatten().tolist() == list(range(anchor))
    del inputs


def test_first_position_never_anchors():
    model = _build()
    inputs = _inputs()
    inputs["loss_mask"] = torch.zeros(_BATCH, _SEQ)
    inputs["loss_mask"][:, :2] = 1  # only anchor 0 would be legal without the f_rnn row before it
    out = model(**inputs)
    assert not out.block_keep_mask.any()


def test_trainer_module_runs_ce_and_tv_without_a_confidence_head():
    model = _build()
    trainer = DSparkTrainerModule(model, loss_decay_gamma=7.0, confidence_head_alpha=0.0)
    metrics = trainer(**_inputs())
    metrics.loss.backward()
    assert torch.isfinite(metrics.loss)
    assert metrics.accept_rate_per_pos_num.shape == (3,)  # recipe block_size = drafted tokens
    assert metrics.confidence_diag_den == 0
    assert model.fuse_weight.grad is not None and model.mask_embed.grad is not None
    assert model.embed_tokens.weight.grad is None and model.lm_head.weight.grad is None


def test_fresh_initialization_is_finite():
    model = Inkling2DSparkConfig.from_options(_text_config(), _MTP_CONFIG, _options()).build()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    out = model(**_inputs())
    out.draft_logits.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


# --------------------------------------------------------------------------- released layout


def _released_shapes(mtp: dict, vocab: int) -> dict[str, tuple[int, ...]]:
    """Tensor shapes of mtp.safetensors as a function of mtp_config (from the released header)."""
    h, nh, nkv, hd, f, r = (
        mtp[k]
        for k in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "markov_rank",
        )
    )
    shapes = {
        "fuse_weight": (mtp["num_hidden_layers"] + 1, len(mtp["target_layer_ids"])),
        "mask_embed": (h,),
        "norm.weight": (h,),
        "markov_head.dflare_norm.weight": (h,),
        "markov_head.hidden_out.weight": (h, r),
        "markov_head.hidden_out.bias": (h,),
        "markov_head.markov_w1.weight": (vocab, r),
        "markov_head.rnn_proj.weight": (3 * r, 2 * r + 2 * h),
        "markov_head.rnn_proj.bias": (3 * r,),
    }
    for i in range(mtp["num_hidden_layers"]):
        pre = f"layers.{i}."
        shapes |= {
            pre + "dflare_norm.weight": (h,),
            pre + "input_layernorm.weight": (h,),
            pre + "post_attention_layernorm.weight": (h,),
            pre + "mlp.gate_proj.weight": (f, h),
            pre + "mlp.up_proj.weight": (f, h),
            pre + "mlp.down_proj.weight": (h, f),
            pre + "self_attn.q_proj.weight": (nh * hd, h),
            pre + "self_attn.k_proj.weight": (nkv * hd, h),
            pre + "self_attn.v_proj.weight": (nkv * hd, h),
            pre + "self_attn.k_proj_ctx.weight": (nkv * hd, h),
            pre + "self_attn.v_proj_ctx.weight": (nkv * hd, h),
            pre + "self_attn.o_proj.weight": (h, nh * hd),
            pre + "self_attn.q_norm.weight": (hd,),
            pre + "self_attn.k_norm.weight": (hd,),
            pre + "self_attn.k_norm_ctx.weight": (hd,),
        }
    return shapes


def test_released_mtp_config_builds_the_released_tensor_set():
    """The 11-12 release: 84 tensors; shapes below are the ones in its safetensors header."""
    mtp = {
        "hidden_size": 4608, "num_hidden_layers": 5, "num_attention_heads": 48, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 18432, "max_position_embeddings": 1048576, "q_bias": False,
        "o_bias": False, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "local_layer_ids": [0, 1, 3, 4],
        "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 11, 17, 23, 29, 35, 41, 48],
    }  # fmt: skip
    text = SimpleNamespace(
        hidden_size=4608, num_hidden_layers=48, vocab_size=201024, unpadded_vocab_size=200058,
        final_logit_softcapping=30.0,
    )  # fmt: skip
    options = _options(
        num_draft_layers=5, target_layer_ids=[4, 10, 16, 22, 28, 34, 40, 47], block_size=7, markov_rank=256
    )
    config = Inkling2DSparkConfig.from_options(text, mtp, options)
    with torch.device("meta"):
        model = Inkling2DSparkModel(config)
    released = model.state_dict_adapter.to_hf(model.state_dict())
    expected = _released_shapes(mtp, 201024)
    assert len(expected) == 84
    assert {key: tuple(t.shape) for key, t in released.items()} == expected
    assert model._pre_shard_hf_state_dict_keys == list(released)
    assert expected["markov_head.rnn_proj.weight"] == (768, 9728) and expected["fuse_weight"] == (6, 8)


def test_released_tensors_round_trip_through_the_adapter(tmp_path):
    model = _build()
    released = model.state_dict_adapter.to_hf(model.state_dict())
    path = tmp_path / "mtp.safetensors"
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in released.items()}, str(path))
    restored = _build(seed=3)
    missing, unexpected = restored.load_state_dict(
        restored.state_dict_adapter.from_hf(load_file(str(path))), strict=False
    )
    assert sorted(missing) == ["embed_tokens.weight", "lm_head.weight"] and not unexpected
    for key, value in released.items():
        assert torch.equal(restored.state_dict()[key], value.to(torch.bfloat16).float()), key


def test_adapter_rejects_tensors_outside_the_release():
    model = _build()
    with pytest.raises(ValueError, match="not part of the Inkling 2 DSpark draft"):
        model.state_dict_adapter.from_hf({"layers.0.self_attn.qkv_proj.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="unquantized"):
        model.state_dict_adapter.convert_single_tensor_to_hf("mask_embed", torch.zeros(1), quantization=True)


def test_export_writes_the_mtp_config(tmp_path):
    model = _build()
    exporter = model._get_consolidated_hf_metadata_exporter(tokenizer=None, original_model_path=None)
    exporter.save(hf_metadata_dir=str(tmp_path), tokenizer=None, original_model_path=None, v4_compatible=False)
    assert json.loads((tmp_path / "config.json").read_text()) == {"mtp_config": _MTP_CONFIG}


# --------------------------------------------------------------------------- contract


@pytest.mark.parametrize(
    "override, match",
    [
        ({"num_draft_layers": 2}, "num_draft_layers=3"),
        ({"target_layer_ids": [2, 3, 4]}, r"target_layer_ids=\(1, 2, 3\)"),
        ({"block_size": 4}, "block_size=3"),
        ({"markov_rank": 8}, "markov_rank=6"),
        ({"markov_head_type": "rnn"}, "markov_head_type=rnn_hidden"),
        ({"confidence_head_alpha": 1.0}, "no confidence head"),
        ({"num_anchors": 0}, "num_anchors"),
    ],
)
def test_options_must_match_the_released_draft(override, match):
    with pytest.raises(ValueError, match=match):
        Inkling2DSparkConfig.from_options(_text_config(), _MTP_CONFIG, _options(**override))


@pytest.mark.parametrize(
    "override, match",
    [
        ({"markov_head_type": "rnn"}, "rnn_hidden"),
        ({"q_bias": True}, "attention biases"),
        ({"local_layer_ids": [0, 3]}, "local_layer_ids"),
        ({"block_size": 1}, "block_size"),
        ({"target_layer_ids": [0, 1, 2]}, r"\[1, 4\]"),
    ],
)
def test_mtp_config_outside_the_released_design_is_rejected(override, match):
    with pytest.raises(ValueError, match=match):
        Inkling2DSparkConfig.from_options(_text_config(), _MTP_CONFIG | override, _options())


def test_missing_mtp_field_is_named():
    mtp = dict(_MTP_CONFIG)
    del mtp["markov_rank"]
    with pytest.raises(ValueError, match="markov_rank"):
        Inkling2DSparkReleasedConfig.from_mtp_config(mtp)


def test_target_hidden_size_must_match():
    text = _text_config()
    text.hidden_size = 64
    with pytest.raises(ValueError, match="hidden_size"):
        Inkling2DSparkConfig.from_options(text, _MTP_CONFIG, _options())


def test_shared_weights_must_stay_frozen_and_carry_the_embedding_norm():
    model = Inkling2DSparkConfig.from_options(_text_config(), _MTP_CONFIG, _options()).build()
    head = torch.nn.Linear(32, _VOCAB, bias=False)
    with pytest.raises(ValueError, match="freeze_embeddings"):
        model.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(_VOCAB, 32), lm_head=head, freeze=False)
    with pytest.raises(ValueError, match="embed_norm"):
        model.initialize_embeddings_and_head(embed_tokens=torch.nn.Embedding(_VOCAB, 32), lm_head=head)
