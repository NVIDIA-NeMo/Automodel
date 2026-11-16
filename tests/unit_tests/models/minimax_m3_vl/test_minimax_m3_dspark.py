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

"""CPU forward test for the MiniMax M3 DSpark draft + its config builder.

Builds a tiny MiniMax M3 draft from a MiniMax M3 VL target config and runs one
forward, covering ``MiniMaxM3VLConfig.build_dspark_draft`` and ``MiniMaxM3DSparkModel``,
plus the contract and export a target's ``dspark/config.json`` selects.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig

from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING  # registers the local configs with AutoConfig
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.minimax_m3_vl import dspark as minimax_m3_dspark
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.dspark import MiniMaxM3DSparkModel
from nemo_automodel.components.speculative.dspark.registry import ModelOwnedDSparkProvider

VOCAB = 256
HIDDEN = 64
HEAD_DIM = 16
TARGET_LAYER_IDS = [1, 3]
RELEASED_FFN = 3 * HIDDEN
RELEASED_WINDOW = 12
RELEASED_LAYERS = 2
RELEASED_BLOCK = 4
RELEASED_MASK = 5
RELEASED_MARKOV = 16
# Per-tensor names copied from MiniMax-M3.1-Flash ``dspark/model.safetensors.index.json``,
# without the ``weight_scale_inv`` entries of its MXFP8 weights.
RELEASED_DRAFT_KEYS = {
    "confidence_head.proj.bias",
    "confidence_head.proj.weight",
    "fc.weight",
    "final_norm.weight",
    "hidden_norm.weight",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
}
RELEASED_LAYER_KEYS = {
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}
RELEASED_KEYS = {f"language_model.model.dspark.{key}" for key in RELEASED_DRAFT_KEYS} | {
    f"language_model.model.dspark.layers.{layer}.decoder_layer.{key}"
    for layer in range(RELEASED_LAYERS)
    for key in RELEASED_LAYER_KEYS
}


def _options(**overrides):
    options = {
        "num_draft_layers": RELEASED_LAYERS,
        "target_layer_ids": TARGET_LAYER_IDS,
        "block_size": RELEASED_BLOCK,
        "mask_token_id": RELEASED_MASK,
        "num_anchors": 8,
        "markov_rank": RELEASED_MARKOV,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
    }
    return SimpleNamespace(**{**options, **overrides})


def _target_config(name_or_path: str = "") -> MiniMaxM3VLConfig:
    text_config = MiniMaxM3VLTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        dense_intermediate_size=2 * HIDDEN,
        shared_intermediate_size=2 * HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        rotary_dim=HEAD_DIM // 2,
        partial_rotary_factor=0.5,
        num_mtp_modules=0,
    )
    config = MiniMaxM3VLConfig(text_config=text_config)
    config._name_or_path = name_or_path
    return config


def _write_released_target(root: Path, **overrides) -> str:
    """Write the part of MiniMax-M3.1-Flash ``dspark/config.json`` the draft reads."""
    target_text_config = _target_config().text_config
    text_config = {
        **{
            name: getattr(target_text_config, name)
            for name in (
                "vocab_size",
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "rms_norm_eps",
                "use_gemma_norm",
                "rope_theta",
                "rotary_dim",
                "partial_rotary_factor",
                "swiglu_alpha",
                "swiglu_limit",
            )
        },
        "dspark_block_size": RELEASED_BLOCK,
        "dspark_markov_rank": RELEASED_MARKOV,
        "dspark_markov_head_type": "vanilla",
        "dspark_noise_token_id": RELEASED_MASK,
        "dspark_use_dense_ffn": True,
        "dspark_ffn_hidden_size": RELEASED_FFN,
        "dspark_target_layer_ids": TARGET_LAYER_IDS,
        "layer_types": ["sliding_attention"] * RELEASED_LAYERS,
        "sliding_window": RELEASED_WINDOW,
        "quantization_config": {"quant_method": "mxfp8"},
        **overrides,
    }
    released = {
        "architectures": ["DSparkMiniMaxDraftModel"],
        "model_type": "minimax_m3_vl",
        "text_config": text_config,
        "quantization_config": {"quant_method": "mxfp8"},
    }
    (root / "dspark").mkdir()
    (root / "dspark" / "config.json").write_text(json.dumps(released))
    return str(root)


def _finish_draft(model: MiniMaxM3DSparkModel, attn_implementation: str = "sdpa") -> MiniMaxM3DSparkModel:
    model.config._attn_implementation = attn_implementation
    model = model.to(dtype=torch.float32).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


def _build_minimax_m3_draft():
    target_config = _target_config()
    assert isinstance(target_config, ModelOwnedDSparkProvider)
    model = target_config.build_dspark_draft(_options())
    assert model.config.architectures == ["MiniMaxM3DSparkModel"]
    assert model.config.num_hidden_layers == 2
    assert model.config.num_mtp_modules == 0
    return _finish_draft(model)


def _build_released_draft(root: Path, attn_implementation: str = "sdpa") -> MiniMaxM3DSparkModel:
    target_config = _target_config(_write_released_target(root))
    return _finish_draft(target_config.build_dspark_draft(_options()), attn_implementation)


def test_minimax_m3_draft_forward_shapes():
    model = _build_minimax_m3_draft()
    b, s, block, anchors = 2, 16, 4, 8
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        torch.manual_seed(7)
        out = model(
            input_ids=torch.randint(0, VOCAB, (b, s), generator=gen),
            target_hidden_states=torch.randn(b, s, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen),
            loss_mask=torch.ones(b, s, dtype=torch.uint8),
            target_last_hidden_states=torch.randn(b, s, HIDDEN, generator=gen),
        )
    assert out.draft_logits.shape == (b, anchors, block, VOCAB)
    assert out.confidence_pred.shape == (b, anchors, block)
    assert torch.isfinite(out.draft_logits).all()


def test_minimax_m3_rotary_device_resynced_after_to():
    """``RotaryEmbedding`` caches its frequencies via ``functools.cache`` keyed on
    the module instance, not a persistent buffer, so plain ``nn.Module._apply``
    (which only visits parameters/buffers) never touches it -- a later
    ``.to(device=...)`` would otherwise silently leave the cached frequencies on
    whatever device was current at construction. Simulate a stale recorded
    device (as if the cache had been populated on the wrong device) and confirm
    ``.to()`` resyncs it to match the model's actual parameter device.
    """
    model = _build_minimax_m3_draft()
    model.rotary_emb.device = torch.device("meta")
    model.rotary_emb._compute_concentration_and_inv_freq.cache_clear()

    model = model.to(dtype=torch.float32)

    assert model.rotary_emb.device == next(model.parameters()).device


def test_minimax_m3_draft_reuses_dense_intermediate_size_field():
    """The draft's MLP width should come from the routed-expert-scale
    ``intermediate_size`` (kept narrow), not the target-only wide
    ``dense_intermediate_size``."""
    model = _build_minimax_m3_draft()
    assert model.layers[0].mlp.gate_proj.out_features == 2 * HIDDEN


def test_minimax_m3_flex_attention_compiles_with_dynamic_shapes(monkeypatch):
    import importlib

    captured = {}

    def compile_fn(fn, **kwargs):
        captured["fn"] = fn
        captured["kwargs"] = kwargs
        return fn

    monkeypatch.setattr(torch, "compile", compile_fn)
    try:
        importlib.reload(minimax_m3_dspark)
        assert captured["fn"] is minimax_m3_dspark.flex_attention
        assert captured["kwargs"] == {"mode": "max-autotune-no-cudagraphs", "dynamic": True}
    finally:
        monkeypatch.undo()
        importlib.reload(minimax_m3_dspark)


def test_recipe_config_path_resolves_the_model_owned_provider(tmp_path):
    """``AutoConfig`` must return the local config, not the Transformers built-in of the same name."""
    assert "MiniMaxM3SparseForConditionalGeneration" in MODEL_ARCH_MAPPING
    target_dir = Path(_write_released_target(tmp_path))
    text_config = _target_config().text_config.to_dict()
    (target_dir / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl", "text_config": text_config}))
    target_config = AutoConfig.from_pretrained(str(target_dir))
    assert isinstance(target_config, ModelOwnedDSparkProvider)
    model = target_config.build_dspark_draft(_options())
    assert model.layers[0].mlp.gate_proj.out_features == RELEASED_FFN


def test_target_without_released_config_keeps_native_draft(tmp_path):
    target_config = _target_config(str(tmp_path))
    model = target_config.build_dspark_draft(_options())
    assert model.layers[0].mlp.gate_proj.out_features == target_config.text_config.intermediate_size
    assert not hasattr(model, "state_dict_adapter")
    assert model._get_consolidated_hf_metadata_exporter(tokenizer=None, original_model_path=None) is None


def test_released_config_sets_ffn_width(tmp_path):
    model = _build_released_draft(tmp_path)
    assert model.layers[0].mlp.gate_proj.out_features == RELEASED_FFN
    assert model.layers[0].mlp.down_proj.in_features == RELEASED_FFN


@pytest.mark.parametrize(
    "override",
    [
        {"num_draft_layers": 1},
        {"target_layer_ids": [0, 3]},
        {"block_size": 3},
        {"mask_token_id": 6},
        {"markov_rank": 8},
        {"markov_head_type": "gated"},
        {"confidence_head_alpha": 0.0},
        {"confidence_head_with_markov": False},
    ],
)
def test_released_config_rejects_recipe_mismatch(tmp_path, override):
    target_config = _target_config(_write_released_target(tmp_path))
    field = next(iter(override))
    with pytest.raises(ValueError, match=field):
        target_config.build_dspark_draft(_options(**override))


@pytest.mark.parametrize(
    "override",
    [{"dspark_use_dense_ffn": False}, {"layer_types": ["sliding_attention", "full_attention"]}],
)
def test_released_config_rejects_other_draft_shapes(tmp_path, override):
    target_config = _target_config(_write_released_target(tmp_path, **override))
    with pytest.raises(ValueError, match="dense, sliding-window"):
        target_config.build_dspark_draft(_options())


@pytest.mark.parametrize(
    "override",
    [
        {"vocab_size": VOCAB + 7},
        {"hidden_size": 2 * HIDDEN},
        {"head_dim": 2 * HEAD_DIM},
        {"rms_norm_eps": 1e-3},
        {"rope_theta": 1.0},
        {"rotary_dim": HEAD_DIM},
        {"swiglu_limit": 1.0},
    ],
)
def test_released_config_rejects_a_different_backbone(tmp_path, override):
    target_config = _target_config(_write_released_target(tmp_path, **override))
    with pytest.raises(ValueError, match=next(iter(override))):
        target_config.build_dspark_draft(_options())


def test_released_config_names_a_missing_field(tmp_path):
    target_dir = Path(_write_released_target(tmp_path))
    released_path = target_dir / "dspark" / "config.json"
    released = json.loads(released_path.read_text())
    del released["text_config"]["partial_rotary_factor"]
    released_path.write_text(json.dumps(released))
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        _target_config(str(target_dir)).build_dspark_draft(_options())


def test_missing_released_config_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _target_config(str(tmp_path)).build_dspark_draft(_options())
    assert "has no dspark/config.json" in caplog.text


def test_released_layout_requires_frozen_shared_weights(tmp_path):
    target_config = _target_config(_write_released_target(tmp_path))
    model = target_config.build_dspark_draft(_options())
    with pytest.raises(ValueError, match="freeze_embeddings"):
        model.initialize_embeddings_and_head(
            embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
            lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
            freeze=False,
        )


@pytest.mark.parametrize("attn_implementation", ["sdpa", "flex_attention"])
def test_released_window_hides_context_outside_it(tmp_path, monkeypatch, attn_implementation):
    """A block sees context at distance < window from each query, and nothing further back."""
    model = _build_released_draft(tmp_path, attn_implementation)
    sequence, anchor = 32, 24
    anchors = torch.tensor([[anchor]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    monkeypatch.setattr(minimax_m3_dspark, "sample_anchor_positions", lambda **_: (anchors, keep))

    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(6, VOCAB, (1, sequence), generator=gen)
    features = torch.randn(1, sequence, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen)

    def block_logits(perturbed_position: int | None = None) -> torch.Tensor:
        context = features.clone()
        if perturbed_position is not None:
            context[:, perturbed_position] += 1.0
        with torch.no_grad():
            return model(
                input_ids=input_ids,
                target_hidden_states=context,
                loss_mask=torch.ones(1, sequence, dtype=torch.uint8),
            ).draft_logits

    baseline = block_logits()
    # The block's first query sits at the anchor and reaches back window - 1 positions.
    oldest_visible = anchor - RELEASED_WINDOW + 1
    torch.testing.assert_close(block_logits(oldest_visible - 1), baseline, rtol=0, atol=0)
    assert not torch.equal(block_logits(oldest_visible), baseline)


def test_released_export_names_match_the_release(tmp_path):
    model = _build_released_draft(tmp_path)
    adapter = model.state_dict_adapter
    exported = adapter.to_hf(model.state_dict())
    assert set(exported) == RELEASED_KEYS
    assert set(adapter.get_hf_state_dict_keys(model.state_dict())) == RELEASED_KEYS

    native = adapter.from_hf(dict(exported))
    assert set(native) == set(model.state_dict()) - {"embed_tokens.weight", "lm_head.weight"}
    for key, value in native.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_adapter_rejects_keys_outside_the_draft(tmp_path):
    model = _build_released_draft(tmp_path)
    with pytest.raises(ValueError, match="language_model.model.layers.0.mlp.weight"):
        model.state_dict_adapter.from_hf({"language_model.model.layers.0.mlp.weight": torch.zeros(1)})


def test_released_checkpoint_exports_and_resumes(tmp_path):
    model = _build_released_draft(tmp_path)
    checkpointer = Checkpointer(
        CheckpointingConfig(
            checkpoint_dir=str(tmp_path),
            model_save_format="safetensors",
            save_consolidated=True,
            model_cache_dir=str(tmp_path / "cache"),
            model_repo_id="test/minimax-m3-dspark",
            model_state_dict_keys=list(model.state_dict()),
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    try:
        checkpointer.save_model(model, str(tmp_path / "trained"))
        checkpointer.async_wait()

        consolidated = tmp_path / "trained" / "model" / "consolidated"
        config = json.loads((consolidated / "config.json").read_text())
        assert config["architectures"] == ["DSparkMiniMaxDraftModel"]
        assert "quantization_config" not in config
        assert "quantization_config" not in config["text_config"]
        index = json.loads((consolidated / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == RELEASED_KEYS
        for key, filename in index["weight_map"].items():
            with safe_open(consolidated / filename, "pt") as f:
                native_key = model.state_dict_adapter.from_hf({key: None}).popitem()[0]
                torch.testing.assert_close(f.get_tensor(key), expected[native_key], rtol=0, atol=0)

        for key, parameter in model.named_parameters():
            if key not in ("embed_tokens.weight", "lm_head.weight"):
                parameter.detach().zero_()
        checkpointer.load_model(model, str(tmp_path / "trained" / "model"))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    finally:
        checkpointer.close()
