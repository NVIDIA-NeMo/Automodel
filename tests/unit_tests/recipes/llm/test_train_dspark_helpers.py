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

"""CPU unit tests for the DSpark recipe's V4-Flash helper knobs.

Covers the recipe-level glue added for the DeepSeek-V4-Flash target:
- ``_apply_draft_activation_checkpointing``: the distributed AC setting wraps
  the trainable draft's attention, MLP, and norm modules before FSDP.
- ``_apply_target_chat_template``: a target whose tokenizer ships no chat
  template (V4-Flash) must take one from ``recipe_args.chat_template`` or fail
  fast, and an explicit template overrides whatever the tokenizer carries.
- ``_resolve_reduced_target_layers``: the ``target_num_hidden_layers``
  diagnostic override is range-checked.
- ``_resolve_dspark_optimizer_spec`` / ``_build_dspark_optimizer``: the
  ``optimizer:`` config is normalized into a ``build_optimizer`` spec and built,
  honoring an explicit ``_target_`` (e.g. TE FusedAdam with
  ``master_weights``/``exp_avg_dtype``/...) instead of always hardcoding plain
  ``torch.optim.AdamW``.
- ``_resolve_warmup_steps``: the ratio-derived warmup length is floored for
  short / small-dataset runs, unless the caller opts out with ``warmup_ratio<=0``.
- ``_resolve_wandb_kwargs`` / ``_init_dspark_wandb``: the examples'
  documentation-only ``enable`` flag is stripped before forwarding to
  ``wandb.init`` and gates whether to log at all; ``_init_dspark_wandb`` also
  gates on rank (``is_main``) and block presence.

(target_layer_ids range/-1/ordering validation is covered by the shared
``common.validate_target_layer_ids``, which HFDSparkTargetModel already calls.)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm.dspark_cache import write_manifest, write_shard, write_target_weights
from nemo_automodel.recipes.llm._dspark_target_build import (
    build_deepseek_v4_backend,
    gather_full_weight_module,
    repair_glm_5_2_qk_rope_head_dim,
    resolve_reduced_target_layers,
)
from nemo_automodel.recipes.llm.train_dspark import (
    TrainDSparkRecipe,
    _apply_draft_activation_checkpointing,
    _apply_target_chat_template,
    _build_dspark_optimizer,
    _extract_mm_kwargs,
    _init_dspark_wandb,
    _resolve_dspark_optimizer_spec,
    _resolve_wandb_kwargs,
    _resolve_warmup_steps,
    _validate_cached_dspark_manifest,
)

JINJA = (
    "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'assistant' %}"
    "{% generation %}{{ m['content'] }}{% endgeneration %}{% endif %}{% endfor %}"
)


def _tok(chat_template=None):
    """A minimal tokenizer stub: ``_has_chat_template`` needs a ``chat_template``
    attribute plus a callable ``apply_chat_template``."""
    return SimpleNamespace(chat_template=chat_template, apply_chat_template=lambda *a, **k: None)


class _DraftLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Linear(4, 4)
        self.input_layernorm = torch.nn.LayerNorm(4)
        self.post_attention_layernorm = torch.nn.LayerNorm(4)


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DraftLayer(), _DraftLayer()])


def test_draft_activation_checkpointing_wraps_trainable_submodules():
    draft = _Draft()
    _apply_draft_activation_checkpointing(draft, True)
    for layer in draft.layers:
        assert hasattr(layer.self_attn, "_checkpoint_wrapped_module")
        assert hasattr(layer.mlp, "_checkpoint_wrapped_module")
        assert hasattr(layer.input_layernorm, "_checkpoint_wrapped_module")
        assert hasattr(layer.post_attention_layernorm, "_checkpoint_wrapped_module")


def test_draft_activation_checkpointing_false_is_noop():
    draft = _Draft()
    original_attention = draft.layers[0].self_attn
    _apply_draft_activation_checkpointing(draft, False)
    assert draft.layers[0].self_attn is original_attention


# ---------------------------------------------------------------------------
# _apply_target_chat_template
# ---------------------------------------------------------------------------


def test_chat_template_set_when_provided_on_templateless_tokenizer():
    tok = _tok(chat_template=None)
    _apply_target_chat_template(tok, JINJA)
    assert tok.chat_template == JINJA


def test_chat_template_override_replaces_existing():
    tok = _tok(chat_template="OLD")
    _apply_target_chat_template(tok, JINJA)
    assert tok.chat_template == JINJA


def test_chat_template_none_with_existing_template_is_noop():
    tok = _tok(chat_template="EXISTING")
    _apply_target_chat_template(tok, None)
    assert tok.chat_template == "EXISTING"


def test_chat_template_none_without_template_raises():
    tok = _tok(chat_template=None)
    with pytest.raises(ValueError, match="no chat template"):
        _apply_target_chat_template(tok, None)


def test_chat_template_non_string_is_coerced(tmp_path):
    # A path-like value is stringified; _resolve_chat_template loads file contents.
    f = tmp_path / "tmpl.jinja"
    f.write_text(JINJA, encoding="utf-8")
    tok = _tok(chat_template=None)
    _apply_target_chat_template(tok, f)  # PosixPath, not str
    assert tok.chat_template == JINJA


# ---------------------------------------------------------------------------
# _resolve_reduced_target_layers
# ---------------------------------------------------------------------------


def test_reduced_layers_none_passes_through():
    assert resolve_reduced_target_layers(43, None) is None


def test_reduced_layers_valid():
    assert resolve_reduced_target_layers(43, 4) == 4


def test_reduced_layers_string_coerced():
    assert resolve_reduced_target_layers(43, "4") == 4


def test_reduced_layers_full_depth_allowed():
    assert resolve_reduced_target_layers(43, 43) == 43


@pytest.mark.parametrize("bad", [0, -1, 44, 100])
def test_reduced_layers_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="target_num_hidden_layers"):
        resolve_reduced_target_layers(43, bad)


# ---------------------------------------------------------------------------
# _resolve_dspark_optimizer_spec
# ---------------------------------------------------------------------------


def _opt_cfg(**fields):
    """A minimal ``optimizer:`` config-node stub: dict-like ``to_dict``/``get``."""
    return SimpleNamespace(to_dict=lambda: dict(fields), get=lambda k, default=None: fields.get(k, default))


def test_optimizer_spec_defaults_to_adamw_when_no_target():
    target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(lr=6e-4, warmup_ratio=0.04, min_lr_ratio=0.1))
    assert target == "torch.optim.AdamW"
    assert kwargs["lr"] == 6e-4
    assert kwargs["betas"] == (0.9, 0.95)
    assert kwargs["weight_decay"] == 0.0
    assert "warmup_ratio" not in kwargs
    assert "min_lr_ratio" not in kwargs


def test_optimizer_spec_respects_explicit_target_and_extra_kwargs():
    target, kwargs = _resolve_dspark_optimizer_spec(
        _opt_cfg(
            _target_="transformer_engine.pytorch.optimizers.FusedAdam",
            lr=1e-5,
            master_weights=True,
            master_weight_dtype="float32",
            exp_avg_dtype="float32",
            exp_avg_sq_dtype="float32",
            store_param_remainders=True,
        )
    )
    assert target == "transformer_engine.pytorch.optimizers.FusedAdam"
    assert kwargs["lr"] == 1e-5
    assert kwargs["master_weights"] is True
    assert kwargs["master_weight_dtype"] == "float32"
    assert kwargs["exp_avg_dtype"] == "float32"
    assert kwargs["exp_avg_sq_dtype"] == "float32"
    assert kwargs["store_param_remainders"] is True


def test_optimizer_spec_preserves_explicit_betas_and_weight_decay():
    _target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01))
    assert kwargs["betas"] == (0.9, 0.999)
    assert kwargs["weight_decay"] == 0.01


def test_optimizer_spec_coerces_lr_to_float():
    _target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(lr="6e-4"))
    assert kwargs["lr"] == pytest.approx(6e-4)
    assert isinstance(kwargs["lr"], float)


def test_optimizer_spec_keeps_real_config_node_target_as_string():
    cfg = ConfigNode({"_target_": "torch.optim.AdamW", "lr": 1e-5})
    target, _kwargs = _resolve_dspark_optimizer_spec(cfg)
    assert target == "torch.optim.AdamW"


def test_optimizer_spec_does_not_force_betas_onto_explicit_target():
    # An explicit _target_ for an optimizer with no `betas` kwarg (e.g. SGD)
    # must not have AdamW's betas/weight_decay defaults forced onto it.
    target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(_target_="torch.optim.SGD", lr=0.1, momentum=0.9))
    assert target == "torch.optim.SGD"
    assert "betas" not in kwargs
    assert "weight_decay" not in kwargs
    assert kwargs["momentum"] == 0.9


# ---------------------------------------------------------------------------
# _build_dspark_optimizer
# ---------------------------------------------------------------------------


def test_build_optimizer_defaults_to_adamw():
    model = torch.nn.Linear(4, 4)
    optimizer = _build_dspark_optimizer(model, _opt_cfg(lr=6e-4))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(6e-4)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)


def test_build_optimizer_respects_explicit_target():
    model = torch.nn.Linear(4, 4)
    optimizer = _build_dspark_optimizer(model, _opt_cfg(_target_="torch.optim.SGD", lr=0.1, momentum=0.9))
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["momentum"] == 0.9


def test_build_optimizer_only_covers_trainable_params():
    model = torch.nn.Linear(4, 4)
    model.bias.requires_grad_(False)
    optimizer = _build_dspark_optimizer(model, _opt_cfg(lr=6e-4))
    (params,) = (group["params"] for group in optimizer.param_groups)
    assert params == [model.weight]


# ---------------------------------------------------------------------------
# _resolve_warmup_steps
# ---------------------------------------------------------------------------


def test_warmup_steps_floors_short_runs():
    # 4% of 100 steps is 4 -- far too little warmup for a freshly-initialized
    # draft; the floor should kick in.
    assert _resolve_warmup_steps(0.04, 100) == 20


def test_warmup_steps_ratio_dominates_for_long_runs():
    # 4% of 10,000 steps is 400, well above the floor -- the ratio wins.
    assert _resolve_warmup_steps(0.04, 10_000) == 400


def test_warmup_steps_zero_ratio_is_explicit_opt_out():
    # The smoke config sets warmup_ratio=0.0 on purpose ("see movement
    # immediately"); the floor must not override that opt-out.
    assert _resolve_warmup_steps(0.0, 100) == 1


def test_warmup_steps_negative_ratio_treated_as_opt_out():
    assert _resolve_warmup_steps(-1.0, 100) == 1


def test_warmup_steps_custom_floor():
    assert _resolve_warmup_steps(0.01, 100, min_warmup_steps=5) == 5
    assert _resolve_warmup_steps(0.5, 100, min_warmup_steps=5) == 50


# ---------------------------------------------------------------------------
# _resolve_wandb_kwargs
# ---------------------------------------------------------------------------


def test_wandb_kwargs_disabled_when_enable_false():
    assert _resolve_wandb_kwargs({"enable": False, "project": "p"}) is None


def test_wandb_kwargs_enabled_strips_enable_key():
    kwargs = _resolve_wandb_kwargs({"enable": True, "project": "p", "group": "g"})
    assert kwargs == {"project": "p", "group": "g"}


def test_wandb_kwargs_defaults_enabled_when_flag_absent():
    kwargs = _resolve_wandb_kwargs({"project": "p"})
    assert kwargs == {"project": "p"}


# ---------------------------------------------------------------------------
# _init_dspark_wandb
# ---------------------------------------------------------------------------


def _patch_wandb_run(monkeypatch, run=object()):
    """Patch the module-level wandb hooks ``_init_dspark_wandb`` calls, returning a spy call log."""
    import nemo_automodel.recipes.llm.train_dspark as train_dspark_module

    calls = {}

    def _fake_init_wandb_run(wandb_kwargs, cfg_dict, default_name):
        calls["wandb_kwargs"] = wandb_kwargs
        calls["cfg_dict"] = cfg_dict
        calls["default_name"] = default_name
        return run

    def _fake_suppress():
        calls["suppressed"] = True

    monkeypatch.setattr(train_dspark_module, "init_wandb_run", _fake_init_wandb_run)
    monkeypatch.setattr(train_dspark_module, "suppress_wandb_log_messages", _fake_suppress)
    return calls


def test_init_wandb_skipped_on_non_main_rank(monkeypatch):
    calls = _patch_wandb_run(monkeypatch)
    result = _init_dspark_wandb(is_main=False, wandb_cfg=_opt_cfg(project="p"), cfg_dict={}, default_name="run")
    assert result is None
    assert calls == {}


def test_init_wandb_skipped_when_block_absent(monkeypatch):
    calls = _patch_wandb_run(monkeypatch)
    result = _init_dspark_wandb(is_main=True, wandb_cfg=None, cfg_dict={}, default_name="run")
    assert result is None
    assert calls == {}


def test_init_wandb_skipped_when_disabled(monkeypatch):
    calls = _patch_wandb_run(monkeypatch)
    result = _init_dspark_wandb(
        is_main=True, wandb_cfg=_opt_cfg(enable=False, project="p"), cfg_dict={}, default_name="run"
    )
    assert result is None
    assert calls == {}


def test_init_wandb_runs_on_main_when_enabled(monkeypatch):
    sentinel_run = object()
    calls = _patch_wandb_run(monkeypatch, run=sentinel_run)
    result = _init_dspark_wandb(
        is_main=True,
        wandb_cfg=_opt_cfg(project="p", group="g"),
        cfg_dict={"lr": 1e-4},
        default_name="dspark_run",
    )
    assert result is sentinel_run
    assert calls["suppressed"] is True
    assert calls["wandb_kwargs"] == {"project": "p", "group": "g"}
    assert calls["cfg_dict"] == {"lr": 1e-4}
    assert calls["default_name"] == "dspark_run"


# ---------------------------------------------------------------------------
# _extract_mm_kwargs (multimodal MiniMax M3 DSpark)
# ---------------------------------------------------------------------------


def test_extract_mm_kwargs_empty_for_text_only_batch():
    batch = {"input_ids": torch.zeros(1), "attention_mask": torch.ones(1), "loss_mask": torch.ones(1)}
    assert _extract_mm_kwargs(batch) == {}


def test_extract_mm_kwargs_passes_through_present_media_keys():
    pixel_values = torch.randn(2, 3, 4, 4)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    batch = {
        "input_ids": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    mm_kwargs = _extract_mm_kwargs(batch)
    assert mm_kwargs == {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def test_extract_mm_kwargs_ignores_unrelated_keys():
    batch = {"input_ids": torch.zeros(1), "seq_lens": torch.tensor([1, 2]), "doc_remaining": torch.tensor([0])}
    assert _extract_mm_kwargs(batch) == {}


# ---------------------------------------------------------------------------
# GLM-5.2 target config repair + reduced-config forwarding
# ---------------------------------------------------------------------------


def test_optimizer_spec_real_config_node_without_target_defaults_to_adamw():
    """Regression: ConfigNode.get_as_string raises KeyError for an absent ``_target_``
    even with a ``None`` default, which crashed every optimizer block omitting it."""
    cfg = ConfigNode({"lr": 6e-4, "betas": [0.9, 0.95], "weight_decay": 0.0, "warmup_ratio": 0.04})
    target, kwargs = _resolve_dspark_optimizer_spec(cfg)
    assert target == "torch.optim.AdamW"
    assert kwargs["lr"] == 6e-4
    assert "warmup_ratio" not in kwargs


def test_repair_glm_qk_rope_restores_clobbered_value():
    cfg = SimpleNamespace(qk_rope_head_dim=192)
    repair_glm_5_2_qk_rope_head_dim(cfg, {"qk_rope_head_dim": 64, "head_dim": 192})
    assert cfg.qk_rope_head_dim == 64


def test_repair_glm_qk_rope_noop_when_already_matching():
    cfg = SimpleNamespace(qk_rope_head_dim=64)
    repair_glm_5_2_qk_rope_head_dim(cfg, {"qk_rope_head_dim": 64})
    assert cfg.qk_rope_head_dim == 64


def test_repair_glm_qk_rope_noop_when_raw_config_omits_field():
    cfg = SimpleNamespace(qk_rope_head_dim=192)
    repair_glm_5_2_qk_rope_head_dim(cfg, {"head_dim": 192})
    assert cfg.qk_rope_head_dim == 192


_TINY_GLM_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    # head_dim (the attention-kernel head dim) alongside the true qk_rope_head_dim, as
    # the published GLM-5.2 config ships them; the HF attribute_map (head_dim ->
    # qk_rope_head_dim) lets the former clobber the latter on load.
    "head_dim": 24,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 16,
    "qk_head_dim": 24,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "v_head_dim": 24,
    "hidden_size": 64,
    "intermediate_size": 48,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "index_head_dim": 16,
    "index_n_heads": 2,
    "index_topk": 8,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "vocab_size": 128,
}


def test_build_glm_5_2_target_forwards_reduced_repaired_config(tmp_path, monkeypatch):
    """Regression: ``from_pretrained`` re-read the checkpoint's own config, silently
    rebuilding the full-depth target and discarding ``target_num_hidden_layers`` (OOM
    on one node). The GLM target build must hand the reduced, repaired config to
    ``from_config`` with ``load_base_model=True``."""
    import json

    import nemo_automodel.recipes.llm._dspark_target_build as tb

    (tmp_path / "config.json").write_text(json.dumps(_TINY_GLM_CONFIG))

    captured = {}

    def _fake_from_config(config=None, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return "target-model"

    monkeypatch.setattr(tb, "NeMoAutoModelForCausalLM", SimpleNamespace(from_config=_fake_from_config))
    monkeypatch.setattr(tb, "create_distributed_setup_from_config", lambda cfg, world_size: "distributed-setup")

    recipe_cfg = _opt_cfg(target_num_hidden_layers=2)
    target_config, target_model, distributed_setup = tb.build_glm_5_2_target(
        cfg=SimpleNamespace(),
        world_size=8,
        device=SimpleNamespace(type="cuda"),
        compute_dtype=torch.bfloat16,
        target_path=str(tmp_path),
        recipe_cfg=recipe_cfg,
        trust_remote_code=False,
    )

    assert target_model == "target-model"
    assert distributed_setup == "distributed-setup"
    assert captured["config"] is target_config
    # The reduction survives (from_pretrained would have re-read the 8-layer config).
    assert target_config.num_hidden_layers == 2
    # The attribute-map clobber is repaired back to the raw checkpoint value.
    assert target_config.qk_rope_head_dim == 8
    assert captured["load_base_model"] is True
    assert captured["distributed_setup"] == "distributed-setup"
    assert captured["torch_dtype"] == torch.bfloat16


def test_build_glm_5_2_target_requires_cuda(tmp_path):
    from nemo_automodel.recipes.llm._dspark_target_build import build_glm_5_2_target

    with pytest.raises(RuntimeError, match="requires CUDA"):
        build_glm_5_2_target(
            cfg=SimpleNamespace(),
            world_size=1,
            device=SimpleNamespace(type="cpu"),
            compute_dtype=torch.float32,
            target_path=str(tmp_path),
            recipe_cfg=_opt_cfg(),
            trust_remote_code=False,
        )


def test_build_deepseek_v4_target_forwards_reduced_config(monkeypatch):
    """The V4 build must hand the (reduced) config to ``from_config`` with the sharded
    distributed_setup and ``load_base_model=True`` (the full 43-layer target OOMs on
    one node, so ``target_num_hidden_layers`` must survive to ``from_config``)."""
    import nemo_automodel.recipes.llm._dspark_target_build as tb

    captured = {}

    def _fake_from_config(config=None, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return "target-model"

    monkeypatch.setattr(
        tb.DeepseekV4Config, "from_pretrained", staticmethod(lambda *a, **k: SimpleNamespace(num_hidden_layers=43))
    )
    monkeypatch.setattr(tb, "NeMoAutoModelForCausalLM", SimpleNamespace(from_config=_fake_from_config))
    monkeypatch.setattr(tb, "create_distributed_setup_from_config", lambda cfg, world_size: "distributed-setup")

    target_config, target_model, distributed_setup = tb.build_deepseek_v4_target(
        cfg=SimpleNamespace(),
        world_size=8,
        device=SimpleNamespace(type="cuda"),
        compute_dtype=torch.bfloat16,
        target_path="v4",
        recipe_cfg=_opt_cfg(target_num_hidden_layers=4),
        trust_remote_code=False,
    )

    assert target_model == "target-model"
    assert distributed_setup == "distributed-setup"
    assert target_config.num_hidden_layers == 4
    assert captured["config"] is target_config
    assert captured["load_base_model"] is True
    assert captured["distributed_setup"] == "distributed-setup"
    assert captured["torch_dtype"] == torch.bfloat16


def test_build_deepseek_v4_target_requires_cuda():
    from nemo_automodel.recipes.llm._dspark_target_build import build_deepseek_v4_target

    with pytest.raises(RuntimeError, match="requires CUDA"):
        build_deepseek_v4_target(
            cfg=SimpleNamespace(),
            world_size=1,
            device=SimpleNamespace(type="cpu"),
            compute_dtype=torch.float32,
            target_path="v4",
            recipe_cfg=_opt_cfg(),
            trust_remote_code=False,
        )


def test_build_deepseek_v4_backend_defaults():
    backend = build_deepseek_v4_backend(_opt_cfg())
    assert backend.attn == "tilelang"
    assert backend.experts == "torch_mm"
    assert backend.dispatcher == "hybridep"
    assert backend.enable_hf_state_dict_adapter is True


def test_gather_full_weight_module_passthrough_and_full_tensor():
    plain = torch.nn.Linear(2, 2)
    assert gather_full_weight_module(plain) is plain  # plain .weight -> unchanged

    gathered = torch.zeros(3)
    dtensor_like = SimpleNamespace(weight=SimpleNamespace(full_tensor=lambda: gathered))
    out = gather_full_weight_module(dtensor_like)
    assert out is not dtensor_like
    assert out.weight is gathered

    no_weight = SimpleNamespace(weight=None)
    assert gather_full_weight_module(no_weight) is no_weight


# ---------------------------------------------------------------------------
# _validate_cached_dspark_manifest
# ---------------------------------------------------------------------------


def _cached_manifest(**overrides):
    manifest = {
        "target_model": "tiny-qwen3",
        "target_model_type": "qwen3",
        "target_vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 6,
        "seq_length": 8,
        "dtype": "fp32",
        "target_hidden_dim": 96,
        "target_last_hidden_dim": 32,
        "target_layer_ids": [1, 3, 5],
    }
    manifest.update(overrides)
    return manifest


def _target_config(**overrides):
    fields = {"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 6}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _validate_cached_manifest(manifest=None, target_config=None, target_layer_ids=None, **kwargs):
    _validate_cached_dspark_manifest(
        "/cache",
        _cached_manifest() if manifest is None else manifest,
        _target_config() if target_config is None else target_config,
        [1, 3, 5] if target_layer_ids is None else target_layer_ids,
        target_model=kwargs.pop("target_model", "tiny-qwen3"),
        target_model_type=kwargs.pop("target_model_type", "qwen3"),
        seq_length=kwargs.pop("seq_length", 8),
        compute_dtype=kwargs.pop("compute_dtype", torch.float32),
    )


def test_cached_dspark_manifest_accepts_matching_shapes():
    _validate_cached_manifest()


def test_cached_dspark_manifest_warns_on_target_path_mismatch(caplog):
    caplog.set_level("WARNING")
    _validate_cached_manifest(
        manifest=_cached_manifest(target_model="/precompute/path/to/target"),
        target_model="/training/path/to/target",
    )
    assert "raw paths can differ across machines" in caplog.text


@pytest.mark.parametrize(
    "manifest,target_config,target_layer_ids,pattern",
    [
        (_cached_manifest(target_model_type="llama"), _target_config(), [1, 3, 5], "target_model_type"),
        (_cached_manifest(target_vocab_size=65), _target_config(), [1, 3, 5], "target_vocab_size"),
        (_cached_manifest(hidden_size=16), _target_config(), [1, 3, 5], "hidden_size"),
        (_cached_manifest(num_hidden_layers=7), _target_config(), [1, 3, 5], "num_hidden_layers"),
        (_cached_manifest(seq_length=16), _target_config(), [1, 3, 5], "seq_length"),
        (_cached_manifest(dtype="int4"), _target_config(), [1, 3, 5], "dtype"),
        (_cached_manifest(dtype="bf16"), _target_config(), [1, 3, 5], "CPU cached training"),
        (_cached_manifest(target_hidden_dim=64), _target_config(), [1, 3, 5], "target_hidden_dim"),
        (_cached_manifest(target_last_hidden_dim=16), _target_config(), [1, 3, 5], "target_last_hidden_dim"),
        (_cached_manifest(target_layer_ids=[1, 2, 3]), _target_config(), [1, 3, 5], "target_layer_ids"),
    ],
)
def test_cached_dspark_manifest_rejects_mismatch(manifest, target_config, target_layer_ids, pattern):
    with pytest.raises(ValueError, match=pattern):
        _validate_cached_manifest(manifest, target_config, target_layer_ids)


def test_cached_dspark_manifest_accepts_bf16_cache_on_cuda_dtype():
    _validate_cached_manifest(manifest=_cached_manifest(dtype="bf16"), compute_dtype=torch.bfloat16)


def test_recipe_cached_path_does_not_load_target_model(monkeypatch, tmp_path):
    """The recipe-level offline path must skip building the live target wrapper."""
    import nemo_automodel.recipes.llm.train_dspark as train_dspark_module

    vocab_size = 64
    hidden_size = 32
    target_layer_ids = [1, 3]
    cache_dir = str(tmp_path / "cache")
    embed = torch.nn.Embedding(vocab_size, hidden_size)
    head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    write_target_weights(cache_dir, embed, head, dtype=torch.float32)
    write_shard(
        cache_dir,
        0,
        {
            "input_ids": torch.randint(0, vocab_size, (1, 8), dtype=torch.long),
            "loss_mask": torch.ones(1, 8, dtype=torch.long),
            "target_hidden_states": torch.randn(1, 8, hidden_size * len(target_layer_ids)),
            "target_last_hidden_states": torch.randn(1, 8, hidden_size),
        },
    )
    write_manifest(
        cache_dir,
        {
            "target_model": "tiny-qwen3",
            "target_model_type": "qwen3",
            "target_vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": 4,
            "seq_length": 8,
            "dtype": "fp32",
            "num_samples": 1,
            "shard_size": 1,
            "target_hidden_dim": hidden_size * len(target_layer_ids),
            "target_last_hidden_dim": hidden_size,
            "target_layer_ids": target_layer_ids,
        },
    )

    class _CfgNode(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def to_dict(self):
            return dict(self)

    cfg = _CfgNode(
        recipe_args=_CfgNode(
            target_model_name_or_path="tiny-qwen3",
            cached_target_path=cache_dir,
            seq_length=8,
            micro_batch_size=1,
            mask_token_id=7,
            num_epochs=1,
            output_dir=str(tmp_path / "out"),
            target_layer_ids=target_layer_ids,
            draft_num_hidden_layers=1,
            num_anchors=4,
            block_size=2,
            markov_rank=8,
            attention_backend="flex_attention",
            trust_remote_code=False,
        ),
        optimizer=_CfgNode(lr=1e-4, warmup_ratio=0.0, min_lr_ratio=0.1),
        checkpoint=_CfgNode(enabled=False),
        raw_config={},
    )
    target_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    target_config.architectures = ["Qwen3ForCausalLM"]

    monkeypatch.setattr(
        train_dspark_module,
        "initialize_distributed",
        lambda **_kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True),
    )
    monkeypatch.setattr(train_dspark_module, "setup_logging", lambda: None)
    monkeypatch.setattr(train_dspark_module, "_read_target_model_type", lambda *_args, **_kwargs: "qwen3")
    monkeypatch.setattr(train_dspark_module.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: target_config)
    monkeypatch.setattr(
        train_dspark_module.NeMoAutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _tok(chat_template=None),
    )
    monkeypatch.setattr(
        train_dspark_module.NeMoAutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("cached_target_path must not load the target model"),
    )
    monkeypatch.setattr(
        train_dspark_module,
        "HFDSparkTargetModel",
        lambda *_args, **_kwargs: pytest.fail("cached_target_path must not build a live target wrapper"),
    )
    monkeypatch.setattr(TrainDSparkRecipe, "_build_checkpointer", lambda self, _target_path: None)
    monkeypatch.setattr(TrainDSparkRecipe, "load_checkpoint", lambda self, restore_from=None: None)

    recipe = TrainDSparkRecipe(cfg)
    recipe.setup()

    assert recipe.target_model is None
    assert recipe.target_wrapper is None
    assert len(recipe.train_dataloader.dataset) == 1
    torch.testing.assert_close(recipe.draft_model.embed_tokens.weight.detach().cpu(), embed.weight.detach())
    torch.testing.assert_close(recipe.draft_model.lm_head.weight.detach().cpu(), head.weight.detach())


# ---------------------------------------------------------------------------
# _should_shard_dense_target: opt-in gate for loading a frozen dense target
# FSDP2-sharded via the standard distributed setup.
# ---------------------------------------------------------------------------


def _make_shard_recipe(strategy="fsdp2", world_size=8):
    recipe = TrainDSparkRecipe({"distributed": {"strategy": strategy}})
    recipe.dist_env = SimpleNamespace(world_size=world_size, is_main=True)
    recipe.compute_dtype = torch.bfloat16
    return recipe


def test_should_shard_dense_target_off_by_default():
    # Existing configs (no shard_dense_target) keep the target replicated.
    recipe = _make_shard_recipe()
    assert recipe._should_shard_dense_target({}) is False


def test_should_shard_dense_target_true_on_fsdp2_multi_rank():
    recipe = _make_shard_recipe()
    assert recipe._should_shard_dense_target({"shard_dense_target": True}) is True


def test_should_shard_dense_target_default_strategy_is_fsdp2():
    # With no explicit distributed.strategy the default is fsdp2, so the flag takes effect.
    recipe = TrainDSparkRecipe({})
    recipe.dist_env = SimpleNamespace(world_size=8, is_main=True)
    recipe.compute_dtype = torch.bfloat16
    assert recipe._should_shard_dense_target({"shard_dense_target": True}) is True


def test_should_shard_dense_target_ignored_on_single_rank():
    recipe = _make_shard_recipe(world_size=1)
    assert recipe._should_shard_dense_target({"shard_dense_target": True}) is False


def test_should_shard_dense_target_ignored_on_ddp():
    recipe = _make_shard_recipe(strategy="ddp")
    assert recipe._should_shard_dense_target({"shard_dense_target": True}) is False


@pytest.mark.parametrize("axis", ["tp_size", "pp_size", "cp_size", "ep_size"])
def test_should_shard_dense_target_rejects_model_parallel_axes(axis):
    # Only a pure FSDP2 data-parallel topology is supported: pp_size>1 builds an
    # AutoPipeline the target wrapper cannot run, and tp/cp/ep are untested here.
    recipe = TrainDSparkRecipe({"distributed": {"strategy": "fsdp2", axis: 2}})
    recipe.dist_env = SimpleNamespace(world_size=8, is_main=True)
    recipe.compute_dtype = torch.bfloat16
    with pytest.raises(ValueError, match=axis):
        recipe._should_shard_dense_target({"shard_dense_target": True})


def test_should_shard_dense_target_allows_explicit_unit_or_null_axes():
    # Explicit 1s or YAML nulls on the model-parallel axes are the supported topology.
    recipe = TrainDSparkRecipe(
        {"distributed": {"strategy": "fsdp2", "tp_size": 1, "pp_size": None, "cp_size": 1, "ep_size": None}}
    )
    recipe.dist_env = SimpleNamespace(world_size=8, is_main=True)
    recipe.compute_dtype = torch.bfloat16
    assert recipe._should_shard_dense_target({"shard_dense_target": True}) is True
