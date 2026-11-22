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

"""Tests of the ERNIE 5.5 configuration, layer schedule and architecture checks."""

import copy
import inspect
import json
import re

import pytest
from transformers import AutoConfig

from nemo_automodel._transformers.model_init import _consume_config_overrides, get_hf_config
from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.models.ernie5_5.config import (
    Ernie5_5Config,
    Ernie5_5ModelPlan,
    decoder_only_config,
    hca_rope_scaling,
    main_rope_theta,
    validate_architecture,
    validate_dsa_rope_layout,
    validate_hca_options,
)
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_config, checkpoint_style_config

pytestmark = pytest.mark.cpu_only

# Flags of the checkpoint's config.json whose only implemented value is built into the layers.
IMPLEMENTED_FLAGS = {
    "qk_norm_type": "per_head",
    "use_gated_attn": True,
    "gated_attn_act": "sigmoid",
    "gated_attn_use_q_lora": True,
    "use_vha_attention": True,
    "add_full_attention_sink_bias": True,
    "attention_bias": False,
    "hybrid_mla_attention": "mqa_dsa",
    "mqa_split_kv_b_proj": True,
    "use_fast_hadamard": True,
    "enable_hyper_connections": True,
    "scoring_func": "sigmoid",
    "routed_scaling_factor_learnable": True,
    "multimax": "lm_head",
}


# Registration and round trip of a checkpoint-style config.json


def test_checkpoint_style_config_resolves_without_remote_code_and_preserves_architecture_metadata(tmp_path):
    original = checkpoint_style_config()
    (tmp_path / "config.json").write_text(json.dumps(original))

    assert resolve_custom_config_cls("ernie5_5_moe") is Ernie5_5Config
    config = get_hf_config(str(tmp_path), "eager", trust_remote_code=False, local_files_only=True)
    assert isinstance(config, Ernie5_5Config)
    assert config.architectures == ["Ernie5_5_MoeForCausalLM"]
    assert config.num_hidden_layers == 43
    assert config.num_nextn_predict_layers == 1
    assert config.compress_ratios == original["compress_ratios"]
    assert config.tie_word_embeddings is False
    assert config.moe_latent_size == original["moe_latent_size"]
    assert config.hc_mult == original["hc_mult"]

    save_dir = tmp_path / "saved"
    config.save_pretrained(save_dir)
    restored = AutoConfig.from_pretrained(save_dir, trust_remote_code=False, local_files_only=True)
    assert isinstance(restored, Ernie5_5Config)
    assert restored.num_nextn_predict_layers == 1
    assert restored.compress_ratios == config.compress_ratios
    assert restored.hc_mult == config.hc_mult
    assert restored.rope_parameters == config.rope_parameters


def test_checkpoint_keys_round_trip_through_to_dict_and_save_pretrained(tmp_path):
    # A checkpoint's file also has keys that the class does not declare, and the version of the library that
    # wrote it. These entries stand for them: two token ids and a top-level rope_type, whose names the library
    # knows, and three names that nothing knows, with a false flag, a float and a string as values.
    undeclared = {
        "transformers_version": "4.40.0",
        "bos_token_id": 11,
        "eos_token_id": 12,
        "rope_type": "rope",
        "flag_the_class_does_not_declare": False,
        "ratio_the_class_does_not_declare": 0.25,
        "name_the_class_does_not_declare": "kept",
    }
    checkpoint = {**checkpoint_style_config(), **undeclared}
    assert len(checkpoint) == len(checkpoint_style_config()) + len(undeclared)
    config = Ernie5_5Config(**checkpoint)
    # Building the configuration does not modify the dictionary it was built from.
    assert checkpoint == {**checkpoint_style_config(), **undeclared}
    config.save_pretrained(tmp_path)
    written = {
        "to_dict": json.loads(config.to_json_string(use_diff=False)),
        "save_pretrained": json.loads((tmp_path / "config.json").read_text()),
    }
    for label, emitted in written.items():
        # Every key is written back with its value; only the version stamp of the library differs.
        changed = sorted(key for key, value in checkpoint.items() if key not in emitted or emitted[key] != value)
        assert changed == ["transformers_version"], label
        # The standardized forms that the library adds carry the same values.
        assert emitted["rope_parameters"] == {
            **checkpoint["rope_scaling"],
            "rope_theta": checkpoint["rope_theta"],
            "rope_type": "yarn",
        }, label
        assert emitted["dtype"] == checkpoint["torch_dtype"], label
    added = sorted(set(written["save_pretrained"]) - set(checkpoint))
    assert added == [
        "dtype",
        "initializer_range",
        "mqa_latent_rope_adjacent_pairing",
        "rope_parameters",
        "sigmoid_gate_fusion",
    ]
    reloaded = Ernie5_5Config(**written["save_pretrained"])
    assert reloaded.to_dict() == config.to_dict()


def test_declared_defaults_are_the_checkpoint_values():
    checkpoint = checkpoint_style_config()
    parameters = inspect.signature(Ernie5_5Config.__init__).parameters
    declared = {name: p.default for name, p in parameters.items() if name not in ("self", "kwargs")}
    assert len(declared) == 73
    # Declared fields that the checkpoint-style mapping does not state, with their defaults.
    assert {name: declared[name] for name in set(declared) - set(checkpoint)} == {
        "initializer_range": 0.02,
        "mqa_latent_rope_adjacent_pairing": None,
        "rope_parameters": None,
        "sigmoid_gate_fusion": False,
    }
    # Five defaults are deliberately not the checkpoint's value. Two more are None and select the checkpoint's value.
    different = {name: value for name, value in declared.items() if name in checkpoint and value != checkpoint[name]}
    assert different == {
        "compress_ratios": None,
        "num_nextn_predict_layers": 0,
        "rope_interleave": None,
        "rotary_base": None,
        "pad_token_id": None,
        "rope_theta": None,
        "rope_scaling": None,
    }
    # The other 62 declared fields of the mapping are the defaults.
    assert len(declared) - 4 - len(different) == 62
    # A configuration built from the mapping holds the three values that the defaults do not supply.
    loaded = checkpoint_config()
    assert (loaded.num_nextn_predict_layers, loaded.rope_interleave, loaded.pad_token_id) == (1, True, 0)
    default = Ernie5_5Config()
    assert default.rope_theta == loaded.rope_theta == 8_000_000
    assert default.rope_parameters == loaded.rope_parameters
    assert default.rotary_base is None


def test_default_architectures_name_the_registered_class(tmp_path):
    assert Ernie5_5Config().architectures == ["Ernie5_5_MoeForCausalLM"]
    assert Ernie5_5Config(architectures=["Other"]).architectures == ["Other"]
    # A configuration written by hand and saved resolves to the same configuration class and architecture.
    Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128, 128]).save_pretrained(tmp_path)
    reloaded = get_hf_config(str(tmp_path), None, local_files_only=True)
    assert isinstance(reloaded, Ernie5_5Config) and reloaded.architectures == ["Ernie5_5_MoeForCausalLM"]


def test_implemented_flags_are_typed_fields_that_default_to_the_implemented_values():
    defaults = Ernie5_5Config().to_dict()
    for name, value in IMPLEMENTED_FLAGS.items():
        assert defaults[name] == value and isinstance(defaults[name], bool) == isinstance(value, bool), name
    assert len(IMPLEMENTED_FLAGS) == 14


def test_rope_settings_of_the_checkpoint_style_config():
    config = checkpoint_config()
    assert main_rope_theta(config) == 8_000_000
    assert hca_rope_scaling(config) == {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    assert config.rotary_base == 8_000_000


def test_rope_theta_override_takes_effect_or_raises(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(checkpoint_style_config()))

    def load(**overrides) -> Ernie5_5Config:
        return get_hf_config(str(source), None, local_files_only=True, **overrides)

    # No override.
    plain = load()
    assert plain.rope_theta == main_rope_theta(plain) == 8_000_000
    # Both keys overridden: the value is the base of the DSA layers and survives saving and reloading.
    both = load(rope_theta=10_000, rotary_base=10_000)
    assert both.rope_theta == main_rope_theta(both) == 10_000
    assert both.rope_parameters["rope_theta"] == 10_000 and both.to_dict()["rope_theta"] == 10_000
    assert hca_rope_scaling(both) == hca_rope_scaling(plain)
    both.save_pretrained(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert saved["rope_theta"] == saved["rotary_base"] == saved["rope_parameters"]["rope_theta"] == 10_000
    reloaded = get_hf_config(str(tmp_path / "saved"), None, local_files_only=True)
    assert reloaded.rope_theta == main_rope_theta(reloaded) == 10_000
    # One key overridden: the file repeats the base in rotary_base, so either override alone is a conflict.
    only_theta = load(rope_theta=10_000)
    assert only_theta.rope_theta == 10_000 and only_theta.rotary_base == 8_000_000
    with pytest.raises(ValueError, match="Conflicting RoPE base values: rope_theta=10000, rotary_base=8000000"):
        main_rope_theta(only_theta)
    only_base = load(rotary_base=10_000)
    with pytest.raises(ValueError, match="Conflicting RoPE base values: rope_theta=8000000, rotary_base=10000"):
        main_rope_theta(only_base)
    # Neither the file nor an earlier configuration object was changed by the overrides.
    assert json.loads((source / "config.json").read_text()) == checkpoint_style_config()
    assert plain.rope_theta == 8_000_000 and plain.rotary_base == 8_000_000


def test_rope_theta_override_alone_takes_effect_without_rotary_base():
    config = Ernie5_5Config(num_hidden_layers=2, compress_ratios=[-2, -2], rope_interleave=True)
    assert config.rotary_base is None and main_rope_theta(config) == 8_000_000
    # The framework applies the overrides of a recipe with this function, after the configuration was loaded.
    applied = _consume_config_overrides(config, {"rope_theta": 10_000.0})
    assert applied == {"rope_theta"}
    assert config.rope_theta == main_rope_theta(config) == 10_000.0
    assert config.to_dict()["rope_theta"] == 10_000.0
    assert hca_rope_scaling(config)["factor"] == 16
    validate_architecture(config)


def test_rotary_settings_given_twice_must_agree():
    saved = Ernie5_5Config(rope_theta=10_000).to_dict()
    assert saved["rope_theta"] == saved["rope_parameters"]["rope_theta"] == 10_000
    assert Ernie5_5Config.from_dict(dict(saved)).rope_theta == 10_000
    with pytest.raises(
        ValueError, match=r"Conflicting RoPE base values: rope_theta=20000, rope_parameters\['rope_theta'\]=10000"
    ):
        Ernie5_5Config.from_dict({**saved, "rope_theta": 20_000})
    with pytest.raises(ValueError, match=r"rope_scaling and rope_parameters differ in \['factor'\]"):
        Ernie5_5Config.from_dict({**saved, "rope_scaling": {**saved["rope_scaling"], "factor": 32}})
    # A released key that is null, as in files saved by earlier versions, counts as not given.
    assert Ernie5_5Config.from_dict({**saved, "rope_theta": None, "rope_scaling": None}).rope_theta == 10_000


# Layer schedule


def test_checkpoint_model_plan_separates_decoder_and_mtp():
    config = checkpoint_config()
    plan = Ernie5_5ModelPlan.from_config(config)
    assert len(plan.decoder_layers) == 43
    assert sum(layer.attention == "hca" for layer in plan.decoder_layers) == 37
    assert [layer.layer_index for layer in plan.decoder_layers if layer.attention == "dsa"] == [7, 14, 21, 28, 35, 42]
    assert plan.decoder_layers[0].feed_forward == "dense"
    assert all(layer.feed_forward == "moe" for layer in plan.decoder_layers[1:])
    assert plan.mtp_attention == ("dsa",)
    assert plan.mtp_shared_last_layer


def test_model_plan_rejects_bad_schedule():
    config = Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128, -2], num_nextn_predict_layers=1)
    with pytest.raises(ValueError, match="one entry per decoder and MTP"):
        Ernie5_5ModelPlan.from_config(config)
    config.compress_ratios = [128, 4, -2]
    with pytest.raises(ValueError, match="Only HCA"):
        Ernie5_5ModelPlan.from_config(config)


@pytest.mark.parametrize(
    "field,value",
    [("num_nextn_predict_layers", False), ("first_k_dense_replace", False), ("moe_layer_freq", True)],
)
def test_plan_rejects_boolean_integer_fields(field, value):
    fields = {"num_hidden_layers": 2, "compress_ratios": [128, -2], "num_nextn_predict_layers": 0}
    assert len(Ernie5_5ModelPlan.from_config(Ernie5_5Config(**fields)).decoder_layers) == 2
    with pytest.raises(ValueError):
        Ernie5_5ModelPlan.from_config(Ernie5_5Config(**{**fields, field: value}))


def test_default_mtp_depth_is_zero_and_the_schedule_needs_no_override():
    config = Ernie5_5Config(num_hidden_layers=3, compress_ratios=[128, 128, -2])
    assert config.num_nextn_predict_layers == 0
    plan = Ernie5_5ModelPlan.from_config(config)
    assert len(plan.decoder_layers) == 3 and plan.mtp_attention == ()
    normalized = decoder_only_config(config)
    assert normalized.compress_ratios == [128, 128, -2] and normalized.mtp_shared_last_layer is False
    # A checkpoint's file states its MTP layer explicitly.
    assert checkpoint_config().num_nextn_predict_layers == 1


def test_plan_rejects_float_compression_ratios():
    config = Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128.0, -2], num_nextn_predict_layers=0)
    with pytest.raises(ValueError, match="Only HCA"):
        Ernie5_5ModelPlan.from_config(config)


# Decoder-only normalization


@pytest.mark.parametrize(
    "overrides,layers",
    [
        ({}, 43),
        ({"num_hidden_layers": 3, "compress_ratios": [128, 128, -2]}, 3),
        ({"num_hidden_layers": 3, "compress_ratios": [128, 128, -2, -2]}, 3),
    ],
)
def test_decoder_only_config_normalizes_a_copy_and_preserves_the_source(overrides, layers):
    config = checkpoint_config(num_nextn_predict_layers=0, **overrides)
    original = copy.deepcopy(config.to_dict())
    normalized = decoder_only_config(config)
    assert config.to_dict() == original
    assert normalized is not config
    assert normalized.num_hidden_layers == layers and normalized.num_nextn_predict_layers == 0
    assert normalized.compress_ratios == config.compress_ratios[:layers]
    assert normalized.mtp_shared_last_layer is False
    assert Ernie5_5ModelPlan.from_config(normalized).mtp_attention == ()
    reloaded = Ernie5_5Config.from_dict(normalized.to_dict())
    assert reloaded.mtp_shared_last_layer is False and len(reloaded.compress_ratios) == layers
    assert reloaded.num_nextn_predict_layers == 0


@pytest.mark.parametrize("depth", [1, 2, None, True])
def test_decoder_only_config_rejects_mtp_layers_and_names_the_override(depth):
    config = checkpoint_config(num_nextn_predict_layers=depth)
    with pytest.raises(ValueError, match=r"MTP depth 0.*`model\.num_nextn_predict_layers: 0`"):
        decoder_only_config(config)


def test_decoder_only_config_rejects_a_malformed_extra_entry():
    config = checkpoint_config(num_nextn_predict_layers=0)
    config.compress_ratios[-1] = 0
    with pytest.raises(ValueError, match="Extra schedule"):
        decoder_only_config(config)
    # An extra entry is accepted only for a shared MTP layer.
    config = checkpoint_config(num_nextn_predict_layers=0, mtp_shared_last_layer=False)
    with pytest.raises(ValueError, match="Extra schedule"):
        decoder_only_config(config)


@pytest.mark.parametrize("ratios", [[128, 128], [128, 128, -2, -2, -2], None])
def test_decoder_only_config_rejects_other_schedule_lengths(ratios):
    config = checkpoint_config(num_nextn_predict_layers=0, num_hidden_layers=3, compress_ratios=ratios)
    with pytest.raises(ValueError, match="schedule length|one entry per decoder layer"):
        decoder_only_config(config)


# Architecture checks


def test_explicit_pairing_takes_precedence_over_legacy_flag():
    validate_dsa_rope_layout(Ernie5_5Config(mqa_latent_rope_adjacent_pairing=True, rope_interleave=False))
    with pytest.raises(NotImplementedError, match="permutation"):
        validate_dsa_rope_layout(Ernie5_5Config(mqa_latent_rope_adjacent_pairing=False, rope_interleave=True))


def test_legacy_pairing_and_ambiguous_or_unsupported_inputs():
    validate_dsa_rope_layout(Ernie5_5Config(rope_interleave=True))
    with pytest.raises(NotImplementedError, match="permutation"):
        validate_dsa_rope_layout(Ernie5_5Config(rope_interleave=False))
    # A configuration that declares neither pairing key is rejected: the default does not supply one.
    with pytest.raises(ValueError, match="boolean"):
        validate_dsa_rope_layout(Ernie5_5Config())
    with pytest.raises(ValueError, match="boolean"):
        validate_dsa_rope_layout(Ernie5_5Config(rope_interleave=None))
    with pytest.raises(ValueError, match="boolean"):
        validate_dsa_rope_layout(Ernie5_5Config(rope_interleave="true"))


def test_only_validated_hca_options_are_allowed():
    validate_hca_options(Ernie5_5Config(use_qk_norm=True, hca_rope_type="yarn"))
    with pytest.raises(NotImplementedError, match="Q/K normalization"):
        validate_hca_options(Ernie5_5Config(use_qk_norm=False, hca_rope_type="yarn"))
    with pytest.raises(NotImplementedError, match="YaRN"):
        validate_hca_options(Ernie5_5Config(use_qk_norm=True, hca_rope_type="rope"))
    with pytest.raises(NotImplementedError, match="sigmoid gate"):
        validate_hca_options(Ernie5_5Config(use_qk_norm=True, hca_rope_type="yarn", sigmoid_gate_fusion=True))


def test_checkpoint_configuration_is_within_guarded_branches():
    config = checkpoint_config()
    validate_dsa_rope_layout(config)
    validate_hca_options(config)
    validate_architecture(decoder_only_config(checkpoint_config(num_nextn_predict_layers=0)))


def test_hca_options_require_the_yarn_settings():
    config = checkpoint_config()
    del config.rope_parameters["factor"]
    with pytest.raises(ValueError, match="YaRN"):
        validate_hca_options(config)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("hidden_act", "silu", "situ"),
        ("n_shared_experts", 2, "one shared expert"),
        ("n_group", 2, "one router group"),
        ("topk_group", 2, "one router group"),
        ("moe_split_feature_routing", False, "split-feature routing"),
        ("latent_moe_use_norm", False, "latent normalization"),
        ("hc_eps", 1e-5, "epsilons"),
        ("rotary_base", 10000.0, "Conflicting RoPE base"),
    ],
)
def test_validate_architecture_rejects_unimplemented_settings(field, value, message):
    validate_architecture(decoder_only_config(checkpoint_config(num_nextn_predict_layers=0)))
    config = decoder_only_config(checkpoint_config(num_nextn_predict_layers=0, **{field: value}))
    with pytest.raises(ValueError, match=message):
        validate_architecture(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("qk_norm_type", "global"),
        ("use_gated_attn", False),
        ("gated_attn_act", "tanh"),
        ("gated_attn_use_q_lora", False),
        ("use_vha_attention", False),
        ("add_full_attention_sink_bias", False),
        ("attention_bias", True),
        ("hybrid_mla_attention", "mla"),
        ("mqa_split_kv_b_proj", False),
        ("use_fast_hadamard", False),
        ("enable_hyper_connections", False),
        ("scoring_func", "softmax"),
        ("routed_scaling_factor_learnable", False),
        ("multimax", "none"),
        # Integers are not accepted in place of booleans, and a missing value is not the implemented one.
        ("use_gated_attn", 1),
        ("attention_bias", 0),
        ("multimax", None),
    ],
)
def test_validate_architecture_rejects_other_variants_of_the_implemented_flags(field, value):
    assert field in IMPLEMENTED_FLAGS
    config = decoder_only_config(checkpoint_config(num_nextn_predict_layers=0, **{field: value}))
    expected = re.escape(f"{field}={value!r} (implemented: {IMPLEMENTED_FLAGS[field]!r})")
    with pytest.raises(NotImplementedError, match=expected):
        validate_architecture(config)


def test_validate_architecture_names_every_differing_flag():
    config = decoder_only_config(
        checkpoint_config(num_nextn_predict_layers=0, scoring_func="softmax", use_fast_hadamard=False)
    )
    with pytest.raises(NotImplementedError, match="use_fast_hadamard=False.*scoring_func='softmax'"):
        validate_architecture(config)


def test_validate_architecture_requires_a_positive_rope_base():
    config = decoder_only_config(checkpoint_config(num_nextn_predict_layers=0))
    config.rope_parameters["rope_theta"] = 0
    with pytest.raises(ValueError, match="positive main RoPE base"):
        validate_architecture(config)
    del config.rope_parameters["rope_theta"]
    with pytest.raises(ValueError, match="positive main RoPE base"):
        validate_architecture(config)


def test_validate_architecture_checks_only_the_attention_kinds_in_the_schedule():
    hca_only = checkpoint_config(
        num_nextn_predict_layers=0, num_hidden_layers=2, compress_ratios=[128, 128], rope_interleave=False
    )
    validate_architecture(hca_only)
    # Without a DSA layer the pairing need not be declared at all.
    validate_architecture(Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128, 128]))
    dsa_only = checkpoint_config(
        num_nextn_predict_layers=0, num_hidden_layers=2, compress_ratios=[-2, -2], hca_rope_type="rope"
    )
    validate_architecture(dsa_only)
    mixed = decoder_only_config(checkpoint_config(num_nextn_predict_layers=0, rope_interleave=False))
    with pytest.raises(NotImplementedError, match="permutation"):
        validate_architecture(mixed)
