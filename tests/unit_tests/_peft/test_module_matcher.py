# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.shared.import_utils import safe_import

HAS_TE, transformer_engine = safe_import("transformer_engine")
import pytest
import torch
import torch.nn as nn

from nemo_automodel.components._peft.module_matcher import ModuleMatcher, _is_linear_module, wildcard_match


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (nn.Linear(10, 10), True),
        (nn.Conv1d(10, 10, 1), False),
        (nn.Conv2d(10, 10, 1), False),
        (nn.Conv3d(10, 10, 1), False),
        (nn.ConvTranspose1d(10, 10, 1), False),
        (nn.ConvTranspose2d(10, 10, 1), False),
        (nn.ConvTranspose3d(10, 10, 1), False),
    ],
)
def test_is_linear_module(module, expected):
    assert _is_linear_module(module) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TE, reason="transformer_engine is not installed")
def test_is_linear_module_transformer_engine():
    assert _is_linear_module(transformer_engine.pytorch.Linear(10, 10))


# ------------------------------------------------------------------ #
# wildcard_match tests                                                #
# ------------------------------------------------------------------ #


class TestWildcardMatch:
    def test_basic_wildcard(self):
        assert wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv") is True

    def test_basic_wildcard_no_match(self):
        assert wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv") is False

    def test_exact_match_no_wildcards(self):
        assert (
            wildcard_match("decoder.layers.0.self_attention.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
            is True
        )

    def test_none_key_returns_false(self):
        result = wildcard_match("*.layers.0.*", None)
        assert result is False

    def test_multi_level_wildcard(self):
        assert wildcard_match("*.*.*", "a.b.c") is True
        assert wildcard_match("*.b.*", "a.b") is True

    def test_segment_wildcard_does_not_extend_layer_index(self):
        # ".*" after a name segment matches that segment's children, not longer
        # segments that share its prefix: layer 1 must not also select layers 10-19.
        assert wildcard_match("*.layers.1.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv") is True
        assert wildcard_match("*.layers.1.*.linear_qkv", "decoder.layers.10.self_attention.linear_qkv") is False
        assert wildcard_match("*.layers.1.*", "model.layers.12.mlp.down_proj") is False
        assert wildcard_match("*.layers.1.*", "model.layers.1.mlp.down_proj") is True

    @pytest.mark.parametrize("pattern", ["*layers.1.*", ".*layers.1.*", "model.layers.1.*"])
    def test_layer_index_wildcard_spellings_select_only_that_layer(self, pattern):
        assert wildcard_match(pattern, "model.layers.1.mlp.down_proj") is True
        assert wildcard_match(pattern, "model.layers.12.mlp.down_proj") is False

    def test_fully_qualified_pattern_matches_through_a_layer_wildcard(self):
        assert wildcard_match("model.layers.*.self_attn.q_proj", "model.layers.7.self_attn.q_proj") is True
        assert wildcard_match("model.layers.*.self_attn.q_proj", "model.layers.7.self_attn.k_proj") is False
        # A pattern that starts with a full segment path matches its children, not a longer sibling name.
        assert wildcard_match("model.layers.*", "model.layers.3.mlp.up_proj") is True
        assert wildcard_match("model.layers.*", "model.layers_norm") is False

    def test_fragment_wildcard_still_matches_any_text(self):
        # The regex-style fragment documented for apply_lora_to_linear_modules must keep matching fc1/fc2.
        assert wildcard_match(".*fc.*", "model.decoder.layers.0.fc1") is True
        assert wildcard_match(".*fc.*", "vision_model.encoder.layers.3.mlp.fc2") is True
        assert wildcard_match("*mlp.*", "model.layers.0.mlp_gate.proj") is True
        assert wildcard_match("*mlp.fc.*", "model.layers.0.mlp.fc1") is True


# ------------------------------------------------------------------ #
# ModuleMatcher validation tests                                      #
# ------------------------------------------------------------------ #


class TestModuleMatcherValidation:
    def test_matches_gate_bias(self):
        m = ModuleMatcher(exclude_modules=["*_proj"])
        assert m.match(nn.Linear(10, 10), name="base_model.model.model.layers.1.mixer.gate.e_score_correction_bias")

    def test_does_not_match_gate_bias(self):
        m = ModuleMatcher(exclude_modules=["*_proj", "*.gate.e_score_correction_bias"])
        assert not m.match(nn.Linear(10, 10), name="base_model.model.model.layers.1.mixer.gate.e_score_correction_bias")
        assert m.match(nn.Linear(10, 10), name="model.lm_head")

    def test_default_target_modules_to_star_proj(self):
        m = ModuleMatcher()
        assert m.target_modules == ["*_proj"]

    def test_rejects_target_and_exclude_together(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModuleMatcher(target_modules=["linear_qkv"], exclude_modules=["linear_proj"])

    def test_rejects_match_all_linear_with_target_modules(self):
        with pytest.raises(ValueError):
            ModuleMatcher(match_all_linear=True, target_modules=["linear_qkv"])

    def test_rejects_match_all_linear_with_exclude_modules(self):
        with pytest.raises(ValueError):
            ModuleMatcher(match_all_linear=True, exclude_modules=["linear_qkv"])

    def test_accepts_match_all_linear_alone(self):
        m = ModuleMatcher(match_all_linear=True)
        assert m.match_all_linear is True

    def test_accepts_target_modules_alone(self):
        m = ModuleMatcher(target_modules=["linear_qkv"])
        assert m.target_modules == ["linear_qkv"]

    def test_accepts_exclude_modules_alone(self):
        m = ModuleMatcher(exclude_modules=["linear_proj"])
        assert m.exclude_modules == ["linear_proj"]

    def test_string_target_modules_coerced_to_list(self):
        m = ModuleMatcher(target_modules="linear_qkv")
        assert m.target_modules == ["linear_qkv"]

    def test_string_exclude_modules_coerced_to_list(self):
        m = ModuleMatcher(exclude_modules="linear_proj")
        assert m.exclude_modules == ["linear_proj"]

    def test_none_target_modules_coerced_to_list(self):
        m = ModuleMatcher(target_modules=None, exclude_modules=["linear_proj"])
        assert m.target_modules == []

    def test_none_exclude_modules_coerced_to_list(self):
        m = ModuleMatcher(target_modules=["linear_qkv"], exclude_modules=None)
        assert m.exclude_modules == []


# ------------------------------------------------------------------ #
# ModuleMatcher.match tests                                           #
# ------------------------------------------------------------------ #


class TestModuleMatcherMatch:
    def test_match_all_linear_matches_linear(self):
        matcher = ModuleMatcher(match_all_linear=True)
        assert matcher.match(nn.Linear(10, 10), name="linear_qkv") is True

    def test_match_all_linear_skips_non_linear(self):
        matcher = ModuleMatcher(match_all_linear=True)
        assert not matcher.match(nn.Conv2d(3, 3, 1), name="conv")

    def test_target_modules_exact_name_match(self):
        matcher = ModuleMatcher(target_modules=["linear_qkv"])
        assert matcher.match(nn.Linear(10, 10), name="linear_qkv") is True

    def test_target_modules_no_match(self):
        matcher = ModuleMatcher(target_modules=["linear_qkv"])
        assert matcher.match(nn.Linear(10, 10), name="linear_proj") is False

    def test_target_modules_wildcard_match(self):
        matcher = ModuleMatcher(target_modules=["*.layers.0.*.linear_qkv"])
        assert (
            matcher.match(
                nn.Linear(10, 10),
                name="linear_qkv",
                prefix="decoder.layers.0.self_attention",
            )
            is True
        )

    def test_target_modules_wildcard_no_match(self):
        matcher = ModuleMatcher(target_modules=["*.layers.0.*.linear_qkv"])
        assert (
            matcher.match(
                nn.Linear(10, 10),
                name="linear_qkv",
                prefix="decoder.layers.1.self_attention",
            )
            is False
        )

    def test_exclude_modules_excludes_by_name(self):
        matcher = ModuleMatcher(exclude_modules=["linear_proj"])
        assert not matcher.match(nn.Linear(10, 10), name="linear_proj")

    def test_exclude_modules_allows_non_excluded_linear(self):
        matcher = ModuleMatcher(exclude_modules=["linear_proj"])
        assert matcher.match(nn.Linear(10, 10), name="linear_qkv") is True

    def test_exclude_modules_skips_non_linear(self):
        matcher = ModuleMatcher(exclude_modules=["conv"])
        assert not matcher.match(nn.Conv2d(3, 3, 1), name="linear_qkv")

    def test_exclude_modules_wildcard(self):
        matcher = ModuleMatcher(exclude_modules=["*.layers.0.*.linear_proj"])
        assert not matcher.match(
            nn.Linear(10, 10),
            name="linear_proj",
            prefix="decoder.layers.0.self_attention",
        )

    def test_target_modules_layer_wildcards_select_only_those_layers(self):
        # Documented usage: LoRA on linear_qkv of the first two layers only.
        matcher = ModuleMatcher(target_modules=["*.layers.0.*.linear_qkv", "*.layers.1.*.linear_qkv"])
        matched = [
            i
            for i in range(24)
            if matcher.match(nn.Linear(10, 10), name=f"decoder.layers.{i}.self_attention.linear_qkv")
        ]
        assert matched == [0, 1]

    def test_exclude_modules_layer_wildcard_excludes_only_that_layer(self):
        matcher = ModuleMatcher(exclude_modules=["*.layers.1.*"])
        excluded = [i for i in range(24) if not matcher.match(nn.Linear(10, 10), name=f"model.layers.{i}.mlp.up_proj")]
        assert excluded == [1]
