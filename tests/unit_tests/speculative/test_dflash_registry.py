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

"""Tests for the DFlash draft-model registry."""

from __future__ import annotations

import pytest

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec


@pytest.mark.parametrize("arch", ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"])
def test_resolve_known_architectures(arch):
    spec = resolve_dflash_draft_spec([arch])
    assert spec.draft_cls is Qwen3DFlashDraftModel


def test_resolve_first_match_wins():
    spec = resolve_dflash_draft_spec(["SomethingUnknown", "Qwen3ForCausalLM"])
    assert spec.draft_cls is Qwen3DFlashDraftModel


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="no DFlash draft spec registered"):
        resolve_dflash_draft_spec(["LlamaForCausalLM"])


@pytest.mark.parametrize(
    "arch",
    [
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_qwen3_family_uses_the_recipe_draft_config_builder(arch):
    """Qwen3-family targets must reach the recipe's ``_build_qwen3_draft_config``.

    That builder projects the target config onto Qwen3 keys (dropping a Qwen3.5
    target's ``partial_rotary_factor``) and applies ``draft_num_attention_heads``,
    ``draft_head_dim`` and ``draft_sliding_window``; a spec-level builder would
    bypass all of it and save a draft config that disagrees with training.
    """
    assert resolve_dflash_draft_spec([arch]).build_draft_config is None
