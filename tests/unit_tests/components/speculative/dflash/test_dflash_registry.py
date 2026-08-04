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
