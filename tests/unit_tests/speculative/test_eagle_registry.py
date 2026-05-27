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

"""Unit tests for the EAGLE draft dispatch registry."""

import pytest

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.registry import (
    EAGLE1_DRAFT_REGISTRY,
    EAGLE3_DRAFT_REGISTRY,
    DraftSpec,
    resolve_eagle1_draft_spec,
    resolve_eagle3_draft_spec,
)


# ── DraftSpec dataclass ──────────────────────────────────────────────────


def test_draft_spec_is_frozen():
    spec = DraftSpec(draft_cls=LlamaEagle3DraftModel)
    with pytest.raises(AttributeError):
        spec.draft_cls = object


def test_draft_spec_stores_class():
    spec = DraftSpec(draft_cls=LlamaEagleDraftModel)
    assert spec.draft_cls is LlamaEagleDraftModel


# ── Registry contents ────────────────────────────────────────────────────


def test_eagle3_registry_contains_llama():
    assert "LlamaForCausalLM" in EAGLE3_DRAFT_REGISTRY
    assert EAGLE3_DRAFT_REGISTRY["LlamaForCausalLM"].draft_cls is LlamaEagle3DraftModel


def test_eagle1_registry_contains_llama():
    assert "LlamaForCausalLM" in EAGLE1_DRAFT_REGISTRY
    assert EAGLE1_DRAFT_REGISTRY["LlamaForCausalLM"].draft_cls is LlamaEagleDraftModel


def test_eagle3_registry_contains_phi3():
    assert "Phi3ForCausalLM" in EAGLE3_DRAFT_REGISTRY


def test_eagle1_registry_contains_phi3():
    assert "Phi3ForCausalLM" in EAGLE1_DRAFT_REGISTRY


def test_eagle3_registry_contains_qwen3_moe():
    assert "Qwen3MoeForCausalLM" in EAGLE3_DRAFT_REGISTRY
    assert EAGLE3_DRAFT_REGISTRY["Qwen3MoeForCausalLM"].draft_cls is LlamaEagle3DraftModel


def test_eagle1_registry_contains_qwen3_moe():
    assert "Qwen3MoeForCausalLM" in EAGLE1_DRAFT_REGISTRY
    assert EAGLE1_DRAFT_REGISTRY["Qwen3MoeForCausalLM"].draft_cls is LlamaEagleDraftModel


# ── resolve_eagle3_draft_spec ────────────────────────────────────────────


def test_resolve_eagle3_llama():
    spec = resolve_eagle3_draft_spec(["LlamaForCausalLM"])
    assert spec.draft_cls is LlamaEagle3DraftModel


def test_resolve_eagle3_first_match_wins():
    spec = resolve_eagle3_draft_spec(["UnknownArch", "LlamaForCausalLM"])
    assert spec.draft_cls is LlamaEagle3DraftModel


def test_resolve_eagle3_unsupported_raises():
    with pytest.raises(ValueError, match="no EAGLE draft spec registered"):
        resolve_eagle3_draft_spec(["CompletelyFakeArch"])


def test_resolve_eagle3_empty_raises():
    with pytest.raises(ValueError, match="no EAGLE draft spec registered"):
        resolve_eagle3_draft_spec([])


# ── resolve_eagle1_draft_spec ────────────────────────────────────────────


def test_resolve_eagle3_qwen3_moe():
    # MoE backbones share the dense Llama-style draft because the draft only
    # consumes post-block hidden states, not the per-expert routing internals.
    spec = resolve_eagle3_draft_spec(["Qwen3MoeForCausalLM"])
    assert spec.draft_cls is LlamaEagle3DraftModel


def test_resolve_eagle1_llama():
    spec = resolve_eagle1_draft_spec(["LlamaForCausalLM"])
    assert spec.draft_cls is LlamaEagleDraftModel


def test_resolve_eagle1_qwen3_moe():
    spec = resolve_eagle1_draft_spec(["Qwen3MoeForCausalLM"])
    assert spec.draft_cls is LlamaEagleDraftModel


def test_resolve_eagle1_unsupported_raises():
    with pytest.raises(ValueError, match="no EAGLE draft spec registered"):
        resolve_eagle1_draft_spec(["CompletelyFakeArch"])


def test_resolve_eagle1_error_lists_supported():
    with pytest.raises(ValueError, match="Supported architectures"):
        resolve_eagle1_draft_spec(["NoSuchModel"])
