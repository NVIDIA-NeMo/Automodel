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

"""Unit tests for the EAGLE draft ``copy_embeddings_from_target`` DTensor branch.

Covers the FSDP2-target compatibility patch on both the EAGLE-3 draft
(``LlamaEagle3DraftModel`` in ``draft_llama.py``) and the EAGLE-1/2 draft
(``LlamaEagleDraftModel`` in ``draft_llama_v12.py``). When the target
embedding weight is a ``DTensor`` (sharded across ranks under FSDP2), the
copy path must gather it via ``.full_tensor()`` before writing into the
(un-sharded) draft parameter. A plain ``nn.Embedding.weight`` (single
rank or pre-sharding) must pass through unchanged.

Pure-CPU, no distributed init: the DTensor branch is exercised with a
mock object that exposes ``full_tensor()``. The test verifies branch
selection and call-count without requiring a multi-rank harness.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel

_VOCAB = 32
_HIDDEN = 16


class _MinimalDraftEagle3:
    """Stand-in exposing ``self.model.embed_tokens`` for the method under test."""

    copy_embeddings_from_target = LlamaEagle3DraftModel.copy_embeddings_from_target

    def __init__(self) -> None:
        inner = nn.Module()
        inner.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model = inner


class _MinimalDraftEagle1:
    """Stand-in exposing ``self.embed_tokens`` for the method under test."""

    copy_embeddings_from_target = LlamaEagleDraftModel.copy_embeddings_from_target

    def __init__(self) -> None:
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)


def _fake_dtensor_embedding(full: torch.Tensor) -> MagicMock:
    """Return an ``nn.Embedding``-like mock whose ``.weight`` is a DTensor stand-in."""
    fake_weight = MagicMock()
    fake_weight.full_tensor = MagicMock(return_value=full)
    fake_emb = MagicMock()
    fake_emb.weight = fake_weight
    return fake_emb


def test_eagle3_copy_plain_tensor() -> None:
    draft = _MinimalDraftEagle3()
    target = nn.Embedding(_VOCAB, _HIDDEN)
    expected = target.weight.detach().clone()
    draft.copy_embeddings_from_target(target)
    assert torch.equal(draft.model.embed_tokens.weight, expected)


def test_eagle3_copy_dtensor_invokes_full_tensor() -> None:
    draft = _MinimalDraftEagle3()
    full = torch.randn(_VOCAB, _HIDDEN)
    fake_emb = _fake_dtensor_embedding(full)
    draft.copy_embeddings_from_target(fake_emb)
    fake_emb.weight.full_tensor.assert_called_once()
    assert torch.equal(draft.model.embed_tokens.weight, full)


def test_eagle1_copy_plain_tensor() -> None:
    draft = _MinimalDraftEagle1()
    target = nn.Embedding(_VOCAB, _HIDDEN)
    expected = target.weight.detach().clone()
    draft.copy_embeddings_from_target(target)
    assert torch.equal(draft.embed_tokens.weight, expected)


def test_eagle1_copy_dtensor_invokes_full_tensor() -> None:
    draft = _MinimalDraftEagle1()
    full = torch.randn(_VOCAB, _HIDDEN)
    fake_emb = _fake_dtensor_embedding(full)
    draft.copy_embeddings_from_target(fake_emb)
    fake_emb.weight.full_tensor.assert_called_once()
    assert torch.equal(draft.embed_tokens.weight, full)
