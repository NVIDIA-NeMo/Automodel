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

"""Attention-backend capability flags for the EAGLE draft models.

``transformers`` gates ``attn_implementation="flash_attention_2"`` on the
class-level ``_supports_flash_attn`` flag (default ``False`` on
``PreTrainedModel``). The EAGLE-3 draft attention implements eager + FA2, so it
must advertise FA2; the EAGLE-1/2 draft attention is eager-only and must not.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.draft_llama import _HAS_FA, LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel


def _tiny_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    config.draft_vocab_size = 16
    config.target_hidden_size = 32
    return config


def test_eagle3_draft_declares_flash_attn_support():
    # This flag is exactly what transformers' _flash_attn_can_dispatch checks;
    # without it, attn_implementation="flash_attention_2" is rejected at init.
    assert LlamaEagle3DraftModel._supports_flash_attn is True
    # The draft attention implements eager + FA2 but NOT SDPA, so SDPA stays off.
    assert LlamaEagle3DraftModel._supports_sdpa is False


def test_eagle3_draft_eager_construction_unchanged():
    # The default (eager) path must keep working.
    model = LlamaEagle3DraftModel(_tiny_config()).to(torch.float32)
    assert model is not None


@pytest.mark.skipif(not _HAS_FA, reason="flash-attn not installed")
def test_eagle3_draft_constructs_with_flash_attention_2():
    config = _tiny_config()
    config.attn_implementation = "flash_attention_2"
    # Must not raise transformers' "does not support Flash Attention 2 yet".
    model = LlamaEagle3DraftModel(config)
    assert model is not None


def test_eagle1_2_draft_does_not_claim_flash_attn():
    # EAGLE-1/2 attention is eager-only; advertising FA2 would let transformers
    # dispatch a backend the layer cannot run. EAGLE-3 opts in via a class-level
    # override; EAGLE-1/2 must not declare its own override (it inherits the base
    # PreTrainedModel default, which is a property in current transformers and so
    # cannot be compared by value at the class level).
    assert LlamaEagle3DraftModel.__dict__.get("_supports_flash_attn") is True
    assert "_supports_flash_attn" not in LlamaEagleDraftModel.__dict__
