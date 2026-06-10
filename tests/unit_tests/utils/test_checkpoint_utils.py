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

from types import SimpleNamespace

import torch.nn as nn

import nemo_automodel.components.checkpoint.utils as checkpoint_utils


def test_is_tied_word_embeddings_prefers_text_config_value():
    class DummyTextConfig:
        def __init__(self, tied: bool) -> None:
            self.tie_word_embeddings = tied

    class DummyConfig:
        def __init__(self) -> None:
            self.tie_word_embeddings = True
            self._text = DummyTextConfig(False)

        def get_text_config(self):
            return self._text

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = DummyConfig()

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


def test_is_tied_word_embeddings_respects_qwen3_vl_moe_exclusion():
    class DummyTextConfig:
        tie_word_embeddings = True

    class DummyConfig:
        tie_word_embeddings = False

        def get_text_config(self):
            return DummyTextConfig()

    class Qwen3VLMoeForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = DummyConfig()

    model = Qwen3VLMoeForConditionalGeneration()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


def test_is_tied_word_embeddings_falls_back_to_top_level_when_no_text_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is True


def test_is_tied_word_embeddings_handles_missing_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


def test_is_tied_word_embeddings_respects_exclusion_list():
    class Qwen3OmniMoeThinkerForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    model = Qwen3OmniMoeThinkerForConditionalGeneration()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


class _DraftLikeModel(nn.Module):
    """Minimal stand-in for an EAGLE-3 draft model.

    Owns ``model.embed_tokens`` (full target vocab) and a separate
    ``lm_head`` (potentially shrunk vocab). Mirrors the FQNs that
    ``get_lm_head_weight_and_name`` and ``get_input_embeddings_weight_and_name``
    look for so the tests exercise the same code paths as the real model.
    """

    def __init__(
        self,
        embed_vocab: int,
        lm_head_vocab: int,
        hidden: int,
        tie_word_embeddings: bool,
        tie_storage: bool = False,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(embed_vocab, hidden)
        self.lm_head = nn.Linear(hidden, lm_head_vocab, bias=False)
        if tie_storage:
            self.lm_head.weight = self.model.embed_tokens.weight


def test_has_local_tied_lm_head_false_when_shapes_disagree():
    """Vocab-shrunk EAGLE-3 draft: embed_tokens [V_t,H] != lm_head [V_d,H]."""
    model = _DraftLikeModel(embed_vocab=128256, lm_head_vocab=8192, hidden=2048, tie_word_embeddings=True)
    assert checkpoint_utils.has_local_tied_lm_head(model) is False


def test_has_local_tied_lm_head_true_when_shapes_match_and_tied():
    """Standard tied-embeddings case: shapes match, flag set, storage aliases."""
    model = _DraftLikeModel(
        embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=True, tie_storage=True
    )
    assert checkpoint_utils.has_local_tied_lm_head(model) is True


def test_has_local_tied_lm_head_false_when_shapes_match_but_storage_is_untied():
    """Matching shapes and config are not enough; the tensors must alias."""
    model = _DraftLikeModel(embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=True)
    assert checkpoint_utils.has_local_tied_lm_head(model) is False


def test_has_local_tied_lm_head_false_when_flag_unset():
    """Even with matching shapes, untied config means not locally tied."""
    model = _DraftLikeModel(embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=False)
    assert checkpoint_utils.has_local_tied_lm_head(model) is False
