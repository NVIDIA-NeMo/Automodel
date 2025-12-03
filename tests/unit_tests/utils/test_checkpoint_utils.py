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


class _ModuleContainer:
    def __init__(self, module: nn.Module) -> None:
        self._module = module
        if hasattr(module, "config"):
            self.config = module.config

    def __getitem__(self, idx: int) -> nn.Module:
        if idx == 0:
            return self._module
        raise IndexError


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

    container = _ModuleContainer(DummyModel())
    assert checkpoint_utils.is_tied_word_embeddings(container) is False


def test_is_tied_word_embeddings_falls_back_to_top_level_when_no_text_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    container = _ModuleContainer(DummyModel())
    assert checkpoint_utils.is_tied_word_embeddings(container) is True


def test_is_tied_word_embeddings_handles_missing_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    container = _ModuleContainer(DummyModel())
    assert checkpoint_utils.is_tied_word_embeddings(container) is False


def test_is_tied_word_embeddings_respects_exclusion_list():
    class Qwen3OmniMoeThinkerForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    container = _ModuleContainer(Qwen3OmniMoeThinkerForConditionalGeneration())
    assert checkpoint_utils.is_tied_word_embeddings(container) is False
