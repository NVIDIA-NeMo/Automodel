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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.training.neftune import NEFTune, _get_input_embeddings


class SimpleModel(nn.Module):
    """Minimal model with an embedding layer for testing."""

    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        return self.linear(x)


class TestNEFTune:
    def test_noise_applied_during_training(self):
        model = SimpleModel()
        model.train()
        input_ids = torch.randint(0, 100, (2, 16))

        torch.manual_seed(42)
        out_clean = model(input_ids)

        neftune = NEFTune(noise_alpha=5.0)
        neftune.activate(model)
        torch.manual_seed(42)
        out_noisy = model(input_ids)

        assert not torch.allclose(out_clean, out_noisy), "NEFTune should change output compared to clean forward"

        neftune.deactivate(model)

    def test_no_noise_during_eval(self):
        model = SimpleModel()
        neftune = NEFTune(noise_alpha=5.0)
        neftune.activate(model)

        model.eval()
        input_ids = torch.randint(0, 100, (2, 16))
        torch.manual_seed(42)
        out1 = model(input_ids)
        torch.manual_seed(42)
        out2 = model(input_ids)
        assert torch.allclose(out1, out2), "NEFTune should not add noise during eval"

        neftune.deactivate(model)

    def test_deactivate_restores_behavior(self):
        model = SimpleModel()
        model.train()

        input_ids = torch.randint(0, 100, (2, 16))
        torch.manual_seed(42)
        original_out = model(input_ids)

        neftune = NEFTune(noise_alpha=5.0)
        neftune.activate(model)
        neftune.deactivate(model)

        torch.manual_seed(42)
        restored_out = model(input_ids)
        assert torch.allclose(original_out, restored_out), "Deactivation should restore original behavior"

    def test_zero_alpha_no_noise(self):
        model = SimpleModel()
        model.train()
        neftune = NEFTune(noise_alpha=0.0)
        neftune.activate(model)

        input_ids = torch.randint(0, 100, (2, 16))
        torch.manual_seed(42)
        out1 = model(input_ids)
        torch.manual_seed(42)
        out2 = model(input_ids)
        assert torch.allclose(out1, out2), "Zero alpha should produce no noise"

        neftune.deactivate(model)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            NEFTune(noise_alpha=-1.0)

    def test_double_activate_raises(self):
        model = SimpleModel()
        neftune = NEFTune(noise_alpha=5.0)
        neftune.activate(model)
        with pytest.raises(RuntimeError, match="already active"):
            neftune.activate(model)
        neftune.deactivate(model)

    def test_deactivate_when_inactive_is_noop(self):
        model = SimpleModel()
        neftune = NEFTune(noise_alpha=5.0)
        neftune.deactivate(model)  # should not raise

    def test_is_active_property(self):
        model = SimpleModel()
        neftune = NEFTune(noise_alpha=5.0)
        assert not neftune.is_active
        neftune.activate(model)
        assert neftune.is_active
        neftune.deactivate(model)
        assert not neftune.is_active


class TestGetInputEmbeddings:
    def test_finds_via_method(self):
        model = SimpleModel()
        emb = _get_input_embeddings(model)
        assert emb is model.embed_tokens

    def test_finds_via_attribute(self):
        model = nn.Module()
        model.embed_tokens = nn.Embedding(100, 32)
        emb = _get_input_embeddings(model)
        assert emb is model.embed_tokens

    def test_returns_none_for_no_embeddings(self):
        model = nn.Linear(10, 10)
        emb = _get_input_embeddings(model)
        assert emb is None
