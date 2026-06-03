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

"""CPU tiny-config checks that the ERNIE 4.5 causal-LM heads support
memory-efficient fused cross-entropy (cut-CE / FusedLinearCrossEntropy).

The training recipe only enables FusedLinearCrossEntropy when (a) the model's
forward exposes a ``logits_to_keep`` parameter and (b) calling
``model(logits_to_keep=1, output_hidden_states=True)`` returns an output that
carries the FINAL hidden states (full sequence) while logits cover only the
last token. These tests assert both conditions on CPU with a tiny config.
"""

from __future__ import annotations

import pytest
import torch

# Skip module if HF doesn't have the configurations available (older transformers).
pytest.importorskip("transformers.models.ernie4_5")
pytest.importorskip("transformers.models.ernie4_5_moe")

from transformers.models.ernie4_5.configuration_ernie4_5 import Ernie4_5Config
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie4_5.model import (
    Ernie4_5_MoeForCausalLM,
    Ernie4_5ForCausalLM,
)
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


@pytest.fixture
def dense_config():
    return Ernie4_5Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,
        tie_word_embeddings=True,
        pad_token_id=0,
    )


@pytest.fixture
def moe_hf_config():
    return Ernie4_5_MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        moe_intermediate_size=16,
        moe_k=2,
        moe_num_experts=4,
        # 0 shared experts keeps MoE.forward on CPU (shared-experts path
        # eagerly allocates a torch.cuda.Stream, which crashes on CPU-only CI).
        moe_num_shared_experts=0,
        moe_layer_start_index=1,
        moe_layer_end_index=3,
        moe_layer_interval=1,
        router_aux_loss_coef=0.001,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _dense_model(dense_config, backend_config):
    torch.manual_seed(0)
    model = Ernie4_5ForCausalLM(dense_config, backend=backend_config)
    return model.to(torch.float32).eval()


def _moe_model(moe_hf_config, backend_config):
    torch.manual_seed(0)
    model = Ernie4_5_MoeForCausalLM(moe_hf_config, backend=backend_config)
    return model.to(torch.float32).eval()


class TestDenseCutCE:
    def test_supports_logits_to_keep(self, dense_config, backend_config):
        model = _dense_model(dense_config, backend_config)
        assert _supports_logits_to_keep(model) is True

    def test_cut_ce_path_returns_full_hidden_states_and_last_token_logits(self, dense_config, backend_config):
        model = _dense_model(dense_config, backend_config)
        batch, seq = 2, 6
        input_ids = torch.randint(0, dense_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

        # (b) hidden states are carried on the output...
        assert ("hidden_states" in out) or out.hidden_states is not None
        hidden_states = out.hidden_states
        # ...and span the FULL sequence (input to lm_head), not just the kept token.
        assert hidden_states.shape == (batch, seq, dense_config.hidden_size)
        # ...while logits correspond to only the last token.
        assert out.logits.shape == (batch, 1, dense_config.vocab_size)

    def test_default_forward_yields_full_length_logits(self, dense_config, backend_config):
        model = _dense_model(dense_config, backend_config)
        batch, seq = 2, 6
        input_ids = torch.randint(0, dense_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = model(input_ids)
        # (c) default behavior: full-length logits, no hidden states emitted.
        assert out.logits.shape == (batch, seq, dense_config.vocab_size)
        assert out.hidden_states is None

    def test_default_logits_unchanged_by_modeloutput_wrapping(self, dense_config, backend_config):
        """logits_to_keep=0 + output_hidden_states falsy must match the last
        row of the logits_to_keep=1 path bit-for-bit (no behavioral drift)."""
        model = _dense_model(dense_config, backend_config)
        batch, seq = 1, 5
        input_ids = torch.randint(0, dense_config.vocab_size, (batch, seq))
        with torch.no_grad():
            full = model(input_ids).logits
            last = model(input_ids, logits_to_keep=1).logits
        assert full.shape == (batch, seq, dense_config.vocab_size)
        torch.testing.assert_close(full[:, -1:, :], last)


class TestMoeCutCE:
    def test_supports_logits_to_keep(self, moe_hf_config, backend_config):
        model = _moe_model(moe_hf_config, backend_config)
        assert _supports_logits_to_keep(model) is True

    def test_cut_ce_path_returns_full_hidden_states_and_last_token_logits(self, moe_hf_config, backend_config):
        model = _moe_model(moe_hf_config, backend_config)
        batch, seq = 1, 6
        input_ids = torch.randint(0, moe_hf_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

        assert ("hidden_states" in out) or out.hidden_states is not None
        hidden_states = out.hidden_states
        assert hidden_states.shape == (batch, seq, moe_hf_config.hidden_size)
        assert out.logits.shape == (batch, 1, moe_hf_config.vocab_size)

    def test_default_forward_yields_full_length_logits(self, moe_hf_config, backend_config):
        model = _moe_model(moe_hf_config, backend_config)
        batch, seq = 1, 6
        input_ids = torch.randint(0, moe_hf_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (batch, seq, moe_hf_config.vocab_size)
        assert out.hidden_states is None
