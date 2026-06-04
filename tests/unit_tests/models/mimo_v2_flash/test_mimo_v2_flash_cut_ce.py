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

"""Memory-efficient fused cross-entropy (cut-CE) support for MiMoV2Flash.

The ``train_ft`` recipe only uses ``FusedLinearCrossEntropy`` when the model's
forward (a) exposes a ``logits_to_keep`` parameter and (b) returns an output
whose ``hidden_states`` carries the FULL-sequence final hidden states while the
logits cover only the last token(s). These tests pin both halves of that
contract on a tiny CPU config.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2FlashForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


@pytest.fixture
def tiny_config():
    """Tiny config exercising the dense-then-MoE pattern across full+sliding layers."""
    return MiMoV2FlashConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        max_position_embeddings=64,
        layernorm_epsilon=1e-6,
        rope_theta=10000.0,
        swa_rope_theta=10000.0,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        partial_rotary_factor=0.5,
        sliding_window=4,
        sliding_window_size=4,
        attention_chunk_size=4,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_layer_freq=[0, 1, 1, 1],  # layer 0 dense; rest MoE
        hybrid_layer_pattern=[0, 1, 0, 1],  # alternating full/sliding
        torch_dtype="float32",
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


def _model(tiny_config, backend_config):
    torch.manual_seed(0)
    model = MiMoV2FlashForCausalLM(tiny_config, backend=backend_config).to(torch.float32).eval()
    # Initialize on CPU so the (otherwise uninitialized) attention-sink buffer and
    # embeddings hold finite values, keeping the numerical assertions meaningful.
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model


def test_supports_logits_to_keep(tiny_config, backend_config):
    """(a) The recipe gates cut-CE on a literal ``logits_to_keep`` forward param."""
    model = _model(tiny_config, backend_config)
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_hidden_states_and_last_token_logits(tiny_config, backend_config):
    """(b) hidden_states span the full sequence; logits cover only the last token."""
    model = _model(tiny_config, backend_config)
    batch, seq = 1, 6
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # hidden_states must be present (the recipe checks both "in out" and the attr).
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    final_hidden_states = out.hidden_states
    # Final hidden states span the FULL sequence so cut-CE can recompute the
    # lm_head projection over every position.
    assert final_hidden_states.shape == (batch, seq, tiny_config.hidden_size)

    # Logits correspond to ONLY the last token (logits_to_keep=1).
    assert out.logits.shape == (batch, 1, tiny_config.vocab_size)

    # The retained logits must equal projecting the last hidden state directly.
    expected_last = model.lm_head(final_hidden_states[:, -1:, :])
    torch.testing.assert_close(out.logits, expected_last)


def test_default_forward_yields_full_length_logits(tiny_config, backend_config):
    """(c) Default forward (logits_to_keep=0) still produces full-length logits."""
    model = _model(tiny_config, backend_config)
    batch, seq = 1, 6
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))

    with torch.no_grad():
        out = model(input_ids)

    assert out.logits.shape == (batch, seq, tiny_config.vocab_size)
    # Default behavior does not attach hidden states.
    assert out.hidden_states is None


def test_logits_unchanged_vs_full_projection(tiny_config, backend_config):
    """logits_to_keep=0 logits are identical to projecting all hidden states."""
    model = _model(tiny_config, backend_config)
    batch, seq = 1, 5
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        ref_logits = model.lm_head(out.hidden_states)

    torch.testing.assert_close(out.logits, ref_logits)
