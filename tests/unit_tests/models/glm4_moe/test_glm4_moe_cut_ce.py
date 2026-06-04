# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Tests that Glm4MoeForCausalLM supports memory-efficient fused cross-entropy.

The fused (cut-) cross-entropy path in the training recipe is only used when the
model's forward (a) exposes a ``logits_to_keep`` parameter and (b) returns an
output carrying the final hidden states. These tests verify both conditions on a
tiny CPU config and that default behavior (full-length logits) is preserved.
"""

import pytest
import torch
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm4_moe.model import Glm4MoeForCausalLM
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _tiny_config() -> Glm4MoeConfig:
    return Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,  # 1 dense layer, rest MoE
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_qk_norm=True,
        partial_rotary_factor=0.5,
        attention_bias=False,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        torch_dtype="float32",
    )


def _cpu_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def model() -> Glm4MoeForCausalLM:
    m = Glm4MoeForCausalLM(_tiny_config(), backend=_cpu_backend())
    # Initialize weights so forward produces finite values (random nn.Module init can
    # yield NaNs through the MoE sigmoid gate on this tiny config).
    m.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return m.to("cpu").to(torch.float32).eval()


def test_supports_logits_to_keep(model):
    """The recipe gates fused-CE on a ``logits_to_keep`` forward parameter."""
    assert _supports_logits_to_keep(model) is True


def test_logits_to_keep_and_hidden_states(model):
    """With ``logits_to_keep=1`` + ``output_hidden_states=True`` the output must
    carry FULL-sequence hidden states while logits cover only the last token."""
    config = model.config
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are present (mirrors the recipe's ``"hidden_states" in out`` check).
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    # Hidden states span the FULL sequence (required so fused-CE can score every label token).
    assert hidden_states.shape == (batch, seq_len, config.hidden_size)

    # Logits correspond to only the last token.
    assert out.logits.shape == (batch, 1, config.vocab_size)


def test_default_forward_full_length_logits(model):
    """Default call (no logits_to_keep / output_hidden_states) yields full-length logits."""
    config = model.config
    batch, seq_len = 2, 5
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    out = model(input_ids)

    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq_len, config.vocab_size)
    # Default must not leak hidden states.
    assert out.hidden_states is None


def test_logits_to_keep_zero_matches_default(model):
    """``logits_to_keep=0`` must produce identical logits to the default path."""
    config = model.config
    batch, seq_len = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        default_logits = model(input_ids).logits
        explicit_logits = model(input_ids, logits_to_keep=0).logits

    assert explicit_logits.shape == (batch, seq_len, config.vocab_size)
    torch.testing.assert_close(default_logits, explicit_logits)
