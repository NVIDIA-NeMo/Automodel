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

"""CPU tiny-config tests for memory-efficient fused cross-entropy support.

Verify that ``GlmMoeDsaForCausalLM.forward`` exposes the ``logits_to_keep``
parameter and returns the final hidden states, which together let the
``train_ft`` recipe take the ``FusedLinearCrossEntropy`` (cut-CE) path instead
of silently falling back to ``MaskedCrossEntropy``.
"""

import importlib.util
import sys
import types

import pytest
import torch

# Mock fast_hadamard_transform before importing deepseek_v32 modules (same as the
# sibling tiny-config test); the kernel is optional and unused on CPU.
try:
    import fast_hadamard_transform  # noqa: F401
except ImportError:
    if "fast_hadamard_transform" not in sys.modules:
        mock_hadamard = types.ModuleType("fast_hadamard_transform")
        mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
        mock_hadamard.hadamard_transform = lambda x, scale: x
        sys.modules["fast_hadamard_transform"] = mock_hadamard

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


def _tiny_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
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
def model() -> GlmMoeDsaForCausalLM:
    torch.manual_seed(0)
    config = _tiny_config()
    model = GlmMoeDsaForCausalLM(config, backend=_cpu_backend())
    # Initialize on CPU in float32 so the forward is exercisable without CUDA.
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def test_supports_logits_to_keep(model):
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_output_carries_full_hidden_states_and_last_token_logits(model):
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (a) the output advertises hidden states (recipe checks ``"hidden_states" in out``).
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    # (b) the hidden states span the FULL sequence (cut-CE recomputes per-token logits).
    assert out.hidden_states is not None
    assert out.hidden_states.shape == (batch, seq_len, model.config.hidden_size)

    # (c) logits correspond to only the last token.
    assert out.logits.shape == (batch, 1, model.config.vocab_size)


def test_default_forward_yields_full_length_logits(model):
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    out = model(input_ids)

    # Default behavior: full-length logits, no hidden states emitted.
    assert out.logits.shape == (batch, seq_len, model.config.vocab_size)
    assert ("hidden_states" not in out) or (out.hidden_states is None)


def test_default_logits_match_unsliced_projection(model):
    """logits_to_keep=0 must reproduce the bare full-sequence projection."""
    batch, seq_len = 1, 5
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    out_default = model(input_ids)
    out_keep_last = model(input_ids, logits_to_keep=1)

    # The last-token slice of the full logits must match the logits_to_keep=1 result.
    torch.testing.assert_close(out_default.logits[:, -1:, :], out_keep_last.logits)
