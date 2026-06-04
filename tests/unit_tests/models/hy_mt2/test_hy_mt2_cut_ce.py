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

"""Memory-efficient fused cross-entropy (cut-CE) support for HyMT2ForCausalLM.

The training recipe only enables ``FusedLinearCrossEntropy`` when the model's
forward (a) accepts a ``logits_to_keep`` parameter and (b) returns an output
carrying the final hidden states. These tests assert both contracts on a tiny
CPU config so the recipe will not silently fall back to MaskedCrossEntropy.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_mt2.config import HyMT2Config
from nemo_automodel.components.models.hy_mt2.model import HyMT2ForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

HIDDEN = 64
INTER = 128
MOE_INTER = 64
N_HEADS = 8
N_KV = 2
HEAD_DIM = 16
N_EXPERTS = 4
VOCAB = 128
SEQ_LEN = 6

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


@pytest.fixture
def config():
    return HyMT2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        expert_hidden_dim=MOE_INTER,
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        num_experts=N_EXPERTS,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        router_scaling_factor=2.826,
        route_norm=True,
        moe_router_use_sigmoid=True,
        moe_router_enable_expert_bias=True,
        # Keep the lm_head in fp32 so the model runs on CPU without bf16 matmul.
        enable_lm_head_fp32=False,
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
        fake_balanced_gate=False,
        gate_precision="float32",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        enable_fsdp_optimizations=False,
    )


@pytest.fixture
def model(config, backend_config):
    m = HyMT2ForCausalLM(config, backend=backend_config)
    # Initialize on CPU in fp32 so the MoE/expert weights are real-valued
    # (default-constructed params are uninitialized and would yield NaNs).
    m.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return m


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB, (1, SEQ_LEN))


def test_supports_logits_to_keep(model):
    """Recipe gate (a): forward exposes a ``logits_to_keep`` parameter."""
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_output_contract(model, input_ids):
    """Recipe gate (b): ``logits_to_keep=1`` + ``output_hidden_states=True``
    yields full-length hidden states and last-token-only logits."""
    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # Hidden states must be present (the recipe checks ``"hidden_states" in out``).
    assert ("hidden_states" in out) or (out.hidden_states is not None)
    assert out.hidden_states is not None

    # Final hidden states span the FULL sequence (cut-CE projects these itself).
    assert out.hidden_states.shape[-2] == SEQ_LEN
    assert out.hidden_states.shape[-1] == HIDDEN

    # Logits correspond to only the last token.
    assert out.logits.shape[-2] == 1
    assert out.logits.shape[-1] == VOCAB


def test_default_forward_yields_full_length_logits(model, input_ids):
    """Default behavior (logits_to_keep=0): full-length logits, no hidden states."""
    out = model(input_ids)
    assert out.logits.shape[-2] == SEQ_LEN
    assert out.logits.shape[-1] == VOCAB
    # output_hidden_states defaults to falsy -> hidden states not carried.
    assert out.hidden_states is None


def test_logits_to_keep_matches_full_tail(model, input_ids):
    """The last-token logits from logits_to_keep=1 equal the tail of the full logits."""
    full = model(input_ids).logits
    tail = model(input_ids, logits_to_keep=1).logits
    torch.testing.assert_close(tail[:, -1, :], full[:, -1, :])
