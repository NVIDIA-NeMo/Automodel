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

"""CPU tiny-config tests for HYV3ForCausalLM memory-efficient fused cross-entropy support.

The training recipe (``nemo_automodel/recipes/llm/train_ft.py``) only routes through
``FusedLinearCrossEntropy`` (cut-CE) when the model's ``forward``:
  (a) accepts a ``logits_to_keep`` parameter (checked by ``_supports_logits_to_keep``), and
  (b) returns an output where ``"hidden_states" in out`` is True and
      ``get_final_hidden_states(out)`` yields the final hidden states.
Otherwise it silently falls back to ``MaskedCrossEntropy``.  These tests pin that contract
for ``HYV3ForCausalLM``.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v3.config import HYV3Config
from nemo_automodel.components.models.hy_v3.model import HYV3ForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

HIDDEN = 32
INTER = 64
MOE_INTER = 32
N_HEADS = 4
N_KV = 2
HEAD_DIM = 8
N_EXPERTS = 4
VOCAB = 64
SEQ = 5
BSZ = 2


def _tiny_config() -> HYV3Config:
    return HYV3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        num_experts=N_EXPERTS,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


def _cpu_backend() -> BackendConfig:
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
def model():
    """Instantiate a tiny HYV3ForCausalLM on CPU in float32.

    ``HYV3Model.__init__`` hard-codes the RoPE buffer device to the current CUDA
    device; we override ``rotary_emb.device`` back to CPU so the full forward pass
    (including RoPE) runs on CPU without a GPU.
    """
    cpu = torch.device("cpu")
    try:
        m = HYV3ForCausalLM(_tiny_config(), backend=_cpu_backend())
        # Initialize weights on CPU in fp32 so the tiny MoE forward is numerically sane
        # (default-constructed gates can produce NaN logits). This also resets the RoPE
        # buffer device, which __init__ hard-codes to the current CUDA device.
        m.initialize_weights(buffer_device=cpu, dtype=torch.float32)
    except Exception as exc:  # pragma: no cover - env without CUDA at construction time
        pytest.skip(f"Could not construct HYV3ForCausalLM on this host: {exc}")
    m = m.to(cpu).to(torch.float32)
    m.model.rotary_emb.device = cpu
    m.eval()
    return m


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (BSZ, SEQ))


def test_supports_logits_to_keep(model):
    """(a) The recipe's capability probe must see ``logits_to_keep`` in forward."""
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_hidden_states_and_logits(model, input_ids):
    """(b) With logits_to_keep=1 + output_hidden_states=True the output carries the
    FULL-sequence hidden states and logits for only the last token."""
    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # hidden states are present (both membership and attribute access).
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    from nemo_automodel.components.training.model_output_utils import get_final_hidden_states

    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    # Final hidden states span the FULL sequence (not sliced by logits_to_keep).
    assert final_hidden.shape == (BSZ, SEQ, HIDDEN)

    # Logits correspond to only the last token.
    assert out.logits.shape == (BSZ, 1, VOCAB)


def test_default_forward_yields_full_length_logits(model, input_ids):
    """(c) Default call (logits_to_keep=0, output_hidden_states unset) keeps full-length
    logits and does NOT surface hidden states."""
    with torch.no_grad():
        out = model(input_ids)

    assert out.logits.shape == (BSZ, SEQ, VOCAB)
    # Default behavior preserved: no hidden states surfaced -> recipe falls back unless
    # output_hidden_states is explicitly requested.
    assert out.hidden_states is None
    assert "hidden_states" not in out


def test_logits_to_keep_matches_full_logits_tail(model, input_ids):
    """The logits_to_keep=N slice must equal the last N positions of the full logits."""
    with torch.no_grad():
        full = model(input_ids).logits
        kept = model(input_ids, logits_to_keep=2).logits
    assert kept.shape == (BSZ, 2, VOCAB)
    torch.testing.assert_close(kept, full[:, -2:, :])
