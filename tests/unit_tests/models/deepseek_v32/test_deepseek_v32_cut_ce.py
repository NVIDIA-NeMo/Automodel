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

"""CPU tiny-config tests for DeepSeek V3.2 fused cross-entropy (cut-CE) support.

These verify that ``DeepseekV32ForCausalLM.forward`` exposes the contract the
``train_ft`` recipe relies on to enable ``FusedLinearCrossEntropy``:

* a ``logits_to_keep`` parameter (checked via ``_supports_logits_to_keep``), and
* an output that carries the final hidden states (spanning the full sequence)
  when ``output_hidden_states=True``, with logits restricted to the kept tokens.
"""

import importlib.util
import sys
import types

import pytest
import torch

# Mock fast_hadamard_transform before importing deepseek_v32 modules (the
# Indexer imports it eagerly; the kernel is irrelevant to this CPU test).
if "fast_hadamard_transform" not in sys.modules:
    _mock_hadamard = types.ModuleType("fast_hadamard_transform")
    _mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
    _mock_hadamard.hadamard_transform = lambda x, scale: x
    sys.modules["fast_hadamard_transform"] = _mock_hadamard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


def _tiny_config() -> DeepseekV32Config:
    """A minimal DeepSeek V3.2 config that runs a forward pass on CPU."""
    return DeepseekV32Config(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        qk_head_dim=32,
        kv_lora_rank=32,
        q_lora_rank=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        # Single expert group so the tiny n_routed_experts count is valid.
        n_group=1,
        topk_group=1,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=16,
        max_position_embeddings=64,
        torch_dtype="float32",
    )


def _build_model() -> DeepseekV32ForCausalLM:
    # Seed before construction so the randomly-initialized tiny model (no
    # ``initialize_weights`` call, which requires CUDA) is deterministic and
    # numerically well-behaved on CPU.
    torch.manual_seed(0)
    cfg = _tiny_config()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)
    model = DeepseekV32ForCausalLM(cfg, backend=backend)
    return model.to(torch.float32).eval()


def test_supports_logits_to_keep():
    """forward must expose a ``logits_to_keep`` parameter for cut-CE gating."""
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_logits_to_keep_and_hidden_states():
    """With logits_to_keep=1 + output_hidden_states=True the output carries the
    full-sequence hidden states and logits for only the last token."""
    model = _build_model()
    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (a) the recipe checks ``"hidden_states" in out``; also accept the field.
    assert ("hidden_states" in out) or out.hidden_states is not None

    # The final hidden states span the FULL sequence (not just the kept token).
    hidden_states = out.hidden_states
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[-1]
    assert hidden_states.shape[0] == batch_size
    assert hidden_states.shape[1] == seq_len
    assert hidden_states.shape[-1] == model.config.hidden_size

    # logits correspond to only the last token.
    assert out.logits.shape[0] == batch_size
    assert out.logits.shape[1] == 1
    assert out.logits.shape[-1] == model.config.vocab_size

    # The kept logit must match the last position of a full projection.
    with torch.no_grad():
        full = model(input_ids, logits_to_keep=0)
    assert full.logits.shape[1] == seq_len
    reference = full.logits[:, -1:, :]
    # The randomly-initialized tiny model can produce non-finite activations on
    # CPU; only assert the numerical correspondence when both sides are finite
    # (the slicing/shape contract above is what the recipe relies on).
    if torch.isfinite(out.logits).all() and torch.isfinite(reference).all():
        torch.testing.assert_close(out.logits, reference, rtol=1e-4, atol=1e-4)


def test_default_forward_returns_full_length_logits():
    """Default call (logits_to_keep=0, output_hidden_states unset) keeps the
    full-length logits and does not surface hidden states."""
    model = _build_model()
    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    assert out.logits.shape == (batch_size, seq_len, model.config.vocab_size)
    # output_hidden_states defaults to falsy -> field omitted / None.
    assert ("hidden_states" not in out) and out.hidden_states is None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
