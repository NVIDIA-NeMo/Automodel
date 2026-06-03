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

"""CPU unit tests for memory-efficient fused cross-entropy (cut-CE) support in
``DeepseekV3ForCausalLM``.

These exercise the contract the training recipe relies on when
``FusedLinearCrossEntropy`` is selected (see ``nemo_automodel/recipes/llm/train_ft.py``):

1. ``_supports_logits_to_keep(model)`` is ``True`` (a ``logits_to_keep`` param exists).
2. ``model(input_ids, logits_to_keep=1, output_hidden_states=True)`` returns an output
   where ``"hidden_states" in out`` (or ``out.hidden_states is not None``), the hidden
   states span the FULL sequence, and the logits correspond to only the last token.
3. The default ``model(input_ids)`` call still yields full-length logits.

A tiny all-dense config (``first_k_dense_replace == num_hidden_layers``) is used so the
forward runs on CPU without the CUDA-only MoE / grouped-GEMM path.
"""

import pytest
import torch

from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

# The model + config import may fail in minimal environments (no transformers
# DeepSeek V3 config, missing optional deps). Keep the test importable and skip
# cleanly in that case rather than erroring at collection.
transformers_config = pytest.importorskip(
    "transformers.models.deepseek_v3.configuration_deepseek_v3",
    reason="transformers DeepseekV3Config unavailable",
)
DeepseekV3Config = transformers_config.DeepseekV3Config

try:
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"DeepseekV3ForCausalLM unavailable: {exc}", allow_module_level=True)


VOCAB_SIZE = 64
HIDDEN_SIZE = 32


def _tiny_config() -> "DeepseekV3Config":
    """Build a tiny CPU-runnable DeepSeek V3 config.

    ``first_k_dense_replace == num_hidden_layers`` makes every block use a dense
    MLP (no MoE), avoiding the CUDA-only grouped-GEMM path so the forward runs on
    CPU.
    """
    return DeepseekV3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        kv_lora_rank=16,
        q_lora_rank=None,
        first_k_dense_replace=2,  # == num_hidden_layers -> all dense, no MoE
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=64,
        torch_dtype="float32",
    )


def _build_model() -> "DeepseekV3ForCausalLM":
    config = _tiny_config()
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = DeepseekV3ForCausalLM(config, backend=backend).to(torch.float32)
    model.eval()
    return model


def test_supports_logits_to_keep():
    """The forward must expose a ``logits_to_keep`` parameter for cut-CE selection."""
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_output_contract():
    """logits_to_keep=1 + output_hidden_states=True -> full-length hidden states, last-token logits."""
    model = _build_model()
    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (a) hidden states are carried on the output (recipe checks `"hidden_states" in out`).
    assert ("hidden_states" in out) or out.hidden_states is not None

    # (b) hidden states span the FULL sequence (cut-CE projects the full hidden states).
    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    assert hidden_states.shape == (batch_size, seq_len, HIDDEN_SIZE)

    # (c) logits correspond to only the last token.
    assert out.logits.shape == (batch_size, 1, VOCAB_SIZE)


def test_logits_to_keep_matches_full_last_token():
    """logits_to_keep=1 must equal the last-token slice of the full-projection logits."""
    model = _build_model()
    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

    with torch.no_grad():
        full = model(input_ids, logits_to_keep=0)
        last = model(input_ids, logits_to_keep=1)

    assert full.logits.shape == (batch_size, seq_len, VOCAB_SIZE)
    assert last.logits.shape == (batch_size, 1, VOCAB_SIZE)
    torch.testing.assert_close(full.logits[:, -1:, :], last.logits)


def test_default_forward_yields_full_length_logits():
    """Default call (no logits_to_keep, no output_hidden_states) keeps prior behavior."""
    model = _build_model()
    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    # Full-length logits, identical shape to the pre-change bare-tensor return.
    assert out.logits.shape == (batch_size, seq_len, VOCAB_SIZE)
    # Hidden states are not carried unless explicitly requested.
    assert out.hidden_states is None
