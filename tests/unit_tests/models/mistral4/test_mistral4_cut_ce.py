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

"""CPU tiny-config tests for Mistral4ForCausalLM memory-efficient fused cross-entropy.

These verify the contract the training recipe (``recipes/llm/train_ft.py``) relies on
to enable :class:`FusedLinearCrossEntropy` (cut cross-entropy):

* ``_supports_logits_to_keep(model)`` is ``True`` (forward exposes ``logits_to_keep``).
* ``model(..., logits_to_keep=1, output_hidden_states=True)`` returns an output that
  carries the FINAL hidden states (spanning the full sequence) and logits restricted
  to the last token.
* The default ``model(input_ids)`` call still produces full-length logits.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.model import Mistral4ForCausalLM
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


@pytest.fixture
def text_config():
    """Tiny Mistral4 text config that runs on CPU in float32."""
    cfg = Mock(spec=[])
    cfg.vocab_size = 256
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.q_lora_rank = 32
    cfg.kv_lora_rank = 16
    cfg.qk_nope_head_dim = 8
    cfg.qk_rope_head_dim = 8
    cfg.qk_head_dim = 16
    cfg.v_head_dim = 16
    cfg.n_routed_experts = 4
    cfg.n_shared_experts = 1
    cfg.num_experts_per_tok = 2
    cfg.n_group = 1
    cfg.topk_group = 1
    cfg.first_k_dense_replace = 0
    cfg.norm_topk_prob = True
    cfg.routed_scaling_factor = 1.0
    cfg.max_position_embeddings = 256
    cfg.rms_norm_eps = 1e-6
    cfg.torch_dtype = torch.float32
    cfg.rope_parameters = {
        "type": "yarn",
        "rope_theta": 10000.0,
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "max_position_embeddings": 256,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale_all_dim": 1.0,
        "mscale": 1.0,
        "llama_4_scaling_beta": 0.1,
    }
    cfg.rope_interleave = True
    return cfg


@pytest.fixture
def backend():
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def model(text_config, backend):
    m = Mistral4ForCausalLM(text_config, backend=backend)
    # Initialize on CPU so the tiny MoE produces finite values (random init can NaN).
    m.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    m.eval()
    return m


def test_supports_logits_to_keep(model):
    """The recipe gates cut-CE on this predicate."""
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_output_carries_full_hidden_states_and_last_token_logits(model, text_config):
    seq_len = 8
    input_ids = torch.randint(0, text_config.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) The output must expose hidden states (recipe checks ``"hidden_states" in out``).
    assert ("hidden_states" in out) or out.hidden_states is not None

    final_hidden_states = get_final_hidden_states(out)
    assert final_hidden_states is not None

    # Hidden states span the FULL sequence (input to lm_head), not just the kept token.
    assert final_hidden_states.shape == (1, seq_len, text_config.hidden_size)

    # Logits correspond to ONLY the last token.
    assert out.logits.shape == (1, 1, text_config.vocab_size)

    # The kept logits must equal lm_head applied to the last hidden-state position,
    # i.e. cut-CE slices the same final hidden states the recipe will project itself.
    expected_last = model.lm_head(final_hidden_states[:, -1:, :])
    torch.testing.assert_close(out.logits, expected_last)


def test_default_forward_yields_full_length_logits(model, text_config):
    seq_len = 8
    input_ids = torch.randint(0, text_config.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    # (c) Default call: full-length logits, identical to the pre-change behavior.
    logits = getattr(out, "logits", out)
    assert logits.shape == (1, seq_len, text_config.vocab_size)
    # output_hidden_states defaults to falsy -> no hidden states emitted.
    assert out.hidden_states is None


def test_default_logits_match_explicit_full_projection(model, text_config):
    """logits_to_keep=0 must project ALL positions (no slicing), identical to lm_head(model(...))."""
    input_ids = torch.randint(0, text_config.vocab_size, (1, 8))
    with torch.no_grad():
        ref = model.lm_head(model.model(input_ids))
        produced = model(input_ids).logits
    torch.testing.assert_close(produced, ref)
