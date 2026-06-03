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

"""Memory-efficient fused cross-entropy (cut-CE) support for Gemma4 MoE.

The training recipe only enables ``FusedLinearCrossEntropy`` when the model
forward (a) exposes a ``logits_to_keep`` parameter and (b) returns an output
carrying the FINAL hidden states (so the fused kernel can re-project them).
These CPU tiny-config tests pin that contract for
``Gemma4ForConditionalGeneration`` (the registered MoE arch).

The Gemma4 MoE decoder layer reuses the HF ``Gemma4TextAttention`` module,
whose signature varies across transformers versions; to keep these tests
version-robust (and CPU-only) we stub the text-model backend to return a
full-sequence ``last_hidden_state``, mirroring the existing
``tests/unit_tests/models/gemma4/test_gemma4_model.py`` forward tests. This
still exercises the real forward path under test: the ``logits_to_keep``
gating of the lm_head and the ``CausalLMOutputWithPast`` construction.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

transformers = pytest.importorskip("transformers")

try:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration
    from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

    _GEMMA4_IMPORTABLE = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard for incomplete envs
    _GEMMA4_IMPORTABLE = False
    _IMPORT_ERROR = repr(exc)

pytestmark = pytest.mark.skipif(
    not _GEMMA4_IMPORTABLE,
    reason=f"gemma4 MoE model/config not importable: {_IMPORT_ERROR}",
)

VOCAB_SIZE = 256
HIDDEN_SIZE = 64


def _make_gemma4_config(**text_overrides):
    """Build a tiny Gemma4Config (MoE-enabled) for CPU unit tests."""
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
        layer_types=["full_attention", "full_attention"],
        sliding_window=128,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
    )
    defaults.update(text_overrides)
    return Gemma4Config(text_config=Gemma4TextConfig(**defaults))


def _backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _build_model():
    cfg = _make_gemma4_config()
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend_config())
    return model.to("cpu").to(torch.float32).eval(), cfg.text_config


@contextmanager
def _stub_backend(model, hidden_states):
    """Stub the MoE text-model backend to return a fixed full-sequence hidden state."""
    with patch.object(
        model.model.language_model,
        "forward",
        return_value=MagicMock(last_hidden_state=hidden_states),
    ):
        yield


def test_supports_logits_to_keep():
    model, _ = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_output_has_full_hidden_states_and_last_token_logits():
    model, text_config = _build_model()
    batch, seq = 2, 6
    input_ids = torch.randint(0, text_config.vocab_size, (batch, seq))
    hidden_states = torch.randn(batch, seq, HIDDEN_SIZE)

    with _stub_backend(model, hidden_states):
        with torch.no_grad():
            out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are exposed via the ModelOutput ...
    assert ("hidden_states" in out) or out.hidden_states is not None
    final_hidden = out.hidden_states
    # ... and span the FULL sequence (needed so the fused CE kernel can re-project).
    assert final_hidden is not None
    assert final_hidden.shape == (batch, seq, HIDDEN_SIZE)
    torch.testing.assert_close(final_hidden, hidden_states)

    # logits correspond to only the LAST token (logits_to_keep=1), not the full seq.
    assert out.logits.shape == (batch, 1, text_config.vocab_size)


def test_cut_ce_logits_match_full_projection_last_token():
    """logits_to_keep=1 must equal projecting the full hidden states then slicing."""
    model, text_config = _build_model()
    batch, seq = 2, 5
    input_ids = torch.randint(0, text_config.vocab_size, (batch, seq))
    hidden_states = torch.randn(batch, seq, HIDDEN_SIZE)

    with _stub_backend(model, hidden_states):
        with torch.no_grad():
            out_keep = model(input_ids, logits_to_keep=1, output_hidden_states=True)
            out_full = model(input_ids)

    torch.testing.assert_close(out_keep.logits, out_full.logits[:, -1:, :])


def test_default_forward_returns_full_length_logits():
    model, text_config = _build_model()
    batch, seq = 2, 7
    input_ids = torch.randint(0, text_config.vocab_size, (batch, seq))
    hidden_states = torch.randn(batch, seq, HIDDEN_SIZE)

    with _stub_backend(model, hidden_states):
        with torch.no_grad():
            out = model(input_ids)

    # (c) default call (logits_to_keep=0) yields full-length logits ...
    assert out.logits.shape == (batch, seq, text_config.vocab_size)
    # ... and does not leak hidden states (output_hidden_states defaults to False).
    assert out.hidden_states is None


def test_cut_ce_handles_2d_thd_hidden_states():
    """Packed/THD [T, H] hidden states must slice on dim 0 before the lm_head."""
    model, text_config = _build_model()
    total_tokens = 9
    input_ids = torch.randint(0, text_config.vocab_size, (total_tokens,))
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE)

    with _stub_backend(model, hidden_states):
        with torch.no_grad():
            out = model(input_ids, logits_to_keep=2, output_hidden_states=True)

    # hidden states preserved at full length ...
    assert out.hidden_states.shape == (total_tokens, HIDDEN_SIZE)
    # ... and logits restricted to the last 2 token positions.
    assert out.logits.shape == (2, text_config.vocab_size)
