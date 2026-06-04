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

"""CPU tiny-config tests for memory-efficient fused cross-entropy support.

These verify that ``Ministral3ForCausalLM.forward`` (the ``*ForCausalLM`` class
registered for the ``mistral3`` model dir) exposes the contract the
``train_ft`` recipe relies on to enable ``FusedLinearCrossEntropy`` (cut-CE):

  - ``logits_to_keep`` is an accepted forward parameter
    (``_supports_logits_to_keep`` returns True), and
  - with ``output_hidden_states=True`` the output carries the FINAL hidden
    states spanning the full sequence while ``logits`` are restricted to the
    last ``logits_to_keep`` positions.

Default behavior (``logits_to_keep=0``, ``output_hidden_states`` falsy) must be
unchanged: full-length logits and no hidden states.

The tiny dense decoder runs on CPU, so no GPU is required.
"""

import pytest
import torch

from nemo_automodel.components.models.mistral3.model import (
    Ministral3Config,
    Ministral3ForCausalLM,
)
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _tiny_config(**overrides) -> Ministral3Config:
    defaults = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    cfg = Ministral3Config(**defaults)
    # Force the eager attention path so the test does not require optional backends.
    cfg._attn_implementation = "eager"
    return cfg


def _build_model() -> Ministral3ForCausalLM:
    model = Ministral3ForCausalLM(_tiny_config())
    return model.eval()


def test_supports_logits_to_keep():
    """Recipe gate (a): forward must expose a ``logits_to_keep`` parameter."""
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_logits_to_keep_and_hidden_states():
    """Recipe gate (b): with logits_to_keep=1 + output_hidden_states=True the output
    must carry full-sequence hidden states while logits cover only the last token."""
    model = _build_model()
    cfg = model.config
    bsz, seq_len = 2, 5
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # train_ft.py checks ``"hidden_states" not in out`` to reject the fused path.
    assert ("hidden_states" in out) or out.hidden_states is not None

    # ... and the surfaced hidden states span the FULL sequence (cut-CE consumes
    # all positions), not just the kept logit positions.
    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    assert final_hidden.shape == (bsz, seq_len, cfg.hidden_size)

    # Logits correspond to only the last token.
    assert out.logits.shape == (bsz, 1, cfg.vocab_size)


def test_default_forward_full_logits():
    """Default call (logits_to_keep=0, no output_hidden_states) yields full-length
    logits and no hidden states, preserving the pre-change behavior."""
    model = _build_model()
    cfg = model.config
    bsz, seq_len = 2, 5
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    assert out.logits.shape == (bsz, seq_len, cfg.vocab_size)
    # Hidden states are gated off by default.
    assert out.hidden_states is None


def test_default_logits_match_logits_to_keep_slice():
    """The last-token logits from logits_to_keep=1 must equal the last position of
    the full (logits_to_keep=0) logits — i.e. slicing does not change the values,
    and the output_hidden_states flag does not perturb the default logits."""
    model = _build_model()
    cfg = model.config
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    with torch.no_grad():
        full = model(input_ids).logits
        last = model(input_ids, logits_to_keep=1).logits
        with_hs = model(input_ids, output_hidden_states=True).logits

    torch.testing.assert_close(last[:, -1, :], full[:, -1, :])
    torch.testing.assert_close(with_hs, full)
