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

These verify that ``BaichuanForCausalLM.forward`` exposes the contract the
``train_ft`` recipe relies on to enable ``FusedLinearCrossEntropy`` (cut-CE):

  - ``logits_to_keep`` is an accepted forward parameter
    (``_supports_logits_to_keep`` returns True), and
  - with ``output_hidden_states=True`` the output carries the FINAL hidden
    states spanning the full sequence while ``logits`` are restricted to the
    last ``logits_to_keep`` positions.

Default behavior (``logits_to_keep=0``, ``output_hidden_states`` falsy) must be
unchanged: full-length logits and no hidden states.
"""

import torch

from nemo_automodel.components.models.baichuan.configuration import BaichuanConfig
from nemo_automodel.components.models.baichuan.model import BaichuanForCausalLM
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


def _tiny_config(**overrides) -> BaichuanConfig:
    defaults = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        use_cache=False,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return BaichuanConfig(**defaults)


def _build_model() -> BaichuanForCausalLM:
    model = BaichuanForCausalLM(_tiny_config())
    model.eval()
    return model


def test_supports_logits_to_keep():
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_logits_to_keep_and_hidden_states():
    model = _build_model()
    cfg = model.config
    bsz, seq_len = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are surfaced on the output object ...
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    # ... and span the FULL sequence (not just the kept logit positions).
    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    assert final_hidden.shape == (bsz, seq_len, cfg.hidden_size)

    # logits correspond to only the last token.
    assert out.logits.shape == (bsz, 1, cfg.vocab_size)


def test_default_forward_yields_full_length_logits():
    model = _build_model()
    cfg = model.config
    bsz, seq_len = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    # (c) default call: full-length logits, hidden states stay off.
    assert out.logits.shape == (bsz, seq_len, cfg.vocab_size)
    assert out.hidden_states is None


def test_default_logits_unchanged_by_output_hidden_states_flag():
    """logits_to_keep=0 must produce identical logits regardless of the flag."""
    model = _build_model()
    cfg = model.config
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    with torch.no_grad():
        base = model(input_ids)
        with_hs = model(input_ids, output_hidden_states=True)

    assert base.logits.shape == with_hs.logits.shape
    torch.testing.assert_close(base.logits, with_hs.logits)
