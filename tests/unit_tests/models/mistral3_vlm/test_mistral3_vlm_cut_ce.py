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

These verify that ``Mistral3FP8VLMForConditionalGeneration.forward`` exposes the
contract the ``train_ft`` recipe relies on to enable ``FusedLinearCrossEntropy``
(cut-CE):

  - ``logits_to_keep`` is an accepted forward parameter
    (``_supports_logits_to_keep`` returns True), and
  - with ``output_hidden_states=True`` the output carries the FINAL hidden states
    spanning the full sequence while ``logits`` are restricted to the last
    ``logits_to_keep`` positions.

Default behavior (``logits_to_keep=0``, ``output_hidden_states`` falsy) must be
unchanged: full-length logits and no hidden states.

The model is exercised text-only (no ``pixel_values``); the LM head and the
final-hidden-state plumbing are shared across modalities, so this is sufficient
to validate the cut-CE contract without a vision tower forward.

``Mistral3FP8VLMForConditionalGeneration`` subclasses HF's
``Mistral3ForConditionalGeneration`` and only differs from it via the FP8
state-dict adapter / rotary-reinit ``__init__`` and the cut-CE ``forward``
override under test. Those FP8/PP load-path behaviors require a quantized
checkpoint and distributed init, so here we construct the model directly from a
tiny non-quantized config to exercise the forward contract on CPU.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.mistral3.modeling_mistral3")

from transformers.models.mistral3.configuration_mistral3 import Mistral3Config  # noqa: E402

from nemo_automodel.components.models.mistral3_vlm.model import (  # noqa: E402
    Mistral3FP8VLMForConditionalGeneration,
)
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states  # noqa: E402
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep  # noqa: E402

# Text-backbone hidden size / vocab the tiny config is built with; the LM head
# projects to the text vocab, so logits last dim == VOCAB_SIZE.
HIDDEN_SIZE = 32
VOCAB_SIZE = 100


def _tiny_config(**overrides) -> Mistral3Config:
    text_config = dict(
        model_type="ministral3",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=64,
        rope_theta=10000.0,
        use_cache=False,
    )
    vision_config = dict(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        image_size=32,
        patch_size=8,
        num_channels=3,
    )
    return Mistral3Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=99,
        # No quantization_config -> HF's __init__ skips the FP8Linear swap, so the
        # model stands up with plain nn.Linear layers on CPU.
        **overrides,
    )


def _build_model() -> Mistral3FP8VLMForConditionalGeneration:
    torch.manual_seed(0)
    model = Mistral3FP8VLMForConditionalGeneration(_tiny_config())
    return model.to(torch.float32).eval()


def test_supports_logits_to_keep():
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_logits_to_keep_and_hidden_states():
    model = _build_model()
    bsz, seq_len = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are surfaced on the output...
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    # ...and span the FULL sequence (input to lm_head), not just the kept token.
    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    assert final_hidden.shape == (bsz, seq_len, HIDDEN_SIZE)

    # ...while logits correspond to only the last token.
    assert out.logits.shape == (bsz, 1, VOCAB_SIZE)


def test_default_forward_yields_full_length_logits():
    model = _build_model()
    bsz, seq_len = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    # (c) default call: full-length logits, hidden states stay off.
    assert out.logits.shape == (bsz, seq_len, VOCAB_SIZE)
    assert out.hidden_states is None


def test_default_logits_unchanged_by_logits_to_keep_slice():
    """logits_to_keep=0 must produce the full logits; the last row must match the
    logits_to_keep=1 path bit-for-bit (no behavioral drift)."""
    model = _build_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))

    with torch.no_grad():
        full = model(input_ids).logits
        last = model(input_ids, logits_to_keep=1).logits

    assert full.shape == (1, 4, VOCAB_SIZE)
    torch.testing.assert_close(full[:, -1:, :], last)
