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

These verify that ``LLaVAOneVision1_5_ForConditionalGeneration.forward`` exposes
the contract the ``train_ft`` recipe relies on to enable
``FusedLinearCrossEntropy`` (cut-CE):

  - ``logits_to_keep`` is an accepted forward parameter
    (``_supports_logits_to_keep`` returns True), and
  - with ``output_hidden_states=True`` the output carries the FINAL hidden
    states spanning the full sequence while ``logits`` are restricted to the
    last ``logits_to_keep`` positions.

Default behavior (``logits_to_keep=0``, ``output_hidden_states`` falsy) must be
unchanged: full-length logits and no hidden states.

The model is exercised text-only (no ``pixel_values``); the LM head and the
final-hidden-state plumbing are shared across modalities, so this is sufficient
to validate the cut-CE contract without a vision tower forward.
"""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3.modeling_qwen3")

from nemo_automodel.components.models.llava_onevision.model import (  # noqa: E402
    LLaVAOneVision1_5_ForConditionalGeneration,
    Llavaonevision1_5Config,
    RiceConfig,
)
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states  # noqa: E402
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep  # noqa: E402

# Text-backbone hidden size / vocab the tiny config is built with; the LM head
# projects to the text vocab, so logits last dim == VOCAB_SIZE.
HIDDEN_SIZE = 64
VOCAB_SIZE = 1000


def _tiny_config(**overrides) -> Llavaonevision1_5Config:
    vision_config = RiceConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=14,
        spatial_merge_size=2,
        text_hidden_size=HIDDEN_SIZE,
    )
    text_config = dict(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB_SIZE,
        rope_theta=1e6,
        max_position_embeddings=128,
        use_cache=False,
    )
    config = Llavaonevision1_5Config(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=98,
        vision_end_token_id=99,
        **overrides,
    )
    return config


def _build_model() -> LLaVAOneVision1_5_ForConditionalGeneration:
    model = LLaVAOneVision1_5_ForConditionalGeneration(_tiny_config(), attn_implementation="eager")
    model.eval()
    return model


def test_supports_logits_to_keep():
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_logits_to_keep_and_hidden_states():
    model = _build_model()
    bsz, seq_len = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (bsz, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are surfaced on the output object ...
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    # ... and span the FULL sequence (not just the kept logit positions).
    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    assert final_hidden.shape == (bsz, seq_len, HIDDEN_SIZE)

    # logits correspond to only the last token.
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


def test_default_logits_unchanged_by_output_hidden_states_flag():
    """logits_to_keep=0 must produce identical logits regardless of the flag."""
    model = _build_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))

    with torch.no_grad():
        base = model(input_ids)
        with_hs = model(input_ids, output_hidden_states=True)

    assert base.logits.shape == with_hs.logits.shape
    torch.testing.assert_close(base.logits, with_hs.logits)
