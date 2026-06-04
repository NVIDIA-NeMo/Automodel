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

"""CPU tiny-config tests for memory-efficient fused cross-entropy (cut-CE) support.

These verify the contract the ``train_ft.py`` recipe relies on to enable
``FusedLinearCrossEntropy`` for ``Qwen2_5OmniThinkerForConditionalGeneration``:

1. ``_supports_logits_to_keep(model)`` is True (forward exposes ``logits_to_keep``).
2. ``model(input_ids, logits_to_keep=1, output_hidden_states=True)`` returns an
   output carrying the FINAL hidden states (spanning the full sequence) and logits
   restricted to the last token.
3. Default ``model(input_ids)`` still produces full-length logits (unchanged behavior).

The forward is exercised text-only (no audio/image/video) so it runs on CPU without
a real checkpoint.
"""

import pytest

torch = pytest.importorskip("torch")

from nemo_automodel.components.training.model_output_utils import get_final_hidden_states  # noqa: E402
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep  # noqa: E402

try:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen2_5_omni.model import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    IMPORT_ERROR = exc


SEQ_LEN = 6
VOCAB_SIZE = 128
HIDDEN_SIZE = 64

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _build_tiny_model():
    """Build a minimal Qwen2.5-Omni Thinker that runs a text-only forward on CPU.

    ``mrope_section`` must sum to ``head_dim // 2`` (here 16 // 2 == 8) for the
    multimodal rotary embedding split to be valid.
    """
    config = Qwen2_5OmniThinkerConfig(
        text_config=dict(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            rope_scaling={"type": "default", "mrope_section": [4, 2, 2]},
        ),
    )
    config.torch_dtype = "float32"

    # No HF state-dict adapter (not needed to exercise the forward).
    backend = BackendConfig(enable_hf_state_dict_adapter=False)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_config(config, backend=backend).to(torch.float32)
    model.eval()
    return model


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import Qwen2.5-Omni model: {IMPORT_ERROR}")
def test_supports_logits_to_keep():
    """forward must expose a ``logits_to_keep`` parameter for cut-CE gating."""
    model = _build_tiny_model()
    assert _supports_logits_to_keep(model) is True


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import Qwen2.5-Omni model: {IMPORT_ERROR}")
def test_cut_ce_outputs_hidden_states_and_last_token_logits():
    """logits_to_keep=1 + output_hidden_states=True yields full hidden states + last-token logits."""
    model = _build_tiny_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # The recipe gates cut-CE on this exact check.
    assert ("hidden_states" in out) or getattr(out, "hidden_states", None) is not None

    final_hidden_states = get_final_hidden_states(out)
    assert final_hidden_states is not None
    # Hidden states span the FULL sequence (lm_head input is not sliced).
    assert final_hidden_states.dim() == 3
    assert final_hidden_states.shape[1] == SEQ_LEN
    assert final_hidden_states.shape[-1] == HIDDEN_SIZE

    # Logits correspond to only the last token.
    logits = getattr(out, "logits", out)
    assert logits.dim() == 3
    assert logits.shape[1] == 1
    assert logits.shape[-1] == VOCAB_SIZE


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import Qwen2.5-Omni model: {IMPORT_ERROR}")
def test_default_forward_full_length_logits():
    """Default forward (no logits_to_keep / output_hidden_states) keeps full-length logits."""
    model = _build_tiny_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        out = model(input_ids)

    logits = getattr(out, "logits", out)
    assert logits.dim() == 3
    assert logits.shape[1] == SEQ_LEN
    assert logits.shape[-1] == VOCAB_SIZE


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import Qwen2.5-Omni model: {IMPORT_ERROR}")
def test_default_logits_match_unsliced_path():
    """logits_to_keep=0 must produce logits identical to the legacy full projection."""
    model = _build_tiny_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        default_logits = getattr(model(input_ids), "logits", None)
        full = model(input_ids, output_hidden_states=True)
        last_token = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # Default logits equal the full projection.
    assert torch.allclose(default_logits, full.logits, atol=1e-5)
    # Last-token logits equal the last position of the full projection.
    assert torch.allclose(last_token.logits, full.logits[:, -1:, :], atol=1e-5)
