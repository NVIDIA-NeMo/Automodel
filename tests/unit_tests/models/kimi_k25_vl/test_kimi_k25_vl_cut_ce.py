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
``FusedLinearCrossEntropy`` for ``KimiK25VLForConditionalGeneration``:

1. ``_supports_logits_to_keep(model)`` is True (forward exposes ``logits_to_keep``).
2. ``model(input_ids, logits_to_keep=1, output_hidden_states=True)`` returns an
   output carrying the FINAL hidden states (spanning the full sequence) and logits
   restricted to the last token.
3. Default ``model(input_ids)`` still produces full-length logits (unchanged behavior).
"""

import pytest
import torch

from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

try:
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    from nemo_automodel.components.models.kimi_k25_vl.model import (
        BackendConfig,
        KimiK25VLConfig,
        KimiK25VLForConditionalGeneration,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    IMPORT_ERROR = exc


SEQ_LEN = 6
VOCAB_SIZE = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _build_tiny_model():
    """Build a minimal KimiK25VL model that runs a text-only forward on CPU."""
    text_config = DeepseekV3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_group=1,
        topk_group=1,
        max_position_embeddings=64,
    )
    vision_config = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=14,
    )
    config = KimiK25VLConfig(text_config=text_config, vision_config=vision_config)
    config.torch_dtype = torch.float32

    # CPU-friendly backend (sdpa attention, torch linear, torch experts/dispatcher) and
    # no HF state-dict adapter (not needed to exercise the forward).
    backend = BackendConfig(enable_hf_state_dict_adapter=False)
    model = KimiK25VLForConditionalGeneration.from_config(config, backend=backend).to(torch.float32)
    model.model.language_model.init_weights(buffer_device=torch.device("cpu"))
    model.eval()
    return model


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import KimiK25VL model: {IMPORT_ERROR}")
def test_supports_logits_to_keep():
    """forward must expose a ``logits_to_keep`` parameter for cut-CE gating."""
    model = _build_tiny_model()
    assert _supports_logits_to_keep(model) is True


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import KimiK25VL model: {IMPORT_ERROR}")
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

    # Logits correspond to only the last token.
    logits = getattr(out, "logits", out)
    assert logits.dim() == 3
    assert logits.shape[1] == 1
    assert logits.shape[-1] == VOCAB_SIZE


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import KimiK25VL model: {IMPORT_ERROR}")
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


@pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"could not import KimiK25VL model: {IMPORT_ERROR}")
def test_default_logits_match_unsliced_path():
    """logits_to_keep=0 must produce logits identical to the legacy full projection."""
    model = _build_tiny_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        default_logits = getattr(model(input_ids), "logits", model(input_ids))
        full = model(input_ids, output_hidden_states=True)
        last_token = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # Default (bare tensor) logits equal the full projection inside the ModelOutput path.
    assert torch.allclose(default_logits, full.logits, atol=1e-5)
    # Last-token logits equal the last position of the full projection.
    assert torch.allclose(last_token.logits, full.logits[:, -1:, :], atol=1e-5)
