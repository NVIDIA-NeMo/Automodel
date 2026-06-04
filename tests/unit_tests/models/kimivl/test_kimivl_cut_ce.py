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

"""CPU tiny-config tests for KimiVL fused cross-entropy (cut-CE) support.

These verify that ``KimiVLForConditionalGeneration.forward`` exposes the
contract the training recipe relies on when it swaps in
``FusedLinearCrossEntropy``:

- ``_supports_logits_to_keep(model)`` is True (forward has a ``logits_to_keep``
  parameter), and
- ``model(logits_to_keep=1, output_hidden_states=True, return_dict=True)``
  returns an output that carries the FINAL hidden states (spanning the full
  sequence) while ``logits`` cover only the requested last token.

The default ``model(input_ids)`` call must still produce full-length logits so
the non-cut-CE (MaskedCrossEntropy) path is unchanged.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

try:
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.kimivl.model import (
        KimiVLConfig,
        KimiVLForConditionalGeneration,
        MoonViTConfig,
    )
    from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.skipif(
        IMPORT_ERROR is not None,
        reason=f"KimiVL / transformers config unavailable: {IMPORT_ERROR}",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA"),
]

VOCAB_SIZE = 128
HIDDEN_SIZE = 64
SEQ_LEN = 6


def _build_tiny_model():
    """Build a tiny CPU KimiVL model with a 2-layer DeepseekV3 text backbone."""
    torch.manual_seed(0)
    text_config = DeepseekV3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        kv_lora_rank=16,
        q_lora_rank=None,
        max_position_embeddings=64,
        torch_dtype="float32",
    )
    config = KimiVLConfig(
        vision_config=MoonViTConfig(
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
        ),
        text_config=text_config,
    )
    backend = BackendConfig(
        linear="torch",
        rms_norm="torch",
        attn="sdpa",
        enable_hf_state_dict_adapter=False,
    )
    model = KimiVLForConditionalGeneration.from_config(config, backend=backend).to(torch.float32)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return _build_tiny_model()


def test_supports_logits_to_keep(tiny_model):
    """The recipe gates cut-CE on a ``logits_to_keep`` parameter in forward."""
    assert _supports_logits_to_keep(tiny_model) is True


def test_cut_ce_output_has_full_hidden_states_and_sliced_logits(tiny_model):
    """logits_to_keep=1 must keep full hidden states but only the last logit."""
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        out = tiny_model(
            input_ids=input_ids,
            logits_to_keep=1,
            output_hidden_states=True,
            return_dict=True,
        )

    # (b) hidden states are carried on the output.
    assert ("hidden_states" in out) or out.hidden_states is not None
    hidden_states = out.hidden_states
    assert hidden_states is not None

    # Hidden states span the FULL sequence (input to lm_head, not sliced).
    assert hidden_states.dim() == 3
    assert hidden_states.shape[0] == 1
    assert hidden_states.shape[1] == SEQ_LEN
    assert hidden_states.shape[2] == HIDDEN_SIZE

    # logits correspond to only the last token.
    logits = getattr(out, "logits", out)
    assert logits.dim() == 3
    assert logits.shape[0] == 1
    assert logits.shape[1] == 1
    assert logits.shape[2] == VOCAB_SIZE


def test_default_forward_returns_full_length_logits(tiny_model):
    """Default call (no logits_to_keep) must still produce full-length logits."""
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        out = tiny_model(input_ids=input_ids)

    # Default path returns a bare tensor (return_dict defaults to False) with
    # logits for every position -- the pre-change behavior.
    logits = getattr(out, "logits", out)
    assert logits.shape[0] == 1
    assert logits.shape[1] == SEQ_LEN
    assert logits.shape[2] == VOCAB_SIZE
