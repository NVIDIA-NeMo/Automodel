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

"""CPU tiny-config tests for Step3.7 memory-efficient fused cross-entropy support.

These assert the contract the ``FusedLinearCrossEntropy`` (cut-CE) path in
``recipes/llm/train_ft.py`` relies on:

* ``_supports_logits_to_keep(model)`` is True (forward exposes ``logits_to_keep``).
* ``model(input_ids, logits_to_keep=1, output_hidden_states=True)`` returns an
  output that carries the *final* hidden states spanning the full sequence, and
  ``logits`` restricted to only the last token.
* The default ``model(input_ids)`` call still yields full-length logits.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
from nemo_automodel.components.models.step3p7.model import Step3p7ForConditionalGeneration
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

VOCAB_SIZE = 32
HIDDEN_SIZE = 8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def small_config(**kwargs):
    """Tiny Step3.7 config (mirrors tests/unit_tests/models/step3p7/test_model.py)."""
    values = dict(
        vision_config={
            "width": 8,
            "layers": 0,
            "heads": 2,
            "num_channels": 3,
            "image_size": 8,
            "patch_size": 2,
            "mlp_ratio": 2.0,
            "hidden_act": "gelu",
            "use_ln_pre": False,
            "use_ln_post": False,
            "use_abs_posemb": False,
            "use_rope2d": False,
        },
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_attention_groups": 1,
            "num_hidden_layers": 0,
            "vocab_size": VOCAB_SIZE,
            "moe_num_experts": 2,
            "moe_top_k": 1,
            "moe_intermediate_size": 4,
            "share_expert_dims": 4,
            "head_dim": 4,
            "torch_dtype": "float32",
            "moe_layers_enum": (),
            "layer_types": [],
        },
        image_token_id=31,
    )
    values.update(kwargs)
    return Step3p7Config(**values)


def backend(**kwargs):
    values = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    values.update(kwargs)
    return BackendConfig(**values)


def _build_model():
    model = Step3p7ForConditionalGeneration(small_config(), backend=backend()).float()
    model.eval()
    return model


def test_supports_logits_to_keep():
    """The fused-CE recipe gates on a ``logits_to_keep`` parameter in forward."""
    model = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_output_has_full_hidden_states_and_last_token_logits():
    """logits_to_keep=1 + output_hidden_states=True: full hidden states, last-token logits."""
    model = _build_model()
    seq_len = 5
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # avoid image token id (31)

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # The recipe checks ``"hidden_states" in out`` and reads ``get_final_hidden_states(out)``.
    assert ("hidden_states" in out) or (getattr(out, "hidden_states", None) is not None)

    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    # Hidden states must span the FULL sequence (so cut-CE can recompute logits).
    assert hidden_states.shape == (1, seq_len, HIDDEN_SIZE)

    # logits correspond to only the last token.
    logits = getattr(out, "logits", out)
    assert logits.shape == (1, 1, VOCAB_SIZE)


def test_default_forward_yields_full_length_logits():
    """Default call (logits_to_keep=0, output_hidden_states unset) keeps full-length logits."""
    model = _build_model()
    seq_len = 5
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids)

    logits = getattr(out, "logits", out)
    assert logits.shape == (1, seq_len, VOCAB_SIZE)


def test_cut_ce_last_token_logits_match_full_projection():
    """The last-token logits under logits_to_keep=1 equal the full projection's last slice."""
    model = _build_model()
    seq_len = 4
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        full = model(input_ids, output_hidden_states=True)
        kept = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    full_logits = getattr(full, "logits", full)
    kept_logits = getattr(kept, "logits", kept)
    torch.testing.assert_close(kept_logits[:, -1, :], full_logits[:, -1, :])
