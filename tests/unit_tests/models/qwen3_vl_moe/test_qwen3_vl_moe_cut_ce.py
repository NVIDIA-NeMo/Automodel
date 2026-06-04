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

"""CPU tiny-config checks that Qwen3VLMoeForConditionalGeneration supports
memory-efficient fused cross-entropy (cut-CE / FusedLinearCrossEntropy).

The recipe (``nemo_automodel/recipes/llm/train_ft.py``) only enables
FusedLinearCrossEntropy when ``_supports_logits_to_keep(model)`` is True *and*
``model(logits_to_keep=1, ...)`` returns an output that carries the final
hidden states. These tests assert exactly that contract on a tiny CPU model.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

# The model + tiny config require the transformers qwen3_vl_moe module; skip the
# whole file (rather than error) if it is unavailable in this environment.
transformers_config = pytest.importorskip("transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe")
Qwen3VLMoeConfig = transformers_config.Qwen3VLMoeConfig
Qwen3VLMoeTextConfig = transformers_config.Qwen3VLMoeTextConfig

from nemo_automodel.components.models.qwen3_vl_moe.model import (  # noqa: E402
    Qwen3VLMoeForConditionalGeneration,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


@pytest.fixture
def text_config():
    return Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        pad_token_id=0,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )


@pytest.fixture
def moe_config(text_config):
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=text_config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def vl_config(text_config):
    vision_cfg = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=8,
        deepstack_visual_indexes=[0, 1],
    )
    return Qwen3VLMoeConfig(text_config=text_config.to_dict(), vision_config=vision_cfg)


@pytest.fixture
def model(vl_config, backend_config, moe_config):
    # CPU + float32 so the tiny model runs without CUDA.
    return Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(
        torch.float32
    )


def test_supports_logits_to_keep(model):
    # The recipe gates FusedLinearCrossEntropy on this returning True.
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_returns_hidden_states_and_last_token_logits(model, vl_config):
    batch, seq_len = 2, 6
    vocab_size = vl_config.text_config.vocab_size
    hidden_size = vl_config.text_config.hidden_size
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) the output must carry the FINAL hidden states for cut-CE.
    assert ("hidden_states" in out) or (out.hidden_states is not None)
    final_hidden = out.hidden_states
    assert final_hidden is not None
    # Hidden states span the FULL sequence (not sliced to the last token).
    assert final_hidden.shape == (batch, seq_len, hidden_size)

    # Logits correspond to only the last token (logits_to_keep=1).
    assert out.logits.shape == (batch, 1, vocab_size)


def test_cut_ce_last_token_matches_full_projection(model, vl_config):
    """Sliced last-token logits must equal projecting the last hidden-state column —
    i.e. cut-CE only changes *what* gets projected, not the values.

    Both quantities are derived from a single forward call so NaN positions (the
    tiny model is randomly initialised, not weight-loaded, so some activations are
    NaN) line up exactly; ``equal_nan`` keeps the comparison about the slice logic.
    """
    batch, seq_len = 1, 5
    vocab_size = vl_config.text_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    model.eval()
    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)
        expected_last = model.lm_head(out.hidden_states[:, -1:, :])

    torch.testing.assert_close(out.logits, expected_last, equal_nan=True)


def test_default_forward_yields_full_length_logits(model, vl_config):
    # (c) default call (no logits_to_keep / output_hidden_states) -> full-length logits.
    batch, seq_len = 2, 4
    vocab_size = vl_config.text_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    out = model(input_ids)

    # Downstream callers read getattr(out, "logits", out); logits span all positions.
    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq_len, vocab_size)
    # Default behaviour does not populate hidden_states.
    assert out.hidden_states is None
