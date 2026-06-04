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

"""Tests that Qwen3_5MoeForConditionalGeneration supports memory-efficient fused
cross-entropy (cut-CE).

The fused (cut-) cross-entropy path in the training recipe
(``nemo_automodel/recipes/llm/train_ft.py``) is only used when the model's forward
(a) exposes a ``logits_to_keep`` parameter and (b) returns an output carrying the
final (full-sequence) hidden states. Otherwise it silently falls back to
``MaskedCrossEntropy``. These tests pin both conditions on a tiny CPU config and
verify that default behavior (full-length logits) is preserved.

The tiny config uses only ``full_attention`` layers so the forward runs on CPU
(the GatedDeltaNet linear-attention kernels require CUDA).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import (
    Qwen3_5MoeForConditionalGeneration,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _text_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        # All full_attention layers so the forward is CPU-runnable.
        layer_types=["full_attention", "full_attention"],
    )


def _vl_config() -> Qwen3_5MoeConfig:
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
    )
    return Qwen3_5MoeConfig(text_config=_text_config().to_dict(), vision_config=vision_cfg)


def _cpu_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _moe_config(text_config: Qwen3_5MoeTextConfig) -> MoEConfig:
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.hidden_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=1,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=text_config.shared_expert_intermediate_size,
    )


@pytest.fixture
def model() -> Qwen3_5MoeForConditionalGeneration:
    text_config = _text_config()
    m = Qwen3_5MoeForConditionalGeneration(
        _vl_config(),
        backend=_cpu_backend(),
        moe_config=_moe_config(text_config),
    )
    # Initialize weights so the forward is finite and deterministic in fp32 (random
    # nn.Module init can yield NaNs / run-to-run variance through the MoE softmax gate
    # on this tiny config).
    m.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return m.to("cpu").to(torch.float32).eval()


def test_supports_logits_to_keep(model):
    """The recipe gates fused-CE on a ``logits_to_keep`` forward parameter."""
    assert _supports_logits_to_keep(model) is True


def test_logits_to_keep_and_hidden_states(model):
    """With ``logits_to_keep=1`` + ``output_hidden_states=True`` the output must
    carry FULL-sequence hidden states while logits cover only the last token."""
    config = model.config.text_config
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        out = model(input_ids=input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are present (mirrors the recipe's ``"hidden_states" in out`` check).
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    # Hidden states span the FULL sequence (required so fused-CE can score every label token).
    assert hidden_states.shape == (batch, seq_len, config.hidden_size)

    # Logits correspond to only the last token.
    assert out.logits.shape == (batch, 1, config.vocab_size)


def test_default_forward_full_length_logits(model):
    """Default call (no logits_to_keep / output_hidden_states) yields full-length logits."""
    config = model.config.text_config
    batch, seq_len = 2, 5
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        out = model(input_ids=input_ids)

    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq_len, config.vocab_size)
    # Default must not leak hidden states.
    assert out.hidden_states is None


def test_logits_to_keep_zero_matches_default(model):
    """``logits_to_keep=0`` must produce logits identical to the default path, and the
    ``logits_to_keep=1`` slice must equal the default path's last-token logits."""
    config = model.config.text_config
    batch, seq_len = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        default_logits = model(input_ids=input_ids).logits
        explicit_zero_logits = model(input_ids=input_ids, logits_to_keep=0).logits
        last_token_logits = model(input_ids=input_ids, logits_to_keep=1).logits

    assert explicit_zero_logits.shape == (batch, seq_len, config.vocab_size)
    torch.testing.assert_close(default_logits, explicit_zero_logits)

    # The kept (last) token's logits match the default path exactly.
    assert last_token_logits.shape == (batch, 1, config.vocab_size)
    torch.testing.assert_close(last_token_logits, default_logits[:, -1:, :])
