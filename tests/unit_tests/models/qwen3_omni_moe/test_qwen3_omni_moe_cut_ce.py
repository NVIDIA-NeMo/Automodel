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

"""CPU unit tests for memory-efficient fused cross-entropy support on the
Qwen3 Omni MoE thinker (``logits_to_keep`` + ``output_hidden_states``).

The thinker's HF ``__init__`` allocates the full multimodal stack (vision +
audio towers), so — mirroring ``test_qwen3_omni_moe_model.py`` — these tests
stub the HF ``__init__`` and patch ``model.model.forward`` to return controlled
hidden states. That keeps everything on CPU while still exercising the real
``Qwen3OmniMoeThinkerForConditionalGeneration.forward`` lm_head / output path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration as HFQwen3OmniMoeThinkerForConditionalGeneration,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_omni_moe.model import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


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
def text_config():
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=5000.0,
        router_aux_loss_coef=0.0,
        use_sliding_window=False,
    )
    cfg.torch_dtype = "float32"
    return cfg


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
def thinker_config(text_config):
    vision_config = SimpleNamespace(spatial_merge_size=2)
    # output_hidden_states defaults to falsy so the "default forward" path is
    # exercised exactly as in production.
    return SimpleNamespace(
        text_config=text_config,
        vision_config=vision_config,
        pad_token_id=0,
        output_hidden_states=False,
    )


def _stub_hf_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    config = args[0] if args else kwargs.get("config")
    self.config = config


def _build_model(thinker_config, backend_config, moe_config):
    model = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_config, moe_config=moe_config, backend=backend_config)
    # The stubbed HF __init__ stores the inner SimpleNamespace; restore it so
    # forward's getattr(self.config, "output_hidden_states", ...) resolves.
    model.config = thinker_config
    return model


@patch.object(HFQwen3OmniMoeThinkerForConditionalGeneration, "__init__", new=_stub_hf_init)
@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_supports_logits_to_keep(rotary_cls, thinker_config, backend_config, moe_config):
    rotary_cls.return_value = MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x)))
    model = _build_model(thinker_config, backend_config, moe_config)
    assert _supports_logits_to_keep(model) is True


@patch.object(HFQwen3OmniMoeThinkerForConditionalGeneration, "__init__", new=_stub_hf_init)
@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_logits_to_keep_and_hidden_states(rotary_cls, thinker_config, backend_config, moe_config):
    """With logits_to_keep=1 + output_hidden_states=True the forward must:
    - return hidden states spanning the FULL sequence, and
    - return logits for only the last token.
    This mirrors the recipe's FusedLinearCrossEntropy call
    ``model(logits_to_keep=1, **batch)`` (train_ft.py ~1443-1456).
    """
    rotary_cls.return_value = MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x)))
    model = _build_model(thinker_config, backend_config, moe_config)

    hidden_size = thinker_config.text_config.hidden_size
    vocab_size = thinker_config.text_config.vocab_size
    batch, seq_len = 2, 5

    hidden = torch.randn(batch, seq_len, hidden_size, dtype=model.lm_head.weight.dtype)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    with patch.object(model.model, "forward", return_value=hidden):
        out = model(input_ids=input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states present, full-length, and only-last-token logits.
    assert ("hidden_states" in out) or (getattr(out, "hidden_states", None) is not None)

    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    assert final_hidden.shape == (batch, seq_len, hidden_size)

    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, 1, vocab_size)
    # Last-token logits must equal projecting only the final hidden position.
    expected_last = model.lm_head(hidden[:, -1:, :])
    torch.testing.assert_close(logits, expected_last)


@patch.object(HFQwen3OmniMoeThinkerForConditionalGeneration, "__init__", new=_stub_hf_init)
@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_default_forward_unchanged(rotary_cls, thinker_config, backend_config, moe_config):
    """(c) Default call (no logits_to_keep, no output_hidden_states) still yields
    full-length logits identical to projecting every position, and returns the
    bare tensor as before (default behavior preserved)."""
    rotary_cls.return_value = MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x)))
    model = _build_model(thinker_config, backend_config, moe_config)

    hidden_size = thinker_config.text_config.hidden_size
    vocab_size = thinker_config.text_config.vocab_size
    batch, seq_len = 2, 4

    hidden = torch.randn(batch, seq_len, hidden_size, dtype=model.lm_head.weight.dtype)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    with patch.object(model.model, "forward", return_value=hidden):
        logits = model(input_ids=input_ids)

    # Default path returns a bare tensor (back-compat), full sequence length.
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch, seq_len, vocab_size)
    torch.testing.assert_close(logits, model.lm_head(hidden))
