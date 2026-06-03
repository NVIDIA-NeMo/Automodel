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

Verifies that ``MiniMaxM2ForCausalLM.forward`` exposes ``logits_to_keep`` and can
return the final hidden states, which is what the train_ft recipe relies on to use
``FusedLinearCrossEntropy`` (cut-CE) instead of silently falling back to
``MaskedCrossEntropy``.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2ForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


@dataclass
class MockMiniMaxM2Config:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rotary_dim: int = 8
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: dict = None
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    scoring_func: str = "sigmoid"
    use_qk_norm: bool = True
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}


@pytest.fixture
def config():
    return MockMiniMaxM2Config()


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def model(config, backend):
    model = MiniMaxM2ForCausalLM(config, backend=backend)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def test_supports_logits_to_keep(model):
    """The recipe gates cut-CE on a ``logits_to_keep`` parameter in forward."""
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_path_returns_hidden_states_and_last_token_logits(config, model):
    """With ``logits_to_keep=1`` + ``output_hidden_states=True`` the output must carry
    full-sequence hidden states and only the last-token logits."""
    batch, seq = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) the recipe checks ``"hidden_states" in out`` and reads get_final_hidden_states.
    assert ("hidden_states" in out) or out.hidden_states is not None
    hidden_states = out.hidden_states
    assert hidden_states is not None
    # Hidden states span the FULL sequence (so cut-CE can fuse over every token).
    assert hidden_states.shape == (batch, seq, config.hidden_size)
    # Logits correspond to only the last token.
    assert out.logits.shape == (batch, 1, config.vocab_size)


def test_logits_to_keep_int_slices_last_n(config, model):
    """A logits_to_keep > 1 keeps exactly the last N positions across both dims."""
    batch, seq = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=3)

    assert out.logits.shape == (batch, 3, config.vocab_size)


def test_default_forward_yields_full_length_logits(config, model):
    """Default call (logits_to_keep=0, output_hidden_states falsy) preserves prior behavior:
    full-length logits and no hidden states attached."""
    batch, seq = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    with torch.no_grad():
        out = model(input_ids)

    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq, config.vocab_size)
    assert out.hidden_states is None


def test_thd_returns_full_hidden_states_and_unsqueezed_logits(config, model):
    """THD (packed 2D) path: hidden states span the full token axis and logits keep the
    ``[1, T, V]`` batch dim, with last-token slicing applied on the 2D hidden states.

    The inner model is mocked to return a 2D ``[T, H]`` tensor (the THD representation);
    this isolates the ForCausalLM logits/hidden-states handling from the packed-attention
    kernel, which cannot run THD on CPU with the SDPA backend.
    """
    total_tokens = 7
    input_ids = torch.randint(0, config.vocab_size, (1, total_tokens))
    position_ids = torch.arange(total_tokens).unsqueeze(0)
    padding_mask = torch.zeros(1, total_tokens, dtype=torch.bool)
    cu_seqlens = torch.tensor([[0, 4, total_tokens]], dtype=torch.int32)

    hidden_2d = torch.randn(total_tokens, config.hidden_size, dtype=torch.float32)
    with torch.no_grad(), patch.object(model.model, "forward", return_value=hidden_2d):
        out = model(
            input_ids,
            position_ids=position_ids,
            padding_mask=padding_mask,
            logits_to_keep=1,
            output_hidden_states=True,
            qkv_format="thd",
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens,
            max_seqlen=torch.tensor([4]),
        )

    # Hidden states are the 2D [T, H] packed representation spanning all tokens.
    assert out.hidden_states is not None
    assert out.hidden_states.shape == (total_tokens, config.hidden_size)
    # Logits keep the THD batch dim and only the last token.
    assert out.logits.shape == (1, 1, config.vocab_size)
