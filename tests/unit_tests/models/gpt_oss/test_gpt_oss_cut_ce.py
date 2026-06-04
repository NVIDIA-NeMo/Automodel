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

"""Memory-efficient fused cross-entropy (cut-CE) support for GptOssForCausalLM.

The training recipe only enables ``FusedLinearCrossEntropy`` when the model's
``forward`` (a) accepts a ``logits_to_keep`` parameter and (b) returns an output
that carries the final hidden states (so the fused kernel can apply ``lm_head``
itself). This module verifies both contracts on a tiny CPU-buildable config.

GPT-OSS only supports ``flex``/``te`` attention and builds its rotary embedding
on the current CUDA device, so a real forward requires a GPU. The tests are
gated on CUDA availability and skip cleanly otherwise; the support-flag check
itself runs anywhere.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.model import GptOssForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _tiny_config():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    return GptOssConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=None,
        layer_types=["full_attention", "sliding_attention"],
        num_local_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        torch_dtype=torch.bfloat16,
    )


def _backend_config():
    # GPT-OSS asserts attn in {"flex", "te"}; flex is the CPU/GPU default in this repo.
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )


def _build_model(device):
    model = GptOssForCausalLM(_tiny_config(), backend=_backend_config())
    model.initialize_weights(device, dtype=torch.bfloat16)
    return model.to(device)


def test_supports_logits_to_keep():
    """forward must expose a ``logits_to_keep`` parameter for the recipe to pick cut-CE."""
    model = GptOssForCausalLM(_tiny_config(), backend=_backend_config())
    assert _supports_logits_to_keep(model) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPT-OSS forward requires CUDA (flex attention)")
def test_cut_ce_hidden_states_and_logits_to_keep():
    """logits_to_keep slices logits to the last token while hidden_states span the full sequence."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = _build_model(device)

    batch_size, seq_len, vocab = 2, 6, model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=device)

    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) hidden states are present and span the FULL sequence.
    assert ("hidden_states" in out) or out.hidden_states is not None
    hidden_states = out.hidden_states
    assert hidden_states is not None
    assert hidden_states.shape == (batch_size, seq_len, model.config.hidden_size)

    # logits correspond to only the last token (logits_to_keep=1).
    assert out.logits.shape == (batch_size, 1, vocab)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPT-OSS forward requires CUDA (flex attention)")
def test_default_forward_full_length_logits():
    """(c) default forward (logits_to_keep=0) still yields full-length logits."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = _build_model(device)

    batch_size, seq_len, vocab = 2, 6, model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=device)

    out = model(input_ids)

    # Default output stays a logits-bearing ModelOutput; callers read getattr(out, "logits", out).
    assert getattr(out, "logits", out).shape == (batch_size, seq_len, vocab)
    # Default does not populate hidden states.
    assert out.hidden_states is None
