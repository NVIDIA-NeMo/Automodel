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

"""CPU tiny-config checks that Qwen3NextForCausalLM supports memory-efficient
fused cross-entropy (cut-CE / FusedLinearCrossEntropy).

The recipe (``nemo_automodel/recipes/llm/train_ft.py``) only routes through
``FusedLinearCrossEntropy`` when:
  (a) ``_supports_logits_to_keep(model)`` is True (forward exposes a
      ``logits_to_keep`` parameter), AND
  (b) ``model(logits_to_keep=1, **batch)`` returns an output where
      ``"hidden_states" in out`` is True and ``get_final_hidden_states(out)``
      yields the FULL-sequence final hidden states.
Otherwise it silently falls back to ``MaskedCrossEntropy``.
"""

from unittest.mock import patch

import torch
import torch.nn as nn
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

# Import the model module up front so the lazy package loader registers the
# ``qwen3_next`` submodule before we patch attributes on it.
import nemo_automodel.components.models.qwen3_next.model as qwen_mod
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss import rope_utils
from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


# Mock for Qwen3NextGatedDeltaNet to avoid torch.get_current_dtype() / kernel
# requirements on CPU (mirrors tests/.../test_qwen3_next_model.py).
class MockQwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dt_bias = nn.Parameter(torch.ones(config.hidden_size))
        self.A_log = nn.Parameter(torch.zeros(config.hidden_size))
        self.in_proj_qkvz = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.in_proj_ba = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        return torch.zeros_like(hidden_states)


def _tiny_config() -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        partial_rotary_factor=1.0,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    )


def _backend_config() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _build_cpu_model():
    """Instantiate Qwen3NextForCausalLM on CPU with a tiny config.

    ``Qwen3NextModel.__init__`` hardcodes a CUDA device for its rotary
    embedding, so we force ``RotaryEmbedding`` onto CPU for the duration of
    construction. The gated-delta-net layer is mocked out to avoid CPU-hostile
    kernels.
    """
    device = torch.device("cpu")
    config = _tiny_config()
    backend = _backend_config()

    orig_rope_init = rope_utils.RotaryEmbedding.__init__

    def _cpu_rope_init(self, *args, **kwargs):
        kwargs["device"] = device
        return orig_rope_init(self, *args, **kwargs)

    with (
        patch.object(qwen_mod, "Qwen3NextGatedDeltaNet", MockQwen3NextGatedDeltaNet),
        patch.object(rope_utils.RotaryEmbedding, "__init__", _cpu_rope_init),
    ):
        model = Qwen3NextForCausalLM(config, backend=backend).to(device)
        model.initialize_weights(buffer_device=device, dtype=torch.float32)
    return model, config, device


def test_supports_logits_to_keep():
    model, _, _ = _build_cpu_model()
    assert _supports_logits_to_keep(model) is True


def test_logits_to_keep_and_hidden_states():
    model, config, device = _build_cpu_model()
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (b) the recipe checks ``"hidden_states" in out`` (HF ModelOutput membership
    # is True only when the field is present AND non-None).
    assert ("hidden_states" in out) or out.hidden_states is not None

    # Final hidden states must span the FULL sequence (cut-CE projects them itself).
    final_hidden = get_final_hidden_states(out)
    assert final_hidden is not None
    assert final_hidden.shape == (batch, seq_len, config.hidden_size)

    # logits_to_keep=1 -> logits correspond to ONLY the last token.
    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, 1, config.vocab_size)


def test_default_forward_yields_full_length_logits():
    model, config, device = _build_cpu_model()
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

    out = model(input_ids)

    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq_len, config.vocab_size)
