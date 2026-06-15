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

"""Regression test: DiffusionGemma RoPE ``inv_freq`` must stay fp32 after bf16 init.

``DiffusionGemmaTextRotaryEmbedding`` (Gemma4-style) registers per-layer-type
``{type}_inv_freq`` fp32 buffers. ``initialize_weights(dtype=torch.bfloat16)`` casts
the model to bf16 via ``cast_model_to_dtype``; the model lists ``"rotary_emb"`` in
``_keep_in_fp32_modules`` so those buffers are restored to fp32. Runs on CPU.

Skipped unless the ``transformers.models.diffusion_gemma`` fork is importable (it
ships in the NGC container, not on PyPI), matching the other diffusion_gemma tests.
"""

import importlib.util

import pytest
import torch

_FORK_AVAILABLE = importlib.util.find_spec("transformers.models.diffusion_gemma") is not None
_GEMMA4_AVAILABLE = importlib.util.find_spec("transformers.models.gemma4") is not None

pytestmark = pytest.mark.skipif(
    not (_FORK_AVAILABLE and _GEMMA4_AVAILABLE),
    reason="requires the transformers diffusion_gemma fork (NGC container only)",
)


def _tiny_text_config_dict():
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=1,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=4096,
        layer_types=["sliding_attention", "full_attention"],
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        final_logit_softcapping=30.0,
        use_bidirectional_attention="vision",
        attention_bias=False,
        attention_dropout=0.0,
    )


def test_initialize_weights_bf16_keeps_rope_inv_freq_fp32():
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion

    config = DiffusionGemmaConfig(text_config=_tiny_text_config_dict(), vision_config=None, canvas_length=8)
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    model = DiffusionGemmaForBlockDiffusion(config, backend=backend, self_conditioning=True, freeze_router=True)
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))

    rope_buffers = [
        (name, buf.dtype) for name, buf in model.named_buffers() if "inv_freq" in name or "freqs_cis" in name
    ]
    assert rope_buffers, "no inv_freq/freqs_cis buffers found on the model"
    for name, dtype in rope_buffers:
        assert dtype == torch.float32, f"rotary buffer {name} was cast to {dtype}, expected float32"

    assert model.model.layers["0"].self_attn.q_proj.weight.dtype == torch.bfloat16
