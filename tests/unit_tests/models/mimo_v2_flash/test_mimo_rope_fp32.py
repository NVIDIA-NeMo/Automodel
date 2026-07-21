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

"""Regression test: MiMo-V2-Flash RoPE ``inv_freq`` must stay fp32 after bf16 init.

``MiMoV2FlashRotaryEmbedding`` registers ``inv_freq`` as an fp32 buffer (full and
sliding-window variants). ``initialize_weights(dtype=torch.bfloat16)`` casts the
model to bf16 via ``cast_model_to_dtype``; the model lists ``"rotary_emb"`` in
``_keep_in_fp32_modules_strict`` (matches ``rotary_emb`` + ``swa_rotary_emb``) so
the rotary buffers are restored to fp32. Runs on CPU.
"""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2FlashForCausalLM


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        max_position_embeddings=64,
        layernorm_epsilon=1e-6,
        rope_theta=10000.0,
        swa_rope_theta=10000.0,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        partial_rotary_factor=0.5,
        sliding_window=4,
        sliding_window_size=4,
        attention_chunk_size=4,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_layer_freq=[0, 1],  # layer 0 dense, layer 1 MoE
        hybrid_layer_pattern=[0, 1],  # layer 0 full, layer 1 sliding
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return MiMoV2FlashConfig(**defaults)


def _cpu_backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def test_initialize_weights_bf16_keeps_rope_inv_freq_fp32():
    model = MiMoV2FlashForCausalLM(_tiny_config(), backend=_cpu_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    rope_buffers = [
        (name, buf.dtype) for name, buf in model.named_buffers() if "inv_freq" in name or "freqs_cis" in name
    ]
    # Full-attention + sliding-window rotary both expose inv_freq.
    assert len(rope_buffers) >= 2, f"expected >=2 rotary buffers, found {rope_buffers}"
    for name, dtype in rope_buffers:
        assert dtype == torch.float32, f"rotary buffer {name} was cast to {dtype}, expected float32"

    assert model.lm_head.weight.dtype == torch.bfloat16
