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

"""Regression test: MiniMax-M3-VL vision-tower RoPE ``inv_freq`` stays fp32 after bf16 init.

The text backbone uses the recompute-from-config ``gpt_oss`` rope (no buffer), but
the vision tower (``MiniMaxM3VisionTransformer``) registers ``inv_freq`` as an fp32
buffer that is NOT under a ``rotary_emb``-named module. ``initialize_weights`` casts
the model to bf16 via ``cast_model_to_dtype``; the model lists ``"inv_freq"`` in
``_keep_in_fp32_modules`` (alongside ``"rotary_emb"``) so the vision buffer is
restored to fp32. Runs on CPU.
"""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration

_TINY_TEXT = dict(
    hidden_size=64,
    intermediate_size=32,
    dense_intermediate_size=48,
    shared_intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rotary_dim=8,
    partial_rotary_factor=0.5,
    vocab_size=128,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    num_local_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    moe_layer_freq=[0, 1, 1],
    use_gemma_norm=True,
    use_qk_norm=True,
    qk_norm_type="per_head",
    scoring_func="sigmoid",
    use_routing_bias=True,
    routed_scaling_factor=2.0,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    num_mtp_modules=0,
)
_SPARSE_ATTN = dict(
    use_sparse_attention=True,
    sparse_index_dim=16,
    sparse_num_index_heads=2,
    sparse_topk_blocks=2,
    sparse_block_size=4,
    sparse_score_type="max",
    sparse_init_block=0,
    sparse_local_block=1,
    sparse_attention_freq=[0, 1, 1],
    sparse_disable_index_value=[0, 1, 1],
)
_TINY_VISION = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=64,
    patch_size=2,
    num_channels=3,
    rope_theta=10000.0,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    img_token_compression_config={"spatial_merge_size": 2, "temporal_patch_size": 2},
)


def _cpu_backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


def test_initialize_weights_bf16_keeps_vision_inv_freq_fp32():
    text = {**_TINY_TEXT, "torch_dtype": "float32", "sparse_attention_config": dict(_SPARSE_ATTN)}
    config = MiniMaxM3VLConfig(
        vision_config=dict(_TINY_VISION),
        text_config=text,
        image_token_index=100,
        video_token_index=101,
        projector_hidden_size=_TINY_TEXT["hidden_size"],
    )
    model = MiniMaxM3SparseForConditionalGeneration(config, backend=_cpu_backend()).eval()
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))

    rope_buffers = [
        (name, buf.dtype) for name, buf in model.named_buffers() if "inv_freq" in name or "freqs_cis" in name
    ]
    # The vision tower contributes the only rotary frequency buffer.
    assert rope_buffers, "no inv_freq/freqs_cis buffers found on the model"
    for name, dtype in rope_buffers:
        assert dtype == torch.float32, f"rotary buffer {name} was cast to {dtype}, expected float32"

    assert model.vision_tower.vision_model.inv_freq.dtype == torch.float32
    assert model.lm_head.weight.dtype == torch.bfloat16
