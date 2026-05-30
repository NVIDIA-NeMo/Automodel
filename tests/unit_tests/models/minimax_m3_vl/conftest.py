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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig

# Tiny config (~layers 0 dense, 1-2 MoE) used across the Stage-1 M3 unit tests.
TINY_CFG = dict(
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
)


@pytest.fixture
def text_config():
    return MiniMaxM3VLTextConfig(torch_dtype="float32", **TINY_CFG)


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


@pytest.fixture
def model(text_config, backend):
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM

    m = MiniMaxM3SparseForCausalLM(text_config, backend=backend).eval()
    m.initialize_weights(dtype=torch.float32)
    return m


# Sparse-attention variant: layer 0 dense attn, layers 1-2 block-sparse (DSA indexer).
# Tiny block_size=4 over seq>=16 exercises real block selection (4 blocks, pick 2).
SPARSE_ATTENTION_CONFIG = dict(
    use_sparse_attention=True,
    sparse_index_dim=16,
    sparse_num_index_heads=2,  # == num_key_value_heads (one idx head per kv head)
    sparse_topk_blocks=2,
    sparse_block_size=4,
    sparse_score_type="max",
    sparse_init_block=0,
    sparse_local_block=1,
    sparse_attention_freq=[0, 1, 1],
    sparse_disable_index_value=[0, 1, 1],
)


@pytest.fixture
def sparse_text_config():
    return MiniMaxM3VLTextConfig(
        torch_dtype="float32", sparse_attention_config=dict(SPARSE_ATTENTION_CONFIG), **TINY_CFG
    )


@pytest.fixture
def sparse_model(sparse_text_config, backend):
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM

    m = MiniMaxM3SparseForCausalLM(sparse_text_config, backend=backend).eval()
    m.initialize_weights(dtype=torch.float32)
    return m


# Tiny vision tower (Conv3d patch 2 + temporal 2; head_dim 8; 2 layers).
VISION_CONFIG = dict(
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
IMAGE_TOKEN_INDEX = 100
VIDEO_TOKEN_INDEX = 101


@pytest.fixture
def vlm_model(backend):
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration

    text = {**TINY_CFG, "torch_dtype": "float32", "sparse_attention_config": dict(SPARSE_ATTENTION_CONFIG)}
    cfg = MiniMaxM3VLConfig(
        vision_config=dict(VISION_CONFIG),
        text_config=text,
        image_token_index=IMAGE_TOKEN_INDEX,
        video_token_index=VIDEO_TOKEN_INDEX,
        projector_hidden_size=TINY_CFG["hidden_size"],
    )
    m = MiniMaxM3SparseForConditionalGeneration(cfg, backend=backend).eval()
    m.initialize_weights(dtype=torch.float32)
    return m
