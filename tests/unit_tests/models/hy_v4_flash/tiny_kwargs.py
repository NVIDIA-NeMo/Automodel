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

"""Pure-data tiny Hy4-flash text configuration shared by the AutoModel tests and the reference dump script."""

TINY_TEXT_KWARGS = dict(
    vocab_size=300,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    routed_expert_hidden_size=32,
    num_hidden_layers=6,
    num_attention_heads=4,
    swa_num_attention_heads=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    swa_qk_nope_head_dim=16,
    swa_qk_rope_head_dim=8,
    swa_v_head_dim=16,
    sliding_window=8,
    index_topk=8,
    index_head_dim=16,
    index_n_heads=4,
    compressed_indexer_compress_ratio=2,
    engram_layer_ids=[2, 4],
    engram_max_ngram_size=3,
    engram_n_head_per_ngram=2,
    engram_n_embed_per_ngram=16,
    engram_layer_head_vocab_sizes={str(i): [[1009, 1013], [1019, 1021]] for i in (2, 4)},
    engram_fp8_embedding=False,  # the PR config defaults to True; the released config.json sets False
    engram_pad_id=2,
    engram_lookup_table_size=290,
    rope_parameters={
        "sliding_attention": {"rope_type": "default", "rope_theta": 1.0e4, "mrope_section": [2, 1, 1]},
        "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1.0e7, "mrope_section": [2, 1, 1]},
    },
    pad_token_id=2,
    bos_token_id=0,
    eos_token_id=1,
)
