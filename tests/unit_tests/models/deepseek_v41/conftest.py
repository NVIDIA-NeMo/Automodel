# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared tiny DeepSeek V4.1 configs and model builders for the unit tests."""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM


def tiny_config(**overrides) -> DeepseekV41Config:
    """Six-layer config exercising SWA, CSA2 Full / Reindex / Reuse, the candidate pool and Engram.

    Layers: 0 SWA, 1 SWA(+Engram), 2 Full(ratio 2), 3 Reuse, 4 Full(ratio 1, candidate source), 5 Reindex.
    """
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        max_position_embeddings=256,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        sliding_window=8,
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 5],
        index_n_heads=2,
        index_head_dim=32,
        index_topk=6,
        candidate_source_layer_id=4,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        engram_layer_ids=[1],
        engram_num_embeddings=[5000],
        engram_max_ngram_size=3,
        engram_vocab_size=200,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=256,  # == vocab_size -> identity token map
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV41Config(**defaults)


def tiny_backend(**overrides) -> BackendConfig:
    kwargs = dict(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    kwargs.update(overrides)
    return BackendConfig(**kwargs)


def build_tiny_model(config: DeepseekV41Config | None = None, *, seed: int = 0, **backend_overrides):
    """Build a tiny fp32 model with deterministic random weights."""
    config = config or tiny_config()
    model = DeepseekV41ForCausalLM(config, backend=tiny_backend(**backend_overrides))
    model = model.float()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if name.endswith(("attn_hc.scale", "ffn_hc.scale")):
                param.fill_(1.0)
            elif name.endswith(("attn_hc.base", "ffn_hc.base")):
                param.zero_()
            elif "engram" in name and name.endswith(("q_weight", "k_weight")):
                param.fill_(1.0)
            elif "norm" in name and param.dim() == 1:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.zero_()
    model.eval()
    return model
