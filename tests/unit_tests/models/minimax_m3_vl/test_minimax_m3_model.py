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

import torch

from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3MLP
from nemo_automodel.components.moe.layers import MoE


def test_forward_shape_and_finite(model):
    cfg = model.config
    bsz, seq = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (bsz, seq, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_per_layer_dense_vs_moe_selection(model):
    layers = model.model.layers
    # moe_layer_freq=[0, 1, 1] -> layer 0 dense, layers 1-2 MoE.
    assert isinstance(layers["0"].mlp, MiniMaxM3MLP)
    assert layers["0"].shared_experts is None
    for li in ("1", "2"):
        assert isinstance(layers[li].mlp, MoE)
        assert isinstance(layers[li].shared_experts, MiniMaxM3MLP)


def test_per_head_qk_norm_shape(model):
    # per_head QK norm normalizes over head_dim, so the weight is head_dim-sized.
    attn = model.model.layers["0"].self_attn
    assert tuple(attn.q_norm.weight.shape) == (model.config.head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (model.config.head_dim,)


def test_gemma_norm_zero_centered(text_config, backend):
    from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm

    norm = MiniMaxM3RMSNorm(8, eps=1e-6, gemma=True)
    norm.reset_parameters()
    # Gemma weight is zero-centered; effective scale is (1 + weight) == 1 here.
    x = torch.randn(4, 8)
    out = norm(x)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, ref, atol=1e-6)
