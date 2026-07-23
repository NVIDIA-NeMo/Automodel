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

"""Composed-forward equivalence for the MiniMax M3 CP sparse attention.

At ``cp_size == 1`` the CP path is degenerate: the all-gather is the identity and
the load-balanced slots are ``arange``, so it reconstructs the full local sequence.
``MiniMaxM3CPSparseAttention._cp_forward`` (gather + reorder + block selection +
FlexAttention) must then match the eager ``MiniMaxM3Attention.forward`` (additive
block-sparse bias + SDPA) on the same inputs. This guards the CP plumbing against
regressions that the primitive-level tests (slots, doc ids, block selection) miss.

Requires CUDA (FlexAttention) + a process group; ``_cp_forward`` is called directly
because ``forward()`` short-circuits to the eager path when ``cp_size <= 1``.
"""

import os

import pytest
import torch

cuda_available = torch.cuda.is_available()


# FlexAttention's compiled kernel requires the sparse block size to be a multiple of
# its internal BLOCK_M (>= 16) and a supported head_dim, so the cp tests/recipes use
# block_size 16+ (the real model uses 128). The tiny conftest config (block_size 4,
# head_dim 16) trips the kernel, hence this flex-valid variant.
_FLEX_CFG = dict(
    hidden_size=256,
    intermediate_size=64,
    dense_intermediate_size=128,
    shared_intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    rotary_dim=32,
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
    sparse_attention_config=dict(
        use_sparse_attention=True,
        sparse_index_dim=64,
        sparse_num_index_heads=2,
        sparse_topk_blocks=2,
        sparse_block_size=128,
        sparse_score_type="max",
        sparse_init_block=0,
        sparse_local_block=1,
        sparse_attention_freq=[0, 1, 1],
        sparse_disable_index_value=[0, 1, 1],
    ),
)


@pytest.mark.skipif(not cuda_available, reason="CP forward equivalence needs CUDA (FlexAttention)")
def test_cp_forward_cp1_matches_eager(backend):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
    from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import MiniMaxM3CPSparseAttention
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created_pg = not dist.is_initialized()
    if created_pg:
        dist.init_process_group("nccl", rank=0, world_size=1)

    try:
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        cfg = MiniMaxM3VLTextConfig(torch_dtype="float32", **_FLEX_CFG)
        model = MiniMaxM3SparseForCausalLM(cfg, backend=backend).eval().to(device)
        model.initialize_weights(dtype=torch.float32)
        text = model.model

        # Pick a sparse-attention layer (its self_attn is the CP-aware subclass).
        attn = next(
            layer.self_attn for layer in text.layers.values() if isinstance(layer.self_attn, MiniMaxM3CPSparseAttention)
        )

        cp_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("cp",))["cp"]

        bsz, seqlen = 1, 256  # even (load-balanced slots); 2 blocks of block_size=128
        hidden = cfg.hidden_size
        x = torch.randn(bsz, seqlen, hidden, device=device, dtype=torch.float32)
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
        freqs_cis = text.make_freqs_cis(position_ids)

        with torch.no_grad():
            attn._cp_mesh = None  # eager reference (additive bias + SDPA)
            eager = attn(x, freqs_cis=freqs_cis, attention_mask=None)

            attn._cp_mesh = cp_mesh  # CP path (gather + reorder + FlexAttention), cp_size==1
            cp = attn._cp_forward(x, freqs_cis=freqs_cis, position_ids=position_ids)

        assert eager.shape == cp.shape == (bsz, seqlen, hidden)
        assert torch.isfinite(cp).all()
        # The two paths apply the *identical* block selection + causal mask to the
        # same q/k/v; they differ only by kernel (FlexAttention Triton vs SDPA), so in
        # fp32 the mean diff is ~rounding and only a few sparse-softmax-sensitive
        # elements drift. A whole-row discrepancy (mis-selected block) would blow up
        # the mean -- this asserts fundamental equivalence, not bit-identity.
        diff = (cp - eager).abs()
        mean_abs = diff.mean().item()
        max_abs = diff.max().item()
        print(f"[cp1-vs-eager] mean_abs={mean_abs:.2e} max_abs={max_abs:.2e}")
        assert mean_abs < 1e-3, f"mean abs diff too large ({mean_abs:.2e}) -> not the same computation"
        assert max_abs < 5e-2, f"max abs diff too large ({max_abs:.2e})"
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()
