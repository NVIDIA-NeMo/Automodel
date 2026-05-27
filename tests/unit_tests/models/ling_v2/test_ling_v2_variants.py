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

"""Per-variant smoke tests: every Ling 2.0 size class (mini / flash / 1T) is
exercised at tiny scale.  This catches variant-specific bugs (such as Ling-1T's
``first_k_dense_replace=4`` or its ``rotary_dim``-instead-of-``partial_rotary_factor``
config) that would otherwise be invisible to a single-variant test."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
from nemo_automodel.components.models.ling_v2.model import BailingMoeV2ForCausalLM
from nemo_automodel.components.moe.config import MoEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for forward smoke")


def _tiny_cfg_for_variant(variant: str) -> BailingMoeV2Config:
    """Tiny replica of each variant's ratio of dense-vs-MoE layers and rope layout."""
    common = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    if variant == "mini":
        return BailingMoeV2Config(num_hidden_layers=2, first_k_dense_replace=1, partial_rotary_factor=0.5, **common)
    if variant == "flash":
        return BailingMoeV2Config(num_hidden_layers=4, first_k_dense_replace=1, partial_rotary_factor=0.5, **common)
    if variant == "1T":
        # Ling-1T sets rotary_dim (not partial_rotary_factor) and uses 4 dense layers.
        return BailingMoeV2Config(num_hidden_layers=6, first_k_dense_replace=4, rotary_dim=8, **common)
    raise ValueError(variant)


def _backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )


def _moe_cfg(cfg: BailingMoeV2Config) -> MoEConfig:
    return MoEConfig(
        dim=cfg.hidden_size,
        inter_dim=cfg.intermediate_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_routed_experts=cfg.num_experts,
        n_shared_experts=cfg.num_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        n_expert_groups=cfg.n_group,
        n_limited_groups=cfg.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.0,
        force_e_score_correction_bias=True,
        score_func=cfg.score_function,
        route_scale=cfg.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=cfg.norm_topk_prob,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        shared_expert_inter_dim=cfg.moe_intermediate_size,
        shared_expert_activation="swiglu",
        softmax_before_topk=False,
        dtype=torch.float32,
    )


def test_pp_keep_self_forward_is_declared():
    """Pipeline parallelism opts out of patch_hf_model_for_pp via this class flag.
    Without it, the framework replaces our forward with a generic HF-style one
    that calls rotary_emb(hidden_states, position_ids) and crashes inside
    apply_rotary_emb because our gpt_oss-style RotaryEmbedding.forward(query, key)
    rotates q and k rather than returning (cos, sin).  See PR #2255 for the
    original symptom (RuntimeError: Sizes of tensors must match... at torch.cat
    in apply_rotary_emb)."""
    assert getattr(BailingMoeV2ForCausalLM, "_pp_keep_self_forward", False) is True, (
        "BailingMoeV2ForCausalLM must declare _pp_keep_self_forward = True so "
        "the PP wrapper preserves the model's own freqs_cis-based forward."
    )


@pytest.mark.parametrize("variant", ["mini", "flash", "1T"])
def test_tiny_variant_forward(variant):
    cfg = _tiny_cfg_for_variant(variant)
    # All three variants must end up with half-RoPE regardless of how the
    # checkpoint expresses it.
    assert cfg.partial_rotary_factor == 0.5
    rope_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    assert rope_dim == 8

    model = BailingMoeV2ForCausalLM(cfg, moe_config=_moe_cfg(cfg), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cuda:0"), dtype=torch.float32)
    model = model.to("cuda:0").eval()

    # Confirm the dense-vs-MoE split matches first_k_dense_replace.
    from nemo_automodel.components.moe.layers import MLP, MoE

    dense_layers = sum(1 for i in range(cfg.num_hidden_layers) if isinstance(model.model.layers[str(i)].mlp, MLP))
    moe_layers = sum(1 for i in range(cfg.num_hidden_layers) if isinstance(model.model.layers[str(i)].mlp, MoE))
    assert dense_layers == cfg.first_k_dense_replace, (
        f"{variant}: expected {cfg.first_k_dense_replace} dense layers, got {dense_layers}"
    )
    assert dense_layers + moe_layers == cfg.num_hidden_layers

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 16), device="cuda:0")
    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all(), f"{variant}: non-finite logits"
    # The model should not collapse to a constant prediction.
    assert logits.argmax(dim=-1).unique().numel() > 1, f"{variant}: degenerate (constant) argmax"
