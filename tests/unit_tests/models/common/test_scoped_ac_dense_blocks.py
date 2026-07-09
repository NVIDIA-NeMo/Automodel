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

"""Scoped activation checkpointing on REAL dense-MLP blocks.

The scoped AC modes ("non_moe"/"non_moe_no_attn") wrap block submodules —
including the dense MLP of e.g. GLM ``first_k_dense_replace`` layers — in
``CheckpointWrapper``. Each custom-MoE block's ``_mlp`` dispatches on the MLP
type, so it must unwrap the checkpoint wrapper before the ``isinstance``
check; without the unwrap the dense block crashes with ``AssertionError`` at
the first forward. The DummyBlock-based parallelizer tests replace both the
blocks and ``ptd_checkpoint_wrapper`` with mocks and cannot catch this, so
these tests run a real block's ``_mlp`` forward+backward through the
production scoped-AC wrapping applied by ``apply_ac``.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.parallelizer import apply_ac

HIDDEN = 64
INTER = 128


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _moe_config() -> MoEConfig:
    """Tiny MoE config; unused by dense blocks but required by the Block signature."""
    return MoEConfig(
        dim=HIDDEN,
        inter_dim=INTER,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        norm_topk_prob=False,
    )


def _glm4_moe_dense_block() -> nn.Module:
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    from nemo_automodel.components.models.glm4_moe.model import Block

    config = Glm4MoeConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=INTER,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
    )
    return Block(layer_idx=0, config=config, moe_config=_moe_config(), backend=_backend())


@dataclass
class _Glm4MoeLiteConfig:
    """Minimal GLM4-MoE-Lite config with the MLA fields the Block requires."""

    vocab_size: int = 128
    hidden_size: int = HIDDEN
    num_hidden_layers: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 64
    torch_dtype: str = "bfloat16"
    num_attention_heads: int = 4
    q_lora_rank: int = 16
    kv_lora_rank: int = 8
    qk_nope_head_dim: int = 8
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16
    rope_scaling: dict | None = None
    intermediate_size: int = INTER
    moe_intermediate_size: int = 32
    n_routed_experts: int = 4
    n_shared_experts: int = 0
    num_experts_per_tok: int = 2
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    mlp_layer_types: list = field(default_factory=lambda: ["dense", "sparse"])
    rope_parameters: dict = field(default_factory=lambda: {"rope_theta": 10000.0, "rope_type": "default"})


def _glm4_moe_lite_dense_block() -> nn.Module:
    from nemo_automodel.components.models.glm4_moe_lite.model import Block

    return Block(layer_idx=0, config=_Glm4MoeLiteConfig(), moe_config=_moe_config(), backend=_backend())


def _glm_moe_dsa_dense_block() -> nn.Module:
    pytest.importorskip("transformers.models.glm_moe_dsa")
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    # GLM DSA layers import fast_hadamard_transform at module import time.
    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError:
        if "fast_hadamard_transform" not in sys.modules:
            mock_hadamard = types.ModuleType("fast_hadamard_transform")
            mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
            mock_hadamard.hadamard_transform = lambda x, scale: x
            sys.modules["fast_hadamard_transform"] = mock_hadamard

    from nemo_automodel.components.models.glm_moe_dsa.model import Block

    config = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=INTER,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense", "sparse"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    return Block(layer_idx=0, config=config, moe_config=_moe_config(), backend=_backend())


def _ling_v2_dense_block() -> nn.Module:
    from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
    from nemo_automodel.components.models.ling_v2.model import Block

    config = BailingMoeV2Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
    )
    return Block(layer_idx=0, config=config, moe_config=_moe_config(), backend=_backend())


def _hy_v3_dense_block() -> nn.Module:
    from nemo_automodel.components.models.hy_v3.config import HYV3Config
    from nemo_automodel.components.models.hy_v3.model import Block

    config = HYV3Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    return Block(layer_idx=0, config=config, moe_config=_moe_config(), backend=_backend())


def _hy_mt2_dense_block() -> nn.Module:
    from nemo_automodel.components.models.hy_mt2.config import HyMT2Config
    from nemo_automodel.components.models.hy_mt2.model import Block

    config = HyMT2Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=32,
        expert_hidden_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )
    return Block(layer_idx=0, config=config, moe_config=_moe_config(), backend=_backend())


_DENSE_BLOCK_FACTORIES = {
    "glm4_moe": _glm4_moe_dense_block,
    "glm4_moe_lite": _glm4_moe_lite_dense_block,
    "glm_moe_dsa": _glm_moe_dsa_dense_block,
    "ling_v2": _ling_v2_dense_block,
    "hy_v3": _hy_v3_dense_block,
    "hy_mt2": _hy_mt2_dense_block,
}


class _TinyLayerModel(nn.Module):
    """Minimal container exposing ``layers`` the way apply_ac iterates blocks."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([block])


@pytest.mark.parametrize("family", sorted(_DENSE_BLOCK_FACTORIES))
def test_scoped_ac_dense_mlp_forward_backward(family):
    """A scoped-AC-wrapped REAL dense block must dispatch _mlp and backprop cleanly."""
    torch.manual_seed(0)
    block = _DENSE_BLOCK_FACTORIES[family]().float()
    # Name-based sanity check: the stubbed-parallelizer tests pop
    # nemo_automodel.components.moe.layers from sys.modules, so a class object
    # imported here at collection time would not be identical to the one the
    # (runtime-imported) model module dispatches on.
    assert type(block.mlp).__name__ == "MLP", "test setup: layer 0 must be a dense-MLP layer"

    model = _TinyLayerModel(block)
    apply_ac(
        model,
        ignore_router=True,
        hidden_size=HIDDEN,
        num_experts=4,
        activation_checkpointing="non_moe",
    )
    assert isinstance(block.mlp, CheckpointWrapper), "scoped AC must wrap the dense MLP"

    x = torch.randn(2, 4, HIDDEN, requires_grad=True)
    out = block._mlp(x=x, padding_mask=None)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    down_proj_weight = block.mlp.down_proj.weight
    assert down_proj_weight.grad is not None
    assert torch.isfinite(down_proj_weight.grad).all()
