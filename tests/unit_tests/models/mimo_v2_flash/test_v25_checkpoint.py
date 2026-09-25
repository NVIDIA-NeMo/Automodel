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

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config
from nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter import MiMoV2FlashStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _adapter(config):
    return MiMoV2FlashStateDictAdapter(
        config=config,
        moe_config=MoEConfig(
            dim=128,
            inter_dim=256,
            moe_inter_dim=128,
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid_with_bias",
            route_scale=1.0,
            norm_topk_prob=True,
        ),
        backend=BackendConfig(attn="sdpa", linear="torch", experts="torch", dispatcher="torch"),
        dtype=torch.float32,
    )


@pytest.mark.parametrize("checkpoint_tp", [2, 4, 8])
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_fp8_fused_qkv_load_preserves_shard_scales_and_native_storage(checkpoint_tp, layer_idx):
    config = SimpleNamespace(
        attention_projection_layout="fused_qkv",
        quantization_config={"quant_method": "fp8"},
        hidden_size=128,
        hybrid_layer_pattern=[0, 1],
        num_attention_heads=2 * checkpoint_tp,
        num_key_value_heads=checkpoint_tp,
        head_dim=128,
        v_head_dim=64,
        swa_num_attention_heads=3 * checkpoint_tp,
        swa_num_key_value_heads=checkpoint_tp,
        swa_head_dim=64,
        swa_v_head_dim=32,
    )
    adapter = _adapter(config)
    # Per-shard full: Q=256, K=128, V=64. Sliding: Q=192, K=64, V=32.
    # Both end partway through an FP8 block, so scales must restart per shard.
    q_rows, k_rows, v_rows = (256, 128, 64) if layer_idx == 0 else (192, 64, 32)
    rows_per_shard = q_rows + k_rows + v_rows
    scale_rows_per_shard = (rows_per_shard + 127) // 128
    prefix = f"model.layers.{layer_idx}.self_attn"
    native = {
        f"{prefix}.{projection}_proj.weight": torch.zeros(checkpoint_tp * rows, 128)
        for projection, rows in zip(("q", "k", "v"), (q_rows, k_rows, v_rows))
    }
    destinations = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    weight_key = f"{prefix}.qkv_proj.weight"
    scale_key = f"{weight_key}_scale_inv"
    assert set(destinations) == {weight_key, scale_key}
    assert destinations[weight_key].shape == (checkpoint_tp * rows_per_shard, 128)
    assert destinations[scale_key].shape == (checkpoint_tp * scale_rows_per_shard, 1)

    expected_q, expected_k, expected_v = [], [], []
    scales = []
    for shard_idx in range(checkpoint_tp):
        scale = torch.arange(
            1 + shard_idx * scale_rows_per_shard,
            1 + (shard_idx + 1) * scale_rows_per_shard,
            dtype=torch.float32,
        ).unsqueeze(1)
        dequantized = scale.repeat_interleave(128, dim=0)[:rows_per_shard].expand(-1, 128)
        expected_q.append(dequantized[:q_rows])
        expected_k.append(dequantized[q_rows : q_rows + k_rows])
        expected_v.append(dequantized[q_rows + k_rows :])
        scales.append(scale)
    destinations[weight_key].fill_(1)
    destinations[scale_key].copy_(torch.cat(scales))
    restored = adapter.from_hf(destinations)

    assert set(restored) == set(native)
    for projection, expected in zip(("q", "k", "v"), (expected_q, expected_k, expected_v)):
        key = f"{prefix}.{projection}_proj.weight"
        assert restored[key] is native[key]
        torch.testing.assert_close(restored[key], torch.cat(expected), rtol=0, atol=0)


def test_v25_pro_load_buffer_shapes_match_published_checkpoint():
    # XiaomiMiMo/MiMo-V2.5-Pro at 21d1ecfecd7bd70f31be25ca49d7edd21f003659,
    # model_pp0_ep0_shard0.safetensors, layer 0. Meta storage avoids allocating
    # the full 27136 x 6144 projection merely to check the loader's contract.
    config = MiMoV2Config(
        hidden_size=6144,
        num_attention_heads=128,
        num_key_value_heads=8,
        head_dim=192,
        v_head_dim=128,
        num_hidden_layers=1,
        hybrid_layer_pattern=[0],
        quantization_config={"quant_method": "fp8"},
    )
    adapter = _adapter(config)
    native = {
        f"model.layers.0.self_attn.{projection}_proj.weight": torch.empty(rows, 6144, device="meta")
        for projection, rows in (("q", 24576), ("k", 1536), ("v", 1024))
    }
    destinations = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    assert destinations["model.layers.0.self_attn.qkv_proj.weight"].shape == (27136, 6144)
    assert destinations["model.layers.0.self_attn.qkv_proj.weight_scale_inv"].shape == (216, 48)


def test_v25_fp8_experts_load_without_mxfp4_packing():
    config = MiMoV2Config(quantization_config={"quant_method": "fp8"})
    adapter = _adapter(config)
    prefix = "model.layers.1.mlp.experts"
    native = {
        f"{prefix}.gate_and_up_projs": torch.zeros(2, 128, 256),
        f"{prefix}.down_projs": torch.zeros(2, 128, 128),
    }
    destinations = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    assert len(destinations) == 12
    for expert_idx in range(2):
        for projection, value in (("gate", 1), ("up", 2), ("down", 3)):
            key = f"{prefix}.{expert_idx}.{projection}_proj.weight"
            assert destinations[key].shape == (128, 128)
            assert destinations[key].dtype == torch.float8_e4m3fn
            assert destinations[key + "_scale_inv"].shape == (1, 1)
            assert destinations[key + "_scale_inv"].dtype == torch.float32
            destinations[key].fill_(value + expert_idx)
            destinations[key + "_scale_inv"].fill_(2)

    restored = adapter.from_hf(destinations)
    assert set(restored) == set(native)
    for expert_idx in range(2):
        gate_up = restored[f"{prefix}.gate_and_up_projs"][expert_idx]
        torch.testing.assert_close(gate_up[:, :128], torch.full((128, 128), float(2 * (1 + expert_idx))))
        torch.testing.assert_close(gate_up[:, 128:], torch.full((128, 128), float(2 * (2 + expert_idx))))
        torch.testing.assert_close(
            restored[f"{prefix}.down_projs"][expert_idx], torch.full((128, 128), float(2 * (3 + expert_idx)))
        )
