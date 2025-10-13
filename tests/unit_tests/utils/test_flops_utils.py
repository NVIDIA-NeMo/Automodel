# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.components.utils import flops_utils


def _moonlight_16b_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=11264,
        vocab_size=163840,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe_intermediate_size=1408,
        num_experts_per_tok=6,
        moe_layer_freq=[0] * 1 + [1] * 26,
        mtp_num_layers=0,
    )


def _deepseek_v3_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        intermediate_size=18432,
        vocab_size=151936,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        moe_layer_freq=[0] * 3 + [1] * 58,
        mtp_num_layers=None,
    )


def _gpt3_cfg() -> SimpleNamespace:
    return SimpleNamespace(hidden_size=2048, num_hidden_layers=16, vocab_size=50257)


def _llama2_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    )


def _llama3_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
    )


def _nemotron_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=24,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    )


def _mixtral_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=14336,
        vocab_size=32000,
        num_experts_per_tok=2,
    )


def _qwen3_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        moe_topk=1,
        head_dim=256,
    )


def _bert_cfg() -> SimpleNamespace:
    return SimpleNamespace(num_hidden_layers=12, hidden_size=768, vocab_size=30522)


def _transformer_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=8192,
        vocab_size=50257,
    )


def _gpt_oss_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts_per_tok=1,
        moe_ffn_hidden_size=8192,
        moe_router_topk=1,
        vocab_size=50257,
        kv_channels=128,
        window_size=[128],
        window_attn_skip_freq=2,
    )


@pytest.mark.parametrize(
    "name, func, cfg_factory, kwargs, expected",
    [
        ("gpt3", flops_utils.gpt3_flops, _gpt3_cfg, dict(gbs=2, seq_len=1024), 11572680327168),
        ("llama2", flops_utils.llama2_flops, _llama2_cfg, dict(gbs=1, seq_len=2048), 84486301679616),
        ("llama3", flops_utils.llama3_flops, _llama3_cfg, dict(gbs=1, seq_len=2048), 90671054585856),
        ("nemotron", flops_utils.nemotron_flops, _nemotron_cfg, dict(gbs=1, seq_len=2048), 50470160695296),
        ("mixtral", flops_utils.mixtral_flops, _mixtral_cfg, dict(gbs=1, seq_len=2048), 169835891785728),
        ("qwen3", flops_utils.qwen3_flops, _qwen3_cfg, dict(gbs=1, seq_len=2048), 110462263885824),
        ("bert", flops_utils.bert_flops, _bert_cfg, dict(gbs=1, seq_len=512), 361920724992),
        ("transformer", flops_utils.transformer_flops, _transformer_cfg, dict(gbs=1, seq_len=1024), 8363320541184),
        ("gpt_oss", flops_utils.gpt_oss_flops, _gpt_oss_cfg, dict(gbs=1, seq_len=1024), 7356800827392),
        ("deepseekv3_moonlight", flops_utils.deepseekv3_flops, _moonlight_16b_config, dict(gbs=1, seq_len=2048), 30625801175040),
        ("deepseekv3_dsv3", flops_utils.deepseekv3_flops, _deepseek_v3_config, dict(gbs=1, seq_len=1024), 233225179889664),
    ],
)
def test_flops_formulas_with_precomputed_values(name, func, cfg_factory, kwargs, expected):
    cfg = cfg_factory()
    actual = int(func(cfg, **kwargs))
    assert actual == expected, f"{name}: expected {expected}, got {actual}"


@pytest.mark.parametrize(
    "tflops, world_size, time_seconds, reference_mfu, expected_mfu",
    [
        # Basic test: 1 TFLOPs per GPU, 1 GPU, 1 second, reference 1 TFLOPs -> 100% MFU
        (1.0, 1, 1.0, 1.0, 100.0),
        # Half efficiency: 0.5 TFLOPs per GPU, 1 GPU, 1 second, reference 1 TFLOPs -> 50% MFU
        (0.5, 1, 1.0, 1.0, 50.0),
        # Multiple GPUs: 1 TFLOPs per GPU, 8 GPUs, 1 second, reference 1 TFLOPs -> 12.5% MFU
        (1.0, 8, 1.0, 1.0, 12.5),
        # Longer time: 10 TFLOPs per GPU, 1 GPU, 10 seconds, reference 1 TFLOPs -> 100% MFU
        (10.0, 1, 10.0, 1.0, 100.0),
        # H100 reference (default): 989 TFLOPs per GPU, 8 GPUs, 1 second, reference 1979 TFLOPs -> 6.25% MFU
        (989.0, 8, 1.0, 1979.0, 6.2468418393127845),
        # Real-world scenario: 500 TFLOPs per GPU, 64 GPUs, 2 seconds, H100 reference -> 0.197% MFU
        (500.0, 64, 2.0, 1979.0, 0.19738504295098536),
    ],
)
def test_calculate_mfu(tflops, world_size, time_seconds, reference_mfu, expected_mfu):
    """Test calculate_mfu function with various scenarios."""
    actual_mfu = flops_utils.calculate_mfu(tflops, world_size, time_seconds, reference_mfu)
    assert pytest.approx(actual_mfu, rel=1e-3) == expected_mfu


def test_calculate_mfu_default_reference():
    """Test calculate_mfu with default H100 reference."""
    # Using default reference_mfu (1979.0 for H100)
    actual_mfu = flops_utils.calculate_mfu(tflops=1979.0, world_size=1, time_seconds=1.0)
    assert pytest.approx(actual_mfu, rel=1e-6) == 100.0
