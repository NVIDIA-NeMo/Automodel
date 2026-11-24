# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""MTP modules declared by the HF config but not run by the built model are not counted."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Glm4MoeLiteConfig, MiniMaxM2Config

from nemo_automodel._transformers.mfu import AutoMFU
from nemo_automodel.components.utils import flops_utils
from nemo_automodel.components.utils.flops_utils import config_without_unbuilt_mtp

pytestmark = pytest.mark.cpu_only

SEQ = 1024


def _minimax_m2_config(**overrides) -> MiniMaxM2Config:
    # Released MiniMax-M2 / M2.5 configs declare three MTP modules.
    kwargs = dict(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        num_local_experts=16,
        num_experts_per_tok=4,
        vocab_size=1024,
        use_mtp=True,
        num_mtp_modules=3,
        mtp_transformer_layers=1,
    )
    kwargs.update(overrides)
    return MiniMaxM2Config(**kwargs)


def _glm4_moe_lite_config(**overrides) -> Glm4MoeLiteConfig:
    # Released GLM-4.7-Flash config declares one MTP layer.
    kwargs = dict(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
        intermediate_size=512,
        moe_intermediate_size=64,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        vocab_size=1024,
        num_nextn_predict_layers=1,
    )
    kwargs.update(overrides)
    return Glm4MoeLiteConfig(**kwargs)


class _Model(torch.nn.Module):
    """Built model stand-in: optional ``mtp_config`` / ``mtp`` mirror Automodel's MTP models."""

    def __init__(self, config, **attrs):
        super().__init__()
        self.config = config
        for name, value in attrs.items():
            setattr(self, name, value)


class _Wrapper(torch.nn.Module):
    """DDP-style wrapper without a ``config`` of its own."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def test_minimax_m2_without_mtp_head_drops_mtp_modules():
    config = _minimax_m2_config()
    result = config_without_unbuilt_mtp(config, _Model(config))

    expected = flops_utils.minimax_m2_flops(_minimax_m2_config(use_mtp=False), gbs=1, seq_len=SEQ)
    assert flops_utils.minimax_m2_flops(result, gbs=1, seq_len=SEQ) == expected
    assert flops_utils.minimax_m2_flops(config, gbs=1, seq_len=SEQ) > expected
    # The caller's config is not modified.
    assert config.use_mtp is True and config.num_mtp_modules == 3


def test_mla_moe_without_mtp_head_drops_nextn_layers():
    config = _glm4_moe_lite_config()
    result = config_without_unbuilt_mtp(config, _Model(config))

    expected = flops_utils.mla_moe_flops(_glm4_moe_lite_config(num_nextn_predict_layers=0), gbs=1, seq_len=SEQ)
    assert flops_utils.mla_moe_flops(result, gbs=1, seq_len=SEQ) == expected
    assert config.num_nextn_predict_layers == 1


def test_disabled_mtp_config_drops_mtp_layers():
    # e.g. Qwen3.5 built with ``num_nextn_predict_layers: 0`` while the HF config keeps one MTP layer.
    config = SimpleNamespace(mtp_num_hidden_layers=1)
    model = _Model(config, mtp_config=SimpleNamespace(enabled=False, num_layers=0), mtp=None)
    assert config_without_unbuilt_mtp(config, model).mtp_num_hidden_layers == 0
    assert config.mtp_num_hidden_layers == 1


def test_enabled_mtp_config_keeps_config():
    config = SimpleNamespace(mtp_num_hidden_layers=1)
    model = _Model(config, mtp_config=SimpleNamespace(enabled=True, num_layers=1))
    assert config_without_unbuilt_mtp(config, model) is config


def test_mtp_head_without_mtp_config_keeps_config():
    config = SimpleNamespace(num_mtp_modules=1, use_mtp=True)
    model = _Model(config, mtp=torch.nn.Identity())
    assert config_without_unbuilt_mtp(config, model) is config


def test_config_without_mtp_fields_is_returned_as_is():
    config = SimpleNamespace(hidden_size=8, num_nextn_predict_layers=0)
    assert config_without_unbuilt_mtp(config, _Model(config)) is config


def test_composite_config_drops_text_config_mtp_layers():
    text_config = SimpleNamespace(mtp_num_hidden_layers=1)
    config = SimpleNamespace(text_config=text_config)
    result = config_without_unbuilt_mtp(config, _Model(config))
    assert result.text_config.mtp_num_hidden_layers == 0
    assert config.text_config is text_config and text_config.mtp_num_hidden_layers == 1


@pytest.mark.parametrize("wrap", [False, True])
def test_auto_mfu_from_model_without_mtp_head(wrap):
    config = _minimax_m2_config()
    model = _Model(config)
    calculator = AutoMFU.from_config(_Wrapper(model) if wrap else model, peak_tflops=989.0)

    expected = flops_utils.minimax_m2_flops(_minimax_m2_config(use_mtp=False), gbs=2, seq_len=SEQ)
    assert calculator.get_flops((2, SEQ)) == expected


def test_auto_mfu_from_model_with_mtp_head_counts_it():
    config = _minimax_m2_config()
    model = _Model(config, mtp_config=SimpleNamespace(enabled=True, num_layers=3))
    calculator = AutoMFU.from_config(_Wrapper(model), peak_tflops=989.0)
    assert calculator.get_flops((2, SEQ)) == flops_utils.minimax_m2_flops(config, gbs=2, seq_len=SEQ)


def test_auto_mfu_from_bare_config_is_unchanged():
    config = _minimax_m2_config()
    calculator = AutoMFU.from_config(config, peak_tflops=989.0)
    assert calculator.get_flops((2, SEQ)) == flops_utils.minimax_m2_flops(config, gbs=2, seq_len=SEQ)
