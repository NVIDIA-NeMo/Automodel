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

"""Framework hooks that DeepSeek V4.1 shares with (or must be excluded from) DeepSeek V4."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel._transformers.capabilities import _is_deepseek_v4
from nemo_automodel.components.distributed.parallelizer import PARALLELIZATION_STRATEGIES
from nemo_automodel.components.models.deepseek_v4 import fsdp as dsv4_fsdp
from nemo_automodel.components.moe.parallelizer import _is_deepseek_v4_model
from nemo_automodel.components.utils.model_utils import freeze_deepseek_v4_indexer_params
from tests.unit_tests.models.deepseek_v41.conftest import build_tiny_model, tiny_config


class _FakeV41Model(nn.Module):
    """Minimal module tree with V4.1 indexer naming (no CUDA needed)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="deepseek_v41")
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict()
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.wq_a = nn.Linear(4, 4, bias=False)
        block.self_attn.indexer = nn.Module()
        block.self_attn.indexer.wq_b = nn.Linear(4, 4, bias=False)
        block.self_attn.indexer.wk = nn.Linear(4, 4, bias=False)
        block.self_attn.compressor = nn.Module()
        block.self_attn.compressor.wkv = nn.Linear(4, 4, bias=False)
        self.model.layers["4"] = block


def test_indexer_params_frozen_for_v41():
    model = _FakeV41Model()
    freeze_deepseek_v4_indexer_params(model)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {
        "model.layers.4.self_attn.indexer.wq_b.weight",
        "model.layers.4.self_attn.indexer.wk.weight",
    }


def test_indexer_freeze_on_real_tiny_model():
    model = build_tiny_model(tiny_config())
    freeze_deepseek_v4_indexer_params(model)
    for name, param in model.named_parameters():
        assert param.requires_grad == (".self_attn.indexer." not in name and "engram.embed" not in name), name


def test_capabilities_do_not_treat_v41_as_v4():
    assert not _is_deepseek_v4(SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v41")))
    assert _is_deepseek_v4(SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4")))

    class DeepseekV41ForCausalLM:  # noqa: N801 - mirrors the real class name
        config = SimpleNamespace(model_type="deepseek_v41")

    class DeepseekV4ForCausalLM:  # noqa: N801
        config = SimpleNamespace(model_type="other")

    assert not _is_deepseek_v4(DeepseekV41ForCausalLM())
    assert _is_deepseek_v4(DeepseekV4ForCausalLM())


def test_fsdp_wrapper_recognizes_v41_modules():
    assert dsv4_fsdp._is_deepseek_v4_module(_FakeV41Model())
    assert _is_deepseek_v4_model(_FakeV41Model())
    plain = nn.Linear(2, 2)
    assert not dsv4_fsdp._is_deepseek_v4_module(plain)
    assert not _is_deepseek_v4_model(plain)


def test_parallelization_strategy_registered():
    assert type(PARALLELIZATION_STRATEGIES["DeepseekV41ForCausalLM"]) is type(
        PARALLELIZATION_STRATEGIES["DeepseekV4ForCausalLM"]
    )


def test_fp32_islands_are_selected_by_the_shared_wrapper():
    model = build_tiny_model(tiny_config())
    # Ratio-2 compressor projections are fp32 islands; ratio-1 ones stay in the model dtype.
    model.model.layers["2"].self_attn.compressor.wkv.float()
    model.model.layers["4"].self_attn.compressor.wkv.to(torch.bfloat16)
    island_ids = {id(module) for module in dsv4_fsdp._iter_dsv4_fp32_modules(model)}
    islands = {name for name, module in model.named_modules() if id(module) in island_ids}
    assert "model.layers.2.self_attn.compressor.wkv" in islands
    assert "model.layers.4.self_attn.compressor.wkv" not in islands
    assert "model.layers.0.attn_hc" in islands and "lm_head" in islands
