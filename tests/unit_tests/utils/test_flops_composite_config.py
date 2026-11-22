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

import pytest
from transformers import LlamaConfig, Mistral3Config, MistralConfig, PretrainedConfig, Qwen3VLConfig

from nemo_automodel.components.models.mistral3.model import Ministral3Config
from nemo_automodel.components.models.mistral4.configuration import Mistral4Config
from nemo_automodel.components.utils.flops_utils import (
    get_flops_formula_for_hf_config,
    llama2_flops,
    mla_moe_flops,
    qwen3_flops,
    transformer_flops,
)

# Ministral-3-3B-Instruct-2512 text backbone: dense GQA + SwiGLU, head_dim * heads != hidden_size.
_MINISTRAL3_3B_TEXT = dict(
    hidden_size=3072,
    intermediate_size=9216,
    num_hidden_layers=26,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=131072,
)


def test_unknown_composite_requires_explicit_text_config() -> None:
    text = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )
    composite = PretrainedConfig(text_config=text)
    assert get_flops_formula_for_hf_config(composite) is None
    formula = get_flops_formula_for_hf_config(composite.get_text_config())
    assert formula is not None
    expected = llama2_flops(text, gbs=2, seq_len=16)
    assert formula(text, gbs=2, seq_len=16) == expected
    assert formula(text, gbs=4, seq_len=16) == 2 * expected
    text.num_hidden_layers = 3
    assert formula(text, gbs=2, seq_len=16) == llama2_flops(text, gbs=2, seq_len=16)


def test_known_composite_keeps_its_registered_formula() -> None:
    assert get_flops_formula_for_hf_config(Qwen3VLConfig()) is qwen3_flops


def test_plain_unknown_config_keeps_transformer_fallback() -> None:
    assert get_flops_formula_for_hf_config(PretrainedConfig()) is transformer_flops


@pytest.mark.parametrize(
    "config",
    [
        # Mistral-Nemo-Base-2407
        MistralConfig(
            hidden_size=5120,
            intermediate_size=14336,
            num_hidden_layers=40,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=131072,
        ),
        Ministral3Config(**_MINISTRAL3_3B_TEXT),
    ],
    ids=["mistral", "ministral3"],
)
def test_dense_mistral_configs_count_swiglu_and_head_dim(config) -> None:
    formula = get_flops_formula_for_hf_config(config)
    assert formula is qwen3_flops
    # The generic fallback treats the MLP as two GEMMs and ignores head_dim.
    assert formula(config, gbs=1, seq_len=4096) > transformer_flops(config, gbs=1, seq_len=4096)


def test_mistral3_with_dense_text_backbone_uses_dense_formula() -> None:
    text = Ministral3Config(**_MINISTRAL3_3B_TEXT)
    config = Mistral3Config(text_config=text)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is qwen3_flops
    assert formula(config, gbs=2, seq_len=128) == qwen3_flops(text, gbs=2, seq_len=128)


def test_mistral3_with_mla_text_backbone_keeps_mla_formula() -> None:
    config = Mistral3Config(text_config=Mistral4Config())
    formula = get_flops_formula_for_hf_config(config)
    assert formula is mla_moe_flops
    assert formula(config, gbs=1, seq_len=128) == mla_moe_flops(config.text_config, gbs=1, seq_len=128)
