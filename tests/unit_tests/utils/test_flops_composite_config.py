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

from transformers import LlamaConfig, PretrainedConfig, Qwen3VLConfig

from nemo_automodel.components.utils.flops_utils import (
    get_flops_formula_for_hf_config,
    llama2_flops,
    qwen3_flops,
    transformer_flops,
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
