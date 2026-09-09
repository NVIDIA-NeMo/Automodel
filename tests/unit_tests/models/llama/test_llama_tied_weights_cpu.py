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

from transformers import LlamaConfig

from nemo_automodel.components.models.llama.model import LlamaForCausalLM


def _tiny_llama_config(tie_word_embeddings: bool) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_llama_ties_lm_head_when_config_requests_tied_embeddings():
    model = LlamaForCausalLM(_tiny_llama_config(tie_word_embeddings=True))

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_llama_leaves_lm_head_untied_when_config_requests_untied_embeddings():
    model = LlamaForCausalLM(_tiny_llama_config(tie_word_embeddings=False))

    assert model.lm_head.weight is not model.model.embed_tokens.weight
