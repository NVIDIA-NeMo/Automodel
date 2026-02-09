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

import pytest
from jinja2.exceptions import TemplateError

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

GEMMA_TOKENIZER_PATH = "/home/TestData/automodel/tokenizers/gemma-2-9b-it"


@pytest.fixture
def conversation_with_system_role():
    """A conversation that includes a system role message."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

@pytest.fixture
def conversation_with_multiple_system_roles():
    """A conversation that includes multiple system role messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "system", "content": "You are a helpful assistant."},
    ]


@pytest.mark.parametrize("force_hf", [True, False])
def test_gemma_tokenizer_system_role_handling(force_hf, conversation_with_system_role):
    """
    Test that NeMoAutoTokenizer handles system role correctly for Gemma-2.

    - force_hf=True: Returns raw HF tokenizer which raises TemplateError on system role
    - force_hf=False: Returns NeMoAutoTokenizer wrapper which maps system->assistant
    """
    tokenizer = NeMoAutoTokenizer.from_pretrained(GEMMA_TOKENIZER_PATH, force_hf=force_hf)

    if force_hf:
        assert not isinstance(tokenizer, NeMoAutoTokenizer)
        # Raw HF tokenizer should raise TemplateError for system role
        with pytest.raises(TemplateError, match="System role not supported"):
            tokenizer.apply_chat_template(
                conversation_with_system_role,
                tokenize=True,
                add_generation_prompt=True,
            )
    else:
        assert isinstance(tokenizer, NeMoAutoTokenizer)
        # NeMoAutoTokenizer should handle system role gracefully (maps to assistant, then drops)
        result = tokenizer.apply_chat_template(
            conversation_with_system_role,
            tokenize=True,
            add_generation_prompt=True,
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

@pytest.mark.parametrize("force_hf", [True, False])
def test_gemma_tokenizer_system_role_handling_with_multiple_system_roles(force_hf, conversation_with_multiple_system_roles):
    """
    Test that NeMoAutoTokenizer handles system role correctly for Gemma-2.

    - force_hf=True: Returns raw HF tokenizer which raises TemplateError on system role
    - force_hf=False: Returns NeMoAutoTokenizer wrapper which maps system->assistant
    """
    tokenizer = NeMoAutoTokenizer.from_pretrained(GEMMA_TOKENIZER_PATH, force_hf=force_hf)

    if force_hf:
        assert not isinstance(tokenizer, NeMoAutoTokenizer)
        # Raw HF tokenizer should raise TemplateError for system role
        with pytest.raises(TemplateError, match="System role not supported"):
            tokenizer.apply_chat_template(conversation_with_multiple_system_roles, tokenize=True, add_generation_prompt=True)
    else:
        assert isinstance(tokenizer, NeMoAutoTokenizer)
        with pytest.raises(ValueError, match="System role appeared in multiple messages."):
            tokenizer.apply_chat_template(conversation_with_multiple_system_roles, tokenize=True, add_generation_prompt=True)
