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

"""Local token resolution and source normalization contracts."""

import pytest

from nemo_automodel.components.datasets.zephon.llm import (
    resolve_token_ids,
)
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


def test_pad_override_and_eos_collision(tokenizer):
    assert resolve_token_ids(tokenizer, pad_token_id=None).pad == tokenizer.eos_token_id
    with pytest.raises(ValueError, match="contradicts"):
        resolve_token_ids(tokenizer, pad_token_id=2)
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_token_ids(tokenizer, pad_token_id=999)
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_token_ids(tokenizer, pad_token_id=1.0)


@pytest.mark.parametrize("pad_id", [True, 1.0])
def test_padding_id_requires_an_integer(tokenizer, pad_id):
    with pytest.raises(ValueError, match="existing tokenizer vocabulary ID"):
        resolve_token_ids(tokenizer, pad_token_id=pad_id)
