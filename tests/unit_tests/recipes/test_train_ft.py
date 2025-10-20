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

import pytest
from nemo_automodel.recipes.llm.train_ft import _get_packed_sequence_config


@pytest.mark.parametrize(
    "has_packed_sequence, is_hf_model, cp_size, return_val, raises",
    [
        (True, True, 1, {"attn_implementation": "flash_attention_2"}, None),
        (True, True, 2, {"attn_implementation": "sdpa"}, ValueError),
        (True, False, 1, {}, None),
        (True, False, 2, {}, None),
        (False, True, 1, {}, None),
        (False, True, 2, {'attn_implementation': 'sdpa'}, None),
        (False, False, 1, {}, None),
        (False, False, 2, {}, None),
    ],
)
def test_get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size, return_val, raises):
    if raises:
        with pytest.raises(raises):
            _get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size)
    else:
        assert _get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size) == return_val