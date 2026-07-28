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

import pytest

from nemo_automodel.components.models.kimi_k3.encoding import build_chat_segments


def test_medium_thinking_effort_is_rendered():
    segments = build_chat_segments(
        [{"role": "user", "content": "Hello"}],
        thinking_effort="medium",
    )

    rendered = "".join(segment.text for segment in segments)
    assert "thinking_effort=medium" in rendered


def test_invalid_thinking_effort_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported thinking_effort='extreme'"):
        build_chat_segments(
            [{"role": "user", "content": "Hello"}],
            thinking_effort="extreme",
        )
