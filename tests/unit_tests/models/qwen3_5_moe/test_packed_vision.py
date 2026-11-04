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

"""Qwen3.5-MoE packed vision must preserve independent-document semantics."""

import pytest

from tests.unit_tests.models.qwen3_5.test_packed_vision import check_packed_media_parity


@pytest.mark.parametrize("media", ["image", "video", "both", "text"])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("pack_sizes", [(2,), (2, 1)])
def test_packed_media_matches_independent_documents(media: str, precomputed: bool, pack_sizes: tuple[int, ...]) -> None:
    check_packed_media_parity(media, precomputed, moe=True, pack_sizes=pack_sizes)
