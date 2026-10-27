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

import logging

import pytest

from tests.unit_tests.models.kimi_linear.test_recipe_packing import _assert_recipe_document_isolation


@pytest.mark.gpu
@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("pack_size", [8, 10, 4096])
def test_recipe_neat_kda_mla_document_isolation(split: str, training: bool, pack_size: int) -> None:
    """Real KDA kernels must reset convolution and recurrence at document boundaries."""
    metrics = _assert_recipe_document_isolation(
        device="cuda", use_kda=True, training=training, pack_size=pack_size, split=split
    )
    logging.getLogger(__name__).info("split=%s training=%s pack_size=%s: %s", split, training, pack_size, metrics)
