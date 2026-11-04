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
import torch

import nemo_automodel.components.models.kimi_k3.model as kimi_k3_model
from nemo_automodel.components.models.kimi_k3.model import _make_causal_mask


def _reference_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(batch_size, 1, -1, -1)


@pytest.fixture(autouse=True)
def clear_mask_cache():
    kimi_k3_model._CAUSAL_MASK_CACHE.clear()
    yield
    kimi_k3_model._CAUSAL_MASK_CACHE.clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_mask_matches_reference(dtype):
    embeds = torch.zeros(2, 5, 8)
    out = _make_causal_mask(embeds, None, dtype=dtype)
    torch.testing.assert_close(out, _reference_mask(2, 5, dtype))


def test_cache_reused_and_sliced_for_shorter_sequences():
    long_embeds = torch.zeros(1, 9, 8)
    _make_causal_mask(long_embeds, None, dtype=torch.float32)
    cached = kimi_k3_model._CAUSAL_MASK_CACHE[(torch.float32, long_embeds.device)]

    short_embeds = torch.zeros(3, 4, 8)
    out = _make_causal_mask(short_embeds, None, dtype=torch.float32)

    # Shorter request slices the existing entry instead of rebuilding it.
    assert kimi_k3_model._CAUSAL_MASK_CACHE[(torch.float32, short_embeds.device)] is cached
    torch.testing.assert_close(out, _reference_mask(3, 4, torch.float32))


def test_cache_grows_for_longer_sequences_and_stays_single_entry():
    _make_causal_mask(torch.zeros(1, 4, 8), None, dtype=torch.float32)
    out = _make_causal_mask(torch.zeros(1, 6, 8), None, dtype=torch.float32)

    torch.testing.assert_close(out, _reference_mask(1, 6, torch.float32))
    cache = kimi_k3_model._CAUSAL_MASK_CACHE
    assert len(cache) == 1
    assert cache[(torch.float32, torch.zeros(1).device)].shape == (6, 6)
