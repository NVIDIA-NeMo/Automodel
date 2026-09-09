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

"""Unit tests for the offline-cache manifest checks of the EAGLE-3 recipe.

The cached trainer streams the precomputed features and ``loss_mask`` /
``position_mask`` as stored, so the target the cache was built for and the
recipe options that shape those masks must match what the manifest recorded.
"""

from types import SimpleNamespace

import pytest

from nemo_automodel.components.datasets.llm.offline_cache import ensure_supervision_options_match
from nemo_automodel.recipes.llm.train_eagle3 import _validate_cached_eagle3_manifest

_CONFIG = SimpleNamespace(vocab_size=64, hidden_size=32)
_OPTIONS = ("mask_reasoning_content", "mask_generation_prompt")


def _manifest(**overrides):
    manifest = {"target_vocab_size": 64, "aux_hidden_dim": 96}
    manifest.update(dict.fromkeys(_OPTIONS, False))
    manifest.update(overrides)
    return manifest


def _validate(manifest, **options):
    _validate_cached_eagle3_manifest("/c", manifest, _CONFIG, **{k: options.get(k, False) for k in _OPTIONS})


@pytest.mark.parametrize("option", _OPTIONS)
@pytest.mark.parametrize("flag", [False, True])
def test_matching_manifest_is_accepted(option, flag):
    _validate(_manifest(**{option: flag}), **{option: flag})


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"target_vocab_size": 65}, "target_vocab_size=65"),
        ({"aux_hidden_dim": 32}, "aux_hidden_dim=32"),
    ],
)
def test_target_mismatch_is_rejected(overrides, pattern):
    with pytest.raises(ValueError, match=pattern):
        _validate(_manifest(**overrides))


@pytest.mark.parametrize("option", _OPTIONS)
@pytest.mark.parametrize(("recorded", "configured"), [(False, True), (True, False)])
def test_mask_option_mismatch_is_rejected(option, recorded, configured):
    with pytest.raises(ValueError, match=rf"{option}={recorded}.*sets {option}={configured}"):
        _validate(_manifest(**{option: recorded}), **{option: configured})


def test_ensure_supervision_options_match_reports_every_mismatch():
    manifest = {"mask_reasoning_content": True, "mask_generation_prompt": False}
    ensure_supervision_options_match(
        manifest,
        {"mask_reasoning_content": 1, "mask_generation_prompt": 0},  # truthiness, not identity
        cache_name="DSpark",
        cache_dir="/c",
        producer_name="precompute_dspark",
    )
    with pytest.raises(
        ValueError, match="mask_reasoning_content=True, mask_generation_prompt=False.*precompute_dspark"
    ):
        ensure_supervision_options_match(
            manifest,
            {"mask_reasoning_content": False, "mask_generation_prompt": True},
            cache_name="DSpark",
            cache_dir="/c",
            producer_name="precompute_dspark",
        )


def test_ensure_supervision_options_match_rejects_unrecorded_option():
    with pytest.raises(ValueError, match="does not record mask_generation_prompt.*precompute_dspark"):
        ensure_supervision_options_match(
            {"mask_reasoning_content": False},
            {"mask_reasoning_content": False, "mask_generation_prompt": False},
            cache_name="DSpark",
            cache_dir="/c",
            producer_name="precompute_dspark",
        )
