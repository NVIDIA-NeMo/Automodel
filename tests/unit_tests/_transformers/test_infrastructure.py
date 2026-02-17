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

from nemo_automodel._transformers.utils import _should_load_before_shard


class TestShouldLoadBeforeShard:
    """Tests for _should_load_before_shard.

    load_before_shard should be True only when ALL of these hold:
      - no pipeline parallelism (autopipeline is None)
      - no tensor parallelism (tp_size <= 1)
      - no expert parallelism (ep_size <= 1)
      - checkpoint needs loading (pretrained_model_name_or_path and load_base_model)
      - no PEFT (peft_config is None)
    """

    # Defaults that satisfy all conditions (single-GPU checkpoint load, no PEFT).
    _DEFAULTS = dict(
        autopipeline=None,
        tp_size=1,
        ep_size=1,
        pretrained_model_name_or_path="/some/path",
        load_base_model=True,
        peft_config=None,
    )

    def test_single_gpu_loads_before_shard(self):
        """With no parallelism and a valid checkpoint path, should load before shard."""
        assert _should_load_before_shard(**self._DEFAULTS) is True

    def test_ep_greater_than_1_skips_load_before_shard(self):
        """With EP > 1, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "ep_size": 2}) is False

    def test_tp_greater_than_1_skips_load_before_shard(self):
        """With TP > 1, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "tp_size": 2}) is False

    def test_pp_skips_load_before_shard(self):
        """With pipeline parallelism, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "autopipeline": object()}) is False

    def test_peft_skips_load_before_shard(self):
        """With PEFT config, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "peft_config": object()}) is False

    def test_no_pretrained_path_skips_load_before_shard(self):
        """Without a pretrained path, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "pretrained_model_name_or_path": ""}) is False

    def test_load_base_model_false_skips_load_before_shard(self):
        """With load_base_model=False, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "load_base_model": False}) is False

    @pytest.mark.parametrize(
        "tp_size,ep_size",
        [
            (2, 2),
            (4, 1),
            (1, 4),
            (2, 4),
        ],
        ids=["tp2_ep2", "tp4_ep1", "tp1_ep4", "tp2_ep4"],
    )
    def test_any_parallelism_skips_load_before_shard(self, tp_size, ep_size):
        """Any TP or EP > 1 should skip load-before-shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "tp_size": tp_size, "ep_size": ep_size}) is False

    def test_all_conditions_false(self):
        """When every condition blocks, result is still False."""
        assert (
            _should_load_before_shard(
                tp_size=2,
                ep_size=4,
                autopipeline=object(),
                pretrained_model_name_or_path="",
                load_base_model=False,
                peft_config=object(),
            )
            is False
        )

    def test_ep_size_exactly_1_allows_load(self):
        """ep_size=1 should not block load-before-shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "ep_size": 1}) is True
