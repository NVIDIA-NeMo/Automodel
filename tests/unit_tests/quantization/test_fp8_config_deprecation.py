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

"""The raw-mapping FP8 builders are deprecated in favor of constructing ``FP8Config``."""

import pytest

from nemo_automodel.components.quantization.fp8 import FP8Config, build_fp8_config, create_fp8_config_from_dict


def test_build_fp8_config_is_deprecated_and_delegates():
    with pytest.warns(FutureWarning, match="construct FP8Config"):
        config = build_fp8_config({"enabled": True, "recipe_name": "rowwise", "filter_fqns": ["lm_head"]})
    assert (config.enabled, config.recipe_name, config.filter_fqns) == (True, "rowwise", ["lm_head"])

    with pytest.warns(FutureWarning, match="construct FP8Config"):
        assert build_fp8_config(None).enabled is False


def test_create_fp8_config_from_dict_is_deprecated_and_delegates():
    with pytest.warns(FutureWarning, match="construct FP8Config"):
        config = create_fp8_config_from_dict({"enabled": True})
    assert isinstance(config, FP8Config)
    assert config.enabled is True and config.filter_fqns == []
