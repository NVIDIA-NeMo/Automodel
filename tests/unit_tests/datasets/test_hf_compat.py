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

from datasets import Features
from datasets.features import features as hf_features

from nemo_automodel.components.datasets.hf_compat import (
    _LIST_FEATURE_DICT,
    patch_datasets_list_feature_deserializer,
)


def test_patch_datasets_list_feature_deserializer_repairs_bad_list_mapping(monkeypatch):
    monkeypatch.setitem(hf_features._FEATURE_TYPES, "List", list)

    patch_datasets_list_feature_deserializer()

    features = Features.from_dict(_LIST_FEATURE_DICT)
    assert "List" in repr(features["input"])
