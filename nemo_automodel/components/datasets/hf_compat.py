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

from dataclasses import is_dataclass
from typing import Any

import datasets
from datasets import Features
from datasets.features import features as hf_features

_LIST_FEATURE_DICT = {
    "input": {
        "feature": {"dtype": "string", "_type": "Value"},
        "_type": "List",
    }
}


def _get_list_feature_type() -> type[Any]:
    list_type = getattr(hf_features, "List", None)
    if list_type is not None and is_dataclass(list_type):
        return list_type

    sequence_type = getattr(hf_features, "Sequence", None)
    if sequence_type is not None:
        return sequence_type

    raise RuntimeError("Could not find a compatible datasets List or Sequence feature type.")


def patch_datasets_list_feature_deserializer() -> None:
    """Patch old ``datasets`` versions to deserialize Hub ``List`` features.

    Some container images still ship ``datasets`` versions whose feature
    deserializer predates Hub metadata that serializes list columns as
    ``{"_type": "List"}``. Those versions can resolve ``List`` to
    ``typing.List`` and then fail with ``TypeError: must be called with a
    dataclass type or instance``. Registering ``List`` in the feature type map
    restores the intended nested-list behavior while leaving newer versions
    unchanged.
    """

    try:
        Features.from_dict(_LIST_FEATURE_DICT)
        return
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if "dataclass type or instance" not in message and "Feature type 'List'" not in message:
            raise

    feature_types = getattr(hf_features, "_FEATURE_TYPES", None)
    if feature_types is None:
        raise RuntimeError("datasets.features.features._FEATURE_TYPES is unavailable.")

    feature_types["List"] = _get_list_feature_type()
    Features.from_dict(_LIST_FEATURE_DICT)


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Call ``datasets.load_dataset`` after applying compatibility patches.

    ``datasets.load_dataset`` is resolved dynamically (via attribute access on the
    module) rather than bound at import time, so callers/tests that monkeypatch
    ``datasets.load_dataset`` are honored.
    """

    patch_datasets_list_feature_deserializer()
    return datasets.load_dataset(*args, **kwargs)
