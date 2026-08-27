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

"""Shared fixtures for the bi-encoder tests."""

import pytest

from nemo_automodel._transformers.registry import ModelRegistry


@pytest.fixture(autouse=True)
def restore_model_registry():
    """Undo registry entries injected by a test.

    Several tests here swap a real architecture for a fake class via
    ``ModelRegistry.model_arch_name_to_cls[arch] = Fake`` so that ``build``
    does not download weights.  ``ModelRegistry`` is a process-wide singleton,
    so without restoration those fakes leak into every later test in the
    session -- which then sees ``FakeBidirectionalModel`` instead of the real
    class (e.g. the capability-declaration contract test in
    ``tests/unit_tests/_transformers/test_model_capabilities.py``).

    ``monkeypatch.setitem`` cannot be used because ``_LazyArchMapping`` is not
    a ``dict`` and lacks ``get``; runtime entries land in its ``_extra`` map,
    so snapshotting that (plus the import cache it feeds) is the accurate undo.
    """
    mapping = ModelRegistry.model_arch_name_to_cls
    extra = dict(mapping._extra)
    loaded = dict(mapping._loaded)
    yield
    mapping._extra.clear()
    mapping._extra.update(extra)
    mapping._loaded.clear()
    mapping._loaded.update(loaded)
