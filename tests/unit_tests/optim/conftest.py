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

import sys

import pytest


@pytest.fixture
def eager_dion(monkeypatch):
    """Run dion's ``@torch.compile(fullgraph=True)`` kernels as their uncompiled Python functions.

    torch raises when a ``fullgraph=True`` function is called while Dynamo is disabled or forced eager,
    so disabling compilation through ``torch._dynamo.config`` or ``torch.compiler.set_stance`` no longer
    works. Swapping in the original callables keeps CPU unit tests free of Inductor compile time while
    still exercising dion's update arithmetic.
    """
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "dion" or name.startswith("dion.")):
            continue
        for attr, value in list(vars(module).items()):
            original = getattr(value, "_torchdynamo_orig_callable", None)
            if original is not None:
                monkeypatch.setattr(module, attr, original)
    yield
