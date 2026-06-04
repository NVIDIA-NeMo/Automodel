# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import torch._dynamo


@pytest.fixture(autouse=True)
def _disable_torch_compile_for_cut_ce(request):
    """Run cut-CE (fused-linear cross-entropy) contract tests with torch.compile off.

    These tests only assert the eager forward output contract: that ``logits_to_keep``
    slices the logits and that the final hidden states are carried. They do not
    validate any compiled kernel.

    The MoE routed-expert path applies the ``@torch.compile``-decorated
    ``weighted_swiglu`` (``moe/megatron/moe_utils.py``) inside a custom
    ``torch.autograd.Function``. Compiling that frame under the CI test harness
    crashes TorchDynamo (SIGSEGV in ``convert_frame._compile``), killing the whole
    test process. Compilation adds nothing to a forward-contract check, so disable
    Dynamo for these tests; the eager output is identical. (The separate
    shared-expert cross-stream race in ``MoE.forward`` is fixed in ``moe/layers.py``,
    not here.) See PR #2397.
    """
    if "_cut_ce.py" not in request.node.nodeid:
        yield
        return
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = prev
