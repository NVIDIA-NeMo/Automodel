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

"""Independent oracle mask for the prefix-tree parity checks.

Shared by ``prefix_tree_flex_parity.py`` and ``prefix_tree_magi_parity.py`` so
the two checks validate against the same reference and cannot silently diverge.
"""

import torch


def build_reference_mask(prompt_len: int, completion_lens: list[int], total: int) -> torch.Tensor:
    """Dense boolean attend-mask, built straight from the rollout structure.

    ``allowed[q, k]`` is True iff query ``q`` may attend to key ``k`` under the
    "prompt ++ completion as an independent causal sequence" semantics:
      * prompt query: causal within the prompt only;
      * completion query: FULL to the prompt, CAUSAL within its own completion.

    This is built from the rollout structure, NOT from the spec under test, so a
    wrong spec or a wrong realization of it is caught.
    """
    node_of = [-1] * prompt_len
    for c, n in enumerate(completion_lens):
        node_of.extend([c] * n)
    assert len(node_of) == total

    allowed = torch.zeros(total, total, dtype=torch.bool)
    for q in range(total):
        qc = node_of[q]
        for k in range(q + 1):  # causal upper bound: never attend to the future
            kc = node_of[k]
            if kc == -1 or kc == qc:
                # prompt key (full for completions, causal for prompt) or same completion.
                allowed[q, k] = True
    return allowed
