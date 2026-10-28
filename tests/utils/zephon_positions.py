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

"""Position invariants shared by scalar and multi-axis model consumers."""

import torch


def assert_joined_position_ranges(original: torch.Tensor, joined: torch.Tensor, document_ids: torch.Tensor) -> None:
    """Check translated geometry and ordered, independent document ranges.

    Args:
        original: Local coordinates [batch, sequence] or [position_axes, batch, sequence].
        joined: Joined coordinates of the same shape, without modifying original.
        document_ids: Subdocument IDs [batch, sequence], zero for excluded slots.
    """
    assert joined.shape == original.shape
    torch.testing.assert_close(joined[..., document_ids.eq(0)], original[..., document_ids.eq(0)])
    for batch_index, ids in enumerate(document_ids):
        previous_maximum = -1
        for document_id in ids.unique():
            if document_id == 0:
                continue
            selected = ids.eq(document_id)
            local = original[..., batch_index, selected]
            translated = joined[..., batch_index, selected]
            assert (translated - local).unique().numel() == 1  # One offset for every token and axis.
            assert translated.min() > previous_maximum
            if previous_maximum == -1:
                torch.testing.assert_close(translated, local)
            previous_maximum = int(translated.max())
