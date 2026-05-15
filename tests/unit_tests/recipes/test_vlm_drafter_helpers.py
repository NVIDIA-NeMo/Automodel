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

"""Unit tests for the VLM-finetune recipe helpers added for the joint drafter path.

Covers the two small additions in ``recipes/vlm/finetune.py``:
  * ``_shift_labels_left`` -- builds drafter-step targets.
  * ``_is_gemma4_joint_target`` -- detects the composite's classmethod target.
"""

import torch

from nemo_automodel.recipes.vlm.finetune import (
    _is_gemma4_joint_target,
    _shift_labels_left,
)


class TestShiftLabelsLeft:
    """Drafter step k predicts position ``t + 1 + k`` of the input sequence,
    so the corresponding labels are the original labels shifted left by
    ``k + 1`` positions, with the trailing ``k + 1`` columns masked to -100."""

    def test_k_zero_is_identity(self):
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        out = _shift_labels_left(labels, 0)
        torch.testing.assert_close(out, labels)

    def test_k_negative_is_identity(self):
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        out = _shift_labels_left(labels, -1)
        torch.testing.assert_close(out, labels)

    def test_shift_one(self):
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        out = _shift_labels_left(labels, 1)
        expected = torch.tensor([[2, 3, 4, 5, -100]])
        torch.testing.assert_close(out, expected)

    def test_shift_two(self):
        labels = torch.tensor([[10, 20, 30, 40, 50]])
        out = _shift_labels_left(labels, 2)
        expected = torch.tensor([[30, 40, 50, -100, -100]])
        torch.testing.assert_close(out, expected)

    def test_shift_at_or_beyond_seq_len_yields_all_masked(self):
        labels = torch.tensor([[1, 2, 3]])
        # shift exactly seq_len: every position becomes -100.
        out = _shift_labels_left(labels, 3)
        assert torch.all(out == -100)
        # shift beyond seq_len: same -- all -100, no out-of-bounds index.
        out_overshoot = _shift_labels_left(labels, 7)
        assert torch.all(out_overshoot == -100)

    def test_preserves_existing_minus_100_mask(self):
        """If the input has -100 padding, the shift carries them through."""
        labels = torch.tensor([[1, 2, -100, 4, -100]])
        out = _shift_labels_left(labels, 1)
        expected = torch.tensor([[2, -100, 4, -100, -100]])
        torch.testing.assert_close(out, expected)

    def test_does_not_mutate_input(self):
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        snapshot = labels.clone()
        _ = _shift_labels_left(labels, 2)
        torch.testing.assert_close(labels, snapshot)

    def test_batch_dim_handled_independently(self):
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        out = _shift_labels_left(labels, 1)
        expected = torch.tensor([[2, 3, 4, -100], [6, 7, 8, -100]])
        torch.testing.assert_close(out, expected)

    def test_dtype_preserved(self):
        labels = torch.zeros((1, 4), dtype=torch.long)
        out = _shift_labels_left(labels, 1)
        assert out.dtype == labels.dtype


class TestIsGemma4JointTarget:
    """The composite's classmethod is recognized so the recipe's build_model
    accepts it the same way it accepts the NeMoAutoModel classmethods."""

    def test_none_returns_false(self):
        assert _is_gemma4_joint_target(None) is False

    def test_unrelated_target_returns_false(self):
        # Any callable that is clearly not the composite's classmethod.
        from torch import nn

        assert _is_gemma4_joint_target(nn.Linear) is False
        assert _is_gemma4_joint_target(lambda: None) is False

    def test_composite_classmethod_returns_true(self):
        from nemo_automodel.components.models.gemma4_drafter.composite import (
            Gemma4WithDrafter,
        )

        # Bound classmethods aren't identity-stable; the helper must use
        # equality, which is why we go through ``in (...)``-style comparison.
        # Re-access the classmethod to make sure the helper handles the
        # non-identity case correctly.
        target = Gemma4WithDrafter.from_pretrained
        assert _is_gemma4_joint_target(target) is True
        assert _is_gemma4_joint_target(Gemma4WithDrafter.from_pretrained) is True

    def test_subclass_classmethod_does_not_match(self):
        """Sanity: an unrelated classmethod must not falsely match."""
        from nemo_automodel._transformers import NeMoAutoModelForCausalLM

        assert _is_gemma4_joint_target(NeMoAutoModelForCausalLM.from_pretrained) is False
