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
    """The VLM collate pipeline pre-shifts labels by 1 so that ``labels[t] ==
    input_ids[t + 1]``. Drafter step ``k`` predicts position ``t + 1 + k`` of
    the original sequence, which maps to ``labels[t + k]`` in the pre-shifted
    convention -- so ``_shift_labels_left`` is called with ``k`` (not ``k + 1``).
    For ``k = 0`` the shift is a no-op."""

    def test_k_zero_is_identity(self):
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        out = _shift_labels_left(labels, 0)
        torch.testing.assert_close(out, labels)

    def test_k_negative_is_identity(self):
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        out = _shift_labels_left(labels, -1)
        torch.testing.assert_close(out, labels)

    def test_post_collate_semantic_alignment(self):
        """Anchor the semantic contract: given input_ids and the VLM-collate
        pre-shifted labels (``labels[t] == input_ids[t + 1]``), the drafter
        step-k target after ``_shift_labels_left(labels, k)`` is
        ``input_ids[t + 1 + k]`` (i.e. the token (k+1) positions ahead of
        ``input_ids[t]``).

        This pins the convention that the off-by-one bug violated -- if a
        future change reintroduces a ``k + 1`` shift, this test fails."""
        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        # Simulate the collate's ``labels = labels[:, 1:]`` shift.
        labels = input_ids[:, 1:]

        # k = 0: drafter target should equal input_ids[t + 1] for every t.
        target_k0 = _shift_labels_left(labels, 0)
        for t in range(labels.shape[-1] - 1):
            assert target_k0[0, t].item() == input_ids[0, t + 1].item(), (
                f"k=0 step at t={t}: expected input_ids[t+1]={input_ids[0, t + 1]}, got {target_k0[0, t]}"
            )
        # Last position drops out (no next token); it stays at the pre-collate value.
        assert target_k0[0, -1].item() == labels[0, -1].item()

        # k = 1: drafter target should equal input_ids[t + 2] for every t in range.
        target_k1 = _shift_labels_left(labels, 1)
        for t in range(labels.shape[-1] - 2):
            assert target_k1[0, t].item() == input_ids[0, t + 2].item(), (
                f"k=1 step at t={t}: expected input_ids[t+2]={input_ids[0, t + 2]}, got {target_k1[0, t]}"
            )
        # Trailing entries are -100 (no valid target k+1 positions ahead).
        assert target_k1[0, -1].item() == -100

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
