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

Covers the small additions in ``recipes/vlm/finetune.py``:
  * ``_shift_labels_left`` -- builds drafter-step targets.
  * ``_is_recipe_target`` -- generic, marker-attribute-driven gate that
    decides whether a YAML ``_target_`` can receive the recipe's
    infrastructure kwargs. Replaces the older Gemma4-specific
    ``_is_gemma4_joint_target`` helper.
  * ``FinetuneRecipeForVLM._maybe_add_drafter_loss`` -- adds the
    ``lambda * sum_k CE(drafter_logits[k], shifted_labels_k)`` term to the
    base loss when the model returns a non-empty ``drafter_logits``.
"""

import types
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import torch

from nemo_automodel.recipes.vlm.finetune import (
    FinetuneRecipeForVLM,
    _is_recipe_target,
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


class TestIsRecipeTarget:
    """``_is_recipe_target`` is the generic gate used by ``build_model`` to
    decide whether a YAML ``_target_`` can receive infrastructure kwargs
    (``device_mesh``, ``distributed_config``, ``peft_config``,
    ``freeze_config``, ``pipeline_config``).

    The contract: anything whose owner class declares
    ``__nemo_recipe_target__ = True`` passes; everything else is rejected.
    Both ``NeMoAutoModelFor*.from_pretrained`` / ``.from_config`` (inherited
    via ``_BaseNeMoAutoModelClass``) and ``Gemma4WithDrafter.from_pretrained``
    are expected to pass after this refactor; raw ``nn.Module`` classes and
    lambdas are not.
    """

    def test_none_returns_false(self):
        assert _is_recipe_target(None) is False

    def test_unmarked_class_returns_false(self):
        """A plain ``nn.Module`` subclass without the marker must not pass."""
        from torch import nn

        assert _is_recipe_target(nn.Linear) is False

    def test_lambda_returns_false(self):
        """Lambdas / regular functions don't carry the marker."""
        assert _is_recipe_target(lambda: None) is False

    def test_unmarked_classmethod_returns_false(self):
        """A classmethod whose owner is not marked must be rejected even
        though it has a ``__self__`` (catches the failure mode where the
        helper accepted *any* bound classmethod)."""

        class _Plain:
            @classmethod
            def from_pretrained(cls):
                return cls()

        assert _is_recipe_target(_Plain.from_pretrained) is False

    def test_marked_class_directly_returns_true(self):
        """``_target_`` pointing at a marked class (not ``.from_pretrained``)
        is also accepted -- the gate is about the kwargs contract, not the
        callable shape."""

        class _Marked:
            __nemo_recipe_target__ = True

        assert _is_recipe_target(_Marked) is True

    def test_marked_classmethod_returns_true(self):
        class _Marked:
            __nemo_recipe_target__ = True

            @classmethod
            def from_pretrained(cls):
                return cls()

        target = _Marked.from_pretrained
        assert _is_recipe_target(target) is True
        # Bound classmethods are not identity-stable across accesses; the
        # marker-attribute check sidesteps the equality issue the old
        # whitelist had with ``target == Cls.from_pretrained``.
        assert _is_recipe_target(_Marked.from_pretrained) is True

    def test_inherited_marker_returns_true(self):
        """The marker is inherited through normal attribute lookup, so a
        subclass of a marked base passes without re-declaring the
        attribute. This is the property that lets a single marker on
        ``_BaseNeMoAutoModelClass`` cover every ``NeMoAutoModelFor*`` class."""

        class _MarkedBase:
            __nemo_recipe_target__ = True

        class _Sub(_MarkedBase):
            @classmethod
            def from_pretrained(cls):
                return cls()

        assert _is_recipe_target(_Sub) is True
        assert _is_recipe_target(_Sub.from_pretrained) is True

    def test_composite_classmethod_returns_true(self):
        """End-to-end check against the real composite class."""
        from nemo_automodel.components.models.gemma4_drafter.composite import (
            Gemma4WithDrafter,
        )

        assert _is_recipe_target(Gemma4WithDrafter.from_pretrained) is True

    def test_nemo_auto_model_classmethods_return_true(self):
        """Every ``NeMoAutoModelFor*`` inherits the marker from
        ``_BaseNeMoAutoModelClass``, so all of these flow through the same
        gate as the composite -- no recipe-side whitelist needed."""
        from nemo_automodel._transformers import (
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForImageTextToText,
            NeMoAutoModelForMultimodalLM,
        )

        for cls in (
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForImageTextToText,
            NeMoAutoModelForMultimodalLM,
        ):
            assert _is_recipe_target(cls.from_pretrained) is True, (
                f"{cls.__name__}.from_pretrained must pass the recipe-target gate"
            )
            assert _is_recipe_target(cls.from_config) is True, (
                f"{cls.__name__}.from_config must pass the recipe-target gate"
            )

    def test_marker_must_be_exactly_true(self):
        """Truthy-but-not-True values (e.g. a non-empty string left over
        from a typo) must be rejected, otherwise the gate becomes a silent
        no-op for classes that happen to set the attribute by accident."""

        class _Truthy:
            __nemo_recipe_target__ = "yes"  # type: ignore[assignment]

        assert _is_recipe_target(_Truthy) is False


@dataclass
class _FakeJointOut:
    """Stand-in for ``Gemma4JointOutput`` -- just the attributes
    ``_maybe_add_drafter_loss`` reads."""

    drafter_logits: Optional[list]
    drafter_loss_weight: float = 0.1


def _identity_loss(*, logits, labels, num_label_tokens):  # noqa: ARG001
    """Toy loss that returns ``logits.sum() / num_label_tokens`` so every
    call's contribution is independent of the labels but still touches the
    autograd graph. The drafter total is then the simple sum of per-step
    sums, which is what ``_maybe_add_drafter_loss`` should produce."""
    return logits.float().sum() / max(int(num_label_tokens), 1)


class TestMaybeAddDrafterLoss:
    """``_maybe_add_drafter_loss`` is the recipe's only joint-loss assembly
    site; it is responsible for:

      * returning ``base_loss`` unchanged when the model is not a joint
        composite (``drafter_logits`` missing OR empty);
      * summing per-step drafter losses with ``shift = k`` against the
        recipe's ``loss_fn`` and weighting by ``drafter_loss_weight``;
      * emitting the rank-0 ``[joint-drafter]`` log line when ``log=True``.
    """

    def _make_recipe(self):
        """Build the smallest object that satisfies the method's attribute
        reads (``self.loss_fn``, ``self.dist_env``)."""
        recipe = types.SimpleNamespace()
        recipe.loss_fn = _identity_loss
        recipe.dist_env = types.SimpleNamespace(is_main=True)
        return recipe

    def test_drafter_logits_missing_returns_base_loss_unchanged(self):
        recipe = self._make_recipe()
        base = torch.tensor(1.5)
        out = types.SimpleNamespace()  # no ``drafter_logits`` attribute
        labels = torch.tensor([[1, 2, 3]])
        result = FinetuneRecipeForVLM._maybe_add_drafter_loss(
            recipe,
            out=out,
            base_loss=base,
            labels=labels,
            model=MagicMock(),
            num_label_tokens=3,
        )
        assert result is base

    def test_drafter_logits_empty_returns_base_loss_unchanged(self):
        recipe = self._make_recipe()
        base = torch.tensor(1.5)
        out = _FakeJointOut(drafter_logits=[])
        labels = torch.tensor([[1, 2, 3]])
        result = FinetuneRecipeForVLM._maybe_add_drafter_loss(
            recipe,
            out=out,
            base_loss=base,
            labels=labels,
            model=MagicMock(),
            num_label_tokens=3,
        )
        assert result is base

    def test_drafter_logits_attribute_set_to_none_returns_base(self):
        """Defensive: a composite output may explicitly set
        ``drafter_logits=None``; this must also short-circuit."""
        recipe = self._make_recipe()
        base = torch.tensor(0.0)
        out = _FakeJointOut(drafter_logits=None)
        result = FinetuneRecipeForVLM._maybe_add_drafter_loss(
            recipe,
            out=out,
            base_loss=base,
            labels=torch.tensor([[1]]),
            model=MagicMock(),
            num_label_tokens=1,
        )
        assert result is base

    def test_adds_weighted_sum_of_per_step_drafter_losses(self):
        """``L_total = L_base + lambda * sum_k L_drafter_k``.

        With the toy ``_identity_loss = logits.sum() / num_label_tokens`` the
        per-step contribution is independent of the labels, so we can predict
        the total in closed form and pin both the magnitude and the
        ``drafter_loss_weight`` plumbing.
        """
        recipe = self._make_recipe()
        base_loss = torch.tensor(2.0)
        # Two drafter steps; logits sums = 6 and 10 -> drafter_total = 16.
        dl_0 = torch.full((1, 4, 3), 0.5)  # sum = 6.0
        dl_1 = torch.full((1, 4, 5), 0.5)  # sum = 10.0
        labels = torch.tensor([[1, 2, 3, 4]])
        out = _FakeJointOut(drafter_logits=[dl_0, dl_1], drafter_loss_weight=0.25)
        result = FinetuneRecipeForVLM._maybe_add_drafter_loss(
            recipe,
            out=out,
            base_loss=base_loss,
            labels=labels,
            model=MagicMock(),
            num_label_tokens=4,
        )
        # Per-step losses: 6/4 = 1.5 and 10/4 = 2.5. Sum = 4.0.
        # Total = 2.0 + 0.25 * 4.0 = 3.0.
        torch.testing.assert_close(result, torch.tensor(3.0))

    def test_uses_shift_labels_left_with_k_index(self):
        """The k-th drafter step must be scored against labels shifted left
        by ``k``. We pin this by capturing the ``labels`` kwarg passed into
        the loss function on every call."""
        recipe = self._make_recipe()
        seen_labels: list = []

        def _capturing_loss(*, logits, labels, num_label_tokens):  # noqa: ARG001
            seen_labels.append(labels.clone())
            return logits.float().sum() / num_label_tokens

        recipe.loss_fn = _capturing_loss
        labels = torch.tensor([[10, 20, 30, 40, 50]])
        dl_0 = torch.zeros((1, 5, 2))
        dl_1 = torch.zeros((1, 5, 2))
        out = _FakeJointOut(drafter_logits=[dl_0, dl_1], drafter_loss_weight=1.0)

        FinetuneRecipeForVLM._maybe_add_drafter_loss(
            recipe,
            out=out,
            base_loss=torch.tensor(0.0),
            labels=labels,
            model=MagicMock(),
            num_label_tokens=5,
        )
        # k=0: identity; k=1: shifted left by 1, last position becomes -100.
        torch.testing.assert_close(seen_labels[0], labels)
        torch.testing.assert_close(seen_labels[1], torch.tensor([[20, 30, 40, 50, -100]]))

    def test_log_main_rank_does_not_raise(self, caplog):
        """``log=True`` on the main rank emits a single ``[joint-drafter]``
        log line. The test pins that the path runs without raising and
        produces the expected breakdown."""
        import logging

        recipe = self._make_recipe()
        out = _FakeJointOut(drafter_logits=[torch.zeros((1, 2, 3))], drafter_loss_weight=0.01)
        with caplog.at_level(logging.INFO, logger="nemo_automodel.recipes.vlm.finetune"):
            FinetuneRecipeForVLM._maybe_add_drafter_loss(
                recipe,
                out=out,
                base_loss=torch.tensor(0.5),
                labels=torch.tensor([[1, 2]]),
                model=MagicMock(),
                num_label_tokens=2,
                log=True,
            )
        # The log message must include the joint-drafter prefix; we don't
        # pin the exact floats since they depend on the toy loss arithmetic.
        assert any("[joint-drafter]" in r.getMessage() for r in caplog.records)

    def test_log_off_main_rank_emits_nothing(self, caplog):
        """``log=True`` but ``dist_env.is_main=False`` (off-main rank): no
        log record from the joint-drafter path."""
        import logging

        recipe = self._make_recipe()
        recipe.dist_env = types.SimpleNamespace(is_main=False)
        out = _FakeJointOut(drafter_logits=[torch.zeros((1, 2, 3))])
        with caplog.at_level(logging.INFO, logger="nemo_automodel.recipes.vlm.finetune"):
            FinetuneRecipeForVLM._maybe_add_drafter_loss(
                recipe,
                out=out,
                base_loss=torch.tensor(0.5),
                labels=torch.tensor([[1, 2]]),
                model=MagicMock(),
                num_label_tokens=2,
                log=True,
            )
        assert not any("[joint-drafter]" in r.getMessage() for r in caplog.records)
