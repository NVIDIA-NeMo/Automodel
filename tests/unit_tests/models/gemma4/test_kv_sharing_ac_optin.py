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

"""Gemma4's opt-in to whole-block activation checkpointing for KV-shared models.

``DefaultParallelizationStrategy`` keeps KV-shared models off whole-block
checkpointing unless the model class declares its shared-K/V store safe under
checkpoint replay. Gemma4 E2B/E4B depend on that declaration to keep attention
inside the recomputed region; without it they drop to
``apply_submodule_checkpointing``, which leaves ``self_attn`` unwrapped and
inflates peak memory -- the regression PR #3513 shipped unnoticed.

The parallelizer reads the flag by name, so nothing else in the suite ties that
name to the class that has to carry it: deleting the attribute from
``gemma4_moe/model.py`` reintroduces the regression with every other test green.
These tests are that tie.
"""

import pytest

from nemo_automodel.components.distributed.parallelizer import _kv_sharing_survives_checkpoint_replay
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

ATTRIBUTE = "kv_sharing_survives_checkpoint_replay"


def test_gemma4_declares_kv_sharing_replay_safe():
    """The E2B/E4B class must carry the opt-in, or it loses attention checkpointing."""
    assert getattr(Gemma4ForConditionalGeneration, ATTRIBUTE, False) is True


def test_parallelizer_sees_the_opt_in_on_a_gemma4_instance():
    """The parallelizer's lookup must resolve the flag on an instance, not just the class.

    Built with ``__new__`` so the assertion stays a CPU-only attribute check:
    the helper only does an attribute lookup, which falls back to the class.
    """
    model = Gemma4ForConditionalGeneration.__new__(Gemma4ForConditionalGeneration)
    assert _kv_sharing_survives_checkpoint_replay(model) is True


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("nemo_automodel.components.models.gemma4_unified.model", "Gemma4UnifiedForConditionalGeneration"),
        ("nemo_automodel.components.models.gemma4_drafter.model", "Gemma4DrafterForCausalLM"),
    ],
)
def test_plain_hf_gemma4_wrappers_do_not_opt_in(module_path, class_name):
    """Sibling Gemma4 wrappers ride plain HF with an accumulating cache and must not opt in.

    They inject no pass-through K/V holder, so a replayed block would call
    ``Cache.update()`` twice and backward would fail.
    """
    module = pytest.importorskip(module_path)
    sibling = getattr(module, class_name)
    assert getattr(sibling, ATTRIBUTE, False) is False
