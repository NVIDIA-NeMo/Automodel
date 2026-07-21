# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""``_load_extra_state`` must re-apply the restored vocab mapping on resume.

setup() pushes the freshly-scanned draft-vocab mapping to the draft and (for a
remote target) the server. A resumed checkpoint may carry a different mapping;
after restoring it onto the trainer module, the recipe must re-push it so the
draft d2t/t2d tables and the remote server match the checkpoint, not the stale
setup-time scan. Otherwise the server keeps projecting full-vocab logits with
the old mapping while the draft trains against the new one -- a silent mismatch.
"""

from types import SimpleNamespace

import torch

from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe


def _write_meta(tmp_path, ids, mask):
    torch.save(
        {"global_step": 7, "epoch": 2, "selected_token_ids": ids, "selected_token_mask": mask},
        tmp_path / "eagle_meta.pt",
    )


def _bare_recipe(module, draft_model, target_wrapper):
    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe._module = lambda: module
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.draft_model = draft_model
    recipe.target_wrapper = target_wrapper
    return recipe


def test_resume_reapplies_vocab_mapping_to_draft_and_target(tmp_path):
    ids = torch.tensor([1, 3, 5], dtype=torch.long)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[ids] = True
    _write_meta(tmp_path, ids, mask)

    module = SimpleNamespace(
        selected_token_ids=torch.zeros(3, dtype=torch.long),
        selected_token_mask=torch.zeros(8, dtype=torch.bool),
    )
    draft_calls = []
    target_calls = []
    draft_model = SimpleNamespace(set_vocab_mapping=lambda i: draft_calls.append(i))
    target_wrapper = SimpleNamespace(set_vocab_mapping=lambda i, m: target_calls.append((i, m)))
    recipe = _bare_recipe(module, draft_model, target_wrapper)

    recipe._load_extra_state(str(tmp_path))

    # meta restored onto the trainer module
    assert recipe.runtime.global_step == 7
    assert recipe._resume_epoch == 2
    assert torch.equal(module.selected_token_ids, ids)
    # and re-pushed to both the draft and the (remote) target, with the restored ids
    assert len(draft_calls) == 1 and torch.equal(draft_calls[0], ids)
    assert len(target_calls) == 1
    assert torch.equal(target_calls[0][0], ids) and torch.equal(target_calls[0][1], mask)


def test_resume_without_target_wrapper_does_not_crash(tmp_path):
    """The cached/offline path has target_wrapper=None; re-applying must be safe."""
    ids = torch.tensor([2, 4, 6], dtype=torch.long)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[ids] = True
    _write_meta(tmp_path, ids, mask)

    module = SimpleNamespace(
        selected_token_ids=torch.zeros(3, dtype=torch.long),
        selected_token_mask=torch.zeros(8, dtype=torch.bool),
    )
    draft_calls = []
    draft_model = SimpleNamespace(set_vocab_mapping=lambda i: draft_calls.append(i))
    recipe = _bare_recipe(module, draft_model, target_wrapper=None)

    recipe._load_extra_state(str(tmp_path))  # must not raise

    assert torch.equal(module.selected_token_ids, ids)
    assert len(draft_calls) == 1
