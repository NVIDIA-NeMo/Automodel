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

"""Auto-resume must not pick up a checkpoint from a differently-expanded run.

With no explicit ``restore_from``, a recipe resumes from the latest checkpoint it finds in
``checkpoint_dir``, and a signature check is what stops it from loading one that belongs to
a different run. Expansion is invisible to the architecture half of that signature -- an
expanded model is built from the same ``model:`` block as its parent -- so the signature
has to carry the expansion configuration too.

Which layers and projections are expanded decides which parameters exist, so it decides
both the model and the optimizer state dict. ``merge_weight`` and ``zero_init_modules`` do
not: they change numerics and initial values, so changing them on resume stays compatible.

Without this, running the parent for comparison in a directory holding an expanded run's
checkpoints fails deep inside the optimizer load with ``Missing key in checkpoint
state_dict: optim.state.model.embed_tokens.weight.step``.
"""

import pytest

from nemo_automodel.recipes.base_recipe import _extract_model_signature, _signatures_match

BASE_MODEL = {
    "model": {
        "pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
    }
}


def _signature(expansion: dict | None) -> dict:
    config = dict(BASE_MODEL)
    if expansion is not None:
        config["expansion"] = expansion
    return _extract_model_signature(config)


EXPANDED = {"enabled": True, "layers": [8, 12], "merge_weight": 0.5}


@pytest.mark.parametrize(
    "other, compatible",
    [
        (EXPANDED, True),
        ({"enabled": True, "layers": [8, 12], "merge_weight": 0.9}, True),
        ({"enabled": True, "layers": [8, 12], "zero_init_modules": ["o_proj"]}, True),
        ({"enabled": True, "layers": [4, 5], "merge_weight": 0.5}, False),
        ({"enabled": True, "layers": [8, 12], "target_modules": ["q_proj"]}, False),
        ({"enabled": False}, False),
        (None, False),
    ],
    ids=[
        "identical",
        "merge_weight_differs",
        "zero_init_differs",
        "layers_differ",
        "target_modules_differ",
        "expansion_disabled",
        "no_expansion_block",
    ],
)
def test_expansion_identity_decides_checkpoint_compatibility(other, compatible):
    """Only the fields that change which parameters exist may break compatibility."""
    assert _signatures_match(_signature(EXPANDED), _signature(other)) is compatible


def test_two_unexpanded_runs_stay_compatible():
    """The guard must not start rejecting checkpoints that predate expansion."""
    assert _signatures_match(_signature(None), _signature({"enabled": False}))
