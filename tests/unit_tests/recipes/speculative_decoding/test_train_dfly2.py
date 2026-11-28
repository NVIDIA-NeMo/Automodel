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

"""Tests for the DFly2 recipe seams over the DSpark recipe.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
overridden seams are exercised: the unsupported-option gate, the model-owned draft
options, and the FSDP units of the FP32 heads.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

from nemo_automodel.recipes.speculative_decoding.train_dfly2 import TrainDFly2Recipe
from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe, _DraftArgs

pytestmark = pytest.mark.cpu_only


def _recipe(**recipe_args) -> TrainDFly2Recipe:
    recipe = TrainDFly2Recipe.__new__(TrainDFly2Recipe)
    recipe.cfg = SimpleNamespace(recipe_args=dict(recipe_args))
    return recipe


def _margs(**overrides) -> _DraftArgs:
    values = dict(
        num_draft_layers=3,
        target_layer_ids=[28, 32, 36],
        block_size=6,
        num_anchors=512,
        mask_token_id=128799,
        markov_rank=256,
        markov_head_type="vanilla",
        confidence_head_alpha=0.0,
        confidence_head_with_markov=True,
        confidence_head_stop_gradient=False,
    )
    values.update(overrides)
    return _DraftArgs(**values)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("cached_target_path", "/cache"),
        ("target_model_backend", "remote"),
        ("packed_sequence_size", 4096),
        ("multimodal", True),
    ],
)
def test_setup_rejects_options_without_a_colocated_unpacked_target(option, value, monkeypatch) -> None:
    called = Mock()
    monkeypatch.setattr(TrainDSparkRecipe, "setup", called)
    with pytest.raises(NotImplementedError, match=option):
        _recipe(**{option: value}).setup()
    called.assert_not_called()


def test_confidence_head_is_off_and_detached_by_default() -> None:
    target_config = SimpleNamespace(build_dfly2_draft=Mock(return_value=Mock()))
    recipe = _recipe()
    recipe.target_model = object()
    # The recipe's own default, as setup() resolves it when the YAML leaves the key out.
    recipe._build_model_owned_draft(
        target_config, _margs(confidence_head_alpha=TrainDFly2Recipe.default_confidence_head_alpha)
    )
    options = target_config.build_dfly2_draft.call_args.args[0]
    assert options.enable_confidence_head is False
    assert options.confidence_head_stop_gradient is True
    assert options.attn_backend == "sdpa"


def test_fsdp_units_include_the_heads_holding_fp32_tensors() -> None:
    draft = SimpleNamespace(
        layers=nn.ModuleList([nn.Linear(1, 1)]),
        mtp_fusion=nn.Linear(1, 1),
        hidden_correction=nn.Linear(1, 1),
        confidence_head=None,
    )
    assert TrainDFly2Recipe._draft_fsdp_units(draft) == [draft.layers[0], draft.mtp_fusion, draft.hidden_correction]
    # The FP32 confidence head must not fall into the root unit, which all-gathers in BF16.
    draft.confidence_head = nn.Linear(1, 1)
    assert TrainDFly2Recipe._draft_fsdp_units(draft)[-1] is draft.confidence_head


def test_warm_start_refuses_a_checkpoint_paired_with_other_layers(tmp_path) -> None:
    import json

    from nemo_automodel.recipes.speculative_decoding.train_dfly2 import check_dfly2_checkpoint_settings

    # The released draft's dfly_config.
    released = {"block_size": 6, "target_kv_layers": [28, 32, 36], "mask_token_id": 128799}
    (tmp_path / "config.json").write_text(json.dumps({"dfly_config": released}))
    check_dfly2_checkpoint_settings(str(tmp_path), target_layer_ids=[28, 32, 36], block_size=6, mask_token_id=128799)
    with pytest.raises(ValueError, match="target_kv_layers"):
        check_dfly2_checkpoint_settings(
            str(tmp_path / "model.safetensors"), target_layer_ids=[24, 28, 32], block_size=6, mask_token_id=128799
        )
    with pytest.raises(ValueError, match="mask_token_id"):
        check_dfly2_checkpoint_settings(str(tmp_path), target_layer_ids=[28, 32, 36], block_size=6, mask_token_id=1)
