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
"""CPU tests for the DSpark ``draft_weights_path`` warm start."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nemo_automodel.recipes.speculative_decoding._dspark_warm_start import load_dspark_draft_weights

LAYER_IDS = [1, 3]
MASK = 7


class _Draft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.fc = nn.Linear(4, 4)


def _saved_state(value: float = 5.0) -> dict[str, torch.Tensor]:
    return {key: torch.full_like(tensor, value) for key, tensor in _Draft().state_dict().items()}


def _load(draft: nn.Module, path, *, keep_target_embeddings: bool = True, layer_ids=LAYER_IDS, mask=MASK) -> None:
    load_dspark_draft_weights(
        draft,
        str(path),
        keep_target_embeddings=keep_target_embeddings,
        target_layer_ids=layer_ids,
        mask_token_id=mask,
    )


def test_frozen_draft_keeps_the_target_embeddings(tmp_path):
    draft = _Draft()
    embed_before = draft.embed_tokens.weight.detach().clone()
    lm_head_before = draft.lm_head.weight.detach().clone()
    save_file(_saved_state(), tmp_path / "draft.safetensors")

    _load(draft, tmp_path / "draft.safetensors")

    assert (draft.fc.weight == 5).all() and (draft.fc.bias == 5).all()
    torch.testing.assert_close(draft.embed_tokens.weight.detach(), embed_before, rtol=0, atol=0)
    torch.testing.assert_close(draft.lm_head.weight.detach(), lm_head_before, rtol=0, atol=0)


def test_frozen_draft_does_not_need_embeddings_in_the_checkpoint(tmp_path):
    state = {key: value for key, value in _saved_state().items() if key.startswith("fc.")}
    save_file(state, tmp_path / "draft.safetensors")
    draft = _Draft()

    _load(draft, tmp_path / "draft.safetensors")

    assert (draft.fc.weight == 5).all()


def test_trainable_embeddings_come_from_the_checkpoint(tmp_path):
    save_file(_saved_state(), tmp_path / "draft.safetensors")
    draft = _Draft()

    _load(draft, tmp_path / "draft.safetensors", keep_target_embeddings=False)

    assert (draft.embed_tokens.weight == 5).all() and (draft.lm_head.weight == 5).all()


def test_trainable_embeddings_must_be_in_the_checkpoint(tmp_path):
    state = {key: value for key, value in _saved_state().items() if key.startswith("fc.")}
    save_file(state, tmp_path / "draft.safetensors")

    with pytest.raises(ValueError, match="2 keys missing"):
        _load(_Draft(), tmp_path / "draft.safetensors", keep_target_embeddings=False)


def test_reads_an_index_sharded_directory(tmp_path):
    state = _saved_state()
    first = {key: value for key, value in state.items() if key.startswith("fc.")}
    second = {key: value for key, value in state.items() if not key.startswith("fc.")}
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {key: "model-00001-of-00002.safetensors" for key in first}
    weight_map.update({key: "model-00002-of-00002.safetensors" for key in second})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    draft = _Draft()

    _load(draft, tmp_path)

    assert (draft.fc.weight == 5).all()


def test_missing_weights_are_an_error(tmp_path):
    state = _saved_state()
    del state["fc.bias"]
    save_file(state, tmp_path / "draft.safetensors")

    with pytest.raises(ValueError, match="1 keys missing.*fc.bias"):
        _load(_Draft(), tmp_path / "draft.safetensors")


def test_unexpected_weights_are_an_error(tmp_path):
    save_file({**_saved_state(), "markov_head.weight": torch.zeros(2)}, tmp_path / "draft.safetensors")

    with pytest.raises(ValueError, match="1 unexpected.*markov_head.weight"):
        _load(_Draft(), tmp_path / "draft.safetensors")


def test_a_different_shape_is_an_error(tmp_path):
    save_file({**_saved_state(), "fc.weight": torch.zeros(4, 5)}, tmp_path / "draft.safetensors")

    with pytest.raises(RuntimeError, match="fc.weight"):
        _load(_Draft(), tmp_path / "draft.safetensors")


def test_a_path_without_safetensors_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no draft weights"):
        _load(_Draft(), tmp_path)


def test_a_draft_key_filter_skips_the_shards_without_draft_weights(tmp_path):
    class _Adapter:
        def is_draft_checkpoint_key(self, key: str) -> bool:
            return key.startswith("fc.")

        def from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return hf_state_dict

    state = _saved_state()
    save_file({key: value for key, value in state.items() if key.startswith("fc.")}, tmp_path / "draft.safetensors")
    # Opening the target shard would raise, so this also proves it stays closed.
    weight_map = {key: "draft.safetensors" for key in state if key.startswith("fc.")}
    weight_map["layers.0.weight"] = "target.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "target.safetensors").write_bytes(b"not a safetensors file")
    draft = _Draft()
    draft.state_dict_adapter = _Adapter()

    _load(draft, tmp_path)

    assert (draft.fc.weight == 5).all()


def _write_earlier_run(root) -> str:
    consolidated = root / "epoch_0_step_8" / "model" / "consolidated"
    consolidated.mkdir(parents=True)
    save_file(_saved_state(), consolidated / "model.safetensors")
    torch.save({"target_layer_ids": LAYER_IDS, "mask_token_id": MASK}, root / "epoch_0_step_8" / "dspark_meta.pt")
    return str(consolidated)


def test_an_earlier_run_with_the_same_settings_loads(tmp_path):
    draft = _Draft()

    _load(draft, _write_earlier_run(tmp_path))

    assert (draft.fc.weight == 5).all()


@pytest.mark.parametrize(
    "overrides, field",
    [({"layer_ids": [2, 3]}, "target_layer_ids"), ({"mask": 6}, "mask_token_id")],
)
def test_an_earlier_run_with_other_settings_is_refused(tmp_path, overrides, field):
    draft = _Draft()
    fc_before = draft.fc.weight.detach().clone()

    with pytest.raises(ValueError, match=field):
        _load(draft, _write_earlier_run(tmp_path), **overrides)

    torch.testing.assert_close(draft.fc.weight.detach(), fc_before, rtol=0, atol=0)


def test_a_file_inside_an_earlier_run_is_checked_too(tmp_path):
    consolidated = _write_earlier_run(tmp_path)

    with pytest.raises(ValueError, match="mask_token_id"):
        _load(_Draft(), f"{consolidated}/model.safetensors", mask=6)


def test_an_earlier_run_without_recorded_settings_loads(tmp_path):
    consolidated = _write_earlier_run(tmp_path)
    torch.save({}, tmp_path / "epoch_0_step_8" / "dspark_meta.pt")
    draft = _Draft()

    _load(draft, consolidated)

    assert (draft.fc.weight == 5).all()
