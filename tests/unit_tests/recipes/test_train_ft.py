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

import pytest
from nemo_automodel.recipes.llm.train_ft import _get_packed_sequence_config, build_validation_dataloader
from nemo_automodel.components.config.loader import ConfigNode
from unittest.mock import patch
import logging


@pytest.mark.parametrize(
    "has_packed_sequence, is_hf_model, cp_size, return_val, raises",
    [
        (True, True, 1, {"attn_implementation": "flash_attention_2"}, None),
        (True, True, 2, {"attn_implementation": "sdpa"}, ValueError),
        (True, False, 1, {}, None),
        (True, False, 2, {}, None),
        (False, True, 1, {}, None),
        (False, True, 2, {'attn_implementation': 'sdpa'}, None),
        (False, False, 1, {}, None),
        (False, False, 2, {}, None),
    ],
)
def test_get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size, return_val, raises):
    if raises:
        with pytest.raises(raises):
            _get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size)
    else:
        assert _get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size) == return_val

def test_build_validation_dataloader_pp_enabled(caplog):
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
        }
    )

    with caplog.at_level(logging.WARNING):
        result = build_validation_dataloader(cfg, dp_world_size=2, dp_rank=0, pp_enabled=True)

    assert result == {}
    assert any("Validation is not supported for pipeline parallelism" in r.message for r in caplog.records)


def test_build_validation_dataloader_collects_and_names_properly():
    # Multiple validation dataset keys with different separators
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
            "distributed": {"cp_size": 3},
            "step_scheduler": {
                "local_batch_size": 8,
                "global_batch_size": 16,
                "max_steps": 123,
                "val_every_steps": 10,
            },
            # Keys to be discovered via cfg.to_dict().keys()
            "validation_dataset": {"some": "cfg"},
            "validation_dataset_val": {"some": "cfg"},
            "validation_dataset-test": {"some": "cfg"},
            "validation_dataset.foo": {"some": "cfg"},
        }
    )

    expected_names = {"default", "val", "test", "foo"}

    with patch("nemo_automodel.recipes.llm.train_ft.build_dataloader", return_value=("dl", "tok")) as mock_build:
        result = build_validation_dataloader(cfg, dp_world_size=4, dp_rank=1, pp_enabled=False)

    # Assert keys are correctly generated
    assert set(result.keys()) == expected_names
    # Values should be the first element of the tuple returned by build_dataloader
    assert set(result.values()) == {"dl"}
    # build_dataloader called once per validation dataset
    assert mock_build.call_count == 4

    # Inspect one call for important kwargs
    _, kwargs = mock_build.call_args
    assert kwargs["dp_world_size"] == 4
    assert kwargs["dp_rank"] == 1
    assert kwargs["pp_enabled"] is False
    assert kwargs["supports_seq_lens"] is True
    assert kwargs["cp_size"] == 3


def test_build_validation_dataloader_no_validation_keys():
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
        }
    )

    with patch("nemo_automodel.recipes.llm.train_ft.build_dataloader") as mock_build:
        result = build_validation_dataloader(cfg, dp_world_size=1, dp_rank=0, pp_enabled=False)

    assert result == {}
    mock_build.assert_not_called()