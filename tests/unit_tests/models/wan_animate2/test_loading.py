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

"""Load a real tiny modular checkpoint entirely offline."""

import json
from unittest.mock import patch

import pytest
import torch

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline


@pytest.mark.parametrize("legacy_index", [False, True])
def test_modular_pipeline_loads_selected_local_weights(tiny_model, tmp_path, legacy_index):
    """A stale pipeline name and remote component revision cannot select other weights."""
    tiny_model.save_pretrained(tmp_path / "transformer")
    index = {
        "_class_name": "WanAnimate2ModularPipeline",
        "_blocks_class_name": "WanAnimate2Blocks",
        "transformer": [
            "diffusers",
            "WanAnimate2Transformer3DModel",
            {
                "type_hint": ["diffusers", "WanAnimate2Transformer3DModel"],
                "pretrained_model_name_or_path": "unused/remote-repository",
                "subfolder": "transformer",
                "revision": "unavailable-revision",
            },
        ],
    }
    (tmp_path / "modular_model_index.json").write_text(json.dumps(index))
    if legacy_index:
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "WanAnimate2Pipeline"}))
    pipe, managers = NeMoAutoDiffusionPipeline.from_pretrained(
        str(tmp_path),
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        components_to_load=["transformer"],
        load_for_training=True,
        local_files_only=True,
    )
    assert not managers
    assert pipe.vae is None
    assert isinstance(pipe.transformer, type(tiny_model))
    assert all(parameter.requires_grad for parameter in pipe.transformer.parameters())
    for name, value in tiny_model.state_dict().items():
        torch.testing.assert_close(value, pipe.transformer.state_dict()[name], rtol=0, atol=0)


def test_supported_standard_pipeline_keeps_existing_loader(tmp_path):
    """A modular index does not change loading of supported standard pipelines."""
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _load_pretrained_pipeline

    (tmp_path / "modular_model_index.json").write_text("{}")
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline"}))
    with patch("nemo_automodel._diffusers.auto_diffusion_pipeline.DiffusionPipeline.from_pretrained") as loader:
        pipe = _load_pretrained_pipeline(str(tmp_path), (), torch_dtype=torch.float32, components_to_load=None)
    assert pipe is loader.return_value
    loader.assert_called_once_with(str(tmp_path), torch_dtype=torch.float32)
