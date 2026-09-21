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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline


@pytest.mark.parametrize("legacy_index", [False, True])
@pytest.mark.parametrize("component_iterator", [False, True])
def test_modular_pipeline_loads_selected_local_weights(tiny_model, tmp_path, legacy_index, component_iterator):
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
        components_to_load=iter(["transformer"]) if component_iterator else ["transformer"],
        load_for_training=True,
        model_type="wan_animate2",
        local_files_only=True,
    )
    assert not managers
    assert pipe.vae is None
    assert isinstance(pipe.transformer, type(tiny_model))
    assert all(parameter.requires_grad for parameter in pipe.transformer.parameters())
    for name, value in tiny_model.state_dict().items():
        torch.testing.assert_close(value, pipe.transformer.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("model_type", [None, "flux", "wan", "ltx2", "unregistered"])
def test_other_models_keep_standard_loader_arguments_and_iterable(tmp_path, model_type):
    """Other models do not inspect modular indices or consume component iterables early."""
    (tmp_path / "modular_model_index.json").write_text("invalid json")
    (tmp_path / "model_index.json").write_text("invalid json")
    components = iter(["transformer"])
    expected_pipe = SimpleNamespace(components={})
    with patch(
        "nemo_automodel._diffusers.auto_diffusion_pipeline.DiffusionPipeline.from_pretrained",
        return_value=expected_pipe,
    ) as loader:
        pipe, managers = NeMoAutoDiffusionPipeline.from_pretrained(
            str(tmp_path),
            "positional-argument",
            model_type=model_type,
            torch_dtype=torch.float32,
            components_to_load=components,
            move_to_device=False,
            revision="requested-revision",
            local_files_only=True,
        )
    assert pipe is expected_pipe
    assert not managers
    assert list(components) == ["transformer"]
    loader.assert_called_once_with(
        str(tmp_path),
        "positional-argument",
        torch_dtype=torch.float32,
        revision="requested-revision",
        local_files_only=True,
    )
