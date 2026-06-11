# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the pipeline-parallel-safe Qwen3.5 dense text backbone.

These exercise ``Qwen3_5TextModelPP.forward`` — the override that lets the HF
text model survive a pipeline split (``self.layers`` rewritten from
``nn.ModuleList`` to ``nn.ModuleDict``; ``norm`` dropped to ``None`` on non-last
stages). The override is verified in isolation by stubbing HF's
``Qwen3_5TextModel.forward`` (the ``super()`` target) so no model weights or
distributed setup are required.
"""

from unittest.mock import patch

import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from nemo_automodel.components.models.qwen3_5.model import Qwen3_5TextModelPP


def _bare_pp_model() -> Qwen3_5TextModelPP:
    """A ``Qwen3_5TextModelPP`` instance without HF's heavy ``__init__``."""
    model = Qwen3_5TextModelPP.__new__(Qwen3_5TextModelPP)
    nn.Module.__init__(model)
    return model


def test_split_stage_presents_sliceable_layers_and_identity_norm():
    """On a split stage (ModuleDict layers, norm=None) HF's forward must see a
    slice-able ModuleList over the same layer objects and a callable norm."""
    model = _bare_pp_model()
    l0, l1 = nn.Linear(2, 2), nn.Linear(2, 2)
    # Splitter keys layers by their original (non-contiguous) index.
    model.layers = nn.ModuleDict({"3": l0, "7": l1})
    model.norm = None

    seen = {}

    def probe(self, *args, **kwargs):
        seen["layers_type"] = type(self.layers)
        seen["layers_objs"] = list(self.layers)
        seen["norm_type"] = type(self.norm)
        seen["kwargs"] = kwargs
        # HF slices ``self.layers[: num_hidden_layers]`` — must not raise.
        _ = self.layers[:64]
        return "HIDDEN"

    with patch.object(Qwen3_5TextModel, "forward", probe):
        out = model.forward(position_ids=None, foo=1)

    assert out == "HIDDEN"
    assert seen["layers_type"] is nn.ModuleList
    assert seen["layers_objs"] == [l0, l1]  # same objects, preserved order
    assert seen["norm_type"] is nn.Identity
    assert seen["kwargs"] == {"position_ids": None, "foo": 1}


def test_split_stage_restores_containers_after_forward():
    """After the wrapped forward, the original ModuleDict / None norm are back."""
    model = _bare_pp_model()
    layers = nn.ModuleDict({"0": nn.Linear(2, 2)})
    model.layers = layers
    model.norm = None

    with patch.object(Qwen3_5TextModel, "forward", lambda self, *a, **k: "X"):
        model.forward()

    assert model.layers is layers
    assert isinstance(model.layers, nn.ModuleDict)
    assert model.norm is None


def test_non_split_model_is_pure_passthrough():
    """Full (non-PP) model: ModuleList layers + real norm are left untouched."""
    model = _bare_pp_model()
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    norm = nn.LayerNorm(2)
    model.layers = layers
    model.norm = norm

    seen = {}

    def probe(self, *args, **kwargs):
        seen["layers"] = self.layers
        seen["norm"] = self.norm
        return "HIDDEN"

    with patch.object(Qwen3_5TextModel, "forward", probe):
        out = model.forward()

    assert out == "HIDDEN"
    # Same objects passed through — no swap occurred.
    assert seen["layers"] is layers
    assert seen["norm"] is norm
    assert model.layers is layers
    assert model.norm is norm


def test_containers_restored_when_forward_raises():
    """The finally-block must restore layers/norm even if HF's forward raises."""
    model = _bare_pp_model()
    layers = nn.ModuleDict({"0": nn.Linear(2, 2)})
    model.layers = layers
    model.norm = None

    def boom(self, *args, **kwargs):
        raise RuntimeError("forward blew up")

    with patch.object(Qwen3_5TextModel, "forward", boom):
        try:
            model.forward()
        except RuntimeError:
            pass

    assert model.layers is layers
    assert model.norm is None


def test_subclass_relationship():
    """Behavioral invariants the __class__ swap in the VLM __init__ relies on."""
    assert issubclass(Qwen3_5TextModelPP, Qwen3_5TextModel)
    # No extra __init__ — the swap reuses the HF-built instance as-is.
    assert "__init__" not in Qwen3_5TextModelPP.__dict__
    assert "forward" in Qwen3_5TextModelPP.__dict__
