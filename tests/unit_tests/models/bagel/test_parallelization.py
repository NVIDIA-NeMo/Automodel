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

"""Tests for BAGEL's model-owned parallelization sidecar."""

import torch.nn as nn

from nemo_automodel.components.models.bagel import parallelization


def _model() -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.model = nn.Module()
    model.model.language_model.model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    model.model.vit_model = nn.Module()
    model.model.vit_model.vision_model = nn.Module()
    model.model.vit_model.vision_model.encoder = nn.Module()
    model.model.vit_model.vision_model.encoder.layers = nn.ModuleList([nn.Linear(2, 2)])
    return model


def test_full_layer_checkpointing_is_model_owned(monkeypatch):
    model = _model()
    wrapped = []

    def fake_wrapper(module, **kwargs):
        wrapper = nn.Module()
        wrapper._checkpoint_wrapped_module = module
        wrapped.append((module, kwargs))
        return wrapper

    monkeypatch.setattr(parallelization, "checkpoint_wrapper", fake_wrapper)
    parallelization._apply_full_layer_checkpointing(model)

    assert len(wrapped) == 3
    assert all(hasattr(layer, "_checkpoint_wrapped_module") for layer in model.model.language_model.model.layers)
    assert all(
        hasattr(layer, "_checkpoint_wrapped_module") for layer in model.model.vit_model.vision_model.encoder.layers
    )
