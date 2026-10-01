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

"""Tests for adapter-owned parallelization metadata."""

import sys

import torch.nn as nn

from nemo_automodel._transformers.model_parallelization import configure_parallelization_metadata
from nemo_automodel.components.distributed.parallelizer import _extract_model_layer_groups


def test_adapter_attaches_layer_metadata_without_model_imports():
    class Qwen2VLForConditionalGeneration(nn.Module):
        pass

    before = {name for name in sys.modules if name.startswith("transformers.models.") and ".modeling_" in name}
    result = configure_parallelization_metadata(Qwen2VLForConditionalGeneration)
    after = {name for name in sys.modules if name.startswith("transformers.models.") and ".modeling_" in name}

    assert result is Qwen2VLForConditionalGeneration
    assert "language" in result.parallel_layer_groups
    assert after == before


def test_adapter_metadata_drives_generic_layer_extraction():
    class Model(nn.Module):
        parallel_layer_groups = {"language": ("decoder.layers",)}

        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    model = Model()
    assert _extract_model_layer_groups(model) == {"language": list(model.decoder.layers)}
