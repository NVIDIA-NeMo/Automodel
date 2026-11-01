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

"""Preserve each D1 release's precision when loading and checking its baseline."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

RECIPE = runpy.run_path(str(Path(__file__).resolve().parents[3] / "examples/decision/d1_posttrain.py"))


class CPUModel(nn.Module):
    """Keep loader-routing tests on CPU while retaining real tensor dtypes."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def cuda(self) -> "CPUModel":
        return self


@pytest.mark.parametrize("release,expected_dtype", [("d1-3b", torch.bfloat16), ("d1-omni", torch.float32)])
def test_native_config_keeps_publisher_recommended_precision(monkeypatch, release, expected_dtype):
    def load(repository, *, revision, code_revision, dtype, **kwargs):
        assert (repository, revision) == RECIPE["RELEASES"][release][:2]
        assert code_revision == revision
        return CPUModel(dtype)

    monkeypatch.setattr(RECIPE["NeMoAutoModelForImageTextToText"], "from_pretrained", load)
    model = RECIPE["NativeD1Config"](release).build()
    assert model.dtype == expected_dtype
    assert not model.training


@pytest.mark.parametrize("expected_dtype", [torch.float32, torch.bfloat16])
def test_publisher_baseline_uses_actual_native_precision(monkeypatch, tmp_path, expected_dtype):
    def load(repository, *, revision, dtype):
        assert repository == "LiquidAI/d1-omni-600M"
        assert dtype == expected_dtype
        return CPUModel(dtype)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    training = RECIPE["D1PostTraining"].__new__(RECIPE["D1PostTraining"])
    training.repository, training.revision, _ = RECIPE["RELEASES"]["d1-omni"]
    training.model = CPUModel(expected_dtype)
    training.head = nn.Linear(3, 1, dtype=torch.float32)
    training.published_class = SimpleNamespace(from_pretrained=load)
    training.omni = True
    training.tokenizer = object()
    training.records = {"validation": []}
    training.check_published_baseline(tmp_path)
    report = json.loads((tmp_path / "published-baseline-parity.json").read_text())
    assert report["dtype"] == str(expected_dtype).removeprefix("torch.")
    assert report["publisher_dtype"] == report["dtype"]
    assert report["readout_dtype"] == "float32"
    assert report["absolute_probability_tolerance"] == 0.02
