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

"""Failure and provenance boundaries of D1's frozen-feature cache."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "decision_d1_feature_cache_recipe", Path(__file__).resolve().parents[3] / "examples/decision/d1_posttrain.py"
)
recipe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipe
spec.loader.exec_module(recipe)


@pytest.fixture
def training(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation", "test"):
        (data / f"{split}.jsonl").write_text("{}\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("# test model source\n")
    head = torch.nn.Linear(2, 2)
    readout = SimpleNamespace(head=head, extract_features=lambda *args: torch.tensor([0.25, 0.5]))
    model = SimpleNamespace(dtype=torch.float32, requires_grad_=lambda _: None, prepare_text_readout=lambda _: readout)
    publisher = type("Publisher", (), {"__module__": "test_publisher.model"})
    monkeypatch.setattr(recipe, "get_class_from_dynamic_module", lambda *a, **k: publisher)
    import_module = recipe.importlib.import_module
    monkeypatch.setattr(
        recipe.importlib,
        "import_module",
        lambda name, *a, **k: SimpleNamespace() if name == "test_publisher.prompt" else import_module(name, *a, **k),
    )
    monkeypatch.setattr(recipe.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        recipe.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=64)),
    )
    monkeypatch.setattr(recipe.NativeD1Config, "build", lambda _: model)
    monkeypatch.setattr(recipe, "load_records", lambda _: [])
    monkeypatch.setattr(recipe.inspect, "getfile", lambda _: str(source / "model.py"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test")
    args = Namespace(
        model="d1-3b",
        data_dir=data,
        cache_dir=tmp_path / "cache",
        input_dir=None,
        output_dir=tmp_path / "out",
        mode="base",
    )
    return recipe.D1PostTraining(args)


@pytest.mark.parametrize("library", ["torch", "transformers"])
def test_feature_cache_rejects_changed_library(training, monkeypatch, library):
    if library == "torch":
        monkeypatch.setattr(torch, "__version__", "changed-version")
    else:
        monkeypatch.setattr(recipe, "transformers_version", "changed-version")
    with pytest.raises(ValueError, match="cache does not match"):
        recipe.D1PostTraining(training.args)


def test_interrupted_feature_write_leaves_no_cache_entry(training, monkeypatch):
    training.examples = {"train": [SimpleNamespace(input_ids=[1, 2])]}
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **k: tensor(*a, **{key: value for key, value in k.items() if key != "device"})
    )
    save = torch.save

    def interrupted(value, stream):
        stream.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted write"):
        training.feature("train", 0)
    assert not (training.cache / "train-0.pt").exists()
    assert list(training.cache.glob(".train-0.pt.*")) == []
    monkeypatch.setattr(torch, "save", save)
    torch.testing.assert_close(training.feature("train", 0), tensor([0.25, 0.5]))
    torch.testing.assert_close(torch.load(training.cache / "train-0.pt", weights_only=True), tensor([0.25, 0.5]))


@pytest.mark.parametrize("library", ["torch", "transformers"])
def test_checkpoint_provenance_rejects_changed_library(training, tmp_path, library):
    prior = dict(training.provenance)
    prior[library] = "different-library-version"
    source = tmp_path / "prior"
    source.mkdir()
    (source / "provenance.json").write_text(json.dumps(prior))
    training.args.input_dir = source
    with pytest.raises(ValueError, match=f"Checkpoint provenance mismatch: {library}"):
        training.run()


def test_recipe_validates_supervision_before_model_encoding(training, monkeypatch):
    record = {
        "id": "bad",
        "state": "observation",
        "questions": {"flag": {"type": "noul"}},
        "rewards": {"flag": {"true": float("inf"), "false": 0.0}},
    }
    monkeypatch.setattr(recipe, "load_records", lambda _: [record])

    def unexpected_encoding(*args, **kwargs):
        raise AssertionError("Invalid supervision reached model encoding")

    monkeypatch.setattr(recipe, "encode_d1_record", unexpected_encoding)
    with pytest.raises(ValueError, match="finite numbers"):
        recipe.D1PostTraining(training.args)


def test_recipe_aligns_separate_targets_in_encoded_question_order(training, monkeypatch):
    record = {
        "id": "r",
        "state": "observation",
        "questions": {"first": {"type": "noul"}, "second": {"type": "noul"}},
        "answers": {"first": "true", "second": "false"},
        "rewards": {"first": {"true": 0.75, "false": -0.5}},
    }
    encoded = [
        recipe.D1TextExample("r", (1, 2), "second", ("true", "false"), (), ((1,), (2,)), 0, 1.0),
        recipe.D1TextExample("r", (3, 4), "first", ("false", "true"), (), ((4,), (3,)), 0, 1.0),
    ]
    monkeypatch.setattr(recipe, "load_records", lambda _: [record])
    monkeypatch.setattr(recipe, "encode_d1_record", lambda *args, **kwargs: encoded)
    run = recipe.D1PostTraining(training.args)
    for split in ("train", "validation", "test"):
        assert run.examples[split] == encoded
        assert [(target.question_id, target.label) for target in run.targets[split]] == [("second", 1), ("first", 1)]
        assert run.targets[split][0].rewards is None
        assert run.targets[split][1].rewards == (-0.5, 0.75)
