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

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


def _processor_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "nemo_automodel/components/models/llama_nemotron_vl/processor.py"


def _load_processor_module():
    module_name = "_llama_nemotron_vl_processor_test"
    spec = importlib.util.spec_from_file_location(module_name, _processor_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    def __init__(self):
        self.model_input_names = ["input_ids", "attention_mask"]
        self.model_max_length = 128
        self.padding_side = "right"
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [[1] for _ in text], "attention_mask": [[1] for _ in text]}


def _make_processor(monkeypatch):
    module = _load_processor_module()
    monkeypatch.setattr(module.ProcessorMixin, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    tokenizer = _FakeTokenizer()
    return module.LlamaNemotronVLProcessor(tokenizer=tokenizer), tokenizer


def test_processor_source_has_no_nemo_automodel_imports():
    tree = ast.parse(_processor_path().read_text())

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    automodel_imports = [
        module for module in imported_modules if module == "nemo_automodel" or module.startswith("nemo_automodel.")
    ]
    assert automodel_imports == []


def test_text_only_call_ignores_image_only_layout_kwargs(monkeypatch):
    processor, tokenizer = _make_processor(monkeypatch)

    output = processor(
        text="hello",
        text_kwargs={"max_length": 7},
        images_kwargs={"pixel_values_layout": "per_image"},
        common_kwargs={"return_tensors": "pt"},
        pixel_values_layout="flat_tiles",
    )

    assert output == {"input_ids": [[1]], "attention_mask": [[1]]}
    assert tokenizer.calls == [
        (
            ["hello"],
            {
                "padding": True,
                "truncation": True,
                "max_length": 7,
                "pad_to_multiple_of": None,
                "return_tensors": "pt",
            },
        )
    ]


def test_text_only_call_rejects_unsupported_image_kwargs(monkeypatch):
    processor, _ = _make_processor(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported images_kwargs"):
        processor(text="hello", images_kwargs={"resize": True})
