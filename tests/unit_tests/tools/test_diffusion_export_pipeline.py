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

import json
import os

import pytest

from tools.diffusion.export_diffusers_pipeline import BASE_COMPONENTS, export_diffusers_pipeline


def _make_base_pipeline(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}))
    for component in BASE_COMPONENTS:
        comp_dir = base_dir / component
        comp_dir.mkdir()
        (comp_dir / "config.json").write_text(json.dumps({"component": component}))


def _make_student(student_dir):
    student_dir.mkdir(parents=True, exist_ok=True)
    (student_dir / "config.json").write_text(json.dumps({"_class_name": "QwenImageTransformer2DModel"}))
    (student_dir / "model-00001-of-00001.safetensors").write_bytes(b"fake-weights")


def test_export_symlinks_by_default(tmp_path):
    base_dir = tmp_path / "base"
    student_dir = tmp_path / "student"
    output_dir = tmp_path / "out"
    _make_base_pipeline(base_dir)
    _make_student(student_dir)

    export_diffusers_pipeline(student_dir, base_dir, output_dir, copy=False)

    assert (output_dir / "model_index.json").is_file()
    assert json.loads((output_dir / "model_index.json").read_text()) == {"_class_name": "QwenImagePipeline"}

    transformer_dir = output_dir / "transformer"
    assert os.path.islink(transformer_dir)
    assert os.path.realpath(transformer_dir) == os.path.realpath(student_dir)

    for component in BASE_COMPONENTS:
        comp_path = output_dir / component
        assert os.path.islink(comp_path)
        assert os.path.realpath(comp_path) == os.path.realpath(base_dir / component)


def test_export_copy_mode_produces_real_files(tmp_path):
    base_dir = tmp_path / "base"
    student_dir = tmp_path / "student"
    output_dir = tmp_path / "out"
    _make_base_pipeline(base_dir)
    _make_student(student_dir)

    export_diffusers_pipeline(student_dir, base_dir, output_dir, copy=True)

    transformer_dir = output_dir / "transformer"
    assert transformer_dir.is_dir()
    assert not os.path.islink(transformer_dir)
    assert (transformer_dir / "model-00001-of-00001.safetensors").read_bytes() == b"fake-weights"

    for component in BASE_COMPONENTS:
        comp_path = output_dir / component
        assert comp_path.is_dir()
        assert not os.path.islink(comp_path)


def test_export_overwrites_existing_output(tmp_path):
    """Re-running export against the same output_dir (e.g. after retraining) must replace
    stale links/files rather than erroring or leaving mixed old/new state."""
    base_dir = tmp_path / "base"
    student_dir = tmp_path / "student"
    other_student_dir = tmp_path / "student_v2"
    output_dir = tmp_path / "out"
    _make_base_pipeline(base_dir)
    _make_student(student_dir)
    _make_student(other_student_dir)
    (other_student_dir / "model-00001-of-00001.safetensors").write_bytes(b"newer-weights")

    export_diffusers_pipeline(student_dir, base_dir, output_dir, copy=False)
    export_diffusers_pipeline(other_student_dir, base_dir, output_dir, copy=False)

    assert os.path.realpath(output_dir / "transformer") == os.path.realpath(other_student_dir)


def test_export_missing_student_path_raises(tmp_path):
    base_dir = tmp_path / "base"
    _make_base_pipeline(base_dir)

    with pytest.raises(FileNotFoundError, match="student_path"):
        export_diffusers_pipeline(tmp_path / "does_not_exist", base_dir, tmp_path / "out")


def test_export_missing_base_pipeline_raises(tmp_path):
    student_dir = tmp_path / "student"
    _make_student(student_dir)

    with pytest.raises(FileNotFoundError, match="base_pipeline_path"):
        export_diffusers_pipeline(student_dir, tmp_path / "does_not_exist", tmp_path / "out")


def test_export_missing_base_component_raises(tmp_path):
    base_dir = tmp_path / "base"
    student_dir = tmp_path / "student"
    base_dir.mkdir()
    (base_dir / "model_index.json").write_text(json.dumps({}))
    # Deliberately omit the component subdirs.
    _make_student(student_dir)

    with pytest.raises(FileNotFoundError, match="vae"):
        export_diffusers_pipeline(student_dir, base_dir, tmp_path / "out")
