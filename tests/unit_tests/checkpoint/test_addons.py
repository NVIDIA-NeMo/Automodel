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

import os

from nemo_automodel.components.checkpoint.addons import _maybe_save_custom_model_code


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_maybe_save_custom_model_code_copies_py_files_and_structure(tmp_path):
    # Arrange: create a nested source tree with .py and non-.py files
    src_root = tmp_path / "src_model_code"
    dst_root = tmp_path / "hf_meta"
    src_root.mkdir(parents=True)
    dst_root.mkdir(parents=True)

    files = {
        "main.py": "print('main')\n",
        "pkg/__init__.py": "# pkg init\n",
        "pkg/subpkg/module.py": "def foo():\n    return 1\n",
        "pkg/readme.txt": "do not copy\n",
    }
    for rel, content in files.items():
        _write(os.path.join(src_root, rel), content)

    # Act
    _maybe_save_custom_model_code(str(src_root), str(dst_root))

    # Assert: .py files copied with preserved structure; non-.py ignored
    assert (dst_root / "main.py").exists()
    assert (dst_root / "pkg" / "__init__.py").exists()
    assert (dst_root / "pkg" / "subpkg" / "module.py").exists()
    assert not (dst_root / "pkg" / "readme.txt").exists()

    # Verify contents match
    with open(dst_root / "pkg" / "subpkg" / "module.py", "r") as f:
        assert "def foo()" in f.read()


def test_maybe_save_custom_model_code_noop_for_none_or_non_dir(tmp_path):
    dst_root = tmp_path / "hf_meta"
    dst_root.mkdir(parents=True)

    # None input should be a no-op
    _maybe_save_custom_model_code(None, str(dst_root))
    assert list(dst_root.rglob("*.py")) == []

    # Non-directory input should be a no-op
    some_file = tmp_path / "not_a_dir.txt"
    some_file.write_text("hello")
    _maybe_save_custom_model_code(str(some_file), str(dst_root))
    assert list(dst_root.rglob("*.py")) == []


