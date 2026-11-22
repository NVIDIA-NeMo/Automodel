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
import sys
import types
from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.config import loader


def _write_payload(path: Path, marker: Path) -> Path:
    """Write a Python file whose top-level code creates ``marker`` when the file runs."""
    path.write_text(
        f"open({str(marker)!r}, 'w').close()\n\n\ndef entry():\n    return 'loaded'\n\n\n_private = entry\n"
    )
    return path


@pytest.mark.parametrize("key", ["_target_", "collate_fn"])
def test_config_file_target_outside_package_requires_opt_in(tmp_path, monkeypatch, key):
    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", False)
    marker = tmp_path / "ran"
    payload = _write_payload(tmp_path / "helper.py", marker)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"dataset": {key: f"{payload}:entry"}}))

    with pytest.raises(ImportError, match="NEMO_ENABLE_USER_MODULES"):
        loader.load_yaml_config(cfg_path)
    # The loader itself enforces the opt-in, so a dotted target that reaches it is refused too.
    with pytest.raises(ImportError, match="NEMO_ENABLE_USER_MODULES"):
        loader.load_module_from_file(payload)
    assert not marker.exists()

    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", True)
    cfg = loader.load_yaml_config(cfg_path)
    assert getattr(cfg.dataset, key)() == "loaded"
    assert marker.exists()


def test_file_target_private_attr_is_rejected_before_the_file_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", True)
    marker = tmp_path / "ran"
    payload = _write_payload(tmp_path / "helper.py", marker)

    with pytest.raises(ImportError, match="_private"):
        loader._resolve_target(f"{payload}:_private")
    assert not marker.exists()


def test_file_target_containment_resolves_symlinks_and_parent_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", False)
    package_dir, outside_dir = tmp_path / "package", tmp_path / "outside"
    package_dir.mkdir()
    outside_dir.mkdir()
    monkeypatch.setattr(loader, "SAFE_BASE_DIR", package_dir.resolve())
    inside_marker, outside_marker = tmp_path / "inside_ran", tmp_path / "outside_ran"
    inside = _write_payload(package_dir / "inside.py", inside_marker)
    outside = _write_payload(outside_dir / "outside.py", outside_marker)
    link = package_dir / "link.py"
    link.symlink_to(outside)

    assert loader._resolve_target(f"{inside}:entry")() == "loaded"
    for escaping in (link, package_dir / ".." / "outside" / "outside.py"):
        with pytest.raises(ImportError, match="NEMO_ENABLE_USER_MODULES"):
            loader._resolve_target(f"{escaping}:entry")
    assert inside_marker.exists()
    assert not outside_marker.exists()


def test_is_allowed_module_respects_prefix_allowlist_when_spec_missing(monkeypatch):
    # Ensure user modules are disabled so allowlist applies
    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", False)

    # Simulate that nothing is importable via find_spec
    monkeypatch.setattr(loader.importlib.util, "find_spec", lambda top_level: None)

    # Ensure a disallowed fake top-level is not already imported
    monkeypatch.delitem(sys.modules, "evilpkg", raising=False)

    # Allowed because top-level prefix 'torch' is in ALLOWED_IMPORT_PREFIXES
    assert loader._is_allowed_module("torch.nn") is True

    # Disallowed because top-level prefix 'evilpkg' is not allowlisted
    assert loader._is_allowed_module("evilpkg.mod") is False


def test_resolve_target_blocks_disallowed_prefix_when_spec_missing(monkeypatch):
    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", False)
    monkeypatch.setattr(loader.importlib.util, "find_spec", lambda top_level: None)

    # Any attempt to import a module should fail (simulate not found)
    def always_fail_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(loader.importlib, "import_module", always_fail_import)

    with pytest.raises(ImportError) as excinfo:
        loader._resolve_target("evilpkg.something")

    assert "Cannot resolve target" in str(excinfo.value)


def test_resolve_target_allows_allowed_prefix_when_import_succeeds(monkeypatch):
    monkeypatch.setattr(loader, "ENABLE_USER_MODULES", False)
    # Make discovery say nothing is importable unless allowlist passes
    monkeypatch.setattr(loader.importlib.util, "find_spec", lambda top_level: None)

    # Create a dummy module for an allowlisted top-level (e.g., 'torch')
    dummy = types.ModuleType("torch")
    sentinel = object()
    setattr(dummy, "fakeattr", sentinel)

    def fake_import(name):
        if name == "torch":
            return dummy
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(loader.importlib, "import_module", fake_import)

    # Since 'torch' is allowlisted, this should resolve successfully
    obj = loader._resolve_target("torch.fakeattr")
    assert obj is sentinel


def test_redact_top_level_sensitive_keys():
    data = {
        "username": "alice",
        "password": "supersecret",
        "api_key": "AKIA...",
        "Auth": "Bearer something",
    }

    redacted = loader._redact(data)

    assert redacted["username"] == "alice"
    assert redacted["password"] == "******"
    assert redacted["api_key"] == "******"
    assert redacted["Auth"] == "******"  # case-insensitive match via substring 'auth'


def test_redact_nested_sensitive_keys_and_lists():
    data = {
        "outer": {
            "token": "t0k3n",
            "inner": {"normal": 1, "SECRET": "dontshow"},
        },
        "items": [
            {"name": "x", "authorization": "AAA"},
            {"nested": {"ApiKey": "BBB"}},
            123,
        ],
    }

    redacted = loader._redact(data)

    assert redacted["outer"]["token"] == "******"
    assert redacted["outer"]["inner"]["normal"] == 1
    assert redacted["outer"]["inner"]["SECRET"] == "******"

    assert redacted["items"][0]["name"] == "x"
    assert redacted["items"][0]["authorization"] == "******"
    assert redacted["items"][1]["nested"]["ApiKey"] == "******"
    assert redacted["items"][2] == 123


def test_redact_does_not_touch_non_sensitive_keys():
    data = {"email": "a@b.com", "profile": {"city": "nyc"}}
    redacted = loader._redact(data)
    assert redacted == {"email": "a@b.com", "profile": {"city": "nyc"}}
