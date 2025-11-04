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
import types
import sys
import builtins
import pytest

from nemo_automodel.components.config import loader


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


