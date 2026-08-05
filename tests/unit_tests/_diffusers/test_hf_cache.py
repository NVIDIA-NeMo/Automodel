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

from unittest.mock import patch

from nemo_automodel._diffusers import _hf_cache
from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

MODULE = "nemo_automodel._diffusers._hf_cache"


def test_resolve_returns_local_path_unchanged(tmp_path):
    # A directory that already exists should be returned verbatim without any
    # Hub interaction.
    with patch(f"{MODULE}.snapshot_download") as mock_sd:
        assert resolve_diffusion_model_dir(str(tmp_path)) == str(tmp_path)
    mock_sd.assert_not_called()


def test_resolve_offline_uses_cache_only(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(_hf_cache, "HF_HUB_AVAILABLE", True)

    with patch(f"{MODULE}.snapshot_download", return_value="/cache/snapshot") as mock_sd:
        resolved = resolve_diffusion_model_dir("some/repo-id")

    # Offline: no cold-cache download, single cache-only resolution.
    assert resolved == "/cache/snapshot"
    mock_sd.assert_called_once_with("some/repo-id", local_files_only=True)


def test_resolve_online_downloads_then_resolves_locally(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(_hf_cache, "HF_HUB_AVAILABLE", True)

    with patch(f"{MODULE}.snapshot_download", return_value="/cache/snapshot") as mock_sd:
        resolved = resolve_diffusion_model_dir("some/repo-id")

    assert resolved == "/cache/snapshot"
    # Online: fetch once (cold cache), then resolve the local dir without revalidation.
    assert mock_sd.call_count == 2
    assert mock_sd.call_args_list[0].args == ("some/repo-id",)
    assert mock_sd.call_args_list[1].kwargs == {"local_files_only": True}


def test_resolve_passthrough_when_hub_unavailable(monkeypatch):
    monkeypatch.setattr(_hf_cache, "HF_HUB_AVAILABLE", False)
    assert resolve_diffusion_model_dir("some/repo-id") == "some/repo-id"
