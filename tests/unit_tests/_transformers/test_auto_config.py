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
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from nemo_automodel._transformers.auto_config import NeMoAutoConfig
from nemo_automodel.components.config.loader import ConfigNode

A, B = "a" * 40, "b" * 40
REPO = "test/config-race"


def test_default_resolution_refreshes_stale_branch(hf_config_hub):
    root, _, _, requests = hf_config_hub
    config = NeMoAutoConfig.from_pretrained(REPO, cache_dir=root)
    assert config.n_embd == 64
    assert config._commit_hash == B
    assert any(request.method == "HEAD" for request in requests)


@pytest.mark.parametrize("kwargs", [{"local_files_only": True}, {"revision": A}])
def test_explicit_cached_revision_does_not_require_network(hf_config_hub, kwargs):
    root, _, _, requests = hf_config_hub
    config = NeMoAutoConfig.from_pretrained(REPO, cache_dir=root, **kwargs)
    assert config.n_embd == 32
    assert config._commit_hash == A
    assert requests == []


def test_force_download_refreshes_cached_file(hf_config_hub):
    root, _, _, requests = hf_config_hub
    config = NeMoAutoConfig.from_pretrained(REPO, cache_dir=root, force_download=True)
    assert config.n_embd == 64
    assert config._commit_hash == B
    assert any(request.method == "GET" for request in requests)


def test_config_dict_and_overrides_use_selected_snapshot(hf_config_hub):
    root, _, _, _ = hf_config_hub
    data, unused = NeMoAutoConfig.get_config_dict(REPO, cache_dir=root, n_layer=3)
    assert data["n_embd"] == 64
    assert data["_commit_hash"] == B
    assert unused["n_layer"] == 3
    config, unused = NeMoAutoConfig.from_pretrained(
        REPO, cache_dir=root, n_layer=3, sentinel=7, return_unused_kwargs=True
    )
    assert config.n_layer == 3
    assert unused["sentinel"] == 7


@pytest.mark.parametrize("as_file", [False, True])
def test_local_config_needs_no_hub(hf_config_hub, as_file):
    _, cache, _, requests = hf_config_hub
    path = cache / "snapshots" / A
    config = NeMoAutoConfig.from_pretrained(path / "config.json" if as_file else path, n_layer=3)
    assert config.n_embd == 32
    assert config.n_layer == 3
    assert requests == []


def test_explicit_yaml_target_preserves_nested_overrides(hf_config_hub):
    root, _, _, _ = hf_config_hub
    cfg = ConfigNode(
        {
            "_target_": "nemo_automodel._transformers.auto_config.NeMoAutoConfig.from_pretrained",
            "pretrained_model_name_or_path": REPO,
            "cache_dir": str(root),
            "n_layer": 3,
        }
    )
    config = cfg.instantiate()
    assert config.n_embd == 64
    assert config.n_layer == 3
    assert config._commit_hash == B


def test_plain_transformers_target_keeps_online_semantics(hf_config_hub):
    root, _, _, requests = hf_config_hub
    config = ConfigNode(
        {"_target_": AutoConfig.from_pretrained, "pretrained_model_name_or_path": REPO, "cache_dir": str(root)}
    ).instantiate()
    assert config.n_embd == 64
    assert requests


def _truncate_ref(ref, download_done, ref_empty, release_writer):
    assert download_done.wait(10)
    with ref.open("w") as stream:
        ref_empty.set()
        assert release_writer.wait(10)
        stream.write(B)


@pytest.mark.parametrize("writer_kind", ["thread", "process"])
@pytest.mark.parametrize("loader", [AutoConfig, NeMoAutoConfig])
def test_concurrent_ref_truncation_after_download(hf_config_hub, monkeypatch, loader, writer_kind):
    """Hold a real writer inside the ref's truncate/write window during the read."""
    root, _, ref, _ = hf_config_hub
    context = multiprocessing.get_context("spawn") if writer_kind == "process" else threading
    download_done, ref_empty, release_writer = (context.Event() for _ in range(3))

    def interleaved_download(*args, **kwargs):
        result = hf_hub_download(*args, **kwargs)
        download_done.set()
        assert ref_empty.wait(5)
        return result

    monkeypatch.setattr("transformers.utils.hub.hf_hub_download", interleaved_download)
    monkeypatch.setattr("nemo_automodel._transformers.auto_config.hf_hub_download", interleaved_download)
    writer_class = context.Process if writer_kind == "process" else context.Thread
    writing = writer_class(target=_truncate_ref, args=(ref, download_done, ref_empty, release_writer))
    writing.start()
    try:
        if loader is AutoConfig:
            # Positive control: the unfixed path reproduces issue #3975.
            with pytest.raises(ValueError, match="Unrecognized model"):
                loader.from_pretrained(REPO, cache_dir=root)
        else:
            config = loader.from_pretrained(REPO, cache_dir=root)
            assert config.n_embd == 64
            assert config._commit_hash == B
    finally:
        release_writer.set()
        writing.join(timeout=10)
    assert not writing.is_alive()
    if writer_kind == "process":
        assert writing.exitcode == 0
    assert ref.read_text() == B


def test_two_threaded_loads_keep_their_downloaded_snapshot(hf_config_hub, monkeypatch):
    root, _, ref, _ = hf_config_hub
    downloaded = threading.Barrier(2)
    ref_changed = threading.Barrier(2)

    def interleaved_download(*args, **kwargs):
        result = hf_hub_download(*args, **kwargs)
        if downloaded.wait(timeout=5) == 0:
            ref.write_text(A)
        ref_changed.wait(timeout=5)
        return result

    monkeypatch.setattr("nemo_automodel._transformers.auto_config.hf_hub_download", interleaved_download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(NeMoAutoConfig.from_pretrained, REPO, cache_dir=root) for _ in range(2)]
        configs = [future.result(timeout=5) for future in futures]
    assert [(config.n_embd, config._commit_hash) for config in configs] == [(64, B), (64, B)]
    assert ref.read_text() == A


def test_shipped_yaml_uses_explicit_nemo_config_adapter():
    root = Path(__file__).resolve().parents[3]
    migrated = []
    for path in (root / "examples").rglob("*.yaml"):
        source = path.read_text()
        assert "transformers.AutoConfig.from_pretrained" not in source, path
        if "auto_config.NeMoAutoConfig.from_pretrained" in source:
            assert yaml.safe_load(source) is not None
            migrated.append(path)
    assert migrated


def test_cold_cache_downloads_current_config(hf_config_hub):
    root, cache, _, requests = hf_config_hub
    (cache / "snapshots" / B / "config.json").unlink()
    config = NeMoAutoConfig.from_pretrained(REPO, cache_dir=root)
    assert config.n_embd == 64
    assert config._commit_hash == B
    assert any(request.method == "GET" for request in requests)


def test_authentication_failure_is_propagated(hf_config_hub, monkeypatch):
    root, _, _, _ = hf_config_hub

    def denied(client, request, *args, **kwargs):
        return httpx.Response(401, request=request, headers={"X-Error-Code": "RepoNotFound"})

    monkeypatch.setattr(httpx.Client, "send", denied)
    from huggingface_hub.errors import RepositoryNotFoundError

    with pytest.raises(RepositoryNotFoundError):
        NeMoAutoConfig.from_pretrained(REPO, cache_dir=root / "empty")


def test_subfolder_and_token_are_preserved(hf_config_hub):
    root, cache, _, requests = hf_config_hub
    nested = cache / "snapshots" / B / "nested"
    nested.mkdir()
    (nested / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_embd": 96}))
    config = NeMoAutoConfig.from_pretrained(REPO, cache_dir=root, subfolder="nested", token="test-token")
    assert config.n_embd == 96
    assert config._commit_hash == B
    assert requests[0].url.path.endswith("/nested/config.json")
    assert requests[0].headers["authorization"] == "Bearer test-token"


@pytest.mark.parametrize("code_revision, expected", [(None, "b"), (A, "a")])
def test_remote_code_follows_resolved_commit_or_explicit_override(hf_config_hub, monkeypatch, code_revision, expected):
    root, cache, _, _ = hf_config_hub
    monkeypatch.setattr("transformers.dynamic_module_utils.HF_MODULES_CACHE", str(root / "modules"))
    for revision in (A, B):
        snapshot = cache / "snapshots" / revision
        (snapshot / "configuration_cache.py").write_text(
            "from transformers import PretrainedConfig\n"
            "class SharedCacheConfig(PretrainedConfig):\n"
            "    model_type = 'cache_test_custom'\n"
            f"    implementation_revision = {revision[0]!r}\n"
        )
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "cache_test_custom",
                    "auto_map": {"AutoConfig": "configuration_cache.SharedCacheConfig"},
                    "hidden_size": 64,
                }
            )
        )
    config = NeMoAutoConfig.from_pretrained(REPO, cache_dir=root, trust_remote_code=True, code_revision=code_revision)
    assert config.hidden_size == 64
    assert config._commit_hash == B
    assert config.implementation_revision == expected
