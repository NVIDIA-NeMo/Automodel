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

import pytest

from nemo_automodel.components.config.loader import ConfigNode


@pytest.fixture(autouse=True)
def enable_cache_first(monkeypatch):
    monkeypatch.delenv("NEMO_AUTOMODEL_HF_LOCAL_FILES_FIRST", raising=False)


def test_warm_hf_cache_loads_config_without_network(tmp_path, monkeypatch):
    from transformers import AutoConfig

    repo_cache = tmp_path / "models--test--model"
    revision = "a" * 40
    snapshot = repo_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 32}))

    def reject_network(*args, **kwargs):
        raise AssertionError("A warm cache must not require a network request")

    monkeypatch.setattr("httpx.Client.send", reject_network)
    cfg = ConfigNode(
        {
            "_target_": AutoConfig.from_pretrained,
            "pretrained_model_name_or_path": "test/model",
            "cache_dir": str(tmp_path),
        }
    )
    result = cfg.instantiate()
    assert result.model_type == "llama"
    assert result.hidden_size == 32
    assert (refs / "main").read_text() == revision


def test_cache_miss_falls_back_to_download(tmp_path):
    cached_config = tmp_path / "config.json"

    def from_pretrained(model_id, *, local_files_only=False):
        if not cached_config.exists():
            if local_files_only:
                raise OSError("Configuration is not cached")
            cached_config.write_text(json.dumps({"model_type": model_id}))
        return json.loads(cached_config.read_text())

    cfg = ConfigNode({"_target_": from_pretrained})
    assert cfg.instantiate("test-model") == {"model_type": "test-model"}
    assert cached_config.is_file()


@pytest.mark.parametrize("local_only", [True, False])
@pytest.mark.parametrize("runtime_override", [True, False])
def test_explicit_cache_policy_is_preserved(local_only, runtime_override):
    def from_pretrained(*, local_files_only):
        return local_files_only

    cfg = ConfigNode(
        {"_target_": from_pretrained, "local_files_only": not local_only if runtime_override else local_only}
    )
    kwargs = {"local_files_only": local_only} if runtime_override else {}
    assert cfg.instantiate(**kwargs) is local_only


def test_environment_opt_out_preserves_default(monkeypatch):
    monkeypatch.setenv("NEMO_AUTOMODEL_HF_LOCAL_FILES_FIRST", "0")

    def from_pretrained(*, local_files_only=False):
        return local_files_only

    assert ConfigNode({"_target_": from_pretrained}).instantiate() is False


def test_callable_without_local_files_only_remains_supported():
    def from_pretrained(model_id):
        return model_id

    assert ConfigNode({"_target_": from_pretrained}).instantiate("test-model") == "test-model"


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_unrelated_failure_is_not_retried(error_type):
    attempts = []
    error = error_type("Invalid model configuration")

    def from_pretrained(**kwargs):
        attempts.append(kwargs)
        raise error

    with pytest.raises(error_type) as caught:
        ConfigNode({"_target_": from_pretrained}).instantiate()
    assert caught.value is error
    assert len(attempts) == 1


def test_download_failure_is_propagated():
    error = OSError("Could not download configuration")

    def from_pretrained(*, local_files_only=False):
        if local_files_only:
            raise OSError("Configuration is not cached")
        raise error

    with pytest.raises(OSError) as caught:
        ConfigNode({"_target_": from_pretrained}).instantiate()
    assert caught.value is error


def test_other_targets_keep_their_default_behavior():
    def factory(*, local_files_only=False):
        return local_files_only

    assert ConfigNode({"_target_": factory}).instantiate() is False
