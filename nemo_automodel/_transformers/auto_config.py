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

"""Hugging Face config loading through Transformers' native revision resolution."""

import os
import re
from pathlib import Path
from typing import Any

from huggingface_hub import ResolvedRevision, hf_hub_download
from transformers import AutoConfig, PretrainedConfig
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import extract_commit_hash, resolve_revision


class NeMoAutoConfig(AutoConfig):
    """Load configs from one snapshot even when native revision resolution fails.

    For a config followed by a separate model or tokenizer load, resolve the
    revision once with ``NeMoAutoConfig.resolve_revision`` and pass it
    to every load as ``revision``. Config objects do not carry Hub loading state.
    """

    @staticmethod
    def resolve_revision(
        pretrained_model_name_or_path: str | os.PathLike | None,
        revision: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        subfolder: str = "",
        force_download: bool = False,
    ) -> str | None:
        """Resolve one snapshot for config, weights, and associated metadata.

        Args:
            pretrained_model_name_or_path: Hub repository or local source; None skips resolution.
            revision: Requested branch, tag, commit, or already resolved revision.
            cache_dir: Hugging Face cache directory.
            token: Hub authentication token or token-discovery policy.
            local_files_only: Whether to use only cached files.
            subfolder: Checkpoint subdirectory containing config.json.
            force_download: Whether to refresh the config file used by fallback resolution.

        Returns:
            The resolved Hub revision, or the original revision for a local source.
        """
        if pretrained_model_name_or_path is None or os.path.exists(pretrained_model_name_or_path):
            return revision
        revision = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        commit_hash = revision.resolved if isinstance(revision, ResolvedRevision) else revision
        if isinstance(commit_hash, str) and re.fullmatch(r"[0-9a-f]{40}", commit_hash):
            return (
                revision
                if isinstance(revision, ResolvedRevision)
                else ResolvedRevision(
                    resolved=commit_hash, initial=revision, repo_id=os.fspath(pretrained_model_name_or_path)
                )
            )
        # Resolution is best-effort. Pin the actual downloaded snapshot when
        # the revision API fails, and return it to every subsequent loader.
        requested_revision = revision.initial if isinstance(revision, ResolvedRevision) else revision
        config_file = hf_hub_download(
            os.fspath(pretrained_model_name_or_path),
            CONFIG_NAME,
            revision=requested_revision,
            cache_dir=cache_dir,
            subfolder=subfolder,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        commit_hash = extract_commit_hash(config_file, None)
        if commit_hash is None:
            raise ValueError(f"Could not resolve the Hub snapshot for {pretrained_model_name_or_path!r}")
        return ResolvedRevision(
            resolved=commit_hash, initial=requested_revision, repo_id=os.fspath(pretrained_model_name_or_path)
        )

    @classmethod
    def _pin_revision(cls, pretrained_model_name_or_path: str | os.PathLike, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = kwargs.copy()
        kwargs["revision"] = cls.resolve_revision(
            pretrained_model_name_or_path,
            kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token", kwargs.get("use_auth_token")),
            local_files_only=kwargs.get("local_files_only", False),
            subfolder=kwargs.get("subfolder", ""),
            force_download=kwargs.get("force_download", False),
        )
        return kwargs

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]:
        """Load a config without re-reading a concurrently updated Hub branch ref.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides.

        Returns:
            The config, or the config and unused options when requested.
        """
        kwargs = cls._pin_revision(pretrained_model_name_or_path, kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the config dictionary for Automodel's custom-config registry.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides.

        Returns:
            The config dictionary and unused loading options.
        """
        kwargs = cls._pin_revision(pretrained_model_name_or_path, kwargs)
        return PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
